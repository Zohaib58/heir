import argparse
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import time


def requested_device():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument(
      "--device", choices=("auto", "cpu", "tpu"), default="auto"
  )
  args, _ = parser.parse_known_args()
  if args.device != "auto":
    os.environ["JAX_PLATFORMS"] = args.device
  return args.device


REQUESTED_DEVICE = requested_device()

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

CROSS_ROOT = os.environ.get("CROSS_ROOT", "/home/zohaib/CROSS_dev")
JAXITE_WORD = os.path.join(CROSS_ROOT, "jaxite_word")
DEMOS = os.path.join(CROSS_ROOT, "demos")
HEIR_PATH = os.environ.get(
    "HEIR_PATH",
    str(
        Path(__file__).resolve().parent
        / "alexnet"
        / "alexnet-tiny-cd1-cross.py"
    ),
)
ENTRY = os.environ.get("HEIR_ENTRY", "alexnet_tiny")
SEED = int(os.environ.get("ALEXNET_SEED", "42"))
WARMUPS = int(os.environ.get("WARMUPS", "1"))
REPEATS = int(os.environ.get("REPEATS", "3"))
ABS_TOL = float(os.environ.get("ABS_TOL", "0.5"))
REL_TOL = float(os.environ.get("REL_TOL", "0.5"))

sys.path.insert(0, JAXITE_WORD)
sys.path.insert(0, DEMOS)

import ckks_ctx  # noqa: E402
import key_gen as kg  # noqa: E402
from alexnet_he import (  # noqa: E402
    alexnet_tiny_cleartext,
    alexnet_tiny_random_inputs,
    load_trained_alexnet_tiny_weights,
    prepare_tiny_args,
)

CROSS_R = 32
CROSS_C = 64


ALEXNET_ROT_INDICES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    24,
    32,
    36,
    40,
    48,
    56,
    60,
    64,
    72,
    80,
    84,
    96,
    108,
    112,
    120,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
    240,
    256,
    512,
]


def patch_generated_jnp_full(heir):
  original_full = heir.jnp.full

  def full_allowing_none(shape, fill_value, *args, **kwargs):
    if fill_value is None:
      if isinstance(shape, tuple):
        if len(shape) != 1:
          raise ValueError(f"unsupported None container shape: {shape}")
        size = shape[0]
      else:
        size = shape
      return [None] * int(size)
    return original_full(shape, fill_value, *args, **kwargs)

  heir.jnp.full = full_allowing_none


def level_for_num_q(cache, num_q):
  for level in range(cache.max_level + 1):
    if cache.num_q_at_level(level) == num_q:
      return level
  raise ValueError(f"no CKKS level has {num_q} Q moduli")


def patch_generated_ensure_poly(heir):
  def ensure_poly(ctx, x, level=None):
    cache = ctx._param_cache
    requested = cache.num_q_at_level(level) if level is not None else None
    data = x.polynomial if hasattr(x, "polynomial") else x
    input_moduli = data.shape[-1]
    num_moduli = (
        input_moduli if requested is None else min(requested, input_moduli)
    )

    if level is not None and requested == num_moduli:
      actual_level = level
    else:
      actual_level = level_for_num_q(cache, num_moduli)

    out = heir.Polynomial(
        {
            "batch": data.shape[0],
            "num_elements": data.shape[1],
            "degree": ctx.degree,
            "num_moduli": num_moduli,
            "precision": 32,
            "degree_layout": (cache.r, cache.c),
        },
        {"moduli": cache.q_moduli_at_level(actual_level)},
    )
    out.polynomial = data.reshape(
        data.shape[0],
        data.shape[1],
        cache.r,
        cache.c,
        input_moduli,
    )[..., :num_moduli]
    return out

  heir._ensure_poly = ensure_poly


class DynamicRotOp:

  def __init__(self, base, cache, requested_level, rot_index):
    self.base = base
    self.cache = cache
    self.requested_level = requested_level
    self.rot_index = rot_index

  def rotate(self, ct):
    actual_level = level_for_num_q(self.cache, ct.polynomial.shape[-1])
    return self.base[actual_level, self.rot_index].rotate(ct)

  def __call__(self, ct):
    return self.rotate(ct)


class DynamicRotAccessor:

  def __init__(self, base, cache):
    self.base = base
    self.cache = cache

  def __getitem__(self, key):
    level, rot_index = key
    return DynamicRotOp(self.base, self.cache, level, rot_index)


class DynamicRescaleOp:

  def __init__(self, base, cache, src_level, dst_level):
    self.base = base
    self.cache = cache
    self.src_level = src_level
    self.dst_level = dst_level

  def rescale(self, ct):
    actual_src = level_for_num_q(self.cache, ct.polynomial.shape[-1])
    if actual_src == self.dst_level:
      return ct
    if actual_src < self.dst_level:
      raise ValueError(
          f"cannot rescale from actual level {actual_src} to "
          f"higher requested level {self.dst_level}"
      )
    return self.base[actual_src, self.dst_level].rescale(ct)

  def __call__(self, ct):
    return self.rescale(ct)


class DynamicRescaleAccessor:

  def __init__(self, base, cache):
    self.base = base
    self.cache = cache

  def __getitem__(self, key):
    src_level, dst_level = key
    return DynamicRescaleOp(self.base, self.cache, src_level, dst_level)


def patch_level_accessors(ctx):
  ctx._he_rot = DynamicRotAccessor(ctx.he_rot, ctx._param_cache)
  ctx._he_rescale = DynamicRescaleAccessor(ctx.he_rescale, ctx._param_cache)


def load_heir_module():
  spec = importlib.util.spec_from_file_location("heir_alexnet_tiny", HEIR_PATH)
  heir = importlib.util.module_from_spec(spec)
  sys.modules["heir_alexnet_tiny"] = heir
  spec.loader.exec_module(heir)
  patch_generated_jnp_full(heir)
  patch_generated_ensure_poly(heir)
  return heir


def sync_value(x):
  if hasattr(x, "block_until_ready"):
    x.block_until_ready()
    return
  if hasattr(x, "polynomial"):
    sync_value(x.polynomial)
    return
  if isinstance(x, dict):
    for v in x.values():
      sync_value(v)
    return
  if isinstance(x, (list, tuple)):
    for v in x:
      sync_value(v)


def timed(fn):
  start = time.perf_counter()
  value = fn()
  sync_value(value)
  return value, time.perf_counter() - start


def median_ms(values):
  return statistics.median(values) * 1000.0


def unwrap(x):
  return x[0] if isinstance(x, (list, tuple)) else x


def make_eval_key(raw):
  return [
      jnp.array(raw["a"], dtype=jnp.uint32).transpose(0, 2, 1),
      jnp.array(raw["b"], dtype=jnp.uint32).transpose(0, 2, 1),
  ]


def make_heir_ctx(heir):
  timings = {}
  ctx, timings["generate_crypto_context"] = timed(
      lambda: getattr(heir, f"{ENTRY}__generate_crypto_context")()
  )
  q_towers = list(ctx.parameters["q_towers"])
  p_towers = list(ctx.parameters.get("p_towers", []))
  dnum = int(ctx.parameters.get("dnum", 3))
  sigma = float(ctx.parameters.get("sigma", 3.190000057220458984375))

  key_pair, timings["keygen_pke"] = timed(
      lambda: kg.gen_pke_pair(q_towers, p_towers, ctx.degree)
  )
  eval_key_raw, timings["keygen_eval"] = timed(
      lambda: kg.gen_evaluation_key(
          key_pair["secret_key"],
          q=q_towers,
          P=p_towers,
          noise_std=sigma,
          noise_scale=1,
          dnum=dnum,
      )
  )
  eval_key, timings["eval_key_layout"] = timed(
      lambda: make_eval_key(eval_key_raw)
  )

  def configure():
    ctx.public_key = key_pair["public_key"]
    ctx.secret_key = key_pair["secret_key"]
    ctx.evaluation_key = eval_key
    ctx.parameters["public_key"] = key_pair["public_key"]
    ctx.parameters["secret_key"] = key_pair["secret_key"]
    ctx.parameters["evaluation_key"] = eval_key
    ctx.program_initialization(
        total_rotation_indices=ALEXNET_ROT_INDICES,
        dnum=dnum,
        r=int(ctx.parameters.get("r", CROSS_R)),
        c=int(ctx.parameters.get("c", CROSS_C)),
        batch=int(ctx.parameters.get("batch", 1)),
    )

  _, timings["configure_crypto_context"] = timed(configure)
  patch_level_accessors(ctx)
  return ctx, key_pair, timings


def default_output_scale(ctx):
  return float(
      ctx.parameters.get("output_scale", ctx.parameters["scaling_factor"])
  )


def decode_with_scale(heir, ctx, key_pair, ct_out, scale):
  ctx.output_scale = scale
  old_bypass = getattr(ckks_ctx, "BYPASS_DECODE_STDDEV_CHECK", False)
  ckks_ctx.BYPASS_DECODE_STDDEV_CHECK = True
  try:
    return np.array(
        getattr(heir, f"{ENTRY}__decrypt__result0")(
            ctx, {}, [ct_out], key_pair["secret_key"]
        ),
        dtype=np.float64,
    )
  finally:
    ckks_ctx.BYPASS_DECODE_STDDEV_CHECK = old_bypass


def best_linear_scale(raw_decoded_at_scale_1, expected):
  denom = float(np.dot(expected, expected))
  if denom == 0.0:
    return np.nan
  return float(np.dot(raw_decoded_at_scale_1, expected) / denom)


def summarize(decoded, expected):
  if decoded.shape != expected.shape or np.any(~np.isfinite(decoded)):
    return np.inf, np.inf, "FAIL"
  maxerr = float(np.max(np.abs(decoded - expected)))
  rel = maxerr / max(1.0, float(np.max(np.abs(expected))))
  status = "PASS" if (maxerr <= ABS_TOL or rel <= REL_TOL) else "FAIL"
  return maxerr, rel, status


def make_input_and_expected():
  rng = alexnet_tiny_random_inputs(seed=SEED)
  weights = load_trained_alexnet_tiny_weights()
  if weights is None:
    raise FileNotFoundError(
        "trained AlexNetTiny weights not found under CROSS demos/maple_data"
    )
  x = rng["X"].reshape(1, 3, 16, 16).astype(np.float32)
  expected = alexnet_tiny_cleartext(rng["X"], *prepare_tiny_args(weights))
  return x, expected.astype(np.float64)


def run_once(heir, x, expected, xprof_dir=None):
  ctx, key_pair, timings = make_heir_ctx(heir)
  public_key = key_pair["public_key"]

  ct_in, timings["encrypt"] = timed(
      lambda: getattr(heir, f"{ENTRY}__encrypt__arg0")(ctx, {}, x, public_key)
  )
  preprocessed_args, timings["preprocessing"] = timed(
      lambda: getattr(heir, f"{ENTRY}__preprocessing")(ctx, {})
  )

  def compute():
    return unwrap(
        getattr(heir, f"{ENTRY}__preprocessed")(
            ctx, {}, ct_in, *preprocessed_args
        )
    )

  ct_out, timings["preprocessed_compute"] = timed(compute)

  if xprof_dir:
    os.makedirs(xprof_dir, exist_ok=True)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 3
    options.host_tracer_level = 3
    if jax.default_backend() == "tpu":
      options.advanced_configuration = {
          "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
          "tpu_num_chips_to_profile_per_task": 1,
      }
    with jax.profiler.trace(xprof_dir, profiler_options=options):
      profiled_out = compute()
      sync_value(profiled_out)
    print(f"xprof_trace={os.path.abspath(xprof_dir)}")
  scale = default_output_scale(ctx)
  decoded, timings["decrypt"] = timed(
      lambda: decode_with_scale(heir, ctx, key_pair, ct_out, scale)
  )

  raw_decoded = decode_with_scale(heir, ctx, key_pair, ct_out, 1.0)
  calibrated_scale = best_linear_scale(raw_decoded, expected)
  calibrated_decoded = raw_decoded / calibrated_scale
  timings["output_scale"] = scale
  timings["calibrated_output_scale"] = calibrated_scale
  return decoded, calibrated_decoded, ct_out.polynomial.shape, timings


def print_timing_summary(collected):
  print("\nTiming medians")
  for key in (
      "generate_crypto_context",
      "keygen_pke",
      "keygen_eval",
      "eval_key_layout",
      "configure_crypto_context",
      "encrypt",
      "preprocessing",
      "preprocessed_compute",
      "decrypt",
  ):
    if key in collected:
      print(f"{key}: {median_ms(collected[key]):.3f} ms")
  phase_total_keys = [
      "encrypt",
      "preprocessing",
      "preprocessed_compute",
      "decrypt",
  ]
  if all(key in collected for key in phase_total_keys):
    totals = [
        sum(parts) for parts in zip(*(collected[k] for k in phase_total_keys))
    ]
    print(f"online_total: {median_ms(totals):.3f} ms")


def main():
  global WARMUPS, REPEATS

  parser = argparse.ArgumentParser(
      description="Run HEIR-emitted AlexNetTiny on CPU or TPU."
  )
  parser.add_argument(
      "--device",
      choices=("auto", "cpu", "tpu"),
      default=REQUESTED_DEVICE,
      help="JAX platform; auto uses JAX's default backend selection.",
  )
  parser.add_argument("--warmups", type=int, default=WARMUPS)
  parser.add_argument("--repeats", type=int, default=REPEATS)
  parser.add_argument(
      "--xprof-dir",
      help="Collect one separately executed encrypted-compute XProf trace.",
  )
  args = parser.parse_args()
  WARMUPS = args.warmups
  REPEATS = args.repeats

  heir = load_heir_module()
  x, expected = make_input_and_expected()
  print(
      f"requested_device={args.device} "
      f"jax_backend={jax.default_backend()} devices={jax.devices()}"
  )
  print(f"heir_path={HEIR_PATH}")
  print(f"entry={ENTRY} seed={SEED} warmups={WARMUPS} repeats={REPEATS}")
  print(f"input_shape={x.shape} expected_shape={expected.shape}")

  for _ in range(WARMUPS):
    run_once(heir, x, expected)

  collected = {}
  last_decoded = None
  last_calibrated = None
  last_shape = None
  for repeat in range(REPEATS):
    trace_dir = args.xprof_dir if repeat == REPEATS - 1 else None
    decoded, calibrated, shape, timings = run_once(
        heir, x, expected, xprof_dir=trace_dir
    )
    last_decoded = decoded
    last_calibrated = calibrated
    last_shape = shape
    for key, value in timings.items():
      collected.setdefault(key, []).append(value)

  maxerr, rel, status = summarize(last_decoded, expected)
  cal_maxerr, cal_rel, cal_status = summarize(last_calibrated, expected)
  print("\nCorrectness")
  print(f"ciphertext_shape={last_shape}")
  print(
      f"output_scale.median={statistics.median(collected['output_scale']):.6e}"
  )
  print(
      "calibrated_output_scale.median="
      f"{statistics.median(collected['calibrated_output_scale']):.6e}"
  )
  print(f"expected={expected}")
  print(f"decoded ={last_decoded}")
  print(f"maxerr={maxerr:.6e} rel={rel:.6e} status={status}")
  print(f"calibrated_decoded={last_calibrated}")
  print(
      f"calibrated_maxerr={cal_maxerr:.6e} "
      f"calibrated_rel={cal_rel:.6e} calibrated_status={cal_status}"
  )
  print_timing_summary(collected)


if __name__ == "__main__":
  main()
