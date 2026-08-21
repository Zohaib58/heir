#!/usr/bin/env python3
"""Run manifest-defined HEIR-generated networks on CROSS."""

from __future__ import annotations

import argparse
import ast
import datetime as datetime_lib
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "benchmark_models.json"
INTEGRITY_OPERATIONS = (
    "matvec",
    "add",
    "add_plain",
    "multiply",
    "square",
    "relinearize",
    "rescale",
    "rotate",
    "plaintext_ciphertext_multiply",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Benchmark a manifest-defined HEIR-generated CROSS model."
  )
  parser.add_argument("--model", required=True)
  parser.add_argument(
      "--device", choices=("auto", "cpu", "tpu"), default="auto"
  )
  parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument(
      "--output-dir", type=Path, default=Path("benchmark-results")
  )
  parser.add_argument("--abs-tol", type=float)
  parser.add_argument("--rel-tol", type=float)
  parser.add_argument("--profile-operations", action="store_true")
  parser.add_argument(
      "--profile-only",
      action="store_true",
      help=(
          "run exactly one instrumented inference and no clean timing"
          " iterations"
      ),
  )
  parser.add_argument("--profile-callsites", action="store_true")
  parser.add_argument("--xprof", action="store_true")
  parser.add_argument(
      "--xprof-only",
      action="store_true",
      help=(
          "capture a bounded encrypted-compute trace, then exit without"
          " correctness"
      ),
  )
  parser.add_argument(
      "--xprof-duration-seconds",
      type=float,
      default=30.0,
      help=(
          "maximum XProf capture duration within encrypted compute"
          " (default: 30)"
      ),
  )
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--seed", type=int)
  parser.add_argument(
      "--validate-only",
      action="store_true",
      help=(
          "validate manifest, generated API, parameters, and integrity without"
          " execution"
      ),
  )
  args = parser.parse_args(argv)
  if args.warmups < 0 or args.repeats <= 0:
    parser.error("--warmups must be nonnegative and --repeats must be positive")
  if args.abs_tol is not None and args.abs_tol < 0:
    parser.error("--abs-tol must be nonnegative")
  if args.rel_tol is not None and args.rel_tol < 0:
    parser.error("--rel-tol must be nonnegative")
  if args.xprof_duration_seconds <= 0:
    parser.error("--xprof-duration-seconds must be positive")
  if args.profile_only:
    args.profile_operations = True
    if args.xprof:
      parser.error("--profile-only and --xprof must be separate executions")
  if args.xprof_only:
    if args.profile_only or args.profile_operations:
      parser.error(
          "--xprof-only and operation profiling must be separate executions"
      )
    args.xprof = True
  return args


def resolve_path(path: str, manifest_path: Path) -> Path:
  candidate = Path(path).expanduser()
  if not candidate.is_absolute():
    candidate = manifest_path.resolve().parent / candidate
  return candidate.resolve()


def validate_model_config(
    model_id: str, config: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
  if not config.get("enabled", False):
    reason = config.get("disabled_reason", "no reason supplied")
    raise ValueError(f"model {model_id!r} is disabled: {reason}")
  required = (
      "mlir_path",
      "generated_python_path",
      "pristine_python_path",
      "entry_point",
      "adapter",
      "input_shape",
      "output_shape",
      "trained_weight_source",
      "seed",
      "correctness",
      "crypto",
      "computation_sha256",
  )
  missing = [name for name in required if name not in config]
  if missing:
    raise ValueError(f"model {model_id!r} is missing fields: {missing}")
  for field in ("mlir_path", "generated_python_path", "pristine_python_path"):
    config[field] = str(resolve_path(config[field], manifest_path))
    if not Path(config[field]).is_file():
      raise FileNotFoundError(f"{field} does not exist: {config[field]}")
  crypto = config["crypto"]
  crypto_required = (
      "ring_degree",
      "slot_count",
      "q_towers",
      "p_towers",
      "scaling_factor",
      "output_scale",
      "dnum",
      "r",
      "c",
      "batch",
      "composite_degree",
  )
  missing_crypto = [name for name in crypto_required if name not in crypto]
  if missing_crypto:
    raise ValueError(f"model {model_id!r} crypto is missing: {missing_crypto}")
  degree = int(crypto["ring_degree"])
  slots = int(crypto["slot_count"])
  if degree != 2 * slots:
    raise ValueError(f"ring_degree={degree} must equal 2 * slot_count={slots}")
  if int(crypto["r"]) * int(crypto["c"]) != degree:
    raise ValueError("crypto r * c must equal ring_degree")
  if degree <= 0 or degree & (degree - 1):
    raise ValueError("ring_degree must be a positive power of two")
  if not crypto["q_towers"] or not crypto["p_towers"]:
    raise ValueError("Q and P towers must be nonempty")
  if len(config["computation_sha256"]) != 64:
    raise ValueError("computation_sha256 must be a SHA-256 hex digest")
  return config


def load_model_config(model_id: str, manifest_path: Path) -> dict[str, Any]:
  data = json.loads(manifest_path.read_text())
  if data.get("schema_version") != 1:
    raise ValueError("unsupported benchmark manifest schema")
  try:
    config = dict(data["models"][model_id])
  except KeyError as exc:
    available = ", ".join(sorted(data.get("models", {})))
    raise ValueError(
        f"unknown model {model_id!r}; available: {available}"
    ) from exc
  if "crypto_profile" in config:
    catalog_path = resolve_path(data["parameter_catalog"], manifest_path)
    catalog = json.loads(catalog_path.read_text())
    try:
      profile = dict(catalog["profiles"][config["crypto_profile"]])
    except KeyError as exc:
      raise ValueError(
          f"unknown crypto profile {config['crypto_profile']!r}"
      ) from exc
    profile["scaling_factor"] = float(profile["scaling_factor"])
    profile.setdefault("output_scale", profile["scaling_factor"])
    profile["output_scale"] = float(profile["output_scale"])
    profile.setdefault("batch", 1)
    profile["ring_degree"] = profile.pop("degree")
    config["crypto"] = profile
  return validate_model_config(model_id, config, manifest_path)


def validate_generated_api(path: Path, entry: str) -> None:
  tree = ast.parse(path.read_text(), filename=str(path))
  functions = {
      node.name for node in tree.body if isinstance(node, ast.FunctionDef)
  }
  required = {
      f"{entry}__generate_crypto_context",
      f"{entry}__configure_crypto_context",
      f"{entry}__encrypt__arg0",
      f"{entry}__preprocessing",
      f"{entry}__preprocessed",
      f"{entry}__decrypt__result0",
  }
  missing = sorted(required - functions)
  if missing:
    raise ValueError(f"generated module is missing functions: {missing}")


def generated_decode_slot_count(path: Path, entry: str) -> int:
  function = _function_node(path, f"{entry}__decrypt__result0")
  candidates = []
  for node in ast.walk(function):
    if not isinstance(node, ast.Call) or not isinstance(
        node.func, ast.Attribute
    ):
      continue
    if node.func.attr != "reshape" or len(node.args) < 2:
      continue
    decoded = node.func.value
    if not isinstance(decoded, ast.Attribute) or decoded.attr != "real":
      continue
    decoded = decoded.value
    if not isinstance(decoded, ast.Call) or not isinstance(
        decoded.func, ast.Attribute
    ):
      continue
    if decoded.func.attr != "decode":
      continue
    last_dimension = node.args[-1]
    if isinstance(last_dimension, ast.Constant) and isinstance(
        last_dimension.value, int
    ):
      candidates.append(last_dimension.value)
  if len(candidates) != 1:
    raise ValueError(
        f"expected one constant decode reshape in {path}, found {candidates}"
    )
  return candidates[0]


def validate_generated_slot_layout(
    path: Path, entry: str, slot_count: int
) -> None:
  generated_slots = generated_decode_slot_count(path, entry)
  if generated_slots != slot_count:
    raise ValueError(
        "generated decode layout does not match CKKS slot count: "
        f"generated={generated_slots}, expected={slot_count}. "
        "Lower with --torch-linalg-to-ckks=ciphertext-degree=<slot_count>, "
        "not the polynomial ring degree."
    )


def _function_node(path: Path, name: str) -> ast.FunctionDef:
  module = ast.parse(path.read_text(), filename=str(path))
  for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name == name:
      return node
  raise ValueError(f"generated function {name!r} not found in {path}")


def computation_fingerprint(path: Path, entry: str) -> dict[str, Any]:
  node = _function_node(path, f"{entry}__preprocessed")
  normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
  counts = {name: 0 for name in INTEGRITY_OPERATIONS}
  for child in ast.walk(node):
    if not isinstance(child, ast.Call) or not isinstance(
        child.func, ast.Attribute
    ):
      continue
    method = child.func.attr
    receiver = ast.dump(child.func.value, include_attributes=False)
    operation = None
    if method == "mul" and "ptct_mul" in receiver:
      operation = "plaintext_ciphertext_multiply"
    elif method == "mul" and "bsgs_matvec" in receiver:
      operation = "matvec"
    elif method in ("mul", "hemul_no_relin") and "he_mul" in receiver:
      operation = "multiply"
    elif method in ("square", "_square_array"):
      operation = "square"
    elif method in ("add", "add_plain", "relinearize", "rescale", "rotate"):
      operation = method
    if operation is not None:
      counts[operation] += 1
  return {
      "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
      "operation_counts": dict(sorted(counts.items())),
  }


def verify_computation_integrity(config: dict[str, Any]) -> dict[str, Any]:
  entry = config["entry_point"]
  pristine = computation_fingerprint(
      Path(config["pristine_python_path"]), entry
  )
  runnable = computation_fingerprint(
      Path(config["generated_python_path"]), entry
  )
  expected = config["computation_sha256"]
  if pristine != runnable:
    raise ValueError(
        "runnable generated computation differs from pristine computation"
    )
  if pristine["sha256"] != expected:
    raise ValueError(
        f"computation hash mismatch: manifest={expected},"
        f" actual={pristine['sha256']}"
    )
  return {"pristine": pristine, "runnable": runnable, "passed": True}


def summarize_samples(samples: list[float]) -> dict[str, Any]:
  return {
      "samples_ms": samples,
      "median_ms": statistics.median(samples),
      "min_ms": min(samples),
      "max_ms": max(samples),
      "stddev_ms": statistics.pstdev(samples),
  }


def json_safe(value: Any) -> Any:
  if isinstance(value, float) and not math.isfinite(value):
    return str(value)
  if isinstance(value, dict):
    return {key: json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(item) for item in value]
  return value


class OperationProfiler:
  """Intrusive operation profiler used only for a separate profile run."""

  def __init__(
      self, sync: Callable[[Any], None], collect_callsites: bool = False
  ):
    self.sync = sync
    self.collect_callsites = collect_callsites
    self.enabled = False
    self.phase = "unscoped"
    self.records: dict[str, dict[str, Any]] = {}
    self._restores: list[Callable[[], None]] = []
    self._wrapped: set[tuple[int, str]] = set()

  def wrap(self, cls: type, method: str, operation: str) -> None:
    key = (id(cls), method)
    if key in self._wrapped or not hasattr(cls, method):
      return
    original = getattr(cls, method)
    profiler = self

    def wrapped(*args, **kwargs):
      if not profiler.enabled:
        return original(*args, **kwargs)
      start = time.perf_counter()
      value = original(*args, **kwargs)
      profiler.sync(value)
      elapsed_ms = (time.perf_counter() - start) * 1000.0
      record_key = f"{profiler.phase}.{operation}"
      record = profiler.records.setdefault(
          record_key, {"count": 0, "total_ms": 0.0, "callsites": {}}
      )
      record["count"] += 1
      record["total_ms"] += elapsed_ms
      if profiler.collect_callsites:
        frame = next(
            (
                f
                for f in inspect.stack()[2:]
                if Path(f.filename) != Path(__file__)
            ),
            None,
        )
        site = (
            "<unknown>"
            if frame is None
            else f"{frame.filename}:{frame.lineno}:{frame.function}"
        )
        callsite = record["callsites"].setdefault(
            site, {"count": 0, "total_ms": 0.0}
        )
        callsite["count"] += 1
        callsite["total_ms"] += elapsed_ms
      return value

    setattr(cls, method, wrapped)
    self._wrapped.add(key)
    self._restores.append(
        lambda cls=cls, method=method, original=original: setattr(
            cls, method, original
        )
    )

  def uninstall(self) -> None:
    for restore in reversed(self._restores):
      restore()
    self._restores.clear()
    self._wrapped.clear()

  def report(self) -> dict[str, Any]:
    result = {}
    for key, record in sorted(self.records.items()):
      row = dict(record)
      row["average_ms"] = row["total_ms"] / row["count"]
      if not row["callsites"]:
        row.pop("callsites")
      result[key] = row
    return result


def import_runtime(cross_root: Path) -> dict[str, Any]:
  jaxite_word = str(cross_root / "jaxite_word")
  if jaxite_word not in sys.path:
    sys.path.insert(0, jaxite_word)
  import bsgs  # pylint: disable=import-outside-toplevel
  import ckks_ctx  # pylint: disable=import-outside-toplevel
  import he_ops  # pylint: disable=import-outside-toplevel
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  import key_gen  # pylint: disable=import-outside-toplevel
  import polynomial  # pylint: disable=import-outside-toplevel

  jax.config.update("jax_enable_x64", True)
  return locals()


def load_generated(path: Path, entry: str) -> Any:
  module_name = f"heir_benchmark_{entry}"
  spec = importlib.util.spec_from_file_location(module_name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load generated module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  required = (
      f"{entry}__generate_crypto_context",
      f"{entry}__configure_crypto_context",
      f"{entry}__encrypt__arg0",
      f"{entry}__preprocessing",
      f"{entry}__preprocessed",
      f"{entry}__decrypt__result0",
  )
  missing = [
      name for name in required if not callable(getattr(module, name, None))
  ]
  if missing:
    raise ValueError(f"generated module is missing functions: {missing}")
  return module


def patch_generated_module(module: Any, runtime: dict[str, Any]) -> None:
  original_full = module.jnp.full

  def full_allowing_none(shape, fill_value, *args, **kwargs):
    if fill_value is None:
      size = shape[0] if isinstance(shape, tuple) else shape
      return [None] * int(size)
    return original_full(shape, fill_value, *args, **kwargs)

  module.jnp.full = full_allowing_none

  def level_for_num_q(cache, num_q):
    for level in range(cache.max_level + 1):
      if cache.num_q_at_level(level) == num_q:
        return level
    raise ValueError(f"no CKKS level has {num_q} Q moduli")

  def ensure_poly(ctx, value, level=None):
    data = value.polynomial if hasattr(value, "polynomial") else value
    requested = (
        ctx._param_cache.num_q_at_level(level)
        if level is not None
        else data.shape[-1]
    )
    num_moduli = min(requested, data.shape[-1])
    actual_level = level_for_num_q(ctx._param_cache, num_moduli)
    result = module.Polynomial(
        {
            "batch": data.shape[0],
            "num_elements": data.shape[1],
            "degree": ctx.degree,
            "num_moduli": num_moduli,
            "precision": 32,
            "degree_layout": (ctx._param_cache.r, ctx._param_cache.c),
        },
        {"moduli": ctx._param_cache.q_moduli_at_level(actual_level)},
    )
    result.polynomial = data.reshape(
        data.shape[0],
        data.shape[1],
        ctx._param_cache.r,
        ctx._param_cache.c,
        data.shape[-1],
    )[..., :num_moduli]
    return result

  module._ensure_poly = ensure_poly


def sync_value(value: Any) -> None:
  if hasattr(value, "block_until_ready"):
    value.block_until_ready()
  elif hasattr(value, "polynomial"):
    sync_value(value.polynomial)
  elif isinstance(value, dict):
    for item in value.values():
      sync_value(item)
  elif isinstance(value, (list, tuple)):
    for item in value:
      sync_value(item)


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
  start = time.perf_counter()
  value = fn()
  sync_value(value)
  return value, (time.perf_counter() - start) * 1000.0


def unwrap(value: Any) -> Any:
  return value[0] if isinstance(value, (list, tuple)) else value


def check_context_parameters(ctx: Any, crypto: dict[str, Any]) -> None:
  actual = ctx.parameters
  expected = {
      "degree": crypto["ring_degree"],
      "num_slots": crypto["slot_count"],
      "q_towers": crypto["q_towers"],
      "p_towers": crypto["p_towers"],
      "scaling_factor": crypto["scaling_factor"],
      "dnum": crypto["dnum"],
      "r": crypto["r"],
      "c": crypto["c"],
      "batch": crypto["batch"],
      "composite_degree": crypto["composite_degree"],
  }
  mismatches = {
      key: {"manifest": value, "generated": actual.get(key)}
      for key, value in expected.items()
      if actual.get(key) != value
  }
  if mismatches:
    raise ValueError(
        f"generated crypto parameters do not match manifest: {mismatches}"
    )


def make_eval_key(raw: dict[str, Any], jnp: Any) -> list[Any]:
  return [
      jnp.array(raw["a"], dtype=jnp.uint32).transpose(0, 2, 1),
      jnp.array(raw["b"], dtype=jnp.uint32).transpose(0, 2, 1),
  ]


def level_for_num_q(cache: Any, num_q: int) -> int:
  for level in range(cache.max_level + 1):
    if cache.num_q_at_level(level) == num_q:
      return level
  raise ValueError(f"no CKKS level has {num_q} Q moduli")


class DynamicRotation:

  def __init__(self, base: Any, cache: Any, rotation: int):
    self.base = base
    self.cache = cache
    self.rotation = rotation

  def rotate(self, ciphertext: Any) -> Any:
    level = level_for_num_q(self.cache, ciphertext.polynomial.shape[-1])
    return self.base[level, self.rotation].rotate(ciphertext)


class DynamicRotationAccessor:

  def __init__(self, base: Any, cache: Any):
    self.base = base
    self.cache = cache

  def __getitem__(self, key: tuple[int, int]) -> DynamicRotation:
    _, rotation = key
    return DynamicRotation(self.base, self.cache, rotation)


class DynamicRescale:

  def __init__(self, base: Any, cache: Any, destination: int):
    self.base = base
    self.cache = cache
    self.destination = destination

  def rescale(self, ciphertext: Any) -> Any:
    source = level_for_num_q(self.cache, ciphertext.polynomial.shape[-1])
    if source == self.destination:
      return ciphertext
    if source < self.destination:
      raise ValueError(
          f"cannot rescale from actual level {source} to higher level "
          f"{self.destination}"
      )
    return self.base[source, self.destination].rescale(ciphertext)


class DynamicRescaleAccessor:

  def __init__(self, base: Any, cache: Any):
    self.base = base
    self.cache = cache

  def __getitem__(self, key: tuple[int, int]) -> DynamicRescale:
    _, destination = key
    return DynamicRescale(self.base, self.cache, destination)


def patch_level_accessors(ctx: Any) -> None:
  ctx._he_rot = DynamicRotationAccessor(ctx.he_rot, ctx._param_cache)
  ctx._he_rescale = DynamicRescaleAccessor(ctx.he_rescale, ctx._param_cache)


def prepare_execution(
    module: Any, config: dict[str, Any], runtime: dict[str, Any]
):
  entry = config["entry_point"]
  key_gen, jnp = runtime["key_gen"], runtime["jnp"]
  timings = {}
  ctx, timings["generate_crypto_context"] = timed(
      lambda: getattr(module, f"{entry}__generate_crypto_context")()
  )
  check_context_parameters(ctx, config["crypto"])
  params = ctx.parameters
  keys, timings["keygen_pke"] = timed(
      lambda: key_gen.gen_pke_pair(
          params["q_towers"], params["p_towers"], ctx.degree
      )
  )
  raw_eval, timings["keygen_eval"] = timed(
      lambda: key_gen.gen_evaluation_key(
          keys["secret_key"],
          q=params["q_towers"],
          P=params["p_towers"],
          noise_std=float(params.get("sigma", 3.190000057220459)),
          noise_scale=1,
          dnum=int(params["dnum"]),
      )
  )
  evaluation_key, timings["eval_key_layout"] = timed(
      lambda: make_eval_key(raw_eval, jnp)
  )
  _, timings["configure_crypto_context"] = timed(
      lambda: getattr(module, f"{entry}__configure_crypto_context")(
          ctx, keys["public_key"], keys["secret_key"], evaluation_key
      )
  )
  patch_level_accessors(ctx)
  return ctx, keys, timings


def decode(module, entry, runtime, ctx, keys, ciphertext, scale):
  old_scale = getattr(ctx, "output_scale", None)
  old_bypass = getattr(runtime["ckks_ctx"], "BYPASS_DECODE_STDDEV_CHECK", False)
  ctx.output_scale = scale
  runtime["ckks_ctx"].BYPASS_DECODE_STDDEV_CHECK = True
  try:
    value = getattr(module, f"{entry}__decrypt__result0")(
        ctx, {}, [ciphertext], keys["secret_key"]
    )
    return runtime["np"].asarray(value, dtype=runtime["np"].float64)
  finally:
    ctx.output_scale = old_scale
    runtime["ckks_ctx"].BYPASS_DECODE_STDDEV_CHECK = old_bypass


def execute_once(
    module,
    config,
    runtime,
    input_value,
    expected,
    profiler=None,
    xprof_path=None,
    xprof_options=None,
    xprof_duration_seconds=30.0,
    terminate_after_xprof=False,
    xprof_metadata_path=None,
):
  entry = config["entry_point"]
  ctx, keys, timings = prepare_execution(module, config, runtime)
  if profiler:
    profiler.phase = "online"
    profiler.enabled = True
  encrypted, timings["encrypt"] = timed(
      lambda: getattr(module, f"{entry}__encrypt__arg0")(
          ctx, {}, input_value, keys["public_key"]
      )
  )
  preprocessed, timings["preprocessing"] = timed(
      lambda: getattr(module, f"{entry}__preprocessing")(ctx, {})
  )
  args = (
      preprocessed
      if isinstance(preprocessed, (list, tuple))
      else (preprocessed,)
  )

  def compute():
    return unwrap(
        getattr(module, f"{entry}__preprocessed")(ctx, {}, encrypted, *args)
    )

  def run_compute():
    if xprof_path is None:
      return compute()
    trace_errors = []

    def stop_trace():
      try:
        runtime["jax"].profiler.stop_trace()
        if terminate_after_xprof:
          metadata = {
              "status": "CAPTURE_COMPLETE",
              "capture_duration_seconds": xprof_duration_seconds,
              "correctness": "skipped; validated in a separate benchmark run",
              "xprof_path": str(Path(xprof_path).resolve()),
          }
          Path(xprof_metadata_path).write_text(
              json.dumps(metadata, indent=2) + "\n"
          )
          print(
              "xprof_status=CAPTURE_COMPLETE"
              f" xprof_path={Path(xprof_path).resolve()}",
              flush=True,
          )
          print(
              f"xprof_metadata={Path(xprof_metadata_path).resolve()}",
              flush=True,
          )
          os._exit(0)
      except Exception as error:  # Propagate failures from the timer thread.
        trace_errors.append(error)

    runtime["jax"].profiler.start_trace(
        str(xprof_path), profiler_options=xprof_options
    )
    timer = threading.Timer(xprof_duration_seconds, stop_trace)
    timer.start()
    try:
      value = compute()
      sync_value(value)
    finally:
      if timer.is_alive():
        timer.cancel()
        stop_trace()
      else:
        timer.join()
    if trace_errors:
      raise RuntimeError(f"XProf capture failed: {trace_errors[0]}")
    return value

  ciphertext, timings["preprocessed_compute"] = timed(run_compute)
  output_scale = float(config["crypto"]["output_scale"])
  decoded, timings["decrypt"] = timed(
      lambda: decode(
          module, entry, runtime, ctx, keys, ciphertext, output_scale
      )
  )
  raw = decode(module, entry, runtime, ctx, keys, ciphertext, 1.0)
  denominator = float(runtime["np"].dot(expected, expected))
  calibrated_scale = (
      float(runtime["np"].dot(raw, expected) / denominator)
      if denominator
      else math.nan
  )
  if math.isfinite(calibrated_scale) and calibrated_scale != 0.0:
    calibrated = raw / calibrated_scale
  else:
    calibrated = runtime["np"].full(raw.shape, runtime["np"].nan)
  timings["online_total"] = sum(
      timings[name]
      for name in (
          "encrypt",
          "preprocessing",
          "preprocessed_compute",
          "decrypt",
      )
  )
  timings["output_scale"] = output_scale
  timings["calibrated_output_scale"] = calibrated_scale
  if profiler:
    profiler.enabled = False
  result = {
      "decoded": decoded,
      "calibrated_decoded": calibrated,
      "ciphertext_shape": list(ciphertext.polynomial.shape),
      "timings": timings,
  }
  return result


def correctness(
    actual,
    expected,
    abs_tol: float,
    rel_tol: float,
    require_argmax_match: bool = False,
) -> dict[str, Any]:
  actual = actual.reshape(expected.shape)
  finite = bool(
      (~(actual != actual)).all() and math.isfinite(float(abs(actual).max()))
  )
  nonzero = bool((actual != 0).any())
  max_error = float(abs(actual - expected).max()) if finite else math.inf
  relative = max_error / max(1.0, float(abs(expected).max()))
  actual_argmax = int(actual.argmax()) if finite else None
  expected_argmax = int(expected.argmax())
  argmax_match = actual_argmax == expected_argmax
  passed = (
      finite
      and nonzero
      and (max_error <= abs_tol or relative <= rel_tol)
      and (argmax_match or not require_argmax_match)
  )
  return {
      "passed": passed,
      "finite": finite,
      "nonzero": nonzero,
      "max_error": max_error,
      "relative_error": relative,
      "actual_argmax": actual_argmax,
      "expected_argmax": expected_argmax,
      "argmax_match": argmax_match,
      "actual": actual.tolist(),
      "expected": expected.tolist(),
  }


def install_operation_profiler(
    profiler: OperationProfiler, runtime: dict[str, Any]
) -> None:
  he_ops = runtime["he_ops"]
  classes = (
      (he_ops._HEAddAtLevel, (("add", "add"), ("add_plain", "add_plain"))),
      (
          he_ops._HEMulAtLevel,
          (
              ("mul", "multiply"),
              ("hemul_no_relin", "multiply_no_relinearize"),
              ("relinearize", "relinearize"),
              ("_square_array", "square"),
          ),
      ),
      (
          he_ops._HEPtCtMulAtLevel,
          (
              ("set_plaintext", "set_plaintext"),
              ("mul", "plaintext_ciphertext_multiply"),
          ),
      ),
      (he_ops._HERescaleAtLevels, (("rescale", "rescale"),)),
      (he_ops._HERotAtLevel, (("rotate", "rotate"),)),
      (runtime["bsgs"]._BSGSMatVecAtLevel, (("mul", "bsgs_matvec"),)),
      (
          runtime["polynomial"].Polynomial,
          (("__init__", "polynomial_allocation"),),
      ),
      (
          runtime["ckks_ctx"].CKKSContext,
          (
              ("encode", "encode"),
              ("encode_at_level", "encode_at_level"),
              ("encrypt", "encrypt"),
              ("decrypt", "decrypt"),
          ),
      ),
  )
  for cls, methods in classes:
    for method, name in methods:
      profiler.wrap(cls, method, name)


def environment_metadata(
    runtime: dict[str, Any], requested_device: str
) -> dict[str, Any]:
  jax = runtime["jax"]
  return {
      "requested_device": requested_device,
      "jax_backend": jax.default_backend(),
      "jax_devices": [str(device) for device in jax.devices()],
      "python": sys.version,
      "cross_root": str(
          Path(os.environ.get("CROSS_ROOT", "/home/zohaib/CROSS_dev")).resolve()
      ),
  }


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  if args.device != "auto":
    os.environ["JAX_PLATFORMS"] = args.device
  config = load_model_config(args.model, args.manifest)
  integrity = verify_computation_integrity(config)
  validate_generated_api(
      Path(config["generated_python_path"]), config["entry_point"]
  )
  validate_generated_slot_layout(
      Path(config["generated_python_path"]),
      config["entry_point"],
      int(config["crypto"]["slot_count"]),
  )
  if args.validate_only:
    print(f"validation_status=PASS model={args.model}")
    print(f"computation_sha256={integrity['runnable']['sha256']}")
    print(
        f"operation_counts={json.dumps(integrity['runnable']['operation_counts'], sort_keys=True)}"
    )
    return 0
  cross_root = Path(
      os.environ.get("CROSS_ROOT", "/home/zohaib/CROSS_dev")
  ).resolve()
  runtime = import_runtime(cross_root)
  runtime["np"] = __import__("numpy")
  if args.device != "auto" and runtime["jax"].default_backend() != args.device:
    raise RuntimeError(
        f"requested {args.device}, but JAX selected"
        f" {runtime['jax'].default_backend()}"
    )
  module = load_generated(
      Path(config["generated_python_path"]), config["entry_point"]
  )
  patch_generated_module(module, runtime)
  from benchmark_model_adapters import get_adapter

  seed = config["seed"] if args.seed is None else args.seed
  input_value, expected = get_adapter(config["adapter"])(seed, str(cross_root))
  if (
      list(input_value.shape) != config["input_shape"]
      or list(expected.shape) != config["output_shape"]
  ):
    raise ValueError("adapter input/output shapes do not match the manifest")
  abs_tol = (
      config["correctness"]["abs_tol"] if args.abs_tol is None else args.abs_tol
  )
  rel_tol = (
      config["correctness"]["rel_tol"] if args.rel_tol is None else args.rel_tol
  )

  timestamp = datetime_lib.datetime.now(datetime_lib.timezone.utc).strftime(
      "%Y%m%dT%H%M%SZ"
  )
  run_dir = (
      args.output_dir
      / f"{args.model}-{config['crypto']['ring_degree']}-{runtime['jax'].default_backend()}-{timestamp}"
  )
  run_dir.mkdir(parents=True, exist_ok=False)
  xprof_path = run_dir / "xprof" if args.xprof else None
  xprof_options = None
  if args.xprof:
    xprof_options = runtime["jax"].profiler.ProfileOptions()
    xprof_options.python_tracer_level = 0
    xprof_options.host_tracer_level = 1
    if runtime["jax"].default_backend() == "tpu":
      xprof_options.advanced_configuration = {
          "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
          "tpu_num_chips_to_profile_per_task": 1,
      }

  operation_profile = None
  if args.profile_only:
    profiler = OperationProfiler(sync_value, args.profile_callsites)
    install_operation_profiler(profiler, runtime)
    try:
      clean_runs = [
          execute_once(
              module, config, runtime, input_value, expected, profiler=profiler
          )
      ]
    finally:
      profiler.enabled = False
      profiler.uninstall()
    operation_profile = profiler.report()
  elif args.xprof_only:
    clean_runs = [
        execute_once(
            module,
            config,
            runtime,
            input_value,
            expected,
            xprof_path=xprof_path,
            xprof_options=xprof_options,
            xprof_duration_seconds=args.xprof_duration_seconds,
            terminate_after_xprof=True,
            xprof_metadata_path=run_dir / "xprof_capture.json",
        )
    ]
  else:
    for _ in range(args.warmups):
      execute_once(module, config, runtime, input_value, expected)
    clean_runs = [
        execute_once(module, config, runtime, input_value, expected)
        for _ in range(args.repeats)
    ]
    if args.xprof:
      execute_once(
          module,
          config,
          runtime,
          input_value,
          expected,
          xprof_path=xprof_path,
          xprof_options=xprof_options,
          xprof_duration_seconds=args.xprof_duration_seconds,
      )
  timing_names = clean_runs[0]["timings"].keys()
  timings = {
      name: summarize_samples([run["timings"][name] for run in clean_runs])
      for name in timing_names
      if not name.endswith("scale")
  }
  require_argmax = config["correctness"].get("require_argmax_match", False)
  ordinary = correctness(
      clean_runs[-1]["decoded"], expected, abs_tol, rel_tol, require_argmax
  )
  calibrated = correctness(
      clean_runs[-1]["calibrated_decoded"],
      expected,
      abs_tol,
      rel_tol,
      require_argmax,
  )

  if args.profile_operations and not args.profile_only:
    profiler = OperationProfiler(sync_value, args.profile_callsites)
    install_operation_profiler(profiler, runtime)
    try:
      execute_once(
          module, config, runtime, input_value, expected, profiler=profiler
      )
    finally:
      profiler.enabled = False
      profiler.uninstall()
    operation_profile = profiler.report()

  result = {
      "model": args.model,
      "timestamp_utc": timestamp,
      "manifest": str(args.manifest.resolve()),
      "configuration": config,
      "environment": environment_metadata(runtime, args.device),
      "iterations": {
          "warmups": (
              0 if (args.profile_only or args.xprof_only) else args.warmups
          ),
          "repeats": (
              1 if (args.profile_only or args.xprof_only) else args.repeats
          ),
      },
      "measurement_mode": (
          "operation_profile"
          if args.profile_only
          else "xprof"
          if args.xprof_only
          else "clean"
      ),
      "integrity": integrity,
      "correctness": {"ordinary": ordinary, "calibrated": calibrated},
      "ciphertext_shape": clean_runs[-1]["ciphertext_shape"],
      "timings": timings,
      "scales": {
          "output": [run["timings"]["output_scale"] for run in clean_runs],
          "calibrated": [
              run["timings"]["calibrated_output_scale"] for run in clean_runs
          ],
      },
      "operation_profile": operation_profile,
      "xprof_path": None if xprof_path is None else str(xprof_path.resolve()),
      "xprof_duration_seconds": (
          args.xprof_duration_seconds if args.xprof else None
      ),
  }
  result_path = run_dir / "result.json"
  result_path.write_text(
      json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n"
  )
  print(f"result_json={result_path.resolve()}")
  print(
      f"ordinary_status={'PASS' if ordinary['passed'] else 'FAIL'} max_error={ordinary['max_error']:.6e}"
  )
  print(
      f"calibrated_status={'PASS' if calibrated['passed'] else 'FAIL'} max_error={calibrated['max_error']:.6e}"
  )
  print(f"compute_median_ms={timings['preprocessed_compute']['median_ms']:.3f}")
  return 0 if ordinary["passed"] and calibrated["passed"] else 1


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (FileNotFoundError, RuntimeError, ValueError) as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(2) from error
