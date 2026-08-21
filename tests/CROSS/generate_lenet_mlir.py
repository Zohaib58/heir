#!/usr/bin/env python3
"""Generate CROSS-compatible LeNet MLIR from trained weight binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


Q_TOWERS = [
    536903681,
    536924161,
    536952833,
    536973313,
    536977409,
    536989697,
    537026561,
    537047041,
    537071617,
]
P_TOWERS = [2147565569, 2147573761, 2147577857, 2147721217]

WEIGHT_FILES = {
    "W1": ("lenet_conv1_W.bin", (4, 1, 5, 5)),
    "b1": ("lenet_conv1_b.bin", (4,)),
    "W2": ("lenet_conv2_W.bin", (8, 4, 5, 5)),
    "b2": ("lenet_conv2_b.bin", (8,)),
    "W3": ("lenet_fc1_W.bin", (32, 392)),
    "b3": ("lenet_fc1_b.bin", (32,)),
    "W4": ("lenet_fc2_W.bin", (10, 32)),
    "b4": ("lenet_fc2_b.bin", (10,)),
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def load_weights(weights_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
  weights = {}
  sources = {}
  errors = []
  for name, (filename, shape) in WEIGHT_FILES.items():
    path = weights_dir / filename
    expected_values = int(np.prod(shape))
    if not path.is_file():
      errors.append(f"missing {path}")
      continue
    values = np.fromfile(path, dtype=np.float64)
    if values.size != expected_values:
      errors.append(
          f"{path}: expected {expected_values} float64 values, got"
          f" {values.size}"
      )
      continue
    if not np.all(np.isfinite(values)):
      errors.append(f"{path}: contains non-finite values")
      continue
    weights[name] = values.reshape(shape).astype(np.float32)
    sources[filename] = {
        "shape": list(shape),
        "values": expected_values,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
  if errors:
    raise ValueError(
        "invalid LeNet weight directory:\n  " + "\n  ".join(errors)
    )
  return weights, sources


def import_lenet(cross_root: Path):
  demos = str(cross_root / "demos")
  if demos not in sys.path:
    sys.path.insert(0, demos)
  import lenet_he  # pylint: disable=import-outside-toplevel

  return lenet_he


def build_constants(
    weights: dict[str, np.ndarray], lenet_he
) -> dict[str, np.ndarray]:
  w1 = weights["W1"].reshape(-1).astype(np.float64)
  w2 = weights["W2"].reshape(-1).astype(np.float64)
  matrix1 = lenet_he.conv_to_matrix(w1, 1, 4, 28, 28, 5, 5, 2, 2, 784)[
      :784, :784
  ]
  matrix2 = lenet_he.conv_to_matrix(w2, 4, 8, 14, 14, 5, 5, 2, 2, 784)[
      :392, :784
  ]
  return {
      "w1": matrix1.astype(np.float32),
      "b1": np.repeat(weights["b1"], 14 * 14).astype(np.float32),
      "w2": matrix2.astype(np.float32),
      "b2": np.repeat(weights["b2"], 7 * 7).astype(np.float32),
      "w3": weights["W3"].astype(np.float32),
      "b3": weights["b3"].astype(np.float32),
      "w4": weights["W4"].astype(np.float32),
      "b4": weights["b4"].astype(np.float32),
  }


def matrix_inference(constants: dict[str, np.ndarray], input_value: np.ndarray):
  values = {}
  values["layer1"] = constants["w1"] @ input_value + constants["b1"]
  values["square1"] = values["layer1"] * values["layer1"]
  values["layer2"] = constants["w2"] @ values["square1"] + constants["b2"]
  values["square2"] = values["layer2"] * values["layer2"]
  values["layer3"] = constants["w3"] @ values["square2"] + constants["b3"]
  values["square3"] = values["layer3"] * values["layer3"]
  values["logits"] = constants["w4"] @ values["square3"] + constants["b4"]
  return values


def verify_constants(
    weights: dict[str, np.ndarray],
    constants: dict[str, np.ndarray],
    lenet_he,
    seed: int = 42,
    trials: int = 3,
) -> dict:
  rng = np.random.default_rng(seed)
  max_error = 0.0
  trials_out = []
  for trial in range(trials):
    input_value = rng.normal(0.0, 0.25, 784).astype(np.float32)
    actual = matrix_inference(constants, input_value)["logits"]
    expected = lenet_he.lenet_cleartext(
        input_value.astype(np.float64),
        weights["W1"].reshape(-1).astype(np.float64),
        weights["W2"].reshape(-1).astype(np.float64),
        weights["W3"].reshape(-1).astype(np.float64),
        weights["W4"].reshape(-1).astype(np.float64),
        weights["b1"].astype(np.float64),
        weights["b2"].astype(np.float64),
        weights["b3"].astype(np.float64),
        weights["b4"].astype(np.float64),
    )
    error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    max_error = max(max_error, error)
    trials_out.append({"trial": trial, "max_abs_error": error})
  if max_error > 1e-4:
    raise ValueError(
        "matrix lowering disagrees with CROSS cleartext: max error"
        f" {max_error:.3e}"
    )
  return {"seed": seed, "trials": trials_out, "max_abs_error": max_error}


def format_dense(values: np.ndarray) -> str:
  values = np.asarray(values, dtype=np.float32)
  if values.ndim == 1:
    return "[" + ", ".join(f"{float(value):.9e}" for value in values) + "]"
  if values.ndim == 2:
    return (
        "[\n"
        + ",\n".join("      " + format_dense(row) for row in values)
        + "\n    ]"
    )
  raise ValueError(f"unsupported dense constant rank: {values.ndim}")


def emit_layer(
    name: str,
    input_name: str,
    input_size: int,
    output_size: int,
    weight_name: str,
    bias_name: str,
    square: bool,
) -> str:
  result = f"""
    %{name}_matvec = linalg.matvec
        ins(%{weight_name}, %{input_name} : tensor<{output_size}x{input_size}xf32>, tensor<{input_size}xf32>)
        outs(%zero_{name} : tensor<{output_size}xf32>) -> tensor<{output_size}xf32>
    %{name}_matvec_barrier_empty = tensor.empty() : tensor<{output_size}xf32>
    %{name}_matvec_barrier = tensor.insert_slice %{name}_matvec into %{name}_matvec_barrier_empty[0] [{output_size}] [1]
        : tensor<{output_size}xf32> into tensor<{output_size}xf32>
    %{name}_bias_empty = tensor.empty() : tensor<{output_size}xf32>
    %{name} = linalg.generic {{
        indexing_maps = [#map1d, #map1d, #map1d],
        iterator_types = ["parallel"]}}
        ins(%{name}_matvec_barrier, %{bias_name} : tensor<{output_size}xf32>, tensor<{output_size}xf32>)
        outs(%{name}_bias_empty : tensor<{output_size}xf32>) {{
    ^bb0(%acc: f32, %bias: f32, %out: f32):
      %sum = arith.addf %acc, %bias : f32
      linalg.yield %sum : f32
    }} -> tensor<{output_size}xf32>
"""
  if not square:
    return result
  return result + f"""
    %{name}_barrier_empty = tensor.empty() : tensor<{output_size}xf32>
    %{name}_barrier = tensor.insert_slice %{name} into %{name}_barrier_empty[0] [{output_size}] [1]
        : tensor<{output_size}xf32> into tensor<{output_size}xf32>
    %square_{name}_empty = tensor.empty() : tensor<{output_size}xf32>
    %square_{name} = linalg.generic {{
        indexing_maps = [#map1d, #map1d],
        iterator_types = ["parallel"]}}
        ins(%{name}_barrier : tensor<{output_size}xf32>)
        outs(%square_{name}_empty : tensor<{output_size}xf32>) {{
    ^bb0(%in: f32, %out: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
    }} -> tensor<{output_size}xf32>
"""


def render_mlir(constants: dict[str, np.ndarray], sources: dict) -> str:
  source_hash = hashlib.sha256(
      json.dumps(sources, sort_keys=True).encode("utf-8")
  ).hexdigest()
  declarations = []
  shapes = {
      "w1": "784x784",
      "b1": "784",
      "w2": "392x784",
      "b2": "392",
      "w3": "32x392",
      "b3": "32",
      "w4": "10x32",
      "b4": "10",
  }
  for name in ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"):
    declarations.append(
        f"    %{name} = arith.constant dense<{format_dense(constants[name])}> "
        f": tensor<{shapes[name]}xf32>"
    )
  zeros = "\n".join(
      f"    %zero_{name} = arith.constant dense<0.000000e+00> :"
      f" tensor<{size}xf32>"
      for name, size in (
          ("layer1", 784),
          ("layer2", 392),
          ("layer3", 32),
          ("logits", 10),
      )
  )
  body = "".join([
      emit_layer("layer1", "flat_input", 784, 784, "w1", "b1", True),
      emit_layer("layer2", "square_layer1", 784, 392, "w2", "b2", True),
      emit_layer("layer3", "square_layer2", 392, 32, "w3", "b3", True),
      emit_layer("logits", "square_layer3", 32, 10, "w4", "b4", False),
  ])
  return f"""// Generated from CROSS LeNet trained binaries.
// Source manifest digest: {source_hash}
#map1d = affine_map<(d0) -> (d0)>

module attributes {{
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<
    logN = 11,
    Q = {Q_TOWERS},
    P = {P_TOWERS},
    logDefaultScale = 58
  >
}} {{
  func.func @lenet(
      %input: tensor<1x1x28x28xf32> {{secret.secret}}) -> tensor<10xf32> {{
{chr(10).join(declarations)}

    %flat_input = tensor.collapse_shape %input [[0, 1, 2, 3]]
        : tensor<1x1x28x28xf32> into tensor<784xf32>
{zeros}
{body}
    return %logits : tensor<10xf32>
  }}
}}
"""


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--cross-root", type=Path, default=Path("/home/zohaib/CROSS_dev")
  )
  parser.add_argument("--weights-dir", type=Path)
  parser.add_argument(
      "--output", type=Path, default=Path("tests/CROSS/lenet/lenet-cross.mlir")
  )
  parser.add_argument("--metadata", type=Path)
  parser.add_argument("--verify-only", action="store_true")
  args = parser.parse_args()

  weights_dir = args.weights_dir or args.cross_root / "demos" / "maple_data"
  weights, sources = load_weights(weights_dir)
  lenet_he = import_lenet(args.cross_root)
  constants = build_constants(weights, lenet_he)
  verification = verify_constants(weights, constants, lenet_he)
  metadata = {
      "model": "CROSS LeNet depth-7",
      "entry_point": "lenet",
      "input_shape": [1, 1, 28, 28],
      "output_shape": [10],
      "weights_dir": str(weights_dir.resolve()),
      "sources": sources,
      "constant_shapes": {
          name: list(value.shape) for name, value in constants.items()
      },
      "verification": verification,
      "crypto": {
          "ring_degree": 2048,
          "slot_count": 1024,
          "q_towers": Q_TOWERS,
          "p_towers": P_TOWERS,
          "scaling_factor": 288276558458736641,
          "dnum": 4,
          "r": 32,
          "c": 64,
          "composite_degree": 1,
      },
  }
  if args.verify_only:
    print(json.dumps(metadata, indent=2))
    return
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(render_mlir(constants, sources))
  metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
  metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
  print(f"wrote {args.output}")
  print(f"wrote {metadata_path}")
  print(f"cleartext max error: {verification['max_abs_error']:.3e}")


if __name__ == "__main__":
  main()
