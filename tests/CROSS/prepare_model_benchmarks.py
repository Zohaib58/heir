#!/usr/bin/env python3
"""Create ring-degree-specific MLIR and lower with degree/2 CKKS slots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "benchmark_params.json"
SCHEME_PARAM_RE = re.compile(
    r"ckks\.schemeParam\s*=\s*#ckks\.scheme_param<.*?>",
    flags=re.DOTALL,
)

MODELS = {
    "alexnet-tiny": {
        "source": ROOT / "alexnet" / "alexnet-tiny-fused-trained-postbias.mlir",
        "entry": "alexnet_tiny",
        "profiles": {
            4096: "alexnet-tiny-4k",
            8192: "cd2-8k",
            16384: "cd2-16k",
            32768: "cd2-32k",
        },
        "output_dir": ROOT / "alexnet" / "generated",
    },
    "lenet": {
        "source": ROOT / "lenet" / "lenet-cross.mlir",
        "entry": "lenet",
        "profiles": {
            2048: "lenet-2k",
            8192: "cd2-8k",
            16384: "cd2-16k",
            32768: "cd2-32k",
        },
        "output_dir": ROOT / "lenet" / "generated",
    },
}


def scheme_param(profile: dict) -> str:
  degree = int(profile["degree"])
  if degree <= 0 or degree & (degree - 1):
    raise ValueError(f"degree must be a positive power of two: {degree}")
  if int(profile["slot_count"]) * 2 != degree:
    raise ValueError(f"slot count must equal degree/2 for degree {degree}")
  if int(profile["r"]) * int(profile["c"]) != degree:
    raise ValueError(f"r*c must equal degree for degree {degree}")
  q = ", ".join(str(value) for value in profile["q_towers"])
  p = ", ".join(str(value) for value in profile["p_towers"])
  return (
      "ckks.schemeParam = #ckks.scheme_param<\n"
      f"    logN = {int(math.log2(degree))},\n"
      f"    Q = [{q}],\n"
      f"    P = [{p}],\n"
      f"    logDefaultScale = {int(profile['log_default_scale'])}\n"
      "  >"
  )


def command(model: str, degree: int, profile: dict, source: Path) -> str:
  spec = MODELS[model]
  prefix = source.with_suffix("")
  slot_count = int(profile["slot_count"])
  options = (
      f"entry-function={spec['entry']} dnum={profile['dnum']} "
      f"r={profile['r']} c={profile['c']} "
      f"composite-degree={profile['composite_degree']} "
      f"scaling-factor={profile['scaling_factor']}"
  )
  return f"""bazel-bin/tools/heir-opt \\
  --torch-linalg-to-ckks=\"ciphertext-degree={slot_count}\" \\
  --ckks-to-lwe \\
  --lwe-to-jaxiteword \\
  --jaxiteword-annotate-encode-rescale-levels \\
  --jaxiteword-configure-crypto-context=\"{options}\" \\
  --preprocessing-to-memref \\
  {source} \\
  -o {prefix}_jaxite_memref.mlir

bazel-bin/tools/heir-translate \\
  --allow-unregistered-dialect \\
  --emit-jaxiteword \\
  {prefix}_jaxite_memref.mlir \\
  -o {prefix}_cross.py

python3 -m py_compile {prefix}_cross.py"""


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--model",
      action="append",
      choices=tuple(MODELS),
      help="model to prepare; repeatable (default: both)",
  )
  parser.add_argument(
      "--degree",
      action="append",
      type=int,
      help="ring degree to prepare; repeatable (default: every model degree)",
  )
  args = parser.parse_args()

  profiles = json.loads(PARAMS_PATH.read_text())["profiles"]
  selected_models = args.model or list(MODELS)
  requested_degrees = None if args.degree is None else set(args.degree)
  commands = []
  for model in selected_models:
    spec = MODELS[model]
    source_text = spec["source"].read_text()
    if len(SCHEME_PARAM_RE.findall(source_text)) != 1:
      raise ValueError(
          f"expected one CKKS scheme parameter in {spec['source']}"
      )
    degrees = [
        degree
        for degree in spec["profiles"]
        if requested_degrees is None or degree in requested_degrees
    ]
    if requested_degrees is not None:
      unsupported = requested_degrees - set(spec["profiles"])
      if unsupported:
        raise ValueError(
            f"{model} does not define degrees {sorted(unsupported)}"
        )
    spec["output_dir"].mkdir(parents=True, exist_ok=True)
    for degree in degrees:
      profile = profiles[spec["profiles"][degree]]
      output = spec["output_dir"] / f"{model}-n{degree}.mlir"
      output.write_text(
          SCHEME_PARAM_RE.sub(scheme_param(profile), source_text, count=1)
      )
      commands.append(command(model, degree, profile, output))
      print(f"prepared {output}")
  print("\n# Run from /home/zohaib/heir-private")
  print(
      "# ckks.schemeParam sets the ring degree; ciphertext-degree sets slots.\n"
  )
  print("\n\n".join(commands))


if __name__ == "__main__":
  main()
