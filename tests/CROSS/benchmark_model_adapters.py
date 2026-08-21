"""Cleartext input/reference adapters for generated network benchmarks."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import sys
from typing import Callable

import numpy as np


Adapter = Callable[[int, str], tuple[np.ndarray, np.ndarray]]


def alexnet_tiny(seed: int, cross_root: str) -> tuple[np.ndarray, np.ndarray]:
  demos = f"{cross_root}/demos"
  if demos not in sys.path:
    sys.path.insert(0, demos)
  from alexnet_he import (  # pylint: disable=import-outside-toplevel
      alexnet_tiny_cleartext,
      alexnet_tiny_random_inputs,
      load_trained_alexnet_tiny_weights,
      prepare_tiny_args,
  )

  rng = alexnet_tiny_random_inputs(seed=seed)
  weights = load_trained_alexnet_tiny_weights()
  if weights is None:
    raise FileNotFoundError(
        f"trained AlexNet-Tiny weights were not found under {demos}/maple_data"
    )
  input_value = rng["X"].reshape(1, 3, 16, 16).astype(np.float32)
  expected = alexnet_tiny_cleartext(
      rng["X"], *prepare_tiny_args(weights)
  ).astype(np.float64)
  return input_value, expected


def _read_idx_images(path: Path) -> np.ndarray:
  with path.open("rb") as stream:
    magic, count, height, width = struct.unpack(">IIII", stream.read(16))
    if magic != 0x803:
      raise ValueError(f"{path}: invalid IDX image magic {magic:#x}")
    values = np.frombuffer(stream.read(), dtype=np.uint8)
  return values.reshape(count, height, width)


def _find_mnist_test_images(cross_root: str) -> Path:
  candidates = []
  if os.environ.get("CROSS_MNIST_DIR"):
    candidates.append(Path(os.environ["CROSS_MNIST_DIR"]))
  candidates.extend([
      Path(cross_root) / "mnist" / "data",
      Path(__file__).resolve().parents[1]
      / "Examples"
      / "common"
      / "mnist"
      / "data",
  ])
  names = ("t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte")
  for directory in candidates:
    for name in names:
      path = directory / name
      if path.is_file():
        return path
  raise FileNotFoundError(
      "MNIST test images were not found; set CROSS_MNIST_DIR to the IDX"
      " directory"
  )


def lenet(seed: int, cross_root: str) -> tuple[np.ndarray, np.ndarray]:
  demos = f"{cross_root}/demos"
  if demos not in sys.path:
    sys.path.insert(0, demos)
  from lenet_he import (  # pylint: disable=import-outside-toplevel
      lenet_cleartext,
      load_trained_lenet_weights,
      normalize_mnist_image,
  )

  weights = load_trained_lenet_weights()
  if weights is None:
    raise FileNotFoundError(
        f"trained LeNet weights were not found under {demos}/maple_data"
    )
  images = _read_idx_images(_find_mnist_test_images(cross_root))
  image = images[seed % len(images)].astype(np.float32) / np.float32(255.0)
  input_value = (
      normalize_mnist_image(image).astype(np.float32).reshape(1, 1, 28, 28)
  )
  model_weights = {
      name: value.astype(np.float32).astype(np.float64)
      for name, value in weights.items()
  }
  expected = lenet_cleartext(
      input_value.reshape(-1).astype(np.float64),
      model_weights["W1"],
      model_weights["W2"],
      model_weights["W3"],
      model_weights["W4"],
      model_weights["b1"],
      model_weights["b2"],
      model_weights["b3"],
      model_weights["b4"],
  ).astype(np.float64)
  return input_value, expected


ADAPTERS: dict[str, Adapter] = {
    "alexnet_tiny": alexnet_tiny,
    "lenet": lenet,
}


def get_adapter(name: str) -> Adapter:
  try:
    return ADAPTERS[name]
  except KeyError as exc:
    raise ValueError(f"unknown model adapter: {name}") from exc
