import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import generate_lenet_mlir as lenet_generator
import run_heir_model_benchmark as benchmark


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "benchmark_models.json"


class BenchmarkInfrastructureTest(unittest.TestCase):

  def test_lenet_generated_graph_and_metadata(self):
    mlir = (ROOT / "lenet" / "lenet-cross.mlir").read_text()
    metadata = json.loads(
        (ROOT / "lenet" / "lenet-cross.metadata.json").read_text()
    )
    self.assertEqual(mlir.count("linalg.matvec"), 4)
    self.assertEqual(mlir.count("arith.addf %acc, %bias"), 4)
    self.assertEqual(mlir.count("arith.mulf %in, %in"), 3)
    self.assertEqual(metadata["constant_shapes"]["w1"], [784, 784])
    self.assertEqual(metadata["constant_shapes"]["w2"], [392, 784])
    self.assertLessEqual(metadata["verification"]["max_abs_error"], 1e-4)

  def test_lenet_weight_loader_reports_all_missing_files(self):
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, "lenet_conv1_W.bin") as raised:
        lenet_generator.load_weights(Path(directory))
      self.assertIn("lenet_fc2_b.bin", str(raised.exception))

  def test_active_manifest_and_integrity(self):
    config = benchmark.load_model_config("alexnet-tiny-4k", MANIFEST)
    self.assertEqual(config["crypto"]["ring_degree"], 4096)
    self.assertEqual(config["crypto"]["slot_count"], 2048)
    result = benchmark.verify_computation_integrity(config)
    self.assertTrue(result["passed"])
    self.assertEqual(result["pristine"], result["runnable"])
    self.assertGreater(result["pristine"]["operation_counts"]["rescale"], 0)
    self.assertGreater(
        result["pristine"]["operation_counts"]["plaintext_ciphertext_multiply"],
        0,
    )

  def test_disabled_manifest_entry_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "is disabled"):
      benchmark.load_model_config("lola-current", MANIFEST)

  def _invalid_manifest(self, field, value, message):
    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      data = json.loads(MANIFEST.read_text())
      config = data["models"]["alexnet-tiny-4k"]
      for name in (
          "mlir_path",
          "generated_python_path",
          "pristine_python_path",
      ):
        config[name] = str((ROOT / config[name]).resolve())
      catalog = json.loads((ROOT / data["parameter_catalog"]).read_text())
      catalog["profiles"][config["crypto_profile"]][field] = value
      catalog_path = directory / "params.json"
      catalog_path.write_text(json.dumps(catalog))
      data["parameter_catalog"] = str(catalog_path)
      path = directory / "manifest.json"
      path.write_text(json.dumps(data))
      with self.assertRaisesRegex(ValueError, message):
        benchmark.load_model_config("alexnet-tiny-4k", path)

  def test_invalid_slot_count_is_rejected(self):
    self._invalid_manifest("slot_count", 1024, r"2 \* slot_count")

  def test_invalid_degree_layout_is_rejected(self):
    self._invalid_manifest("r", 32, r"r \* c")

  def test_parameter_only_change_preserves_integrity(self):
    source = """
def model__preprocessed(ctx, state, ct):
  return ctx.he_rescale[1, 0].rescale(ct)

def model__generate_crypto_context():
  return {"degree": 16}
"""
    with tempfile.TemporaryDirectory() as directory:
      pristine = Path(directory) / "pristine.py"
      runnable = Path(directory) / "runnable.py"
      pristine.write_text(source)
      runnable.write_text(source.replace("16", "32"))
      self.assertEqual(
          benchmark.computation_fingerprint(pristine, "model"),
          benchmark.computation_fingerprint(runnable, "model"),
      )

  def test_computation_change_is_detected(self):
    with tempfile.TemporaryDirectory() as directory:
      pristine = Path(directory) / "pristine.py"
      runnable = Path(directory) / "runnable.py"
      pristine.write_text(
          "def model__preprocessed(ctx, state, ct):\n"
          "  return ctx.he_rescale[1, 0].rescale(ct)\n"
      )
      runnable.write_text(
          "def model__preprocessed(ctx, state, ct):\n"
          "  return ctx.he_add[1].add(ct, ct)\n"
      )
      config = {
          "entry_point": "model",
          "pristine_python_path": str(pristine),
          "generated_python_path": str(runnable),
          "computation_sha256": benchmark.computation_fingerprint(
              pristine, "model"
          )["sha256"],
      }
      with self.assertRaisesRegex(ValueError, "differs"):
        benchmark.verify_computation_integrity(config)

  def test_timing_summary(self):
    summary = benchmark.summarize_samples([1.0, 2.0, 3.0])
    self.assertEqual(summary["samples_ms"], [1.0, 2.0, 3.0])
    self.assertEqual(summary["median_ms"], 2.0)
    self.assertEqual(summary["min_ms"], 1.0)
    self.assertEqual(summary["max_ms"], 3.0)
    self.assertAlmostEqual(summary["stddev_ms"], 0.816496580927726)

  def test_profile_only_implies_operation_profiling(self):
    args = benchmark.parse_args(["--model", "mock", "--profile-only"])
    self.assertTrue(args.profile_only)
    self.assertTrue(args.profile_operations)

  def test_profiler_restores_wrapped_method(self):
    class FakeOperation:

      def add(self, left, right):
        return left + right

    original = FakeOperation.add
    profiler = benchmark.OperationProfiler(lambda value: None)
    profiler.wrap(FakeOperation, "add", "add")
    profiler.enabled = True
    self.assertEqual(FakeOperation().add(2, 3), 5)
    profiler.uninstall()
    self.assertIs(FakeOperation.add, original)
    report = profiler.report()["unscoped.add"]
    self.assertEqual(report["count"], 1)
    self.assertGreaterEqual(report["average_ms"], 0)

  def test_dynamic_rescale_uses_ciphertext_level(self):
    class Cache:
      max_level = 2

      def num_q_at_level(self, level):
        return level + 1

    class Ciphertext:

      class Data:
        shape = (1, 2, 2)

      polynomial = Data()

    class Operation:

      def __init__(self, key):
        self.key = key

      def rescale(self, ciphertext):
        return self.key

    class Accessor:

      def __getitem__(self, key):
        return Operation(key)

    dynamic = benchmark.DynamicRescaleAccessor(Accessor(), Cache())
    self.assertEqual(dynamic[2, 0].rescale(Ciphertext()), (1, 0))

  def test_correctness_requires_finite_nonzero_output(self):
    expected = np.array([1.0, 2.0])
    self.assertTrue(
        benchmark.correctness(expected.copy(), expected, 1e-3, 1e-3)["passed"]
    )
    self.assertFalse(
        benchmark.correctness(np.zeros(2), expected, 10.0, 10.0)["passed"]
    )
    self.assertFalse(
        benchmark.correctness(np.array([np.nan, 2.0]), expected, 10.0, 10.0)[
            "passed"
        ]
    )
    self.assertFalse(
        benchmark.correctness(
            np.array([3.0, 1.0]),
            expected,
            10.0,
            10.0,
            require_argmax_match=True,
        )["passed"]
    )

  def test_generated_module_has_current_api(self):
    config = benchmark.load_model_config("alexnet-tiny-4k", MANIFEST)
    tree = ast.parse(Path(config["generated_python_path"]).read_text())
    functions = {
        node.name: len(node.args.args)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    entry = config["entry_point"]
    self.assertEqual(functions[f"{entry}__generate_crypto_context"], 0)
    self.assertEqual(functions[f"{entry}__configure_crypto_context"], 4)

  def test_generated_decode_layout_must_use_slot_count(self):
    source = (
        "def model__decrypt__result0(ctx, state, value, key):\n"
        "  return ctx.decode(value).real.reshape(1, 4096)\n"
    )
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "generated.py"
      path.write_text(source)
      with self.assertRaisesRegex(ValueError, "generated=4096, expected=2048"):
        benchmark.validate_generated_slot_layout(path, "model", 2048)

  def test_mocked_cpu_end_to_end_writes_result(self):
    generated_source = (
        "def model__generate_crypto_context():\n  pass\n\n"
        "def model__configure_crypto_context(ctx, pk, sk, ek):\n  pass\n\n"
        "def model__encrypt__arg0(ctx, state, value, key):\n  pass\n\n"
        "def model__preprocessing(ctx, state):\n  pass\n\n"
        "def model__preprocessed(ctx, state, ct):\n"
        "  return ctx.he_add[1].add(ct, ct)\n\n"
        "def model__decrypt__result0(ctx, state, value, key):\n"
        "  return ctx.decode(value).real.reshape(1, 8)\n"
    )
    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      mlir = directory / "model.mlir"
      pristine = directory / "pristine.py"
      runnable = directory / "runnable.py"
      output = directory / "results"
      mlir.write_text("module {}\n")
      pristine.write_text(generated_source)
      runnable.write_text(generated_source)
      fingerprint = benchmark.computation_fingerprint(pristine, "model")
      manifest = directory / "manifest.json"
      manifest.write_text(
          json.dumps({
              "schema_version": 1,
              "models": {
                  "mock": {
                      "enabled": True,
                      "mlir_path": str(mlir),
                      "generated_python_path": str(runnable),
                      "pristine_python_path": str(pristine),
                      "entry_point": "model",
                      "adapter": "mock_adapter",
                      "input_shape": [2],
                      "output_shape": [2],
                      "trained_weight_source": "mock",
                      "seed": 1,
                      "correctness": {"abs_tol": 0.001, "rel_tol": 0.001},
                      "crypto": {
                          "ring_degree": 16,
                          "slot_count": 8,
                          "q_towers": [17],
                          "p_towers": [19],
                          "scaling_factor": 4,
                          "output_scale": 4,
                          "dnum": 1,
                          "r": 4,
                          "c": 4,
                          "batch": 1,
                          "composite_degree": 1,
                      },
                      "computation_sha256": fingerprint["sha256"],
                  }
              },
          })
      )

      class FakeJax:

        @staticmethod
        def default_backend():
          return "cpu"

      expected = np.array([1.0, 2.0])
      fake_run = {
          "decoded": expected.copy(),
          "calibrated_decoded": expected.copy(),
          "ciphertext_shape": [1, 2, 4, 4, 1],
          "timings": {
              "generate_crypto_context": 1.0,
              "keygen_pke": 1.0,
              "keygen_eval": 1.0,
              "eval_key_layout": 1.0,
              "configure_crypto_context": 1.0,
              "encrypt": 1.0,
              "preprocessing": 1.0,
              "preprocessed_compute": 2.0,
              "decrypt": 1.0,
              "online_total": 5.0,
              "output_scale": 4.0,
              "calibrated_output_scale": 4.0,
          },
      }
      import benchmark_model_adapters

      old_adapters = dict(benchmark_model_adapters.ADAPTERS)
      benchmark_model_adapters.ADAPTERS["mock_adapter"] = (
          lambda seed, cross_root: (np.array([3.0, 4.0]), expected)
      )
      try:
        with (
            mock.patch.object(
                benchmark, "import_runtime", return_value={"jax": FakeJax()}
            ),
            mock.patch.object(
                benchmark, "load_generated", return_value=object()
            ),
            mock.patch.object(benchmark, "patch_generated_module"),
            mock.patch.object(benchmark, "execute_once", return_value=fake_run),
            mock.patch.object(
                benchmark,
                "environment_metadata",
                return_value={"jax_backend": "cpu"},
            ),
        ):
          status = benchmark.main([
              "--model",
              "mock",
              "--device",
              "cpu",
              "--warmups",
              "0",
              "--repeats",
              "1",
              "--manifest",
              str(manifest),
              "--output-dir",
              str(output),
          ])
      finally:
        benchmark_model_adapters.ADAPTERS.clear()
        benchmark_model_adapters.ADAPTERS.update(old_adapters)
      self.assertEqual(status, 0)
      result_paths = list(output.glob("*/result.json"))
      self.assertEqual(len(result_paths), 1)
      result = json.loads(result_paths[0].read_text())
      self.assertTrue(result["correctness"]["ordinary"]["passed"])


if __name__ == "__main__":
  unittest.main()
