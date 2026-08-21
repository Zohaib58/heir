# AlexNet-Tiny common LWE boundary

`alexnet-tiny-common-lwe.mlir` is the AlexNet-Tiny program immediately after
HEIR's `--ckks-to-lwe` conversion. It was generated with:

```bash
bazel-bin/tools/heir-opt \
  --torch-linalg-to-ckks=ciphertext-degree=2048 \
  --ckks-to-lwe \
  tests/CROSS/alexnet/alexnet-tiny-fused-trained-postbias.mlir \
  -o tests/CROSS/alexnet/alexnet-tiny-common-lwe.mlir
```

This is the useful common boundary for comparing backend lowering. Packing,
matrix lowering, rotations, level assignment, and scaling decisions have already
been made. Backend-specific OpenFHE, Lattigo, or JaxiteWord operations have not
yet been introduced.

The program is not a pure LWE-only trace. HEIR intentionally leaves
`ckks.rotate`, `ckks.rescale`, and `ckks.relinearize` operations at this
boundary while arithmetic operations are represented mainly by `lwe.radd`,
`lwe.radd_plain`, `lwe.rmul`, and `lwe.rmul_plain`. It also retains the
`preprocessing` dialect so each backend can lower plaintext preprocessing in its
own way.

The source MLIR embeds the CROSS parameter set, so this artifact contains its
Q/P towers, `logN = 11`, and `logDefaultScale = 58`. The operation trace is the
common part; cryptographic parameters may need to be regenerated or replaced for
another backend.

Current operation highlights:

```text
lwe.radd             466
lwe.radd_plain         4
lwe.rmul               3
lwe.rmul_plain       464
ckks.rotate            77
ckks.rescale           28
ckks.relinearize        3
```

OpenFHE, Lattigo, and CROSS/JaxiteWord have lowering paths from this conceptual
boundary. This checkout does not currently contain a registered CKKS/LWE to
Cheddar conversion, so the file should not yet be described as directly
lowerable to Cheddar.

<!-- mdformat global-off -->
