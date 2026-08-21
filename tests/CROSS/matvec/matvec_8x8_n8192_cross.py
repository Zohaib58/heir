import jax
import jax.numpy as jnp
import key_gen
import numpy as np
from polynomial import Polynomial
import ckks_ctx as ckks


def _encode_slots(ctx, values):
  values = np.asarray(values)
  if values.shape[0] == ctx.num_slots:
    return ctx.encode(values)
  padded = np.zeros((ctx.num_slots,), dtype=values.dtype)
  padded[: values.shape[0]] = values
  return ctx.encode(padded)


def _decode_active(ctx, pt, active_slots):
  return ctx.decode(pt, is_ntt=False).real[:active_slots]


def matvec_identity__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> np.ndarray:
  v2 = np.full((8,), 1.000000e00, dtype=np.float32)
  v3 = np.full((8,), 0.000000e00, dtype=np.float32)
  v4 = np.full((28,), None, dtype=object)
  pt = _encode_slots(v0, v3)
  v5 = 0
  v4[0] = pt
  pt1 = _encode_slots(v0, v2)
  v6 = 1
  v4[1] = pt1
  return v4


def matvec_identity__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 6
  v5 = 3
  v6 = 2
  v7 = 1
  v8 = 0
  ct = v2[0]
  ct1 = v0.he_rot[v0.max_level, 1].rotate(ct)
  v9 = 0
  pt = v3[0]
  ct2_pt_ntt = pt.polynomial[0, 0, ..., : ct1.num_moduli].astype(jnp.uint32)
  ct2_ptct = v0.ptct_mul[v0.max_level]
  ct2_ptct.set_plaintext(ct2_pt_ntt)
  ct2 = ct2_ptct.mul(ct1, use_bat=False)
  ct3 = v0.he_rot[v0.max_level, 2].rotate(ct)
  ct4_pt_ntt = pt.polynomial[0, 0, ..., : ct3.num_moduli].astype(jnp.uint32)
  ct4_ptct = v0.ptct_mul[v0.max_level]
  ct4_ptct.set_plaintext(ct4_pt_ntt)
  ct4 = ct4_ptct.mul(ct3, use_bat=False)
  ct5_pt_ntt = pt.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5 = ct5_ptct.mul(ct, use_bat=False)
  ct6 = v0.he_add[v0.max_level].add(ct5, ct2)
  ct7 = v0.he_add[v0.max_level].add(ct6, ct4)
  ct8 = v0.he_rot[v0.max_level, 3].rotate(ct7)
  ct9 = v0.he_rot[v0.max_level, 6].rotate(ct6)
  v10 = 1
  pt1 = v3[1]
  ct10_pt_ntt = pt1.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct10_ptct = v0.ptct_mul[v0.max_level]
  ct10_ptct.set_plaintext(ct10_pt_ntt)
  ct10 = ct10_ptct.mul(ct, use_bat=False)
  ct11 = v0.he_add[v0.max_level].add(ct10, ct2)
  ct12 = v0.he_add[v0.max_level].add(ct4, ct8)
  ct13 = v0.he_add[v0.max_level].add(ct12, ct9)
  ct14 = v0.he_add[v0.max_level].add(ct11, ct13)
  v11 = [None] * 1
  ct15 = v0.he_rescale[v0.max_level, v0.max_level - 1].rescale(ct14)
  v11[0] = ct15
  v12 = v11
  return v12


def matvec_identity(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  v3 = matvec_identity__preprocessing(v0, v1)
  v4 = matvec_identity__preprocessed(v0, v1, v2, v3)
  return v4


def matvec_identity__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = _encode_slots(v0, v15)
  v0.public_key = v3
  ct = v0.encrypt(pt)
  v16 = [ct]
  return v16


def matvec_identity__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 7
  v8 = 0
  v9 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  pt = v0.decrypt(ct)
  v10 = _decode_active(v0, pt, 8).reshape(1, 8)
  v11 = v9.copy()
  for v12 in range(0, 8):
    v14 = v7 - v12
    v15 = int(v14)
    v16 = v10[0, v15]
    v11[v15] = v16
  return v11


def matvec_shift__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> np.ndarray:
  v2 = np.full((8,), 1.000000e00, dtype=np.float32)
  v3 = np.full((8,), 0.000000e00, dtype=np.float32)
  v4 = np.full((28,), None, dtype=object)
  pt = _encode_slots(v0, v3)
  v5 = 2
  v4[2] = pt
  pt1 = _encode_slots(v0, v2)
  v6 = 3
  v4[3] = pt1
  return v4


def matvec_shift__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 6
  v5 = 3
  v6 = 2
  v7 = 1
  v8 = 0
  v9 = 2
  pt = v3[2]
  ct = v2[0]
  ct1_pt_ntt = pt.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1 = ct1_ptct.mul(ct, use_bat=False)
  ct2 = v0.he_rot[v0.max_level, 1].rotate(ct)
  ct3 = v0.he_rot[v0.max_level, 2].rotate(ct)
  ct4_pt_ntt = pt.polynomial[0, 0, ..., : ct3.num_moduli].astype(jnp.uint32)
  ct4_ptct = v0.ptct_mul[v0.max_level]
  ct4_ptct.set_plaintext(ct4_pt_ntt)
  ct4 = ct4_ptct.mul(ct3, use_bat=False)
  ct5_pt_ntt = pt.polynomial[0, 0, ..., : ct2.num_moduli].astype(jnp.uint32)
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5 = ct5_ptct.mul(ct2, use_bat=False)
  ct6 = v0.he_add[v0.max_level].add(ct1, ct5)
  ct7 = v0.he_add[v0.max_level].add(ct6, ct4)
  ct8 = v0.he_rot[v0.max_level, 3].rotate(ct7)
  ct9 = v0.he_rot[v0.max_level, 6].rotate(ct6)
  v10 = 3
  pt1 = v3[3]
  ct10_pt_ntt = pt1.polynomial[0, 0, ..., : ct2.num_moduli].astype(jnp.uint32)
  ct10_ptct = v0.ptct_mul[v0.max_level]
  ct10_ptct.set_plaintext(ct10_pt_ntt)
  ct10 = ct10_ptct.mul(ct2, use_bat=False)
  ct11 = v0.he_add[v0.max_level].add(ct1, ct10)
  ct12 = v0.he_add[v0.max_level].add(ct4, ct8)
  ct13 = v0.he_add[v0.max_level].add(ct12, ct9)
  ct14 = v0.he_add[v0.max_level].add(ct11, ct13)
  v11 = [None] * 1
  ct15 = v0.he_rescale[v0.max_level, v0.max_level - 1].rescale(ct14)
  v11[0] = ct15
  v12 = v11
  return v12


def matvec_shift(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  v3 = matvec_shift__preprocessing(v0, v1)
  v4 = matvec_shift__preprocessed(v0, v1, v2, v3)
  return v4


def matvec_shift__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = _encode_slots(v0, v15)
  v0.public_key = v3
  ct = v0.encrypt(pt)
  v16 = [ct]
  return v16


def matvec_shift__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 7
  v8 = 0
  v9 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  pt = v0.decrypt(ct)
  v10 = _decode_active(v0, pt, 8).reshape(1, 8)
  v11 = v9.copy()
  for v12 in range(0, 8):
    v14 = v7 - v12
    v15 = int(v14)
    v16 = v10[0, v15]
    v11[v15] = v16
  return v11


def matvec_random__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> np.ndarray:
  v2 = np.array(
      [
          8.116263e-01,
          1.445338e00,
          9.206955e-01,
          1.077045e00,
          6.787661e-01,
          1.358792e00,
          1.236010e00,
          7.778313e-01,
      ],
      dtype=np.float32,
  )
  v3 = np.array(
      [
          1.906357e00,
          1.391105e-01,
          6.533354e-01,
          1.225588e00,
          2.855770e-01,
          6.922510e-01,
          1.851561e00,
          2.681358e-01,
      ],
      dtype=np.float32,
  )
  v4 = np.array(
      [
          1.490788e00,
          1.942829e00,
          1.262521e00,
          1.882558e-01,
          1.400043e00,
          1.088129e00,
          1.138749e00,
          4.723674e-01,
      ],
      dtype=np.float32,
  )
  v5 = np.array(
      [
          3.318726e-01,
          4.512235e-01,
          1.859318e-01,
          1.237451e00,
          1.681641e00,
          3.650383e-01,
          1.254335e00,
          9.362897e-01,
      ],
      dtype=np.float32,
  )
  v6 = np.array(
      [
          1.040836e00,
          1.942211e00,
          7.181276e-01,
          3.964354e-01,
          5.034443e-01,
          6.550748e-01,
          4.239958e-01,
          2.235980e-01,
      ],
      dtype=np.float32,
  )
  v7 = np.array(
      [
          1.653382e-01,
          1.572752e00,
          8.384869e-01,
          3.963896e-01,
          4.454674e-01,
          7.960875e-01,
          9.665329e-01,
          1.902883e00,
      ],
      dtype=np.float32,
  )
  v8 = np.array(
      [
          6.780602e-01,
          1.591834e00,
          1.934701e00,
          1.827709e00,
          1.885048e00,
          6.155632e-01,
          2.103589e-01,
          4.484686e-01,
      ],
      dtype=np.float32,
  )
  v9 = np.array(
      [
          1.097037e00,
          4.793802e-01,
          1.635955e00,
          5.916820e-01,
          1.800172e00,
          1.674601e00,
          1.745735e00,
          1.242118e00,
      ],
      dtype=np.float32,
  )
  v10 = np.full((28,), None, dtype=object)
  pt = _encode_slots(v0, v2)
  v11 = 4
  v10[4] = pt
  pt1 = _encode_slots(v0, v3)
  v12 = 5
  v10[5] = pt1
  pt2 = _encode_slots(v0, v4)
  v13 = 6
  v10[6] = pt2
  pt3 = _encode_slots(v0, v5)
  v14 = 7
  v10[7] = pt3
  pt4 = _encode_slots(v0, v6)
  v15 = 8
  v10[8] = pt4
  pt5 = _encode_slots(v0, v7)
  v16 = 9
  v10[9] = pt5
  pt6 = _encode_slots(v0, v8)
  v17 = 10
  v10[10] = pt6
  pt7 = _encode_slots(v0, v9)
  v18 = 11
  v10[11] = pt7
  return v10


def matvec_random__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 6
  v5 = 3
  v6 = 2
  v7 = 1
  v8 = 0
  v9 = 4
  pt = v3[4]
  ct = v2[0]
  ct1_pt_ntt = pt.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1 = ct1_ptct.mul(ct, use_bat=False)
  ct2 = v0.he_rot[v0.max_level, 1].rotate(ct)
  v10 = 5
  pt1 = v3[5]
  ct3_pt_ntt = pt1.polynomial[0, 0, ..., : ct2.num_moduli].astype(jnp.uint32)
  ct3_ptct = v0.ptct_mul[v0.max_level]
  ct3_ptct.set_plaintext(ct3_pt_ntt)
  ct3 = ct3_ptct.mul(ct2, use_bat=False)
  ct4 = v0.he_rot[v0.max_level, 2].rotate(ct)
  v11 = 6
  pt2 = v3[6]
  ct5_pt_ntt = pt2.polynomial[0, 0, ..., : ct4.num_moduli].astype(jnp.uint32)
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5 = ct5_ptct.mul(ct4, use_bat=False)
  v12 = 7
  pt3 = v3[7]
  ct6_pt_ntt = pt3.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct6_ptct = v0.ptct_mul[v0.max_level]
  ct6_ptct.set_plaintext(ct6_pt_ntt)
  ct6 = ct6_ptct.mul(ct, use_bat=False)
  v13 = 8
  pt4 = v3[8]
  ct7_pt_ntt = pt4.polynomial[0, 0, ..., : ct2.num_moduli].astype(jnp.uint32)
  ct7_ptct = v0.ptct_mul[v0.max_level]
  ct7_ptct.set_plaintext(ct7_pt_ntt)
  ct7 = ct7_ptct.mul(ct2, use_bat=False)
  v14 = 9
  pt5 = v3[9]
  ct8_pt_ntt = pt5.polynomial[0, 0, ..., : ct4.num_moduli].astype(jnp.uint32)
  ct8_ptct = v0.ptct_mul[v0.max_level]
  ct8_ptct.set_plaintext(ct8_pt_ntt)
  ct8 = ct8_ptct.mul(ct4, use_bat=False)
  ct9 = v0.he_add[v0.max_level].add(ct6, ct7)
  ct10 = v0.he_add[v0.max_level].add(ct9, ct8)
  ct11 = v0.he_rot[v0.max_level, 3].rotate(ct10)
  v15 = 10
  pt6 = v3[10]
  ct12_pt_ntt = pt6.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct12_ptct = v0.ptct_mul[v0.max_level]
  ct12_ptct.set_plaintext(ct12_pt_ntt)
  ct12 = ct12_ptct.mul(ct, use_bat=False)
  v16 = 11
  pt7 = v3[11]
  ct13_pt_ntt = pt7.polynomial[0, 0, ..., : ct2.num_moduli].astype(jnp.uint32)
  ct13_ptct = v0.ptct_mul[v0.max_level]
  ct13_ptct.set_plaintext(ct13_pt_ntt)
  ct13 = ct13_ptct.mul(ct2, use_bat=False)
  ct14 = v0.he_add[v0.max_level].add(ct12, ct13)
  ct15 = v0.he_rot[v0.max_level, 6].rotate(ct14)
  ct16 = v0.he_add[v0.max_level].add(ct1, ct3)
  ct17 = v0.he_add[v0.max_level].add(ct5, ct11)
  ct18 = v0.he_add[v0.max_level].add(ct17, ct15)
  ct19 = v0.he_add[v0.max_level].add(ct16, ct18)
  v17 = [None] * 1
  ct20 = v0.he_rescale[v0.max_level, v0.max_level - 1].rescale(ct19)
  v17[0] = ct20
  v18 = v17
  return v18


def matvec_random(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  v3 = matvec_random__preprocessing(v0, v1)
  v4 = matvec_random__preprocessed(v0, v1, v2, v3)
  return v4


def matvec_random__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = _encode_slots(v0, v15)
  v0.public_key = v3
  ct = v0.encrypt(pt)
  v16 = [ct]
  return v16


def matvec_random__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 7
  v8 = 0
  v9 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  pt = v0.decrypt(ct)
  v10 = _decode_active(v0, pt, 8).reshape(1, 8)
  v11 = v9.copy()
  for v12 in range(0, 8):
    v14 = v7 - v12
    v15 = int(v14)
    v16 = v10[0, v15]
    v11[v15] = v16
  return v11


def matvec_chain__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> np.ndarray:
  v2 = np.array(
      [
          1.340000e00,
          1.220000e00,
          1.050000e00,
          1.500000e00,
          1.010000e00,
          8.800000e-01,
          5.000000e-01,
          1.060000e00,
      ],
      dtype=np.float32,
  )
  v3 = np.array(
      [
          5.800000e-01,
          5.200000e-01,
          8.900000e-01,
          8.600000e-01,
          1.170000e00,
          8.200000e-01,
          1.490000e00,
          1.410000e00,
      ],
      dtype=np.float32,
  )
  v4 = np.array(
      [
          1.260000e00,
          1.090000e00,
          1.430000e00,
          1.260000e00,
          6.100000e-01,
          8.500000e-01,
          6.700000e-01,
          7.100000e-01,
      ],
      dtype=np.float32,
  )
  v5 = np.array(
      [
          8.200000e-01,
          1.330000e00,
          7.900000e-01,
          7.400000e-01,
          1.060000e00,
          1.340000e00,
          1.090000e00,
          6.300000e-01,
      ],
      dtype=np.float32,
  )
  v6 = np.array(
      [
          1.160000e00,
          8.400000e-01,
          1.020000e00,
          6.900000e-01,
          6.600000e-01,
          8.600000e-01,
          1.190000e00,
          6.500000e-01,
      ],
      dtype=np.float32,
  )
  v7 = np.array(
      [
          1.350000e00,
          1.050000e00,
          1.400000e00,
          1.070000e00,
          6.500000e-01,
          5.400000e-01,
          8.000000e-01,
          9.000000e-01,
      ],
      dtype=np.float32,
  )
  v8 = np.array(
      [
          8.200000e-01,
          9.000000e-01,
          7.400000e-01,
          1.050000e00,
          1.080000e00,
          1.480000e00,
          6.000000e-01,
          1.200000e00,
      ],
      dtype=np.float32,
  )
  v9 = np.array(
      [
          1.190000e00,
          1.200000e00,
          8.400000e-01,
          1.350000e00,
          1.020000e00,
          7.600000e-01,
          1.390000e00,
          1.130000e00,
      ],
      dtype=np.float32,
  )
  v10 = np.array(
      [
          1.200000e00,
          8.900000e-01,
          1.030000e00,
          7.300000e-01,
          9.300000e-01,
          7.500000e-01,
          8.400000e-01,
          1.170000e00,
      ],
      dtype=np.float32,
  )
  v11 = np.array(
      [
          7.900000e-01,
          8.400000e-01,
          1.030000e00,
          7.900000e-01,
          1.390000e00,
          9.800000e-01,
          8.000000e-01,
          9.200000e-01,
      ],
      dtype=np.float32,
  )
  v12 = np.array(
      [
          7.300000e-01,
          1.230000e00,
          1.130000e00,
          1.130000e00,
          1.440000e00,
          1.490000e00,
          1.020000e00,
          1.180000e00,
      ],
      dtype=np.float32,
  )
  v13 = np.array(
      [
          1.120000e00,
          1.110000e00,
          1.380000e00,
          1.050000e00,
          9.400000e-01,
          1.350000e00,
          5.900000e-01,
          1.000000e00,
      ],
      dtype=np.float32,
  )
  v14 = np.array(
      [
          6.200000e-01,
          6.200000e-01,
          1.010000e00,
          1.220000e00,
          5.600000e-01,
          1.220000e00,
          9.300000e-01,
          9.300000e-01,
      ],
      dtype=np.float32,
  )
  v15 = np.array(
      [
          8.200000e-01,
          1.330000e00,
          1.170000e00,
          9.200000e-01,
          9.000000e-01,
          1.110000e00,
          1.220000e00,
          9.900000e-01,
      ],
      dtype=np.float32,
  )
  v16 = np.array(
      [
          6.800000e-01,
          8.200000e-01,
          9.300000e-01,
          9.100000e-01,
          1.100000e00,
          1.090000e00,
          1.480000e00,
          1.240000e00,
      ],
      dtype=np.float32,
  )
  v17 = np.array(
      [
          6.800000e-01,
          8.600000e-01,
          8.100000e-01,
          1.370000e00,
          1.050000e00,
          1.120000e00,
          1.180000e00,
          9.800000e-01,
      ],
      dtype=np.float32,
  )
  v18 = np.full((28,), None, dtype=object)
  pt = _encode_slots(v0, v2)
  v19 = 12
  v18[12] = pt
  pt1 = _encode_slots(v0, v3)
  v20 = 13
  v18[13] = pt1
  pt2 = _encode_slots(v0, v4)
  v21 = 14
  v18[14] = pt2
  pt3 = _encode_slots(v0, v5)
  v22 = 15
  v18[15] = pt3
  pt4 = _encode_slots(v0, v6)
  v23 = 16
  v18[16] = pt4
  pt5 = _encode_slots(v0, v7)
  v24 = 17
  v18[17] = pt5
  pt6 = _encode_slots(v0, v8)
  v25 = 18
  v18[18] = pt6
  pt7 = _encode_slots(v0, v9)
  v26 = 19
  v18[19] = pt7
  pt8 = _encode_slots(v0, v10)
  v27 = 20
  v18[20] = pt8
  pt9 = _encode_slots(v0, v11)
  v28 = 21
  v18[21] = pt9
  pt10 = _encode_slots(v0, v12)
  v29 = 22
  v18[22] = pt10
  pt11 = _encode_slots(v0, v13)
  v30 = 23
  v18[23] = pt11
  pt12 = _encode_slots(v0, v14)
  v31 = 24
  v18[24] = pt12
  pt13 = _encode_slots(v0, v15)
  v32 = 25
  v18[25] = pt13
  pt14 = _encode_slots(v0, v16)
  v33 = 26
  v18[26] = pt14
  pt15 = _encode_slots(v0, v17)
  v34 = 27
  v18[27] = pt15
  return v18


def matvec_chain__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 6
  v5 = 3
  v6 = 2
  v7 = 1
  v8 = 0
  v9 = 12
  pt = v3[12]
  ct = v2[0]
  ct1_pt_ntt = pt.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1 = ct1_ptct.mul(ct, use_bat=False)
  ct2 = v0.he_rot[v0.max_level, 1].rotate(ct)
  v10 = 13
  pt1 = v3[13]
  ct3_pt_ntt = pt1.polynomial[0, 0, ..., : ct2.num_moduli].astype(jnp.uint32)
  ct3_ptct = v0.ptct_mul[v0.max_level]
  ct3_ptct.set_plaintext(ct3_pt_ntt)
  ct3 = ct3_ptct.mul(ct2, use_bat=False)
  ct4 = v0.he_rot[v0.max_level, 2].rotate(ct)
  v11 = 14
  pt2 = v3[14]
  ct5_pt_ntt = pt2.polynomial[0, 0, ..., : ct4.num_moduli].astype(jnp.uint32)
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5 = ct5_ptct.mul(ct4, use_bat=False)
  v12 = 15
  pt3 = v3[15]
  ct6_pt_ntt = pt3.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct6_ptct = v0.ptct_mul[v0.max_level]
  ct6_ptct.set_plaintext(ct6_pt_ntt)
  ct6 = ct6_ptct.mul(ct, use_bat=False)
  v13 = 16
  pt4 = v3[16]
  ct7_pt_ntt = pt4.polynomial[0, 0, ..., : ct2.num_moduli].astype(jnp.uint32)
  ct7_ptct = v0.ptct_mul[v0.max_level]
  ct7_ptct.set_plaintext(ct7_pt_ntt)
  ct7 = ct7_ptct.mul(ct2, use_bat=False)
  v14 = 17
  pt5 = v3[17]
  ct8_pt_ntt = pt5.polynomial[0, 0, ..., : ct4.num_moduli].astype(jnp.uint32)
  ct8_ptct = v0.ptct_mul[v0.max_level]
  ct8_ptct.set_plaintext(ct8_pt_ntt)
  ct8 = ct8_ptct.mul(ct4, use_bat=False)
  ct9 = v0.he_add[v0.max_level].add(ct6, ct7)
  ct10 = v0.he_add[v0.max_level].add(ct9, ct8)
  ct11 = v0.he_rot[v0.max_level, 3].rotate(ct10)
  v15 = 18
  pt6 = v3[18]
  ct12_pt_ntt = pt6.polynomial[0, 0, ..., : ct.num_moduli].astype(jnp.uint32)
  ct12_ptct = v0.ptct_mul[v0.max_level]
  ct12_ptct.set_plaintext(ct12_pt_ntt)
  ct12 = ct12_ptct.mul(ct, use_bat=False)
  v16 = 19
  pt7 = v3[19]
  ct13_pt_ntt = pt7.polynomial[0, 0, ..., : ct2.num_moduli].astype(jnp.uint32)
  ct13_ptct = v0.ptct_mul[v0.max_level]
  ct13_ptct.set_plaintext(ct13_pt_ntt)
  ct13 = ct13_ptct.mul(ct2, use_bat=False)
  ct14 = v0.he_add[v0.max_level].add(ct12, ct13)
  ct15 = v0.he_rot[v0.max_level, 6].rotate(ct14)
  ct16 = v0.he_add[v0.max_level].add(ct1, ct3)
  ct17 = v0.he_add[v0.max_level].add(ct5, ct11)
  ct18 = v0.he_add[v0.max_level].add(ct17, ct15)
  ct19 = v0.he_add[v0.max_level].add(ct16, ct18)
  ct20 = v0.he_rescale[v0.max_level, v0.max_level - 1].rescale(ct19)
  v17 = 20
  pt8 = v3[20]
  ct21_pt_ntt = pt8.polynomial[0, 0, ..., : ct20.num_moduli].astype(jnp.uint32)
  ct21_ptct = v0.ptct_mul[v0.max_level - 1]
  ct21_ptct.set_plaintext(ct21_pt_ntt)
  ct21 = ct21_ptct.mul(ct20, use_bat=False)
  ct22 = v0.he_rot[v0.max_level, 1].rotate(ct19)
  ct23 = v0.he_rescale[v0.max_level, v0.max_level - 1].rescale(ct22)
  v18 = 21
  pt9 = v3[21]
  ct24_pt_ntt = pt9.polynomial[0, 0, ..., : ct23.num_moduli].astype(jnp.uint32)
  ct24_ptct = v0.ptct_mul[v0.max_level - 1]
  ct24_ptct.set_plaintext(ct24_pt_ntt)
  ct24 = ct24_ptct.mul(ct23, use_bat=False)
  ct25 = v0.he_rot[v0.max_level, 2].rotate(ct19)
  ct26 = v0.he_rescale[v0.max_level, v0.max_level - 1].rescale(ct25)
  v19 = 22
  pt10 = v3[22]
  ct27_pt_ntt = pt10.polynomial[0, 0, ..., : ct26.num_moduli].astype(jnp.uint32)
  ct27_ptct = v0.ptct_mul[v0.max_level - 1]
  ct27_ptct.set_plaintext(ct27_pt_ntt)
  ct27 = ct27_ptct.mul(ct26, use_bat=False)
  v20 = 23
  pt11 = v3[23]
  ct28_pt_ntt = pt11.polynomial[0, 0, ..., : ct20.num_moduli].astype(jnp.uint32)
  ct28_ptct = v0.ptct_mul[v0.max_level - 1]
  ct28_ptct.set_plaintext(ct28_pt_ntt)
  ct28 = ct28_ptct.mul(ct20, use_bat=False)
  v21 = 24
  pt12 = v3[24]
  ct29_pt_ntt = pt12.polynomial[0, 0, ..., : ct23.num_moduli].astype(jnp.uint32)
  ct29_ptct = v0.ptct_mul[v0.max_level - 1]
  ct29_ptct.set_plaintext(ct29_pt_ntt)
  ct29 = ct29_ptct.mul(ct23, use_bat=False)
  v22 = 25
  pt13 = v3[25]
  ct30_pt_ntt = pt13.polynomial[0, 0, ..., : ct26.num_moduli].astype(jnp.uint32)
  ct30_ptct = v0.ptct_mul[v0.max_level - 1]
  ct30_ptct.set_plaintext(ct30_pt_ntt)
  ct30 = ct30_ptct.mul(ct26, use_bat=False)
  ct31 = v0.he_add[v0.max_level - 1].add(ct28, ct29)
  ct32 = v0.he_add[v0.max_level - 1].add(ct31, ct30)
  ct33 = v0.he_rot[v0.max_level - 1, 3].rotate(ct32)
  v23 = 26
  pt14 = v3[26]
  ct34_pt_ntt = pt14.polynomial[0, 0, ..., : ct20.num_moduli].astype(jnp.uint32)
  ct34_ptct = v0.ptct_mul[v0.max_level - 1]
  ct34_ptct.set_plaintext(ct34_pt_ntt)
  ct34 = ct34_ptct.mul(ct20, use_bat=False)
  v24 = 27
  pt15 = v3[27]
  ct35_pt_ntt = pt15.polynomial[0, 0, ..., : ct23.num_moduli].astype(jnp.uint32)
  ct35_ptct = v0.ptct_mul[v0.max_level - 1]
  ct35_ptct.set_plaintext(ct35_pt_ntt)
  ct35 = ct35_ptct.mul(ct23, use_bat=False)
  ct36 = v0.he_add[v0.max_level - 1].add(ct34, ct35)
  ct37 = v0.he_rot[v0.max_level - 1, 6].rotate(ct36)
  ct38 = v0.he_add[v0.max_level - 1].add(ct21, ct24)
  ct39 = v0.he_add[v0.max_level - 1].add(ct27, ct33)
  ct40 = v0.he_add[v0.max_level - 1].add(ct39, ct37)
  ct41 = v0.he_add[v0.max_level - 1].add(ct38, ct40)
  v25 = [None] * 1
  ct42 = v0.he_rescale[v0.max_level - 1, v0.max_level - 2].rescale(ct41)
  v25[0] = ct42
  v26 = v25
  return v26


def matvec_chain(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  v3 = matvec_chain__preprocessing(v0, v1)
  v4 = matvec_chain__preprocessed(v0, v1, v2, v3)
  return v4


def matvec_chain__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = _encode_slots(v0, v15)
  v0.public_key = v3
  ct = v0.encrypt(pt)
  v16 = [ct]
  return v16


def matvec_chain__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 7
  v8 = 0
  v9 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  pt = v0.decrypt(ct)
  v10 = _decode_active(v0, pt, 8).reshape(1, 8)
  v11 = v9.copy()
  for v12 in range(0, 8):
    v14 = v7 - v12
    v15 = int(v14)
    v16 = v10[0, v15]
    v11[v15] = v16
  return v11


def matvec_identity__generate_crypto_context() -> ckks.CKKSContext:
  params = {
      "degree": 8192,
      "num_slots": 4096,
      "batch": 1,
      "r": 64,
      "c": 128,
      "dnum": 3,
      "scaling_factor": 1073692673,
      "output_scale": 1073692673,
      "q_towers": [
          2147352577,
          1073430529,
          1073872897,
          1073479681,
          1073643521,
          1073692673,
      ],
      "p_towers": [1073299457, 1073233921, 1073184769, 1073135617],
      "composite_degree": 1,
      "p": 30,
      "max_bits_in_word": 61,
      "max_bits_value": 9223372036854775295,
      "noise_scale_degree": 1,
      "CKKS_M_FACTOR": 1,
  }
  v0 = ckks.CKKSContext(params)
  return v0


def matvec_identity__configure_crypto_context(
    v0: ckks.CKKSContext,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: dict,
):
  v0.public_key = v1
  v0.secret_key = v2
  v0.evaluation_key = v3
  v0.parameters["public_key"] = v1
  v0.parameters["secret_key"] = v2
  v0.parameters["evaluation_key"] = v3
  v0.program_initialization(
      total_rotation_indices=[1, 2, 3, 6], dnum=3, r=64, c=128, batch=1
  )
