!Z1073184769_i64 = !mod_arith.int<1073184769 : i64>
!Z1073872897_i64 = !mod_arith.int<1073872897 : i64>
!Z2147352577_i64 = !mod_arith.int<2147352577 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 30>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 60>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 7 }">
#modulus_chain_L5_C0 = #lwe.modulus_chain<elements = <2147352577 : i64, 1073184769 : i64, 1073872897 : i64, 1073971201 : i64, 1073479681 : i64, 1073643521 : i64>, current = 0>
#modulus_chain_L5_C1 = #lwe.modulus_chain<elements = <2147352577 : i64, 1073184769 : i64, 1073872897 : i64, 1073971201 : i64, 1073479681 : i64, 1073643521 : i64>, current = 1>
#modulus_chain_L5_C2 = #lwe.modulus_chain<elements = <2147352577 : i64, 1073184769 : i64, 1073872897 : i64, 1073971201 : i64, 1073479681 : i64, 1073643521 : i64>, current = 2>
#ring_f64_1_x8 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
!rns_L0 = !rns.rns<!Z2147352577_i64>
!rns_L1 = !rns.rns<!Z2147352577_i64, !Z1073184769_i64>
!rns_L2 = !rns.rns<!Z2147352577_i64, !Z1073184769_i64, !Z1073872897_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>>
#ring_rns_L0_1_x8 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**8>>
#ring_rns_L2_1_x8 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**8>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x8, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x8, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L5_C0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L5_C1>
!ct_L1_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L5_C1>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L5_C2>
!ct_L2_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L5_C2>
module attributes {scheme.ckks} {
  func.func @matvec_identity__preprocessing(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>) -> memref<28x!pt> attributes {client.pack_func = {func_name = "matvec_identity"}} {
    %cst = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %alloc = memref.alloc() : memref<28x!pt>
    %pt = jaxiteword.encode %arg0, %cst_0 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c0 = arith.constant 0 : index
    memref.store %pt, %alloc[%c0] : memref<28x!pt>
    %pt_1 = jaxiteword.encode %arg0, %cst : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c1 = arith.constant 1 : index
    memref.store %pt_1, %alloc[%c1] : memref<28x!pt>
    return %alloc : memref<28x!pt>
  }
  func.func @matvec_identity__preprocessed(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2>, %arg3: memref<28x!pt>) -> tensor<1x!ct_L1> attributes {client.preprocessed_func = {func_name = "matvec_identity"}} {
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L2>
    %ct = jaxiteword.rot %arg0, %extracted, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %c0_0 = arith.constant 0 : index
    %pt = memref.load %arg3[%c0_0] : memref<28x!pt>
    %ct_1 = jaxiteword.mul_plain %arg0, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_2 = jaxiteword.rot %arg0, %extracted, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_3 = jaxiteword.mul_plain %arg0, %ct_2, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_4 = jaxiteword.mul_plain %arg0, %extracted, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_5 = jaxiteword.add %arg0, %ct_4, %ct_1 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_6 = jaxiteword.add %arg0, %ct_5, %ct_3 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_7 = jaxiteword.rot %arg0, %ct_6, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_8 = jaxiteword.rot %arg0, %ct_5, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %c1_9 = arith.constant 1 : index
    %pt_10 = memref.load %arg3[%c1_9] : memref<28x!pt>
    %ct_11 = jaxiteword.mul_plain %arg0, %extracted, %pt_10 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_12 = jaxiteword.add %arg0, %ct_11, %ct_1 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_13 = jaxiteword.add %arg0, %ct_3, %ct_7 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_14 = jaxiteword.add %arg0, %ct_13, %ct_8 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_15 = jaxiteword.add %arg0, %ct_12, %ct_14 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %0 = tensor.empty() : tensor<1x!ct_L1>
    %ct_16 = jaxiteword.mod_reduce %arg0, %ct_15 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %inserted = tensor.insert %ct_16 into %0[%c0] : tensor<1x!ct_L1>
    return %inserted : tensor<1x!ct_L1>
  }
  func.func @matvec_identity(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 8>}, tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) {
    %0 = call @matvec_identity__preprocessing(%arg0, %arg1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>) -> memref<28x!pt>
    %1 = call @matvec_identity__preprocessed(%arg0, %arg1, %arg2, %0) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>, tensor<1x!ct_L2>, memref<28x!pt>) -> tensor<1x!ct_L1>
    return %1 : tensor<1x!ct_L1>
  }
  func.func @matvec_identity__encrypt__arg0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<8xf32>, %arg3: !jaxiteword.public_key<>) -> tensor<1x!ct_L2> attributes {client.enc_func = {func_name = "matvec_identity", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg4 : i32 to index
      %extracted = tensor.extract %arg2[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg5[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %extracted_slice : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %ct = jaxiteword.encrypt %arg0, %pt, %arg3 : (!jaxiteword.crypto_context<>, !pt, !jaxiteword.public_key<>) -> !ct_L2
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L2>
    return %from_elements : tensor<1x!ct_L2>
  }
  func.func @matvec_identity__decrypt__result0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L1>, %arg3: !jaxiteword.private_key<>) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec_identity", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L1>
    %pt = jaxiteword.decrypt %arg0, %extracted, %arg3 : (!jaxiteword.crypto_context<>, !ct_L1, !jaxiteword.private_key<>) -> !pt
    %0 = jaxiteword.decode %arg0, %pt : (!jaxiteword.crypto_context<>, !pt) -> tensor<1x8xf32>
    %1 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<8xf32>)  : i32 {
      %2 = arith.subi %c7_i32, %arg4 : i32
      %3 = arith.index_cast %2 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %3] : tensor<1x8xf32>
      %inserted = tensor.insert %extracted_0 into %arg5[%3] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %1 : tensor<8xf32>
  }
  func.func @matvec_shift__preprocessing(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>) -> memref<28x!pt> attributes {client.pack_func = {func_name = "matvec_shift"}} {
    %cst = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %alloc = memref.alloc() : memref<28x!pt>
    %pt = jaxiteword.encode %arg0, %cst_0 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c2 = arith.constant 2 : index
    memref.store %pt, %alloc[%c2] : memref<28x!pt>
    %pt_1 = jaxiteword.encode %arg0, %cst : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c3 = arith.constant 3 : index
    memref.store %pt_1, %alloc[%c3] : memref<28x!pt>
    return %alloc : memref<28x!pt>
  }
  func.func @matvec_shift__preprocessed(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2>, %arg3: memref<28x!pt>) -> tensor<1x!ct_L1> attributes {client.preprocessed_func = {func_name = "matvec_shift"}} {
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_0 = arith.constant 2 : index
    %pt = memref.load %arg3[%c2_0] : memref<28x!pt>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L2>
    %ct = jaxiteword.mul_plain %arg0, %extracted, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_1 = jaxiteword.rot %arg0, %extracted, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_2 = jaxiteword.rot %arg0, %extracted, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %ct_3 = jaxiteword.mul_plain %arg0, %ct_2, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_4 = jaxiteword.mul_plain %arg0, %ct_1, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_5 = jaxiteword.add %arg0, %ct, %ct_4 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_6 = jaxiteword.add %arg0, %ct_5, %ct_3 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_7 = jaxiteword.rot %arg0, %ct_6, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_8 = jaxiteword.rot %arg0, %ct_5, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %c3_9 = arith.constant 3 : index
    %pt_10 = memref.load %arg3[%c3_9] : memref<28x!pt>
    %ct_11 = jaxiteword.mul_plain %arg0, %ct_1, %pt_10 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_12 = jaxiteword.add %arg0, %ct, %ct_11 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_13 = jaxiteword.add %arg0, %ct_3, %ct_7 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_14 = jaxiteword.add %arg0, %ct_13, %ct_8 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_15 = jaxiteword.add %arg0, %ct_12, %ct_14 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %0 = tensor.empty() : tensor<1x!ct_L1>
    %ct_16 = jaxiteword.mod_reduce %arg0, %ct_15 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %inserted = tensor.insert %ct_16 into %0[%c0] : tensor<1x!ct_L1>
    return %inserted : tensor<1x!ct_L1>
  }
  func.func @matvec_shift(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 8>}, tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) {
    %0 = call @matvec_shift__preprocessing(%arg0, %arg1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>) -> memref<28x!pt>
    %1 = call @matvec_shift__preprocessed(%arg0, %arg1, %arg2, %0) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>, tensor<1x!ct_L2>, memref<28x!pt>) -> tensor<1x!ct_L1>
    return %1 : tensor<1x!ct_L1>
  }
  func.func @matvec_shift__encrypt__arg0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<8xf32>, %arg3: !jaxiteword.public_key<>) -> tensor<1x!ct_L2> attributes {client.enc_func = {func_name = "matvec_shift", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg4 : i32 to index
      %extracted = tensor.extract %arg2[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg5[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %extracted_slice : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %ct = jaxiteword.encrypt %arg0, %pt, %arg3 : (!jaxiteword.crypto_context<>, !pt, !jaxiteword.public_key<>) -> !ct_L2
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L2>
    return %from_elements : tensor<1x!ct_L2>
  }
  func.func @matvec_shift__decrypt__result0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L1>, %arg3: !jaxiteword.private_key<>) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec_shift", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L1>
    %pt = jaxiteword.decrypt %arg0, %extracted, %arg3 : (!jaxiteword.crypto_context<>, !ct_L1, !jaxiteword.private_key<>) -> !pt
    %0 = jaxiteword.decode %arg0, %pt : (!jaxiteword.crypto_context<>, !pt) -> tensor<1x8xf32>
    %1 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<8xf32>)  : i32 {
      %2 = arith.subi %c7_i32, %arg4 : i32
      %3 = arith.index_cast %2 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %3] : tensor<1x8xf32>
      %inserted = tensor.insert %extracted_0 into %arg5[%3] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %1 : tensor<8xf32>
  }
  func.func @matvec_random__preprocessing(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>) -> memref<28x!pt> attributes {client.pack_func = {func_name = "matvec_random"}} {
    %cst = arith.constant dense<[0.811626255, 1.44533789, 0.920695543, 1.07704544, 0.678766131, 1.3587923, 1.236010e+00, 0.777831316]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[1.90635717, 0.139110535, 0.653335392, 1.22558773, 0.285577029, 6.922510e-01, 1.85156107, 0.268135756]> : tensor<8xf32>
    %cst_1 = arith.constant dense<[1.49078846, 1.94282877, 1.26252055, 0.188255787, 1.40004277, 1.08812928, 1.13874948, 0.472367436]> : tensor<8xf32>
    %cst_2 = arith.constant dense<[0.331872642, 0.451223463, 0.185931846, 1.23745108, 1.68164098, 0.365038335, 1.25433517, 0.936289727]> : tensor<8xf32>
    %cst_3 = arith.constant dense<[1.0408361, 1.94221079, 0.718127608, 0.39643541, 0.503444314, 0.655074835, 0.423995823, 0.223598033]> : tensor<8xf32>
    %cst_4 = arith.constant dense<[0.165338188, 1.57275236, 0.83848685, 0.396389604, 0.445467442, 0.796087503, 0.966532945, 1.90288258]> : tensor<8xf32>
    %cst_5 = arith.constant dense<[0.678060233, 1.59183431, 1.93470085, 1.82770872, 1.88504803, 0.615563154, 0.210358858, 0.448468566]> : tensor<8xf32>
    %cst_6 = arith.constant dense<[1.0970372, 0.47938019, 1.63595498, 0.591681957, 1.80017197, 1.67460132, 1.74573469, 1.24211848]> : tensor<8xf32>
    %alloc = memref.alloc() : memref<28x!pt>
    %pt = jaxiteword.encode %arg0, %cst : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c4 = arith.constant 4 : index
    memref.store %pt, %alloc[%c4] : memref<28x!pt>
    %pt_7 = jaxiteword.encode %arg0, %cst_0 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c5 = arith.constant 5 : index
    memref.store %pt_7, %alloc[%c5] : memref<28x!pt>
    %pt_8 = jaxiteword.encode %arg0, %cst_1 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c6 = arith.constant 6 : index
    memref.store %pt_8, %alloc[%c6] : memref<28x!pt>
    %pt_9 = jaxiteword.encode %arg0, %cst_2 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c7 = arith.constant 7 : index
    memref.store %pt_9, %alloc[%c7] : memref<28x!pt>
    %pt_10 = jaxiteword.encode %arg0, %cst_3 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c8 = arith.constant 8 : index
    memref.store %pt_10, %alloc[%c8] : memref<28x!pt>
    %pt_11 = jaxiteword.encode %arg0, %cst_4 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c9 = arith.constant 9 : index
    memref.store %pt_11, %alloc[%c9] : memref<28x!pt>
    %pt_12 = jaxiteword.encode %arg0, %cst_5 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c10 = arith.constant 10 : index
    memref.store %pt_12, %alloc[%c10] : memref<28x!pt>
    %pt_13 = jaxiteword.encode %arg0, %cst_6 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c11 = arith.constant 11 : index
    memref.store %pt_13, %alloc[%c11] : memref<28x!pt>
    return %alloc : memref<28x!pt>
  }
  func.func @matvec_random__preprocessed(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2>, %arg3: memref<28x!pt>) -> tensor<1x!ct_L1> attributes {client.preprocessed_func = {func_name = "matvec_random"}} {
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %pt = memref.load %arg3[%c4] : memref<28x!pt>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L2>
    %ct = jaxiteword.mul_plain %arg0, %extracted, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_0 = jaxiteword.rot %arg0, %extracted, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %c5 = arith.constant 5 : index
    %pt_1 = memref.load %arg3[%c5] : memref<28x!pt>
    %ct_2 = jaxiteword.mul_plain %arg0, %ct_0, %pt_1 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_3 = jaxiteword.rot %arg0, %extracted, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %c6_4 = arith.constant 6 : index
    %pt_5 = memref.load %arg3[%c6_4] : memref<28x!pt>
    %ct_6 = jaxiteword.mul_plain %arg0, %ct_3, %pt_5 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %c7 = arith.constant 7 : index
    %pt_7 = memref.load %arg3[%c7] : memref<28x!pt>
    %ct_8 = jaxiteword.mul_plain %arg0, %extracted, %pt_7 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %c8 = arith.constant 8 : index
    %pt_9 = memref.load %arg3[%c8] : memref<28x!pt>
    %ct_10 = jaxiteword.mul_plain %arg0, %ct_0, %pt_9 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %c9 = arith.constant 9 : index
    %pt_11 = memref.load %arg3[%c9] : memref<28x!pt>
    %ct_12 = jaxiteword.mul_plain %arg0, %ct_3, %pt_11 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_13 = jaxiteword.add %arg0, %ct_8, %ct_10 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_14 = jaxiteword.add %arg0, %ct_13, %ct_12 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_15 = jaxiteword.rot %arg0, %ct_14, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %c10 = arith.constant 10 : index
    %pt_16 = memref.load %arg3[%c10] : memref<28x!pt>
    %ct_17 = jaxiteword.mul_plain %arg0, %extracted, %pt_16 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %c11 = arith.constant 11 : index
    %pt_18 = memref.load %arg3[%c11] : memref<28x!pt>
    %ct_19 = jaxiteword.mul_plain %arg0, %ct_0, %pt_18 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_20 = jaxiteword.add %arg0, %ct_17, %ct_19 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_21 = jaxiteword.rot %arg0, %ct_20, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_22 = jaxiteword.add %arg0, %ct, %ct_2 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_23 = jaxiteword.add %arg0, %ct_6, %ct_15 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_24 = jaxiteword.add %arg0, %ct_23, %ct_21 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_25 = jaxiteword.add %arg0, %ct_22, %ct_24 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %0 = tensor.empty() : tensor<1x!ct_L1>
    %ct_26 = jaxiteword.mod_reduce %arg0, %ct_25 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %inserted = tensor.insert %ct_26 into %0[%c0] : tensor<1x!ct_L1>
    return %inserted : tensor<1x!ct_L1>
  }
  func.func @matvec_random(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 8>}, tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) {
    %0 = call @matvec_random__preprocessing(%arg0, %arg1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>) -> memref<28x!pt>
    %1 = call @matvec_random__preprocessed(%arg0, %arg1, %arg2, %0) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>, tensor<1x!ct_L2>, memref<28x!pt>) -> tensor<1x!ct_L1>
    return %1 : tensor<1x!ct_L1>
  }
  func.func @matvec_random__encrypt__arg0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<8xf32>, %arg3: !jaxiteword.public_key<>) -> tensor<1x!ct_L2> attributes {client.enc_func = {func_name = "matvec_random", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg4 : i32 to index
      %extracted = tensor.extract %arg2[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg5[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %extracted_slice : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %ct = jaxiteword.encrypt %arg0, %pt, %arg3 : (!jaxiteword.crypto_context<>, !pt, !jaxiteword.public_key<>) -> !ct_L2
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L2>
    return %from_elements : tensor<1x!ct_L2>
  }
  func.func @matvec_random__decrypt__result0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L1>, %arg3: !jaxiteword.private_key<>) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec_random", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L1>
    %pt = jaxiteword.decrypt %arg0, %extracted, %arg3 : (!jaxiteword.crypto_context<>, !ct_L1, !jaxiteword.private_key<>) -> !pt
    %0 = jaxiteword.decode %arg0, %pt : (!jaxiteword.crypto_context<>, !pt) -> tensor<1x8xf32>
    %1 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<8xf32>)  : i32 {
      %2 = arith.subi %c7_i32, %arg4 : i32
      %3 = arith.index_cast %2 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %3] : tensor<1x8xf32>
      %inserted = tensor.insert %extracted_0 into %arg5[%3] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %1 : tensor<8xf32>
  }
  func.func @matvec_chain__preprocessing(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>) -> memref<28x!pt> attributes {client.pack_func = {func_name = "matvec_chain"}} {
    %cst = arith.constant dense<[1.340000e+00, 1.220000e+00, 1.050000e+00, 1.500000e+00, 1.010000e+00, 0.879999995, 5.000000e-01, 1.060000e+00]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[5.800000e-01, 5.200000e-01, 0.889999985, 8.600000e-01, 1.170000e+00, 0.819999992, 1.490000e+00, 1.410000e+00]> : tensor<8xf32>
    %cst_1 = arith.constant dense<[1.260000e+00, 1.090000e+00, 1.430000e+00, 1.260000e+00, 6.100000e-01, 8.500000e-01, 6.700000e-01, 0.709999978]> : tensor<8xf32>
    %cst_2 = arith.constant dense<[0.819999992, 1.330000e+00, 7.900000e-01, 7.400000e-01, 1.060000e+00, 1.340000e+00, 1.090000e+00, 6.300000e-01]> : tensor<8xf32>
    %cst_3 = arith.constant dense<[1.160000e+00, 0.839999973, 1.020000e+00, 0.689999997, 6.600000e-01, 8.600000e-01, 1.190000e+00, 6.500000e-01]> : tensor<8xf32>
    %cst_4 = arith.constant dense<[1.350000e+00, 1.050000e+00, 1.400000e+00, 1.070000e+00, 6.500000e-01, 5.400000e-01, 8.000000e-01, 0.899999976]> : tensor<8xf32>
    %cst_5 = arith.constant dense<[0.819999992, 0.899999976, 7.400000e-01, 1.050000e+00, 1.080000e+00, 1.480000e+00, 6.000000e-01, 1.200000e+00]> : tensor<8xf32>
    %cst_6 = arith.constant dense<[1.190000e+00, 1.200000e+00, 0.839999973, 1.350000e+00, 1.020000e+00, 7.600000e-01, 1.390000e+00, 1.130000e+00]> : tensor<8xf32>
    %cst_7 = arith.constant dense<[1.200000e+00, 0.889999985, 1.030000e+00, 7.300000e-01, 9.300000e-01, 7.500000e-01, 0.839999973, 1.170000e+00]> : tensor<8xf32>
    %cst_8 = arith.constant dense<[7.900000e-01, 0.839999973, 1.030000e+00, 7.900000e-01, 1.390000e+00, 9.800000e-01, 8.000000e-01, 9.200000e-01]> : tensor<8xf32>
    %cst_9 = arith.constant dense<[7.300000e-01, 1.230000e+00, 1.130000e+00, 1.130000e+00, 1.440000e+00, 1.490000e+00, 1.020000e+00, 1.180000e+00]> : tensor<8xf32>
    %cst_10 = arith.constant dense<[1.120000e+00, 1.110000e+00, 1.380000e+00, 1.050000e+00, 0.939999997, 1.350000e+00, 5.900000e-01, 1.000000e+00]> : tensor<8xf32>
    %cst_11 = arith.constant dense<[6.200000e-01, 6.200000e-01, 1.010000e+00, 1.220000e+00, 5.600000e-01, 1.220000e+00, 9.300000e-01, 9.300000e-01]> : tensor<8xf32>
    %cst_12 = arith.constant dense<[0.819999992, 1.330000e+00, 1.170000e+00, 9.200000e-01, 0.899999976, 1.110000e+00, 1.220000e+00, 9.900000e-01]> : tensor<8xf32>
    %cst_13 = arith.constant dense<[6.800000e-01, 0.819999992, 9.300000e-01, 9.100000e-01, 1.100000e+00, 1.090000e+00, 1.480000e+00, 1.240000e+00]> : tensor<8xf32>
    %cst_14 = arith.constant dense<[6.800000e-01, 8.600000e-01, 8.100000e-01, 1.370000e+00, 1.050000e+00, 1.120000e+00, 1.180000e+00, 9.800000e-01]> : tensor<8xf32>
    %alloc = memref.alloc() : memref<28x!pt>
    %pt = jaxiteword.encode %arg0, %cst : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c12 = arith.constant 12 : index
    memref.store %pt, %alloc[%c12] : memref<28x!pt>
    %pt_15 = jaxiteword.encode %arg0, %cst_0 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c13 = arith.constant 13 : index
    memref.store %pt_15, %alloc[%c13] : memref<28x!pt>
    %pt_16 = jaxiteword.encode %arg0, %cst_1 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c14 = arith.constant 14 : index
    memref.store %pt_16, %alloc[%c14] : memref<28x!pt>
    %pt_17 = jaxiteword.encode %arg0, %cst_2 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c15 = arith.constant 15 : index
    memref.store %pt_17, %alloc[%c15] : memref<28x!pt>
    %pt_18 = jaxiteword.encode %arg0, %cst_3 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c16 = arith.constant 16 : index
    memref.store %pt_18, %alloc[%c16] : memref<28x!pt>
    %pt_19 = jaxiteword.encode %arg0, %cst_4 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c17 = arith.constant 17 : index
    memref.store %pt_19, %alloc[%c17] : memref<28x!pt>
    %pt_20 = jaxiteword.encode %arg0, %cst_5 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c18 = arith.constant 18 : index
    memref.store %pt_20, %alloc[%c18] : memref<28x!pt>
    %pt_21 = jaxiteword.encode %arg0, %cst_6 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c19 = arith.constant 19 : index
    memref.store %pt_21, %alloc[%c19] : memref<28x!pt>
    %pt_22 = jaxiteword.encode %arg0, %cst_7 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c20 = arith.constant 20 : index
    memref.store %pt_22, %alloc[%c20] : memref<28x!pt>
    %pt_23 = jaxiteword.encode %arg0, %cst_8 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c21 = arith.constant 21 : index
    memref.store %pt_23, %alloc[%c21] : memref<28x!pt>
    %pt_24 = jaxiteword.encode %arg0, %cst_9 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c22 = arith.constant 22 : index
    memref.store %pt_24, %alloc[%c22] : memref<28x!pt>
    %pt_25 = jaxiteword.encode %arg0, %cst_10 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c23 = arith.constant 23 : index
    memref.store %pt_25, %alloc[%c23] : memref<28x!pt>
    %pt_26 = jaxiteword.encode %arg0, %cst_11 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c24 = arith.constant 24 : index
    memref.store %pt_26, %alloc[%c24] : memref<28x!pt>
    %pt_27 = jaxiteword.encode %arg0, %cst_12 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c25 = arith.constant 25 : index
    memref.store %pt_27, %alloc[%c25] : memref<28x!pt>
    %pt_28 = jaxiteword.encode %arg0, %cst_13 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c26 = arith.constant 26 : index
    memref.store %pt_28, %alloc[%c26] : memref<28x!pt>
    %pt_29 = jaxiteword.encode %arg0, %cst_14 : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %c27 = arith.constant 27 : index
    memref.store %pt_29, %alloc[%c27] : memref<28x!pt>
    return %alloc : memref<28x!pt>
  }
  func.func @matvec_chain__preprocessed(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2>, %arg3: memref<28x!pt>) -> tensor<1x!ct_L0> attributes {client.preprocessed_func = {func_name = "matvec_chain"}} {
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %pt = memref.load %arg3[%c12] : memref<28x!pt>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L2>
    %ct = jaxiteword.mul_plain %arg0, %extracted, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_0 = jaxiteword.rot %arg0, %extracted, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %c13 = arith.constant 13 : index
    %pt_1 = memref.load %arg3[%c13] : memref<28x!pt>
    %ct_2 = jaxiteword.mul_plain %arg0, %ct_0, %pt_1 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_3 = jaxiteword.rot %arg0, %extracted, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2, !jaxiteword.eval_key<>) -> !ct_L2
    %c14 = arith.constant 14 : index
    %pt_4 = memref.load %arg3[%c14] : memref<28x!pt>
    %ct_5 = jaxiteword.mul_plain %arg0, %ct_3, %pt_4 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %c15 = arith.constant 15 : index
    %pt_6 = memref.load %arg3[%c15] : memref<28x!pt>
    %ct_7 = jaxiteword.mul_plain %arg0, %extracted, %pt_6 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %c16 = arith.constant 16 : index
    %pt_8 = memref.load %arg3[%c16] : memref<28x!pt>
    %ct_9 = jaxiteword.mul_plain %arg0, %ct_0, %pt_8 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %c17 = arith.constant 17 : index
    %pt_10 = memref.load %arg3[%c17] : memref<28x!pt>
    %ct_11 = jaxiteword.mul_plain %arg0, %ct_3, %pt_10 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_12 = jaxiteword.add %arg0, %ct_7, %ct_9 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_13 = jaxiteword.add %arg0, %ct_12, %ct_11 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_14 = jaxiteword.rot %arg0, %ct_13, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %c18 = arith.constant 18 : index
    %pt_15 = memref.load %arg3[%c18] : memref<28x!pt>
    %ct_16 = jaxiteword.mul_plain %arg0, %extracted, %pt_15 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %c19 = arith.constant 19 : index
    %pt_17 = memref.load %arg3[%c19] : memref<28x!pt>
    %ct_18 = jaxiteword.mul_plain %arg0, %ct_0, %pt_17 : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2_1
    %ct_19 = jaxiteword.add %arg0, %ct_16, %ct_18 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_20 = jaxiteword.rot %arg0, %ct_19, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_21 = jaxiteword.add %arg0, %ct, %ct_2 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_22 = jaxiteword.add %arg0, %ct_5, %ct_14 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_23 = jaxiteword.add %arg0, %ct_22, %ct_20 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_24 = jaxiteword.add %arg0, %ct_21, %ct_23 : (!jaxiteword.crypto_context<>, !ct_L2_1, !ct_L2_1) -> !ct_L2_1
    %ct_25 = jaxiteword.mod_reduce %arg0, %ct_24 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %c20 = arith.constant 20 : index
    %pt_26 = memref.load %arg3[%c20] : memref<28x!pt>
    %ct_27 = jaxiteword.mul_plain %arg0, %ct_25, %pt_26 : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1_1
    %ct_28 = jaxiteword.rot %arg0, %ct_24, %arg1 {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_29 = jaxiteword.mod_reduce %arg0, %ct_28 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %c21 = arith.constant 21 : index
    %pt_30 = memref.load %arg3[%c21] : memref<28x!pt>
    %ct_31 = jaxiteword.mul_plain %arg0, %ct_29, %pt_30 : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1_1
    %ct_32 = jaxiteword.rot %arg0, %ct_24, %arg1 {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L2_1, !jaxiteword.eval_key<>) -> !ct_L2_1
    %ct_33 = jaxiteword.mod_reduce %arg0, %ct_32 : (!jaxiteword.crypto_context<>, !ct_L2_1) -> !ct_L1
    %c22 = arith.constant 22 : index
    %pt_34 = memref.load %arg3[%c22] : memref<28x!pt>
    %ct_35 = jaxiteword.mul_plain %arg0, %ct_33, %pt_34 : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1_1
    %c23 = arith.constant 23 : index
    %pt_36 = memref.load %arg3[%c23] : memref<28x!pt>
    %ct_37 = jaxiteword.mul_plain %arg0, %ct_25, %pt_36 : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1_1
    %c24 = arith.constant 24 : index
    %pt_38 = memref.load %arg3[%c24] : memref<28x!pt>
    %ct_39 = jaxiteword.mul_plain %arg0, %ct_29, %pt_38 : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1_1
    %c25 = arith.constant 25 : index
    %pt_40 = memref.load %arg3[%c25] : memref<28x!pt>
    %ct_41 = jaxiteword.mul_plain %arg0, %ct_33, %pt_40 : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1_1
    %ct_42 = jaxiteword.add %arg0, %ct_37, %ct_39 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_43 = jaxiteword.add %arg0, %ct_42, %ct_41 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_44 = jaxiteword.rot %arg0, %ct_43, %arg1 {index = 3 : i64} : (!jaxiteword.crypto_context<>, !ct_L1_1, !jaxiteword.eval_key<>) -> !ct_L1_1
    %c26 = arith.constant 26 : index
    %pt_45 = memref.load %arg3[%c26] : memref<28x!pt>
    %ct_46 = jaxiteword.mul_plain %arg0, %ct_25, %pt_45 : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1_1
    %c27 = arith.constant 27 : index
    %pt_47 = memref.load %arg3[%c27] : memref<28x!pt>
    %ct_48 = jaxiteword.mul_plain %arg0, %ct_29, %pt_47 : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1_1
    %ct_49 = jaxiteword.add %arg0, %ct_46, %ct_48 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_50 = jaxiteword.rot %arg0, %ct_49, %arg1 {index = 6 : i64} : (!jaxiteword.crypto_context<>, !ct_L1_1, !jaxiteword.eval_key<>) -> !ct_L1_1
    %ct_51 = jaxiteword.add %arg0, %ct_27, %ct_31 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_52 = jaxiteword.add %arg0, %ct_35, %ct_44 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_53 = jaxiteword.add %arg0, %ct_52, %ct_50 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %ct_54 = jaxiteword.add %arg0, %ct_51, %ct_53 : (!jaxiteword.crypto_context<>, !ct_L1_1, !ct_L1_1) -> !ct_L1_1
    %0 = tensor.empty() : tensor<1x!ct_L0>
    %ct_55 = jaxiteword.mod_reduce %arg0, %ct_54 : (!jaxiteword.crypto_context<>, !ct_L1_1) -> !ct_L0
    %inserted = tensor.insert %ct_55 into %0[%c0] : tensor<1x!ct_L0>
    return %inserted : tensor<1x!ct_L0>
  }
  func.func @matvec_chain(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L2> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 8>}, tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L0> {tensor_ext.original_type = #original_type}) {
    %0 = call @matvec_chain__preprocessing(%arg0, %arg1) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>) -> memref<28x!pt>
    %1 = call @matvec_chain__preprocessed(%arg0, %arg1, %arg2, %0) : (!jaxiteword.crypto_context<>, !jaxiteword.eval_key<>, tensor<1x!ct_L2>, memref<28x!pt>) -> tensor<1x!ct_L0>
    return %1 : tensor<1x!ct_L0>
  }
  func.func @matvec_chain__encrypt__arg0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<8xf32>, %arg3: !jaxiteword.public_key<>) -> tensor<1x!ct_L2> attributes {client.enc_func = {func_name = "matvec_chain", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg4 : i32 to index
      %extracted = tensor.extract %arg2[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg5[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %pt = jaxiteword.encode %arg0, %extracted_slice : (!jaxiteword.crypto_context<>, tensor<8xf32>) -> !pt
    %ct = jaxiteword.encrypt %arg0, %pt, %arg3 : (!jaxiteword.crypto_context<>, !pt, !jaxiteword.public_key<>) -> !ct_L2
    %from_elements = tensor.from_elements %ct : tensor<1x!ct_L2>
    return %from_elements : tensor<1x!ct_L2>
  }
  func.func @matvec_chain__decrypt__result0(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L0>, %arg3: !jaxiteword.private_key<>) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec_chain", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L0>
    %pt = jaxiteword.decrypt %arg0, %extracted, %arg3 : (!jaxiteword.crypto_context<>, !ct_L0, !jaxiteword.private_key<>) -> !pt
    %0 = jaxiteword.decode %arg0, %pt : (!jaxiteword.crypto_context<>, !pt) -> tensor<1x8xf32>
    %1 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<8xf32>)  : i32 {
      %2 = arith.subi %c7_i32, %arg4 : i32
      %3 = arith.index_cast %2 : i32 to index
      %extracted_0 = tensor.extract %0[%c0, %3] : tensor<1x8xf32>
      %inserted = tensor.insert %extracted_0 into %arg5[%3] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %1 : tensor<8xf32>
  }
  func.func @matvec_identity__generate_crypto_context() -> !jaxiteword.crypto_context<> {
    %0 = jaxiteword.gen_params  {batch = 1 : i32, c = 128 : i32, compositeDegree = 1 : i32, degree = 16384 : i64, dnum = 3 : i32, numSlots = 8192 : i64, pTowers = array<i64: 1073053697, 1072857089, 1072496641, 1071513601>, qTowers = array<i64: 2147352577, 1073184769, 1073872897, 1073971201, 1073479681, 1073643521>, r = 128 : i32, scalingFactor = 0x41D0000000000000 : f64} : () -> !jaxiteword.crypto_context<>
    return %0 : !jaxiteword.crypto_context<>
  }
  func.func @matvec_identity__configure_crypto_context(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.public_key<>, %arg2: !jaxiteword.private_key<>, %arg3: !jaxiteword.eval_key<>) {
    jaxiteword.program_initialization %arg0, %arg1, %arg2, %arg3 {batch = 1 : i32, c = 128 : i32, dnum = 3 : i32, r = 128 : i32, totalRotationIndices = array<i64: 1, 2, 3, 6>} : (!jaxiteword.crypto_context<>, !jaxiteword.public_key<>, !jaxiteword.private_key<>, !jaxiteword.eval_key<>) -> ()
    return
  }
}
