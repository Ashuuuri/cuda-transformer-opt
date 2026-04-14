// quant_utils.cu — Shared quantization utilities (FP16 <-> INT8 conversion).
// Owner: Heling
//
// Suggested utilities:
//
//   // Per-tensor symmetric quantization: FP16 → INT8
//   void quantize_fp16_to_int8(
//       const half* input, int8_t* output, float* scale, int n
//   );
//
//   // Dequantization: INT8 → FP16
//   void dequantize_int8_to_fp16(
//       const int8_t* input, half* output, float scale, int n
//   );
//
// These are shared by int8_attention.cu and int8_mlp.cu.
//
// TODO: Implement quantization/dequantization helpers.

#include <cuda_fp16.h>
#include <cstdint>
