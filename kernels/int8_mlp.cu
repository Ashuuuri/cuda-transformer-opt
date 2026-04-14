// int8_mlp.cu — INT8 quantized MLP CUDA kernel.
// Owner: Heling
//
// Required interface (do not change the signature):
//
//   void int8_mlp_forward(
//       const int8_t* x, const int8_t* W1, const int8_t* W2, int8_t* out,
//       const float* scale_x, const float* scale_W1, const float* scale_W2,
//       int batch, int seq_len, int d_model, int d_ff
//   );
//
// Inputs are row-major contiguous:
//   x:   (batch, seq_len, d_model)
//   W1:  (d_model, d_ff)
//   W2:  (d_ff, d_model)
//   out: (batch, seq_len, d_model)
//
// Per-tensor symmetric quantization: real_value = int8_value * scale.
// GELU uses tanh approximation (dequantize → GELU → requantize).
//
// TODO: Implement INT8 two-layer MLP kernel.

#include <cuda_fp16.h>
#include <cstdint>

void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, int8_t* out,
    const float* scale_x, const float* scale_W1, const float* scale_W2,
    int batch, int seq_len, int d_model, int d_ff
) {
    // TODO: implement
}
