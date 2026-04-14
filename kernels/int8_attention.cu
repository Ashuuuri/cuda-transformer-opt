// int8_attention.cu — INT8 quantized attention CUDA kernel.
// Owner: Heling
//
// Required interface (do not change the signature):
//
//   void int8_attention_forward(
//       const int8_t* Q, const int8_t* K, const int8_t* V, int8_t* out,
//       const float* scale_Q, const float* scale_K, const float* scale_V,
//       int batch, int heads, int seq_len, int head_dim
//   );
//
// Inputs are row-major contiguous with shape (batch, heads, seq_len, head_dim).
// Per-tensor symmetric quantization: real_value = int8_value * scale.
// Scale = 1/sqrt(head_dim), computed inside the kernel.
//
// TODO: Implement INT8 scaled dot-product attention kernel.

#include <cuda_fp16.h>
#include <cstdint>

void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, int8_t* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int batch, int heads, int seq_len, int head_dim
) {
    // TODO: implement
}
