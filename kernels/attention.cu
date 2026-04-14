// attention.cu — FP16 fused attention CUDA kernel.
// Owner: Jonathan
//
// Required interface (do not change the signature):
//
//   void attention_forward(
//       const half* Q, const half* K, const half* V, half* out,
//       int batch, int heads, int seq_len, int head_dim
//   );
//
// Inputs are row-major contiguous with shape (batch, heads, seq_len, head_dim).
// Scale = 1/sqrt(head_dim), computed inside the kernel.
//
// TODO: Implement optimized scaled dot-product attention kernel.

#include <cuda_fp16.h>

void attention_forward(
    const half* Q, const half* K, const half* V, half* out,
    int batch, int heads, int seq_len, int head_dim
) {
    // TODO: implement
}
