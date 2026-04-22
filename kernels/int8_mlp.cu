// int8_mlp.cu — INT8 quantized MLP CUDA kernel (NAIVE version).
// Owner: Heling
//
// Required interface (do not change the signature):
//
//   void int8_mlp_forward(
//       const int8_t* x, const int8_t* W1, const int8_t* W2, half* out,
//       float scale_x, float scale_W1, float scale_W2,
//       int batch, int seq_len, int d_model, int d_ff
//   );
//
// Inputs are row-major contiguous:
//   x:   (batch * seq_len, d_model)        INT8
//   W1:  (d_model, d_ff)                   INT8
//   W2:  (d_ff, d_model)                   INT8
//   out: (batch * seq_len, d_model)        FP16
//
// Data flow:
//   x_int8 @ W1_int8 (INT32 acc)
//     → dequant to FP16 (× scale_x × scale_W1)
//     → GELU (tanh approximation)
//     → requant to INT8 (find new scale)
//   hidden_int8 @ W2_int8 (INT32 acc)
//     → dequant to FP16 (× scale_hidden × scale_W2) → output
//
// This is a NAIVE implementation: one thread per output element, no tensor
// cores, no tiling. Used to validate correctness before optimization.

#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>

// Forward declarations from quant_utils.cu
extern "C" void quantize_fp16_to_int8(
    const half* input, int8_t* output, float* scale_out, int n
);

// ── Naive INT8 GEMM with INT32 accumulation ────────────────────────────
// C[M, N] = A[M, K] × B[K, N], all row-major.
// One thread computes one output element.
__global__ void int8_gemm_naive(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int32_t acc = 0;
    for (int k = 0; k < K; k++) {
        acc += (int32_t)A[row * K + k] * (int32_t)B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ── Dequantize INT32 → FP16, then apply tanh-approximation GELU ────────
// scale = scale_A * scale_B (combined scale of the two INT8 inputs).
__global__ void dequant_gelu_kernel(
    const int32_t* input, half* output, float scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = (float)input[idx] * scale;

    // GELU tanh approximation:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float k0 = 0.7978845608f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    float g = 0.5f * x * (1.0f + tanhf(inner));

    output[idx] = __float2half(g);
}

// ── Dequantize INT32 → FP16 (no activation) ────────────────────────────
__global__ void dequant_kernel(
    const int32_t* input, half* output, float scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = (float)input[idx] * scale;
    output[idx] = __float2half(x);
}

// ── Public API ─────────────────────────────────────────────────────────
extern "C" void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, half* out,
    float scale_x, float scale_W1, float scale_W2,
    int batch, int seq_len, int d_model, int d_ff
) {
    int M = batch * seq_len;          // tokens
    int K = d_model;
    int N1 = d_ff;
    int N2 = d_model;

    // ── GEMM 1: hidden_int32 = x @ W1 ──────────────────────────────────
    int32_t* hidden_int32;
    cudaMalloc(&hidden_int32, (size_t)M * N1 * sizeof(int32_t));

    dim3 block1(16, 16);
    dim3 grid1((N1 + 15) / 16, (M + 15) / 16);
    int8_gemm_naive<<<grid1, block1>>>(x, W1, hidden_int32, M, N1, K);

    // ── Dequant + GELU: hidden_fp16 = gelu(hidden_int32 * scale_x * scale_W1) ──
    half* hidden_fp16;
    cudaMalloc(&hidden_fp16, (size_t)M * N1 * sizeof(half));

    int n1 = M * N1;
    int threads = 256;
    int blocks = (n1 + threads - 1) / threads;
    dequant_gelu_kernel<<<blocks, threads>>>(hidden_int32, hidden_fp16, scale_x * scale_W1, n1);

    cudaFree(hidden_int32);

    // ── Requantize hidden_fp16 → hidden_int8 ───────────────────────────
    int8_t* hidden_int8;
    float* d_scale_hidden;
    cudaMalloc(&hidden_int8, (size_t)M * N1 * sizeof(int8_t));
    cudaMalloc(&d_scale_hidden, sizeof(float));

    quantize_fp16_to_int8(hidden_fp16, hidden_int8, d_scale_hidden, n1);

    float scale_hidden;
    cudaMemcpy(&scale_hidden, d_scale_hidden, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(hidden_fp16);
    cudaFree(d_scale_hidden);

    // ── GEMM 2: out_int32 = hidden_int8 @ W2 ───────────────────────────
    int32_t* out_int32;
    cudaMalloc(&out_int32, (size_t)M * N2 * sizeof(int32_t));

    dim3 block2(16, 16);
    dim3 grid2((N2 + 15) / 16, (M + 15) / 16);
    int8_gemm_naive<<<grid2, block2>>>(hidden_int8, W2, out_int32, M, N2, d_ff);

    cudaFree(hidden_int8);

    // ── Dequant: out = out_int32 * scale_hidden * scale_W2 ─────────────
    int n2 = M * N2;
    blocks = (n2 + threads - 1) / threads;
    dequant_kernel<<<blocks, threads>>>(out_int32, out, scale_hidden * scale_W2, n2);

    cudaFree(out_int32);
}
