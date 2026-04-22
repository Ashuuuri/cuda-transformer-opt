// int8_mlp.cu — INT8 quantized MLP CUDA kernel.
// Owner: Heling
//
// Optimization log:
//   v1: cudaMalloc/cudaFree each forward.
//   v2: cached workspace (static struct).
//   v3 (this): WMMA INT8 tensor core GEMM; scale stays on GPU (no D2H copy).

#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>
#include <mma.h>

using namespace nvcuda::wmma;

extern "C" void quantize_fp16_to_int8(
    const half* input, int8_t* output, float* scale_out, int n
);

// ── WMMA INT8 GEMM: C[M,N] = A[M,K] × B[K,N] (all row-major) ────────────
// One warp per 16×16 output tile. INT8 inputs, INT32 accumulation.
__global__ void int8_gemm_wmma(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    int row = blockIdx.y * 16;   // tile-row in A / C
    int col = blockIdx.x * 16;   // tile-col in B / C
    if (row >= M || col >= N) return;

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t>        c_frag;
    fill_fragment(c_frag, 0);

    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag, A + row * K + k, K);
        load_matrix_sync(b_frag, B + k   * N + col, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + row * N + col, c_frag, N, mem_row_major);
}

// ── Dequantize INT32 → FP16, apply tanh GELU ───────────────────────────
__global__ void dequant_gelu_kernel(
    const int32_t* input, half* output, float scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = (float)input[idx] * scale;
    const float k0 = 0.7978845608f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    float g = 0.5f * x * (1.0f + tanhf(inner));
    output[idx] = __float2half(g);
}

// ── Dequantize INT32 → FP16 using a DEVICE-side scale pointer ─────────
// scale = (*scale_A_ptr) * scale_B. Avoids D2H copy of the intermediate scale.
static __global__ void dequant_kernel_dev_mlp(
    const int32_t* input, half* output,
    const float* scale_A_ptr, float scale_B, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float scale = (*scale_A_ptr) * scale_B;
    output[idx] = __float2half((float)input[idx] * scale);
}

// ── Cached workspace ───────────────────────────────────────────────────
struct MlpWorkspace {
    int32_t* hidden_int32 = nullptr;
    half*    hidden_fp16  = nullptr;
    int8_t*  hidden_int8  = nullptr;
    int32_t* out_int32    = nullptr;
    float*   d_scale      = nullptr;
    size_t   hidden_cap   = 0;
    size_t   out_cap      = 0;

    void ensure(size_t hidden_n, size_t out_n) {
        if (hidden_n > hidden_cap) {
            if (hidden_int32) cudaFree(hidden_int32);
            if (hidden_fp16)  cudaFree(hidden_fp16);
            if (hidden_int8)  cudaFree(hidden_int8);
            cudaMalloc(&hidden_int32, hidden_n * sizeof(int32_t));
            cudaMalloc(&hidden_fp16,  hidden_n * sizeof(half));
            cudaMalloc(&hidden_int8,  hidden_n * sizeof(int8_t));
            hidden_cap = hidden_n;
        }
        if (out_n > out_cap) {
            if (out_int32) cudaFree(out_int32);
            cudaMalloc(&out_int32, out_n * sizeof(int32_t));
            out_cap = out_n;
        }
        if (!d_scale) cudaMalloc(&d_scale, sizeof(float));
    }
};

static MlpWorkspace g_mlp_ws;

// ── Public API ─────────────────────────────────────────────────────────
extern "C" void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, half* out,
    float scale_x, float scale_W1, float scale_W2,
    int batch, int seq_len, int d_model, int d_ff
) {
    int M  = batch * seq_len;
    int K  = d_model;
    int N1 = d_ff;
    int N2 = d_model;

    size_t hidden_n = (size_t)M * N1;
    size_t out_n    = (size_t)M * N2;

    g_mlp_ws.ensure(hidden_n, out_n);

    const int threads = 256;

    // GEMM 1: hidden_int32 = x @ W1  (WMMA)
    {
        dim3 block(32);                         // one warp per block
        dim3 grid((N1 + 15) / 16, (M + 15) / 16);
        int8_gemm_wmma<<<grid, block>>>(x, W1, g_mlp_ws.hidden_int32, M, N1, K);
    }

    // dequant + GELU → hidden_fp16
    {
        int blocks = (hidden_n + threads - 1) / threads;
        dequant_gelu_kernel<<<blocks, threads>>>(
            g_mlp_ws.hidden_int32, g_mlp_ws.hidden_fp16,
            scale_x * scale_W1, (int)hidden_n
        );
    }

    // quantize hidden_fp16 → hidden_int8, scale lands in g_mlp_ws.d_scale (device)
    quantize_fp16_to_int8(
        g_mlp_ws.hidden_fp16, g_mlp_ws.hidden_int8,
        g_mlp_ws.d_scale, (int)hidden_n
    );

    // GEMM 2: out_int32 = hidden_int8 @ W2  (WMMA)
    {
        dim3 block(32);
        dim3 grid((N2 + 15) / 16, (M + 15) / 16);
        int8_gemm_wmma<<<grid, block>>>(g_mlp_ws.hidden_int8, W2, g_mlp_ws.out_int32, M, N2, d_ff);
    }

    // dequant → FP16 output using device-side scale (no D2H copy)
    {
        int blocks = (out_n + threads - 1) / threads;
        dequant_kernel_dev_mlp<<<blocks, threads>>>(
            g_mlp_ws.out_int32, out,
            g_mlp_ws.d_scale, scale_W2, (int)out_n
        );
    }
}
