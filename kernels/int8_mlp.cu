// int8_mlp.cu — INT8 quantized MLP CUDA kernel.
// Owner: Heling
//
// Optimization log:
//   v1-v4: see commit history.
//   v5: epilogue fusion (GEMM+dequant+activation).
//   v5.1: quantize scale stays on GPU (no CPU sync in quant_utils).
//   v6 (this): warp-level tiling — each warp computes a 32×32 output tile
//              with 2×2 accumulator fragments, reusing each loaded A/B
//              fragment twice across mma calls. Halves HBM fragment loads.

#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>
#include <mma.h>

using namespace nvcuda::wmma;

extern "C" void quantize_fp16_to_int8(
    const half* input, int8_t* output, float* scale_out, int n
);

// ── Fused GEMM + dequant + GELU, 32×32 warp tile ───────────────────────
// C[M,N] = gelu(dequant(A[M,K] @ B[K,N], scale))
// Each warp holds 4 accumulator fragments (2×2 of 16×16), reusing A and B
// fragments between them.
static __global__ void int8_gemm_dequant_gelu_wmma(
    const int8_t* A, const int8_t* B, half* C,
    int M, int N, int K, float scale
) {
    int row = blockIdx.y * 32;
    int col = blockIdx.x * 32;
    if (row >= M || col >= N) return;

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag[2];
    fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag[2];
    fragment<accumulator, 16, 16, 16, int32_t>        c_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0);

    for (int k = 0; k < K; k += 16) {
        // Two A fragments (top and bottom rows of the 32×32 tile)
        load_matrix_sync(a_frag[0], A + (row + 0)  * K + k, K);
        load_matrix_sync(a_frag[1], A + (row + 16) * K + k, K);
        // Two B fragments (left and right cols)
        load_matrix_sync(b_frag[0], B + k * N + (col + 0),  N);
        load_matrix_sync(b_frag[1], B + k * N + (col + 16), N);

        // 4 mma ops sharing loaded fragments
        mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
        mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
        mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
        mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
    }

    // Epilogue: stage the 32×32 INT32 tile in smem, then dequant + GELU
    __shared__ int32_t c_smem[32 * 32];
    store_matrix_sync(c_smem +  0 * 32 +  0, c_frag[0][0], 32, mem_row_major);
    store_matrix_sync(c_smem +  0 * 32 + 16, c_frag[0][1], 32, mem_row_major);
    store_matrix_sync(c_smem + 16 * 32 +  0, c_frag[1][0], 32, mem_row_major);
    store_matrix_sync(c_smem + 16 * 32 + 16, c_frag[1][1], 32, mem_row_major);
    __syncwarp();

    const float k0 = 0.7978845608f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    int lane = threadIdx.x;
    #pragma unroll
    for (int i = lane; i < 1024; i += 32) {
        int r = i >> 5;
        int c = i & 31;
        if (row + r < M && col + c < N) {
            float f = (float)c_smem[i] * scale;
            float inner = k0 * (f + k1 * f * f * f);
            float g = 0.5f * f * (1.0f + tanhf(inner));
            C[(row + r) * N + (col + c)] = __float2half(g);
        }
    }
}

// ── Fused GEMM + dequant, 32×32 warp tile ──────────────────────────────
static __global__ void int8_gemm_dequant_wmma(
    const int8_t* A, const int8_t* B, half* C,
    int M, int N, int K,
    const float* scale_A_ptr, float scale_B
) {
    int row = blockIdx.y * 32;
    int col = blockIdx.x * 32;
    if (row >= M || col >= N) return;

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag[2];
    fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag[2];
    fragment<accumulator, 16, 16, 16, int32_t>        c_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0);

    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag[0], A + (row + 0)  * K + k, K);
        load_matrix_sync(a_frag[1], A + (row + 16) * K + k, K);
        load_matrix_sync(b_frag[0], B + k * N + (col + 0),  N);
        load_matrix_sync(b_frag[1], B + k * N + (col + 16), N);

        mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
        mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
        mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
        mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
    }

    __shared__ int32_t c_smem[32 * 32];
    store_matrix_sync(c_smem +  0 * 32 +  0, c_frag[0][0], 32, mem_row_major);
    store_matrix_sync(c_smem +  0 * 32 + 16, c_frag[0][1], 32, mem_row_major);
    store_matrix_sync(c_smem + 16 * 32 +  0, c_frag[1][0], 32, mem_row_major);
    store_matrix_sync(c_smem + 16 * 32 + 16, c_frag[1][1], 32, mem_row_major);
    __syncwarp();

    float scale = (*scale_A_ptr) * scale_B;
    int lane = threadIdx.x;
    #pragma unroll
    for (int i = lane; i < 1024; i += 32) {
        int r = i >> 5;
        int c = i & 31;
        if (row + r < M && col + c < N) {
            float f = (float)c_smem[i] * scale;
            C[(row + r) * N + (col + c)] = __float2half(f);
        }
    }
}

// ── Cached workspace ───────────────────────────────────────────────────
struct MlpWorkspace {
    half*    hidden_fp16 = nullptr;
    int8_t*  hidden_int8 = nullptr;
    float*   d_scale     = nullptr;
    size_t   hidden_cap  = 0;

    void ensure(size_t hidden_n) {
        if (hidden_n > hidden_cap) {
            if (hidden_fp16) cudaFree(hidden_fp16);
            if (hidden_int8) cudaFree(hidden_int8);
            cudaMalloc(&hidden_fp16, hidden_n * sizeof(half));
            cudaMalloc(&hidden_int8, hidden_n * sizeof(int8_t));
            hidden_cap = hidden_n;
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
    g_mlp_ws.ensure(hidden_n);

    // Fused GEMM 1 (32×32 warp tile)
    {
        dim3 block(32);
        dim3 grid((N1 + 31) / 32, (M + 31) / 32);
        int8_gemm_dequant_gelu_wmma<<<grid, block>>>(
            x, W1, g_mlp_ws.hidden_fp16,
            M, N1, K, scale_x * scale_W1
        );
    }

    quantize_fp16_to_int8(
        g_mlp_ws.hidden_fp16, g_mlp_ws.hidden_int8,
        g_mlp_ws.d_scale, (int)hidden_n
    );

    // Fused GEMM 2 (32×32 warp tile)
    {
        dim3 block(32);
        dim3 grid((N2 + 31) / 32, (M + 31) / 32);
        int8_gemm_dequant_wmma<<<grid, block>>>(
            g_mlp_ws.hidden_int8, W2, out,
            M, N2, d_ff,
            g_mlp_ws.d_scale, scale_W2
        );
    }
}