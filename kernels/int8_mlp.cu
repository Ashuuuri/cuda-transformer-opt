// int8_mlp.cu — INT8 quantized two-layer MLP (Linear → GELU → Linear).
// Owner: Heling
//
// Architecture mirrors Sherry's FP16 mlp.cu:
//   64×64 block tile, 8 warps, 16×16×16 WMMA, cp.async double-buffer (STAGE_K=32).
//   Fragment types changed to int8_t/int32_t; epilogue dequantizes INT32→FP16.
//
//   Forward pass:
//     GEMM1: int8 x × int8 W1  →  INT32 acc × (scale_x·scale_W1) → GELU → FP16 hidden
//     Quant: FP16 hidden → INT8 hidden  (dynamic per-tensor scale_hidden)
//     GEMM2: int8 hidden × int8 W2  →  INT32 acc × (scale_hidden·scale_W2) → FP16
//     Quant: FP16 out → INT8 out  (dynamic per-tensor scale_out)
//
// Ablation flags:
//   INT8_MLP_STAGE_K=N   change K-stage depth (default 32)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include "int8_common.cuh"

using namespace nvcuda;

// ── Tile / block shape (mirrors Sherry's mlp.cu) ───────────────────────
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16
#ifndef INT8_MLP_STAGE_K
#define INT8_MLP_STAGE_K 32
#endif
#define STAGE_K         INT8_MLP_STAGE_K
#define BLOCK_M         64    // 4 row-tiles × 16
#define BLOCK_N         64    // 4 col-tiles × 16
#define WARP_COL_TILES   2
#define WARP_COL_GROUPS  2    // BLOCK_N/WMMA_N / WARP_COL_TILES = 4/2
#define WARPS_PER_BLOCK  8    // BLOCK_M/WMMA_M × WARP_COL_GROUPS = 4×2
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)  // 256
#define SCALAR_TILE     16    // scalar fallback tile size

// INT8 smem padding: 16 bytes per row — same byte-level protection as
// Sherry's 8-half (16-byte) skew, expressed in int8 elements.
#define SMEM_SKEW       16
#define A_SMEM_STRIDE   (STAGE_K + SMEM_SKEW)   // 48
#define B_SMEM_STRIDE   (BLOCK_N  + SMEM_SKEW)   // 80

// INT8: 16 bytes per cp.async = 16 elements (vs 8 for FP16).
#define A_COPIES_PER_ROW  (STAGE_K / 16)   // 2
#define B_COPIES_PER_ROW  (BLOCK_N  / 16)  // 4

// ── Scalar fallback ────────────────────────────────────────────────────
// Accumulates INT8 products as float, writes FP16 output.
template <bool apply_gelu>
__global__ void gemm_int8_scalar_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    half* __restrict__         C,
    int M, int N, int K, float scale
) {
    __shared__ float sA[SCALAR_TILE][SCALAR_TILE];
    __shared__ float sB[SCALAR_TILE][SCALAR_TILE];

    const int row = blockIdx.y * SCALAR_TILE + threadIdx.y;
    const int col = blockIdx.x * SCALAR_TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + SCALAR_TILE - 1) / SCALAR_TILE; ++t) {
        const int kA = t * SCALAR_TILE + threadIdx.x;
        const int kB = t * SCALAR_TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? (float)A[row*K + kA] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (kB  < K && col < N) ? (float)B[kB *N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < SCALAR_TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc * scale;
        if (apply_gelu) v = int8_gelu(v);
        C[row * N + col] = __float2half_rn(v);
    }
}

// ── Async tile loader (INT8 double-buffer) ─────────────────────────────
// Loads one BLOCK_M×STAGE_K slab of A and one STAGE_K×BLOCK_N slab of B
// into shared memory via cp.async.  Only called when dimensions are multiples
// of the tile sizes (enforced by the use_wmma check in int8_mlp_forward).
__device__ __forceinline__ void load_int8_tile_async(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* sA, int8_t* sB,
    int block_m, int block_n, int k0, int K, int N
) {
    const int A_COPIES = BLOCK_M * A_COPIES_PER_ROW;    // 128
    const int B_COPIES = STAGE_K * B_COPIES_PER_ROW;    // 128
    const int TOTAL    = A_COPIES + B_COPIES;

    for (int c = threadIdx.x; c < TOTAL; c += THREADS_PER_BLOCK) {
        if (c < A_COPIES) {
            const int row = c / A_COPIES_PER_ROW;
            const int col = (c % A_COPIES_PER_ROW) * 16;
            i8_cp_async_16B(sA + row * A_SMEM_STRIDE + col,
                            A  + (block_m + row) * K + k0 + col);
        } else {
            const int bc  = c - A_COPIES;
            const int row = bc / B_COPIES_PER_ROW;
            const int col = (bc % B_COPIES_PER_ROW) * 16;
            i8_cp_async_16B(sB + row * B_SMEM_STRIDE + col,
                            B  + (k0 + row) * N + block_n + col);
        }
    }
}

// ── INT8 WMMA GEMM kernel ───────────────────────────────────────────────
// C (FP16) = dequant( A (INT8) × B (INT8) )
//   real[i,j] = int32_acc[i,j] * scale   [+ GELU]
template <bool apply_gelu>
__global__ void gemm_int8_wmma_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    half* __restrict__         C,
    int M, int N, int K, float scale
) {
    __shared__ __align__(16) int8_t  sA[2][BLOCK_M * A_SMEM_STRIDE];
    __shared__ __align__(16) int8_t  sB[2][STAGE_K * B_SMEM_STRIDE];
    __shared__               int32_t c_smem[WARPS_PER_BLOCK * WARP_COL_TILES * WMMA_M * WMMA_N];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m  = warp_id / WARP_COL_GROUPS;
    const int warp_ng = warp_id % WARP_COL_GROUPS;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    const int tile_m  = block_m + warp_m * WMMA_M;
    const int tile_n0 = block_n + warp_ng * WARP_COL_TILES * WMMA_N;
    const int tile_n1 = tile_n0 + WMMA_N;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag0, b_frag1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc0, acc1;
    wmma::fill_fragment(acc0, (int32_t)0);
    wmma::fill_fragment(acc1, (int32_t)0);

    int stage = 0;
    load_int8_tile_async(A, B, sA[0], sB[0], block_m, block_n, 0, K, N);
    i8_cp_async_commit();

    for (int k0 = 0; k0 < K; k0 += STAGE_K) {
        i8_cp_async_wait_all();
        __syncthreads();

        const int next_k0 = k0 + STAGE_K;
        const int next_s  = stage ^ 1;
        if (next_k0 < K) {
            load_int8_tile_async(A, B, sA[next_s], sB[next_s],
                                 block_m, block_n, next_k0, K, N);
            i8_cp_async_commit();
        }

        #pragma unroll
        for (int kk = 0; kk < STAGE_K; kk += WMMA_K) {
            const int8_t* a_ptr  = sA[stage] + warp_m * WMMA_M * A_SMEM_STRIDE + kk;
            const int8_t* b_ptr0 = sB[stage] + kk * B_SMEM_STRIDE + warp_ng * WARP_COL_TILES * WMMA_N;
            const int8_t* b_ptr1 = b_ptr0 + WMMA_N;
            wmma::load_matrix_sync(a_frag,  a_ptr,  A_SMEM_STRIDE);
            wmma::load_matrix_sync(b_frag0, b_ptr0, B_SMEM_STRIDE);
            wmma::load_matrix_sync(b_frag1, b_ptr1, B_SMEM_STRIDE);
            wmma::mma_sync(acc0, a_frag, b_frag0, acc0);
            wmma::mma_sync(acc1, a_frag, b_frag1, acc1);
        }

        __syncthreads();
        stage = next_s;
    }

    // Epilogue: INT32 → float → ×scale → [GELU] → FP16
    int32_t* c0 = c_smem + warp_id * WARP_COL_TILES * WMMA_M * WMMA_N;
    int32_t* c1 = c0 + WMMA_M * WMMA_N;
    wmma::store_matrix_sync(c0, acc0, WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(c1, acc1, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
        const int r = i / WMMA_N, c = i % WMMA_N;
        float v0 = (float)c0[i] * scale;
        float v1 = (float)c1[i] * scale;
        if (apply_gelu) { v0 = int8_gelu(v0); v1 = int8_gelu(v1); }
        C[(tile_m + r) * N + (tile_n0 + c)] = __float2half_rn(v0);
        C[(tile_m + r) * N + (tile_n1 + c)] = __float2half_rn(v1);
    }
}

// ── File-scope device buffer cache ────────────────────────────────────
// Kept at file scope so int8_mlp_get_output_scale() can access s_scale_out.
static half*   s_hidden_fp16  = nullptr;
static int8_t* s_hidden_int8  = nullptr;
static half*   s_out_fp16     = nullptr;
static float*  s_scale_hidden = nullptr;
static float*  s_scale_out    = nullptr;
static size_t  s_hidden_cap   = 0;
static size_t  s_out_cap      = 0;

// ── Public interface ───────────────────────────────────────────────────
void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, int8_t* out,
    const float* scale_x, const float* scale_W1, const float* scale_W2,
    int batch, int seq_len, int d_model, int d_ff
) {
    const int T = batch * seq_len;

    const size_t hidden_fp16_sz = (size_t)T * d_ff    * sizeof(half);
    const size_t hidden_i8_sz   = (size_t)T * d_ff    * sizeof(int8_t);
    const size_t out_fp16_sz    = (size_t)T * d_model * sizeof(half);

    if (hidden_fp16_sz > s_hidden_cap) {
        cudaFree(s_hidden_fp16);  cudaFree(s_hidden_int8);  cudaFree(s_scale_hidden);
        cudaMalloc(&s_hidden_fp16,  hidden_fp16_sz);
        cudaMalloc(&s_hidden_int8,  hidden_i8_sz);
        cudaMalloc(&s_scale_hidden, sizeof(float));
        s_hidden_cap = hidden_fp16_sz;
    }
    if (out_fp16_sz > s_out_cap) {
        cudaFree(s_out_fp16);  cudaFree(s_scale_out);
        cudaMalloc(&s_out_fp16,  out_fp16_sz);
        cudaMalloc(&s_scale_out, sizeof(float));
        s_out_cap = out_fp16_sz;
    }

    // Read input scales to host once; avoids repeated device reads in epilogues.
    float h_sx, h_sw1, h_sw2;
    cudaMemcpy(&h_sx,  scale_x,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sw1, scale_W1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sw2, scale_W2, sizeof(float), cudaMemcpyDeviceToHost);

    const bool use_wmma =
        (T      % BLOCK_M == 0) && (d_model % BLOCK_N == 0) &&
        (d_ff   % BLOCK_N == 0) && (d_model % STAGE_K == 0) &&
        (d_ff   % STAGE_K == 0);

    // ── GEMM 1: x @ W1 → FP16 hidden (with GELU) ───────────────────────
    if (use_wmma) {
        dim3 grid(d_ff / BLOCK_N, T / BLOCK_M);
        gemm_int8_wmma_kernel<true><<<grid, THREADS_PER_BLOCK>>>(
            x, W1, s_hidden_fp16, T, d_ff, d_model, h_sx * h_sw1);
    } else {
        dim3 grid((d_ff + SCALAR_TILE-1)/SCALAR_TILE, (T + SCALAR_TILE-1)/SCALAR_TILE);
        gemm_int8_scalar_kernel<true><<<grid, dim3(SCALAR_TILE, SCALAR_TILE)>>>(
            x, W1, s_hidden_fp16, T, d_ff, d_model, h_sx * h_sw1);
    }

    // ── Quantize hidden (FP16 → INT8, computes scale_hidden) ────────────
    quantize_fp16_to_int8(s_hidden_fp16, s_hidden_int8, s_scale_hidden, T * d_ff);

    float h_sh;
    cudaMemcpy(&h_sh, s_scale_hidden, sizeof(float), cudaMemcpyDeviceToHost);

    // ── GEMM 2: hidden @ W2 → FP16 out ─────────────────────────────────
    if (use_wmma) {
        dim3 grid(d_model / BLOCK_N, T / BLOCK_M);
        gemm_int8_wmma_kernel<false><<<grid, THREADS_PER_BLOCK>>>(
            s_hidden_int8, W2, s_out_fp16, T, d_model, d_ff, h_sh * h_sw2);
    } else {
        dim3 grid((d_model + SCALAR_TILE-1)/SCALAR_TILE, (T + SCALAR_TILE-1)/SCALAR_TILE);
        gemm_int8_scalar_kernel<false><<<grid, dim3(SCALAR_TILE, SCALAR_TILE)>>>(
            s_hidden_int8, W2, s_out_fp16, T, d_model, d_ff, h_sh * h_sw2);
    }

    // ── Quantize output (FP16 → INT8) ────────────────────────────────────
    quantize_fp16_to_int8(s_out_fp16, out, s_scale_out, T * d_model);
}

// Returns the per-tensor output scale used in the most recent int8_mlp_forward call.
// Needed by the Python binding to correctly dequantize for correctness checking.
void int8_mlp_get_output_scale(float* host_out) {
    if (s_scale_out)
        cudaMemcpy(host_out, s_scale_out, sizeof(float), cudaMemcpyDeviceToHost);
    else
        *host_out = 1.f;
}
