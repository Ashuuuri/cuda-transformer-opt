// mlp.cu — FP16 fused MLP CUDA kernel (Linear -> GELU -> Linear).
// Owner: Shengjing
//
// Required interface (do not change the signature):
//
//   void mlp_forward(
//       const half* x, const half* W1, const half* W2, half* out,
//       int batch, int seq_len, int d_model, int d_ff
//   );
//
// Inputs are row-major contiguous:
//   x:   (batch, seq_len, d_model)
//   W1:  (d_model, d_ff)
//   W2:  (d_ff, d_model)
//   out: (batch, seq_len, d_model)
//
// GELU uses tanh approximation:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Optimization: Tensor Core WMMA
//   - Kernel 1: hidden = GELU(x @ W1)  [fuses matmul + activation epilogue]
//   - Kernel 2: out    = hidden @ W2
//   The fast path uses 16x16x16 WMMA FP16 GEMMs with FP32 accumulation.
//   A scalar fallback is kept for tiny/non-multiple-of-16 tests.
//   The hidden buffer is statically cached to avoid cudaMalloc overhead
//   on repeated calls (important for accurate benchmarking).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cmath>

using namespace nvcuda;

// ── Tile size ──────────────────────────────────────────────────────────
// 16×16 tiles of float32 → 2 × (16×16×4) = 2 KB shared memory per block.
// Small enough to keep high SM occupancy on A100.
#define TILE 16

// ── WMMA tile/block shape ──────────────────────────────────────────────
// Each warp computes one 16x16 C tile. A block covers 64x32 output values
// using 8 warps (256 threads), which keeps enough independent tiles in flight
// without pushing block size to the 512-thread limit.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_ROW_TILES 4
#define BLOCK_COL_TILES 2
#define WARPS_PER_BLOCK (BLOCK_ROW_TILES * BLOCK_COL_TILES)
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

// ── GELU (tanh approximation) ──────────────────────────────────────────
__device__ __forceinline__ float gelu_tanh(float x) {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoef        = 0.044715f;
    float inner = kSqrt2OverPi * (x + kCoef * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ── Tiled FP16 GEMM with optional fused GELU ──────────────────────────
// Computes: C = A @ B  (apply_gelu=false)
//        or C = GELU(A @ B) (apply_gelu=true)
// A: (M, K) row-major half
// B: (K, N) row-major half
// C: (M, N) row-major half
//
// Accumulation is done in float32 to match PyTorch FP16 baseline accuracy.
template <bool apply_gelu>
__global__ void gemm_fp16_scalar_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__       C,
    int M, int N, int K
) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;

    const int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; t++) {
        // Collaboratively load one TILE×TILE chunk of A and B into shared mem.
        int kA = t * TILE + threadIdx.x;   // A column this thread loads
        int kB = t * TILE + threadIdx.y;   // B row this thread loads

        sA[threadIdx.y][threadIdx.x] = (row < M && kA < K)
                                       ? __half2float(A[row * K + kA]) : 0.f;
        sB[threadIdx.y][threadIdx.x] = (kB < K && col < N)
                                       ? __half2float(B[kB * N + col]) : 0.f;
        __syncthreads();

        // Accumulate dot product for this tile.
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        if (apply_gelu) acc = gelu_tanh(acc);
        C[row * N + col] = __float2half(acc);
    }
}

// ── WMMA FP16 GEMM with optional fused GELU epilogue ───────────────────
// Fast path for dimensions divisible by 16:
//   A: (M, K), B: (K, N), C: (M, N), all row-major half.
// Tensor Cores perform FP16xFP16 -> FP32 accumulation. The accumulator tile
// is passed through GELU when requested, then converted once to FP16 output.
template <bool apply_gelu>
__global__ void gemm_fp16_wmma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__       C,
    int M, int N, int K
) {
    __shared__ float c_smem[WARPS_PER_BLOCK * WMMA_M * WMMA_N];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int warp_m = warp_id / BLOCK_COL_TILES;
    const int warp_n = warp_id % BLOCK_COL_TILES;

    const int tile_m = (blockIdx.y * BLOCK_ROW_TILES + warp_m) * WMMA_M;
    const int tile_n = (blockIdx.x * BLOCK_COL_TILES + warp_n) * WMMA_N;

    if (tile_m >= M || tile_n >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const half* a_tile = A + tile_m * K + k0;
        const half* b_tile = B + k0 * N + tile_n;

        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (apply_gelu) {
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = gelu_tanh(acc_frag.x[i]);
        }
    }

    float* c_tile = c_smem + warp_id * WMMA_M * WMMA_N;
    wmma::store_matrix_sync(c_tile, acc_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += 32) {
        const int r = idx / WMMA_N;
        const int c = idx % WMMA_N;
        C[(tile_m + r) * N + (tile_n + c)] = __float2half_rn(c_tile[idx]);
    }
}

// ── Public interface ───────────────────────────────────────────────────
void mlp_forward(
    const half* x, const half* W1, const half* W2, half* out,
    int batch, int seq_len, int d_model, int d_ff
) {
    const int T = batch * seq_len;   // total tokens

    // Cache the hidden buffer across calls to avoid cudaMalloc overhead
    // during benchmarking runs.
    static half*  s_hidden      = nullptr;
    static size_t s_hidden_size = 0;

    const size_t needed = (size_t)T * d_ff * sizeof(half);
    if (needed > s_hidden_size) {
        if (s_hidden) cudaFree(s_hidden);
        cudaMalloc(&s_hidden, needed);
        s_hidden_size = needed;
    }

    const dim3 block(TILE, TILE);
    const bool use_wmma =
        (T % WMMA_M == 0) &&
        (d_model % WMMA_K == 0) &&
        (d_ff % WMMA_N == 0);

    // ── Kernel 1: hidden = GELU(x @ W1) ────────────────────────────────
    // x      : (T, d_model)
    // W1     : (d_model, d_ff)
    // hidden : (T, d_ff)
    if (use_wmma) {
        dim3 grid((d_ff + BLOCK_COL_TILES * WMMA_N - 1) /
                      (BLOCK_COL_TILES * WMMA_N),
                  (T + BLOCK_ROW_TILES * WMMA_M - 1) /
                      (BLOCK_ROW_TILES * WMMA_M));
        gemm_fp16_wmma_kernel<true><<<grid, THREADS_PER_BLOCK>>>(
            x, W1, s_hidden, T, d_ff, d_model);
    } else {
        dim3 grid((d_ff + TILE - 1) / TILE,
                  (T    + TILE - 1) / TILE);
        gemm_fp16_scalar_kernel<true><<<grid, block>>>(
            x, W1, s_hidden, T, d_ff, d_model);
    }

    // ── Kernel 2: out = hidden @ W2 ─────────────────────────────────────
    // hidden : (T, d_ff)
    // W2     : (d_ff, d_model)
    // out    : (T, d_model)
    if (use_wmma) {
        dim3 grid((d_model + BLOCK_COL_TILES * WMMA_N - 1) /
                      (BLOCK_COL_TILES * WMMA_N),
                  (T + BLOCK_ROW_TILES * WMMA_M - 1) /
                      (BLOCK_ROW_TILES * WMMA_M));
        gemm_fp16_wmma_kernel<false><<<grid, THREADS_PER_BLOCK>>>(
            s_hidden, W2, out, T, d_model, d_ff);
    } else {
        dim3 grid((d_model + TILE - 1) / TILE,
                  (T       + TILE - 1) / TILE);
        gemm_fp16_scalar_kernel<false><<<grid, block>>>(
            s_hidden, W2, out, T, d_model, d_ff);
    }
}
