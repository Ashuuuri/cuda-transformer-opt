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
//   The fast path uses 64x64 block tiles built from 16x16x16 WMMA FP16
//   Tensor Core operations with FP32 accumulation.
//   A/B tiles are staged through shared memory with cp.async on sm_80+.
//   The K stage is 32 wide, so each async copy stage feeds two WMMA steps.
//   Shared-memory strides use a WMMA-compatible skew/swizzle to reduce bank
//   conflicts.
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
// Each warp computes two neighboring 16x16 C tiles in N. A block still covers
// the proposal baseline 64x64 output tile, but uses 8 warps (256 threads)
// instead of 16, reducing scheduling/synchronization overhead.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define STAGE_K 32
#define STAGE_K_TILES (STAGE_K / WMMA_K)
#define BLOCK_ROW_TILES 4
#define BLOCK_COL_TILES 4
#define WARP_COL_TILES 2
#define WARP_COL_GROUPS (BLOCK_COL_TILES / WARP_COL_TILES)
#define BLOCK_M (BLOCK_ROW_TILES * WMMA_M)
#define BLOCK_N (BLOCK_COL_TILES * WMMA_N)
#define WARPS_PER_BLOCK (BLOCK_ROW_TILES * WARP_COL_GROUPS)
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

// WMMA-compatible shared-memory swizzle: keep each row 16-byte aligned for
// cp.async, but skew row strides to avoid the worst ldmatrix bank conflicts.
#define SMEM_SKEW 8
#define A_SMEM_STRIDE (STAGE_K + SMEM_SKEW)
#define B_SMEM_STRIDE (BLOCK_N + SMEM_SKEW)

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

// ── cp.async helpers ───────────────────────────────────────────────────
__device__ __forceinline__ unsigned smem_u32addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16B(void* smem_ptr,
                                             const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const unsigned smem_addr = smem_u32addr(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"(smem_addr), "l"(gmem_ptr));
#else
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void stage_mlp_tile_async(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ sA,
    half* __restrict__ sB,
    int block_m, int block_n, int k0, int K, int N
) {
    const int A_VEC_COPIES = BLOCK_M * (STAGE_K / 8);
    const int B_VEC_COPIES = STAGE_K * (BLOCK_N / 8);
    const int TOTAL_COPIES = A_VEC_COPIES + B_VEC_COPIES;

    for (int copy = threadIdx.x; copy < TOTAL_COPIES;
         copy += THREADS_PER_BLOCK) {
        if (copy < A_VEC_COPIES) {
            const int row = copy / (STAGE_K / 8);
            const int col = (copy % (STAGE_K / 8)) * 8;
            half* dst = sA + row * A_SMEM_STRIDE + col;
            const half* src = A + (block_m + row) * K + k0 + col;
            cp_async_16B(dst, src);
        } else {
            const int bcopy = copy - A_VEC_COPIES;
            const int row = bcopy / (BLOCK_N / 8);
            const int col = (bcopy % (BLOCK_N / 8)) * 8;
            half* dst = sB + row * B_SMEM_STRIDE + col;
            const half* src = B + (k0 + row) * N + block_n + col;
            cp_async_16B(dst, src);
        }
    }
}

// ── WMMA FP16 GEMM with optional fused GELU epilogue ───────────────────
// Fast path for dimensions divisible by the 64x64 block tile:
//   A: (M, K), B: (K, N), C: (M, N), all row-major half.
// Tensor Cores perform FP16xFP16 -> FP32 accumulation. The accumulator tile
// is passed through GELU when requested, then converted once to FP16 output.
// A/B are double-buffered in shared memory and loaded with cp.async.
template <bool apply_gelu>
__global__ void gemm_fp16_wmma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__       C,
    int M, int N, int K
) {
    __shared__ __align__(16) half sA[2][BLOCK_M * A_SMEM_STRIDE];
    __shared__ __align__(16) half sB[2][STAGE_K * B_SMEM_STRIDE];
    __shared__ float c_smem[WARPS_PER_BLOCK * WARP_COL_TILES *
                            WMMA_M * WMMA_N];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int warp_m = warp_id / WARP_COL_GROUPS;
    const int warp_n_group = warp_id % WARP_COL_GROUPS;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    const int tile_m = block_m + warp_m * WMMA_M;
    const int tile_n0 = block_n + warp_n_group * WARP_COL_TILES * WMMA_N;
    const int tile_n1 = tile_n0 + WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag0;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag0;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag1;

    wmma::fill_fragment(acc_frag0, 0.0f);
    wmma::fill_fragment(acc_frag1, 0.0f);

    int stage = 0;
    stage_mlp_tile_async(A, B, sA[stage], sB[stage],
                         block_m, block_n, 0, K, N);
    cp_async_commit();

    for (int k0 = 0; k0 < K; k0 += STAGE_K) {
        cp_async_wait_all();
        __syncthreads();

        const int next_k0 = k0 + STAGE_K;
        const int next_stage = stage ^ 1;
        if (next_k0 < K) {
            stage_mlp_tile_async(A, B, sA[next_stage], sB[next_stage],
                                 block_m, block_n, next_k0, K, N);
            cp_async_commit();
        }

        #pragma unroll
        for (int kk = 0; kk < STAGE_K; kk += WMMA_K) {
            const half* a_tile = sA[stage] +
                warp_m * WMMA_M * A_SMEM_STRIDE + kk;
            const half* b_tile0 = sB[stage] + kk * B_SMEM_STRIDE +
                warp_n_group * WARP_COL_TILES * WMMA_N;
            const half* b_tile1 = b_tile0 + WMMA_N;

            wmma::load_matrix_sync(a_frag, a_tile, A_SMEM_STRIDE);
            wmma::load_matrix_sync(b_frag0, b_tile0, B_SMEM_STRIDE);
            wmma::load_matrix_sync(b_frag1, b_tile1, B_SMEM_STRIDE);
            wmma::mma_sync(acc_frag0, a_frag, b_frag0, acc_frag0);
            wmma::mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);
        }

        __syncthreads();
        stage = next_stage;
    }

    if (apply_gelu) {
        #pragma unroll
        for (int i = 0; i < acc_frag0.num_elements; i++) {
            acc_frag0.x[i] = gelu_tanh(acc_frag0.x[i]);
            acc_frag1.x[i] = gelu_tanh(acc_frag1.x[i]);
        }
    }

    float* c_tile0 = c_smem +
        warp_id * WARP_COL_TILES * WMMA_M * WMMA_N;
    float* c_tile1 = c_tile0 + WMMA_M * WMMA_N;
    wmma::store_matrix_sync(c_tile0, acc_frag0, WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(c_tile1, acc_frag1, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += 32) {
        const int r = idx / WMMA_N;
        const int c = idx % WMMA_N;
        C[(tile_m + r) * N + (tile_n0 + c)] = __float2half_rn(c_tile0[idx]);
        C[(tile_m + r) * N + (tile_n1 + c)] = __float2half_rn(c_tile1[idx]);
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
        (T % BLOCK_M == 0) &&
        (d_model % STAGE_K == 0) &&
        (d_ff % STAGE_K == 0) &&
        (d_model % BLOCK_N == 0) &&
        (d_ff % BLOCK_N == 0);

    // ── Kernel 1: hidden = GELU(x @ W1) ────────────────────────────────
    // x      : (T, d_model)
    // W1     : (d_model, d_ff)
    // hidden : (T, d_ff)
    if (use_wmma) {
        dim3 grid(d_ff / BLOCK_N, T / BLOCK_M);
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
        dim3 grid(d_model / BLOCK_N, T / BLOCK_M);
        gemm_fp16_wmma_kernel<false><<<grid, THREADS_PER_BLOCK>>>(
            s_hidden, W2, out, T, d_model, d_ff);
    } else {
        dim3 grid((d_model + TILE - 1) / TILE,
                  (T       + TILE - 1) / TILE);
        gemm_fp16_scalar_kernel<false><<<grid, block>>>(
            s_hidden, W2, out, T, d_model, d_ff);
    }
}
