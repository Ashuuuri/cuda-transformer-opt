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
// Optimization: kernel fusion
//   - Kernel 1: hidden = GELU(x @ W1)  [fuses matmul + activation]
//   - Kernel 2: out    = hidden @ W2
//   Both kernels use 16x16 shared-memory tiling to reduce HBM traffic.
//   The hidden buffer is statically cached to avoid cudaMalloc overhead
//   on repeated calls (important for accurate benchmarking).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ── Tile size ──────────────────────────────────────────────────────────
// 16×16 tiles of float32 → 2 × (16×16×4) = 2 KB shared memory per block.
// Small enough to keep high SM occupancy on A100.
#define TILE 16

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
__global__ void gemm_fp16_kernel(
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

    // ── Kernel 1: hidden = GELU(x @ W1) ────────────────────────────────
    // x      : (T, d_model)
    // W1     : (d_model, d_ff)
    // hidden : (T, d_ff)
    {
        dim3 grid((d_ff    + TILE - 1) / TILE,
                  (T       + TILE - 1) / TILE);
        gemm_fp16_kernel<true><<<grid, block>>>(x, W1, s_hidden, T, d_ff, d_model);
    }

    // ── Kernel 2: out = hidden @ W2 ─────────────────────────────────────
    // hidden : (T, d_ff)
    // W2     : (d_ff, d_model)
    // out    : (T, d_model)
    {
        dim3 grid((d_model + TILE - 1) / TILE,
                  (T       + TILE - 1) / TILE);
        gemm_fp16_kernel<false><<<grid, block>>>(s_hidden, W2, out, T, d_model, d_ff);
    }
}
