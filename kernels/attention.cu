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
// Ablation flags (pass via -D at compile time):
//   ATTN_WMMA=0       use scalar fused kernel instead of WMMA
//   ATTN_FAST_MATH=0  use expf instead of __expf
//   ATTN_SWIZZLE=0    disable smem row padding (shows bank conflict cost)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>

#ifndef ATTN_FAST_MATH
#define ATTN_FAST_MATH 1
#endif

#ifndef ATTN_WMMA
#define ATTN_WMMA 1
#endif

#ifndef ATTN_SWIZZLE
#define ATTN_SWIZZLE 1
#endif

#if ATTN_FAST_MATH
#  define ATTN_EXP __expf
#else
#  define ATTN_EXP expf
#endif


// Scalar fused kernel
// One block per query row. Online softmax over TILE_KV-sized KV chunks.
// Used as fallback when head_dim is not a multiple of 16, or with ATTN_WMMA=0.

#define TILE_KV    64
#define FUSED_BDIM 128

__global__ void fused_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ out,
    int seq_len, int head_dim, float scale
) {
    const int bh = blockIdx.y;
    const int qi = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* s_Q = smem;
    float* s_K = s_Q + head_dim;
    float* s_V = s_K + TILE_KV * head_dim;
    float* s_scores = s_V + TILE_KV * head_dim;
    float* s_out = s_scores + TILE_KV;
    float* s_reduce = s_out + head_dim;

    for (int d = tid; d < head_dim; d += FUSED_BDIM) {
        s_Q[d] = __half2float(Q[((size_t)bh * seq_len + qi) * head_dim + d]);
        s_out[d] = 0.f;
    }
    __syncthreads();

    float running_max = -1e38f;
    float running_sum = 0.f;

    for (int t = 0; t < (seq_len + TILE_KV - 1) / TILE_KV; t++) {
        const int tile_start = t * TILE_KV;
        const int tile_len = min(TILE_KV, seq_len - tile_start);

        for (int idx = tid; idx < tile_len * head_dim; idx += FUSED_BDIM) {
            const int r = idx / head_dim, d = idx % head_dim;
            s_K[r * head_dim + d] = __half2float(K[((size_t)bh * seq_len + tile_start + r) * head_dim + d]);
            s_V[r * head_dim + d] = __half2float(V[((size_t)bh * seq_len + tile_start + r) * head_dim + d]);
        }
        __syncthreads();

        for (int j = tid; j < tile_len; j += FUSED_BDIM) {
            float acc = 0.f;
            for (int d = 0; d < head_dim; d++)
                acc += s_Q[d] * s_K[j * head_dim + d];
            s_scores[j] = acc * scale;
        }
        __syncthreads();

        float lmax = -1e38f;
        for (int j = tid; j < tile_len; j += FUSED_BDIM)
            lmax = fmaxf(lmax, s_scores[j]);
        s_reduce[tid] = lmax;
        __syncthreads();
        for (int stride = FUSED_BDIM >> 1; stride >= 1; stride >>= 1) {
            if (tid < stride)
                s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
            __syncthreads();
        }

        const float new_max = fmaxf(running_max, s_reduce[0]);
        const float corr = ATTN_EXP(running_max - new_max);
        for (int d = tid; d < head_dim; d += FUSED_BDIM)
            s_out[d] *= corr;
        running_sum *= corr;

        float tile_sum = 0.f;
        for (int j = 0; j < tile_len; j++) {
            const float w = ATTN_EXP(s_scores[j] - new_max);
            tile_sum += w;
            for (int d = tid; d < head_dim; d += FUSED_BDIM)
                s_out[d] += w * s_V[j * head_dim + d];
        }
        running_max = new_max;
        running_sum += tile_sum;
        __syncthreads();
    }

    const float inv_sum = 1.f / running_sum;
    for (int d = tid; d < head_dim; d += FUSED_BDIM)
        out[((size_t)bh * seq_len + qi) * head_dim + d] = __float2half(s_out[d] * inv_sum);
}

static void fused_attention_forward(
    const half* Q, const half* K, const half* V, half* out,
    int batch, int heads, int seq_len, int head_dim
) {
    const size_t smem_bytes =
        ((size_t)head_dim +
         (size_t)TILE_KV * head_dim * 2 +
         TILE_KV + head_dim + FUSED_BDIM) * sizeof(float);

    cudaFuncSetAttribute(fused_attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    dim3 grid(seq_len, batch * heads);
    fused_attention_kernel<<<grid, FUSED_BDIM, smem_bytes>>>(
        Q, K, V, out, seq_len, head_dim, 1.f / sqrtf((float)head_dim));
}


#if ATTN_WMMA

// WMMA Split-Q kernel with cp.async double-buffer pipeline.
// Each block handles 32 query rows (2 warps * 16 rows). Each warp tracks its
// own online softmax state in registers — no cross-warp sync needed.
// While computing tile t, tile t+1 loads asynchronously via cp.async.cg.
// Requires seq_len % WMMA_TILE_KV == 0.
//
// Smem layout (all half, rows padded by SMEM_PAD to reduce bank conflicts):
//   s_Q        [WMMA_TILE_Q  * (head_dim + SMEM_PAD)]
//   s_K0/s_K1  [WMMA_TILE_KV * (head_dim + SMEM_PAD)] each  (double-buffered)
//   s_V0/s_V1  [WMMA_TILE_KV * (head_dim + SMEM_PAD)] each  (double-buffered)
//   s_scores_h [WMMA_TILE_Q  * (WMMA_TILE_KV + SMEM_PAD)]
//
// sm_80 wmma 16x16x16 float accumulator layout, thread lane t (0..31):
//   x[0]: (t/4,   (t%4)*2)     x[1]: (t/4,   (t%4)*2+1)
//   x[2]: (t/4+8, (t%4)*2)     x[3]: (t/4+8, (t%4)*2+1)
//   x[4]: (t/4,   (t%4)*2+8)   x[5]: (t/4,   (t%4)*2+9)
//   x[6]: (t/4+8, (t%4)*2+8)   x[7]: (t/4+8, (t%4)*2+9)

using namespace nvcuda::wmma;

#define WMMA_M       16
#define WMMA_N       16
#define WMMA_K_DIM   16
#define WMMA_TILE_Q  32
#define WMMA_TILE_KV 32
#define WMMA_BDIM    64
// Pad each smem row by 8 halves (16 bytes) to reduce bank conflicts.
// Rows whose byte-stride is a multiple of 128 (32 bank widths) alias to the
// same banks; PAD=8 breaks that alignment and reduces conflicts.
// Disable with -DATTN_SWIZZLE=0 to measure the bank conflict cost.
#if ATTN_SWIZZLE
#  define SMEM_PAD 8
#else
#  define SMEM_PAD 0
#endif

// cp.async helpers (Ampere+). cp.async.cg bypasses L1, goes through L2.
// commit_group() seals a stage, wait_group<N>() stalls until at most N remain in flight.

static __device__ __forceinline__
void cp_async16(half* dst, const half* src) {
    unsigned d = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(d), "l"(src) : "memory"
    );
}
static __device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
static __device__ __forceinline__ void cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
}
static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

__global__ __launch_bounds__(WMMA_BDIM, 2)
void wmma_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ out,
    int seq_len, int head_dim, float scale
) {
    const int bh = blockIdx.y;
    const int qi_base = blockIdx.x * WMMA_TILE_Q;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    extern __shared__ char smem_raw[];
    const int qs = head_dim + SMEM_PAD;           // padded Q/K/V row stride
    const int ss = WMMA_TILE_KV + SMEM_PAD;       // padded scores row stride
    half* s_Q = (half*) smem_raw;
    half* s_K0 = s_Q + WMMA_TILE_Q * qs;
    half* s_K1 = s_K0 + WMMA_TILE_KV * qs;
    half* s_V0 = s_K1 + WMMA_TILE_KV * qs;
    half* s_V1 = s_V0 + WMMA_TILE_KV * qs;
    half* s_scores_h = s_V1 + WMMA_TILE_KV * qs;

    for (int idx = tid; idx < WMMA_TILE_Q * head_dim; idx += WMMA_BDIM) {
        const int row = idx / head_dim;
        const int col = idx % head_dim;
        const int qi = qi_base + row;
        s_Q[row * qs + col] = (qi < seq_len)
            ? Q[((size_t)bh * seq_len + qi) * head_dim + col]
            : __float2half(0.f);
    }

    const int frow0 = lane / 4;
    const int frow1 = lane / 4 + 8;
    const int fcol_lo = (lane % 4) * 2;
    float rmax0 = -1e38f, rmax1 = -1e38f;
    float rsum0 = 0.f, rsum1 = 0.f;
    const int n_slices = head_dim / WMMA_N;
    const int n_kv_groups = WMMA_TILE_KV / WMMA_N;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K_DIM, float> frag_out[16];
    for (int s = 0; s < n_slices; ++s) fill_fragment(frag_out[s], 0.f);

    __syncthreads();

    const int num_tiles = seq_len / WMMA_TILE_KV;

    // kick off tile 0 load before entering the loop
    {
        const size_t base = ((size_t)bh * seq_len) * head_dim;
        for (int idx = tid; idx < WMMA_TILE_KV * (head_dim / 8); idx += WMMA_BDIM) {
            const int row = idx / (head_dim / 8);
            const int col = (idx % (head_dim / 8)) * 8;
            cp_async16(s_K0 + row * qs + col, K + base + row * head_dim + col);
            cp_async16(s_V0 + row * qs + col, V + base + row * head_dim + col);
        }
        cp_async_fence();
    }

    for (int t = 0; t < num_tiles; t++) {
        half* s_K_cur = (t % 2 == 0) ? s_K0 : s_K1;
        half* s_V_cur = (t % 2 == 0) ? s_V0 : s_V1;
        half* s_K_nxt = (t % 2 == 0) ? s_K1 : s_K0;
        half* s_V_nxt = (t % 2 == 0) ? s_V1 : s_V0;

        if (t + 1 < num_tiles) {
            const size_t base = ((size_t)bh * seq_len + (t + 1) * WMMA_TILE_KV) * head_dim;
            for (int idx = tid; idx < WMMA_TILE_KV * (head_dim / 8); idx += WMMA_BDIM) {
                const int row = idx / (head_dim / 8);
                const int col = (idx % (head_dim / 8)) * 8;
                cp_async16(s_K_nxt + row * qs + col, K + base + row * head_dim + col);
                cp_async16(s_V_nxt + row * qs + col, V + base + row * head_dim + col);
            }
            cp_async_fence();
            cp_async_wait_one();  // wait for tile t; tile t+1 continues in HW
        } else {
            cp_async_wait_all();
        }
        __syncthreads();  // [1/2] tile t's K, V visible to all threads

        // QK^T - stays in registers
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K_DIM, float> frag_qk[WMMA_TILE_KV / WMMA_N];
        for (int g = 0; g < n_kv_groups; ++g) {
            fill_fragment(frag_qk[g], 0.f);
            for (int k = 0; k < head_dim; k += WMMA_K_DIM) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K_DIM, half, row_major> frag_Q;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K_DIM, half, col_major> frag_K;
                load_matrix_sync(frag_Q, s_Q + warp_id * WMMA_M * qs + k, qs);
                load_matrix_sync(frag_K, s_K_cur + g * WMMA_N * qs + k, qs);
                mma_sync(frag_qk[g], frag_Q, frag_K, frag_qk[g]);
            }
        }

        // Scale + per-row max (warp shuffle, no smem)
        // All tiles are full (seq_len % WMMA_TILE_KV == 0), so no masking needed.
        float lmax0 = -1e38f, lmax1 = -1e38f;
        for (int g = 0; g < n_kv_groups; ++g) {
            const float s0 = frag_qk[g].x[0] * scale;
            const float s1 = frag_qk[g].x[1] * scale;
            const float s8 = frag_qk[g].x[4] * scale;
            const float s9 = frag_qk[g].x[5] * scale;
            frag_qk[g].x[0] = s0; frag_qk[g].x[1] = s1;
            frag_qk[g].x[4] = s8; frag_qk[g].x[5] = s9;
            lmax0 = fmaxf(lmax0, fmaxf(fmaxf(s0, s1), fmaxf(s8, s9)));

            const float t0 = frag_qk[g].x[2] * scale;
            const float t1 = frag_qk[g].x[3] * scale;
            const float t8 = frag_qk[g].x[6] * scale;
            const float t9 = frag_qk[g].x[7] * scale;
            frag_qk[g].x[2] = t0; frag_qk[g].x[3] = t1;
            frag_qk[g].x[6] = t8; frag_qk[g].x[7] = t9;
            lmax1 = fmaxf(lmax1, fmaxf(fmaxf(t0, t1), fmaxf(t8, t9)));
        }
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 1));
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 2));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 1));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 2));

        const float new_max0 = fmaxf(rmax0, lmax0);
        const float new_max1 = fmaxf(rmax1, lmax1);
        const float corr0 = ATTN_EXP(rmax0 - new_max0);
        const float corr1 = ATTN_EXP(rmax1 - new_max1);
        rmax0 = new_max0; rmax1 = new_max1;
        rsum0 *= corr0;   rsum1 *= corr1;

        for (int s = 0; s < n_slices; ++s) {
            frag_out[s].x[0] *= corr0; frag_out[s].x[1] *= corr0;
            frag_out[s].x[2] *= corr1; frag_out[s].x[3] *= corr1;
            frag_out[s].x[4] *= corr0; frag_out[s].x[5] *= corr0;
            frag_out[s].x[6] *= corr1; frag_out[s].x[7] *= corr1;
        }

        // Softmax weights -> s_scores_h; accumulate row sums in registers
        float lsum0 = 0.f, lsum1 = 0.f;
        for (int g = 0; g < n_kv_groups; ++g) {
            const int qr0 = warp_id * WMMA_M + frow0;
            const int qr1 = warp_id * WMMA_M + frow1;
            const int kc = g * WMMA_N + fcol_lo;

            const float w0 = ATTN_EXP(frag_qk[g].x[0] - rmax0);
            const float w1 = ATTN_EXP(frag_qk[g].x[1] - rmax0);
            const float w8 = ATTN_EXP(frag_qk[g].x[4] - rmax0);
            const float w9 = ATTN_EXP(frag_qk[g].x[5] - rmax0);
            lsum0 += w0 + w1 + w8 + w9;
            s_scores_h[qr0 * ss + kc] = __float2half(w0);
            s_scores_h[qr0 * ss + kc + 1] = __float2half(w1);
            s_scores_h[qr0 * ss + kc + 8] = __float2half(w8);
            s_scores_h[qr0 * ss + kc + 9] = __float2half(w9);

            const float u0 = ATTN_EXP(frag_qk[g].x[2] - rmax1);
            const float u1 = ATTN_EXP(frag_qk[g].x[3] - rmax1);
            const float u8 = ATTN_EXP(frag_qk[g].x[6] - rmax1);
            const float u9 = ATTN_EXP(frag_qk[g].x[7] - rmax1);
            lsum1 += u0 + u1 + u8 + u9;
            s_scores_h[qr1 * ss + kc] = __float2half(u0);
            s_scores_h[qr1 * ss + kc + 1] = __float2half(u1);
            s_scores_h[qr1 * ss + kc + 8] = __float2half(u8);
            s_scores_h[qr1 * ss + kc + 9] = __float2half(u9);
        }
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 1);
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 2);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 1);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 2);
        rsum0 += lsum0;
        rsum1 += lsum1;

        // score*V, each warp reads its own rows of s_scores_h
        __syncwarp();
        for (int s = 0; s < n_slices; ++s) {
            const int d_base = s * WMMA_N;
            for (int k = 0; k < WMMA_TILE_KV; k += WMMA_K_DIM) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K_DIM, half, row_major> frag_w;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K_DIM, half, row_major> frag_Vs;
                load_matrix_sync(frag_w,  s_scores_h + warp_id * WMMA_M * ss + k, ss);
                load_matrix_sync(frag_Vs, s_V_cur + k * qs + d_base, qs);
                mma_sync(frag_out[s], frag_w, frag_Vs, frag_out[s]);
            }
        }
        __syncthreads();  // [2/2] done reading s_V_cur
    }

    const float inv0 = 1.f / rsum0;
    const float inv1 = 1.f / rsum1;
    for (int s = 0; s < n_slices; ++s) {
        const int d_base = s * WMMA_N;
        const int qi0 = qi_base + warp_id * WMMA_M + frow0;
        const int qi1 = qi_base + warp_id * WMMA_M + frow1;
        const size_t b0 = ((size_t)bh * seq_len + qi0) * head_dim;
        const size_t b1 = ((size_t)bh * seq_len + qi1) * head_dim;
        if (qi0 < seq_len) {
            out[b0 + d_base + fcol_lo] = __float2half(frag_out[s].x[0] * inv0);
            out[b0 + d_base + fcol_lo + 1] = __float2half(frag_out[s].x[1] * inv0);
            out[b0 + d_base + fcol_lo + 8] = __float2half(frag_out[s].x[4] * inv0);
            out[b0 + d_base + fcol_lo + 9] = __float2half(frag_out[s].x[5] * inv0);
        }
        if (qi1 < seq_len) {
            out[b1 + d_base + fcol_lo] = __float2half(frag_out[s].x[2] * inv1);
            out[b1 + d_base + fcol_lo + 1] = __float2half(frag_out[s].x[3] * inv1);
            out[b1 + d_base + fcol_lo + 8] = __float2half(frag_out[s].x[6] * inv1);
            out[b1 + d_base + fcol_lo + 9] = __float2half(frag_out[s].x[7] * inv1);
        }
    }
}

static void wmma_attention_forward(
    const half* Q, const half* K, const half* V, half* out,
    int batch, int heads, int seq_len, int head_dim
) {
    if (head_dim < WMMA_K_DIM || head_dim % WMMA_K_DIM != 0) {
        fused_attention_forward(Q, K, V, out, batch, heads, seq_len, head_dim);
        return;
    }

    const float scale = 1.f / sqrtf((float)head_dim);
    const int BH = batch * heads;
    dim3 grid((seq_len + WMMA_TILE_Q - 1) / WMMA_TILE_Q, BH);

    // s_Q + 2*s_K + 2*s_V + s_scores_h (all rows padded by SMEM_PAD)
    const size_t smem =
        ((size_t)(WMMA_TILE_Q + 4 * WMMA_TILE_KV) * (head_dim + SMEM_PAD) +
         (size_t) WMMA_TILE_Q * (WMMA_TILE_KV + SMEM_PAD)) * sizeof(half);

    cudaFuncSetAttribute(wmma_attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    wmma_attention_kernel<<<grid, WMMA_BDIM, smem>>>(
        Q, K, V, out, seq_len, head_dim, scale);
}

#endif  // ATTN_WMMA

// Public interface

void attention_forward(
    const half* Q, const half* K, const half* V, half* out,
    int batch, int heads, int seq_len, int head_dim
) {
#if ATTN_WMMA
    wmma_attention_forward(Q, K, V, out, batch, heads, seq_len, head_dim);
#else
    fused_attention_forward(Q, K, V, out, batch, heads, seq_len, head_dim);
#endif
}
