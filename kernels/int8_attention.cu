// int8_attention.cu — INT8 quantized scaled dot-product attention.
// Owner: Heling
//
// Per-token symmetric quantization:
//   Each token row has its own scale factor → scale arrays of size [batch*heads*seq_len].
//   QK^T element [i][j] = int32_acc * scale_Q[i] * scale_K[j] * attn_scale
//   V dequant row r: V_fp16[r][c] = V_int8[r][c] * scale_V[r]
//   Output is FP16 (can't factor out a single scale from weighted V rows).
//
// Two paths:
//   WMMA path  (INT8_ATTN_WMMA=1, default):
//     Q,K in smem as INT8 → INT8 WMMA for QK^T (INT32 accum)
//     Per-element scale, online softmax, FP16 WMMA for attn×V
//
//   Scalar path (INT8_ATTN_WMMA=0, or non-aligned dims):
//     Same algorithm, loop-based dot products.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <cstdint>
#include "int8_common.cuh"

#ifndef INT8_ATTN_WMMA
#define INT8_ATTN_WMMA 1
#endif

// ── Scalar path ─────────────────────────────────────────────────────────
#define SCALAR_TILE_KV  64
#define SCALAR_BDIM    128

__global__ void int8_fused_attention_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    half* __restrict__         out,
    const float* __restrict__ d_scale_Q,   // [batch*heads*seq_len]
    const float* __restrict__ d_scale_K,   // [batch*heads*seq_len]
    const float* __restrict__ d_scale_V,   // [batch*heads*seq_len]
    int seq_len, int head_dim,
    float attn_scale
) {
    const int bh  = blockIdx.y;
    const int qi  = blockIdx.x;
    const int tid = threadIdx.x;

    // Per-token scale for this query row
    const float sQ = __ldg(&d_scale_Q[bh * seq_len + qi]);

    extern __shared__ float smem[];
    float* s_Q      = smem;
    float* s_K      = s_Q + head_dim;
    float* s_V      = s_K + SCALAR_TILE_KV * head_dim;
    float* s_scores = s_V + SCALAR_TILE_KV * head_dim;
    float* s_out    = s_scores + SCALAR_TILE_KV;
    float* s_reduce = s_out + head_dim;

    // Load Q row (keep as float, scale already captured)
    for (int d = tid; d < head_dim; d += SCALAR_BDIM) {
        s_Q[d] = (float)Q[((size_t)bh * seq_len + qi) * head_dim + d];
        s_out[d] = 0.f;
    }
    __syncthreads();

    float rmax = -1e38f, rsum = 0.f;

    for (int t = 0; t < (seq_len + SCALAR_TILE_KV - 1) / SCALAR_TILE_KV; ++t) {
        const int ts   = t * SCALAR_TILE_KV;
        const int tlen = min(SCALAR_TILE_KV, seq_len - ts);

        // Load K tile (raw int8 → float, no scale yet)
        // Load V tile with per-row dequantization
        for (int idx = tid; idx < tlen * head_dim; idx += SCALAR_BDIM) {
            const int r = idx / head_dim, d = idx % head_dim;
            const size_t base = ((size_t)bh * seq_len + ts + r) * head_dim + d;
            s_K[r * head_dim + d] = (float)K[base];
            float sV_row = __ldg(&d_scale_V[bh * seq_len + ts + r]);
            s_V[r * head_dim + d] = (float)V[base] * sV_row;
        }
        __syncthreads();

        // Compute QK^T scores with per-token scales
        for (int j = tid; j < tlen; j += SCALAR_BDIM) {
            float acc = 0.f;
            for (int d = 0; d < head_dim; ++d) acc += s_Q[d] * s_K[j * head_dim + d];
            float sK_j = __ldg(&d_scale_K[bh * seq_len + ts + j]);
            s_scores[j] = acc * sQ * sK_j * attn_scale;
        }
        __syncthreads();

        // Online softmax: find tile max
        float lmax = -1e38f;
        for (int j = tid; j < tlen; j += SCALAR_BDIM) lmax = fmaxf(lmax, s_scores[j]);
        s_reduce[tid] = lmax;
        __syncthreads();
        for (int s = SCALAR_BDIM >> 1; s >= 1; s >>= 1) {
            if (tid < s) s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + s]);
            __syncthreads();
        }

        const float new_max = fmaxf(rmax, s_reduce[0]);
        const float corr    = __expf(rmax - new_max);
        for (int d = tid; d < head_dim; d += SCALAR_BDIM) s_out[d] *= corr;
        rsum *= corr;

        float tile_sum = 0.f;
        for (int j = 0; j < tlen; ++j) {
            const float w = __expf(s_scores[j] - new_max);
            tile_sum += w;
            for (int d = tid; d < head_dim; d += SCALAR_BDIM)
                s_out[d] += w * s_V[j * head_dim + d];
        }
        rmax = new_max;
        rsum += tile_sum;
        __syncthreads();
    }

    // Normalize and write FP16 output
    const float inv_rsum = 1.f / rsum;
    for (int d = tid; d < head_dim; d += SCALAR_BDIM) {
        out[((size_t)bh * seq_len + qi) * head_dim + d] =
            __float2half(s_out[d] * inv_rsum);
    }
}

// ── WMMA path ──────────────────────────────────────────────────────────
#if INT8_ATTN_WMMA

using namespace nvcuda::wmma;

#define ATTN_WMMA_M     16
#define ATTN_WMMA_N     16
#define ATTN_WMMA_K     16
#define ATTN_TILE_Q     64   // 4 warps × 16 rows each
#define ATTN_TILE_KV    32
#define ATTN_BDIM      128   // 4 warps: better latency hiding + faster KV loading
#define ATTN_SMEM_PAD    8
#define ATTN_I8_PAD     16  // break INT8 smem bank conflicts (stride must not be multiple of 128 bytes)

// Maximum head_dim supported = ATTN_WMMA_N * ATTN_MAX_SLICES = 16 * 16 = 256.
#define ATTN_MAX_SLICES 16

// Template on MIN_BLOCKS for __launch_bounds__:
//   head_dim ≤ 64  → 4 blocks/SM (low register pressure, ~150 regs/thread)
//   head_dim ≤ 128 → 3 blocks/SM (~180 regs/thread)
//   head_dim > 128 → 2 blocks/SM (high register pressure, ~220 regs/thread)
template<int MIN_BLOCKS>
__global__ __launch_bounds__(ATTN_BDIM, MIN_BLOCKS)
void int8_wmma_attention_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    half* __restrict__         out,
    const float* __restrict__ d_scale_Q,   // [batch*heads*seq_len]
    const float* __restrict__ d_scale_K,   // [batch*heads*seq_len]
    const float* __restrict__ d_scale_V,   // [batch*heads*seq_len]
    int seq_len, int head_dim,
    float attn_scale
) {
    const int bh      = blockIdx.y;
    const int qi_base = blockIdx.x * ATTN_TILE_Q;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    extern __shared__ char smem_raw[];
    // INT8 row stride: pad to break bank conflicts.
    // Without padding, head_dim=256 → stride 256 bytes → 256%128=0 → all rows hit same banks.
    // With +16 bytes, stride=272 → 272%128=16 → rows cycle through different bank sets.
    const int ks_i8 = head_dim + ATTN_I8_PAD;
    // FP16 strides (in half elements).
    const int vs = head_dim + ATTN_SMEM_PAD;      // V row stride
    const int ss = ATTN_TILE_KV + ATTN_SMEM_PAD;  // scores row stride

    // Smem layout: [Q INT8][K INT8][V FP16][scores FP16][K_scales float]
    // V scales are read directly from global via __ldg (L1 cached, only TILE_KV unique values).
    int8_t* s_Q_i8   = (int8_t*) smem_raw;
    int8_t* s_K_i8   = s_Q_i8 + ATTN_TILE_Q  * ks_i8;
    half*   s_V_fp16 = (half*)(s_K_i8 + ATTN_TILE_KV * ks_i8);
    half*   s_scores = s_V_fp16 + ATTN_TILE_KV * vs;
    // Per-token K scales for current KV tile (needed for per-element QK^T scaling)
    float*  s_scale_K = (float*)(s_scores + ATTN_TILE_Q * ss);

    // Load Q tile as INT8 (vectorized 16-byte loads).
    // head_dim is always a multiple of 16, so we can use int4 (128-bit) loads.
    {
        const int q_vecs = ATTN_TILE_Q * head_dim / 16;  // total 16-byte chunks
        const int4 zero4 = make_int4(0, 0, 0, 0);
        for (int vi = tid; vi < q_vecs; vi += ATTN_BDIM) {
            const int byte_off = vi * 16;
            const int row = byte_off / head_dim;
            const int col = byte_off % head_dim;
            const int qi  = qi_base + row;
            int4 val = (qi < seq_len)
                ? *reinterpret_cast<const int4*>(&Q[((size_t)bh * seq_len + qi) * head_dim + col])
                : zero4;
            *reinterpret_cast<int4*>(&s_Q_i8[row * ks_i8 + col]) = val;
        }
    }

    // Load per-token Q scales for the rows this warp handles.
    // Fragment element layout (m16n16k16 accumulator, sm_80):
    //   frow0 = lane/4       (rows 0-7 of the 16x16 fragment)
    //   frow1 = lane/4 + 8   (rows 8-15)
    //   fcol_lo = (lane%4)*2
    const int frow0   = lane / 4;
    const int frow1   = lane / 4 + 8;
    const int fcol_lo = (lane % 4) * 2;

    // Each thread handles 2 query rows in the fragment.
    // Pre-fold attn_scale into Q scales: saves 1 FMUL per element in the hot loop.
    const int qi0_global = qi_base + warp_id * ATTN_WMMA_M + frow0;
    const int qi1_global = qi_base + warp_id * ATTN_WMMA_M + frow1;
    const float r_sQ0 = (qi0_global < seq_len)
        ? __ldg(&d_scale_Q[bh * seq_len + qi0_global]) * attn_scale : 0.f;
    const float r_sQ1 = (qi1_global < seq_len)
        ? __ldg(&d_scale_Q[bh * seq_len + qi1_global]) * attn_scale : 0.f;

    float rmax0 = -1e38f, rmax1 = -1e38f;
    float rsum0 = 0.f,    rsum1 = 0.f;

    const int n_slices    = head_dim / ATTN_WMMA_N;
    const int n_kv_groups = ATTN_TILE_KV / ATTN_WMMA_N;

    fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, float> frag_out[ATTN_MAX_SLICES];
    for (int s = 0; s < n_slices; ++s) fill_fragment(frag_out[s], 0.f);

    __syncthreads();

    const int num_tiles = seq_len / ATTN_TILE_KV;

    for (int t = 0; t < num_tiles; ++t) {
        const int kv_tile_start = t * ATTN_TILE_KV;
        const size_t kv_base = ((size_t)bh * seq_len + kv_tile_start) * head_dim;

        // Load K (INT8) + V (dequant to FP16) — vectorized 16-byte loads.
        const size_t scale_base = (size_t)bh * seq_len + kv_tile_start;
        {
            const int kv_vecs = ATTN_TILE_KV * head_dim / 16;
            for (int vi = tid; vi < kv_vecs; vi += ATTN_BDIM) {
                const int byte_off = vi * 16;
                const int row = byte_off / head_dim;
                const int col = byte_off % head_dim;
                const size_t goff = kv_base + row * head_dim + col;
                // K: 16-byte vectorized load + store to padded smem
                *reinterpret_cast<int4*>(&s_K_i8[row * ks_i8 + col]) =
                    *reinterpret_cast<const int4*>(&K[goff]);
                // V: 16-byte vectorized load, per-row dequant, write FP16
                int4 v_vec = *reinterpret_cast<const int4*>(&V[goff]);
                const float sv = __ldg(&d_scale_V[scale_base + row]);
                const int8_t* vb = reinterpret_cast<const int8_t*>(&v_vec);
                half* vdst = &s_V_fp16[row * vs + col];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    reinterpret_cast<half2*>(vdst)[i] = __halves2half2(
                        __float2half((float)vb[2*i]     * sv),
                        __float2half((float)vb[2*i + 1] * sv));
                }
            }
        }
        // Load per-token K scales to smem (needed for per-element QK^T scaling)
        for (int idx = tid; idx < ATTN_TILE_KV; idx += ATTN_BDIM) {
            s_scale_K[idx] = __ldg(&d_scale_K[scale_base + idx]);
        }
        __syncthreads();

        // QK^T via INT8 WMMA -> INT32 accumulator.
        fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int32_t>
            frag_qk[ATTN_TILE_KV / ATTN_WMMA_N];
        #pragma unroll
        for (int g = 0; g < n_kv_groups; ++g) {
            fill_fragment(frag_qk[g], (int32_t)0);
            #pragma unroll 4
            for (int k = 0; k < head_dim; k += ATTN_WMMA_K) {
                fragment<matrix_a, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int8_t, row_major> fq;
                fragment<matrix_b, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int8_t, col_major> fk;
                load_matrix_sync(fq, s_Q_i8 + warp_id * ATTN_WMMA_M * ks_i8 + k, ks_i8);
                load_matrix_sync(fk, s_K_i8 + g * ATTN_WMMA_N * ks_i8 + k,       ks_i8);
                mma_sync(frag_qk[g], fq, fk, frag_qk[g]);
            }
        }

        // Cast INT32 accumulators to float with per-element scaling.
        // Element mapping (m16n16k16 accumulator on sm_80):
        //   x[0]: row=frow0, col=g*16+fcol_lo
        //   x[1]: row=frow0, col=g*16+fcol_lo+1
        //   x[2]: row=frow1, col=g*16+fcol_lo
        //   x[3]: row=frow1, col=g*16+fcol_lo+1
        //   x[4]: row=frow0, col=g*16+fcol_lo+8
        //   x[5]: row=frow0, col=g*16+fcol_lo+9
        //   x[6]: row=frow1, col=g*16+fcol_lo+8
        //   x[7]: row=frow1, col=g*16+fcol_lo+9
        float sf[ATTN_TILE_KV / ATTN_WMMA_N][8];
        float lmax0 = -1e38f, lmax1 = -1e38f;
        for (int g = 0; g < n_kv_groups; ++g) {
            // Per-token K scales for the columns this thread touches
            const int kc_base = g * ATTN_WMMA_N;
            float sK_c0 = s_scale_K[kc_base + fcol_lo];
            float sK_c1 = s_scale_K[kc_base + fcol_lo + 1];
            float sK_c8 = s_scale_K[kc_base + fcol_lo + 8];
            float sK_c9 = s_scale_K[kc_base + fcol_lo + 9];

            // scale = (sQ * attn_scale)[row] * sK[col]  — attn_scale pre-folded into r_sQ
            sf[g][0] = (float)frag_qk[g].x[0] * r_sQ0 * sK_c0;
            sf[g][1] = (float)frag_qk[g].x[1] * r_sQ0 * sK_c1;
            sf[g][2] = (float)frag_qk[g].x[2] * r_sQ1 * sK_c0;
            sf[g][3] = (float)frag_qk[g].x[3] * r_sQ1 * sK_c1;
            sf[g][4] = (float)frag_qk[g].x[4] * r_sQ0 * sK_c8;
            sf[g][5] = (float)frag_qk[g].x[5] * r_sQ0 * sK_c9;
            sf[g][6] = (float)frag_qk[g].x[6] * r_sQ1 * sK_c8;
            sf[g][7] = (float)frag_qk[g].x[7] * r_sQ1 * sK_c9;

            lmax0 = fmaxf(lmax0, fmaxf(fmaxf(sf[g][0],sf[g][1]), fmaxf(sf[g][4],sf[g][5])));
            lmax1 = fmaxf(lmax1, fmaxf(fmaxf(sf[g][2],sf[g][3]), fmaxf(sf[g][6],sf[g][7])));
        }
        // Reduce max across threads sharing the same row (lanes with same lane/4).
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 1));
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 2));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 1));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 2));

        // Online softmax correction
        const float new_max0 = fmaxf(rmax0, lmax0), corr0 = __expf(rmax0 - new_max0);
        const float new_max1 = fmaxf(rmax1, lmax1), corr1 = __expf(rmax1 - new_max1);
        rmax0 = new_max0; rmax1 = new_max1;
        rsum0 *= corr0;   rsum1 *= corr1;
        for (int s = 0; s < n_slices; ++s) {
            frag_out[s].x[0]*=corr0; frag_out[s].x[1]*=corr0;
            frag_out[s].x[2]*=corr1; frag_out[s].x[3]*=corr1;
            frag_out[s].x[4]*=corr0; frag_out[s].x[5]*=corr0;
            frag_out[s].x[6]*=corr1; frag_out[s].x[7]*=corr1;
        }

        // Softmax weights -> s_scores (FP16, for attn x V WMMA).
        float lsum0 = 0.f, lsum1 = 0.f;
        for (int g = 0; g < n_kv_groups; ++g) {
            const int qr0 = warp_id * ATTN_WMMA_M + frow0;
            const int qr1 = warp_id * ATTN_WMMA_M + frow1;
            const int kc  = g * ATTN_WMMA_N + fcol_lo;
            const float w0=__expf(sf[g][0]-rmax0), w1=__expf(sf[g][1]-rmax0);
            const float w8=__expf(sf[g][4]-rmax0), w9=__expf(sf[g][5]-rmax0);
            lsum0 += w0+w1+w8+w9;
            s_scores[qr0*ss+kc]=__float2half(w0); s_scores[qr0*ss+kc+1]=__float2half(w1);
            s_scores[qr0*ss+kc+8]=__float2half(w8); s_scores[qr0*ss+kc+9]=__float2half(w9);
            const float u0=__expf(sf[g][2]-rmax1), u1=__expf(sf[g][3]-rmax1);
            const float u8=__expf(sf[g][6]-rmax1), u9=__expf(sf[g][7]-rmax1);
            lsum1 += u0+u1+u8+u9;
            s_scores[qr1*ss+kc]=__float2half(u0); s_scores[qr1*ss+kc+1]=__float2half(u1);
            s_scores[qr1*ss+kc+8]=__float2half(u8); s_scores[qr1*ss+kc+9]=__float2half(u9);
        }
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 1);
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 2);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 1);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 2);
        rsum0 += lsum0;  rsum1 += lsum1;

        // attn x V via FP16 WMMA (s_V_fp16 already dequantized per-row).
        __syncwarp();
        for (int s = 0; s < n_slices; ++s) {
            const int d_base = s * ATTN_WMMA_N;
            #pragma unroll
            for (int k = 0; k < ATTN_TILE_KV; k += ATTN_WMMA_K) {
                fragment<matrix_a, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fw;
                fragment<matrix_b, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fv;
                load_matrix_sync(fw, s_scores + warp_id * ATTN_WMMA_M * ss + k, ss);
                load_matrix_sync(fv, s_V_fp16 + k * vs + d_base,                vs);
                mma_sync(frag_out[s], fw, fv, frag_out[s]);
            }
        }
        __syncthreads();
    }

    // Normalize and write FP16 output.
    const float inv0 = 1.f / rsum0, inv1 = 1.f / rsum1;
    for (int s = 0; s < n_slices; ++s) {
        const int d    = s * ATTN_WMMA_N;
        const int qi0  = qi_base + warp_id * ATTN_WMMA_M + frow0;
        const int qi1  = qi_base + warp_id * ATTN_WMMA_M + frow1;
        const size_t b0 = ((size_t)bh * seq_len + qi0) * head_dim;
        const size_t b1 = ((size_t)bh * seq_len + qi1) * head_dim;
        if (qi0 < seq_len) {
            out[b0+d+fcol_lo]   = __float2half(frag_out[s].x[0]*inv0);
            out[b0+d+fcol_lo+1] = __float2half(frag_out[s].x[1]*inv0);
            out[b0+d+fcol_lo+8] = __float2half(frag_out[s].x[4]*inv0);
            out[b0+d+fcol_lo+9] = __float2half(frag_out[s].x[5]*inv0);
        }
        if (qi1 < seq_len) {
            out[b1+d+fcol_lo]   = __float2half(frag_out[s].x[2]*inv1);
            out[b1+d+fcol_lo+1] = __float2half(frag_out[s].x[3]*inv1);
            out[b1+d+fcol_lo+8] = __float2half(frag_out[s].x[6]*inv1);
            out[b1+d+fcol_lo+9] = __float2half(frag_out[s].x[7]*inv1);
        }
    }
}

#endif  // INT8_ATTN_WMMA

// ── Public interface ───────────────────────────────────────────────────
// scale_Q/K/V are device arrays of size [batch*heads*seq_len] — per-token scales.
// attn_scale = 1/sqrt(head_dim) is host-computable.
// Output is FP16 (half*).
void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int batch, int heads, int seq_len, int head_dim
) {
    const float attn_scale = 1.f / sqrtf((float)head_dim);

#if INT8_ATTN_WMMA
    if (head_dim % ATTN_WMMA_K == 0 && seq_len % ATTN_TILE_KV == 0) {
        const int BH = batch * heads;
        dim3 grid((seq_len + ATTN_TILE_Q - 1) / ATTN_TILE_Q, BH);
        // Smem: Q(INT8 padded) + K(INT8 padded) + V(FP16) + scores(FP16) + K_scales(float)
        const int ks_i8_host = head_dim + ATTN_I8_PAD;
        const size_t smem =
            (size_t) ATTN_TILE_Q  * ks_i8_host * sizeof(int8_t) +
            (size_t) ATTN_TILE_KV * ks_i8_host * sizeof(int8_t) +
            (size_t) ATTN_TILE_KV * (head_dim + ATTN_SMEM_PAD) * sizeof(half) +
            (size_t) ATTN_TILE_Q  * (ATTN_TILE_KV + ATTN_SMEM_PAD) * sizeof(half) +
            (size_t) ATTN_TILE_KV * sizeof(float);  // K_scales only
        // Select kernel variant by head_dim: higher occupancy for smaller head_dim.
        // A100: 65536 regs/SM, 164 KB smem/SM.
        //   head_dim ≤ 64:  smem ~18KB → 4 blocks fit, regs ~150 → OK at 128 regs/thread
        //   head_dim ≤ 128: smem ~34KB → 3 blocks fit, regs ~180 → OK at 170 regs/thread
        //   head_dim > 128: smem ~48KB → 2 blocks max, regs ~220
        if (head_dim <= 64) {
            cudaFuncSetAttribute(int8_wmma_attention_kernel<4>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            int8_wmma_attention_kernel<4><<<grid, ATTN_BDIM, smem>>>(
                Q, K, V, out, scale_Q, scale_K, scale_V,
                seq_len, head_dim, attn_scale);
        } else if (head_dim <= 128) {
            cudaFuncSetAttribute(int8_wmma_attention_kernel<3>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            int8_wmma_attention_kernel<3><<<grid, ATTN_BDIM, smem>>>(
                Q, K, V, out, scale_Q, scale_K, scale_V,
                seq_len, head_dim, attn_scale);
        } else {
            cudaFuncSetAttribute(int8_wmma_attention_kernel<2>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            int8_wmma_attention_kernel<2><<<grid, ATTN_BDIM, smem>>>(
                Q, K, V, out, scale_Q, scale_K, scale_V,
                seq_len, head_dim, attn_scale);
        }
        return;
    }
#endif

    const size_t smem =
        ((size_t)head_dim + (size_t)SCALAR_TILE_KV * head_dim * 2 +
         SCALAR_TILE_KV + head_dim + SCALAR_BDIM) * sizeof(float);
    cudaFuncSetAttribute(int8_fused_attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(seq_len, batch * heads);
    int8_fused_attention_kernel<<<grid, SCALAR_BDIM, smem>>>(
        Q, K, V, out, scale_Q, scale_K, scale_V,
        seq_len, head_dim, attn_scale);
}
