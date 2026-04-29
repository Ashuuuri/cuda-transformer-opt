// int8_attention.cu — INT8 quantized scaled dot-product attention.
// Owner: Heling
//
// Two paths (same dispatch logic as Jonathan's attention.cu):
//
//   WMMA path  (INT8_ATTN_WMMA=1, default):
//     INT8 Q,K,V → dequant to FP16 in smem → FP16 WMMA QK^T + softmax + attn×V
//     → quantize output to INT8 using scale_V.
//     Bandwidth savings from 2× smaller I/O; compute stays in FP16 to avoid
//     INT32 accumulator layout assumptions in the online-softmax register reads.
//
//   Scalar path (INT8_ATTN_WMMA=0, or non-aligned head_dim / seq_len):
//     Same algorithm as Jonathan's fused_attention_kernel; Q,K,V dequantized
//     at load, output quantized at write.
//
//   Output scale = scale_V:
//     Attention output is a convex combination of V rows, so
//     |output| ≤ max|V_real| = 127 × scale_V → no overflow when using scale_V.
//
// Ablation flag:
//   INT8_ATTN_WMMA=0   force scalar path

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
    int8_t* __restrict__       out,
    const float* __restrict__ d_scale_Q,
    const float* __restrict__ d_scale_K,
    const float* __restrict__ d_scale_V,
    int seq_len, int head_dim,
    float attn_scale
) {
    const float scale_Q       = __ldg(d_scale_Q);
    const float scale_K       = __ldg(d_scale_K);
    const float scale_V       = __ldg(d_scale_V);
    const float inv_out_scale = 1.f / scale_V;
    const int bh  = blockIdx.y;
    const int qi  = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* s_Q      = smem;
    float* s_K      = s_Q + head_dim;
    float* s_V      = s_K + SCALAR_TILE_KV * head_dim;
    float* s_scores = s_V + SCALAR_TILE_KV * head_dim;
    float* s_out    = s_scores + SCALAR_TILE_KV;
    float* s_reduce = s_out + head_dim;

    for (int d = tid; d < head_dim; d += SCALAR_BDIM) {
        s_Q[d]   = (float)Q[((size_t)bh * seq_len + qi) * head_dim + d] * scale_Q;
        s_out[d] = 0.f;
    }
    __syncthreads();

    float rmax = -1e38f, rsum = 0.f;

    for (int t = 0; t < (seq_len + SCALAR_TILE_KV - 1) / SCALAR_TILE_KV; ++t) {
        const int ts  = t * SCALAR_TILE_KV;
        const int tlen = min(SCALAR_TILE_KV, seq_len - ts);

        for (int idx = tid; idx < tlen * head_dim; idx += SCALAR_BDIM) {
            const int r = idx / head_dim, d = idx % head_dim;
            const size_t base = ((size_t)bh * seq_len + ts + r) * head_dim + d;
            s_K[r * head_dim + d] = (float)K[base] * scale_K;
            s_V[r * head_dim + d] = (float)V[base] * scale_V;
        }
        __syncthreads();

        for (int j = tid; j < tlen; j += SCALAR_BDIM) {
            float acc = 0.f;
            for (int d = 0; d < head_dim; ++d) acc += s_Q[d] * s_K[j * head_dim + d];
            s_scores[j] = acc * attn_scale;
        }
        __syncthreads();

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

    const float inv_rsum = 1.f / rsum;
    for (int d = tid; d < head_dim; d += SCALAR_BDIM) {
        out[((size_t)bh * seq_len + qi) * head_dim + d] =
            f32_to_i8(s_out[d] * inv_rsum, inv_out_scale);
    }
}

// ── WMMA path ──────────────────────────────────────────────────────────
#if INT8_ATTN_WMMA

using namespace nvcuda::wmma;

#define ATTN_WMMA_M     16
#define ATTN_WMMA_N     16
#define ATTN_WMMA_K     16
#define ATTN_TILE_Q     32
#define ATTN_TILE_KV    32
#define ATTN_BDIM       64
#define ATTN_SMEM_PAD    8   // matches Jonathan's SMEM_PAD

// Maximum head_dim supported = ATTN_WMMA_N × ATTN_MAX_SLICES = 16 × 16 = 256.
#define ATTN_MAX_SLICES 16

__global__ __launch_bounds__(ATTN_BDIM, 2)
void int8_wmma_attention_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    int8_t* __restrict__       out,
    const float* __restrict__ d_scale_Q,
    const float* __restrict__ d_scale_K,
    const float* __restrict__ d_scale_V,
    int seq_len, int head_dim,
    float attn_scale
) {
    const float scale_Q       = __ldg(d_scale_Q);
    const float scale_K       = __ldg(d_scale_K);
    const float scale_V       = __ldg(d_scale_V);
    const float inv_out_scale = 1.f / scale_V;
    // Combined scale applied to INT32 QK^T accumulator to get attention scores.
    const float qk_scale      = scale_Q * scale_K * attn_scale;
    const int bh      = blockIdx.y;
    const int qi_base = blockIdx.x * ATTN_TILE_Q;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    extern __shared__ char smem_raw[];
    // INT8 row stride: head_dim bytes (head_dim is always a multiple of 16 for WMMA).
    const int ks_i8 = head_dim;
    // FP16 strides (in half elements).
    const int vs = head_dim + ATTN_SMEM_PAD;      // V row stride
    const int ss = ATTN_TILE_KV + ATTN_SMEM_PAD;  // scores row stride

    // Smem layout: [Q INT8][K INT8][V FP16][scores FP16]
    // Q and K together = (TILE_Q + TILE_KV) * head_dim bytes, always even → FP16 alignment OK.
    int8_t* s_Q_i8   = (int8_t*) smem_raw;
    int8_t* s_K_i8   = s_Q_i8 + ATTN_TILE_Q  * ks_i8;
    half*   s_V_fp16 = (half*)(s_K_i8 + ATTN_TILE_KV * ks_i8);
    half*   s_scores = s_V_fp16 + ATTN_TILE_KV * vs;

    // Load Q tile as INT8 — no dequantization, scale folded into qk_scale.
    for (int idx = tid; idx < ATTN_TILE_Q * head_dim; idx += ATTN_BDIM) {
        const int row = idx / head_dim, col = idx % head_dim;
        const int qi  = qi_base + row;
        s_Q_i8[row * ks_i8 + col] = (qi < seq_len)
            ? Q[((size_t)bh * seq_len + qi) * head_dim + col]
            : (int8_t)0;
    }

    const int frow0   = lane / 4;
    const int frow1   = lane / 4 + 8;
    const int fcol_lo = (lane % 4) * 2;

    float rmax0 = -1e38f, rmax1 = -1e38f;
    float rsum0 = 0.f,    rsum1 = 0.f;

    const int n_slices    = head_dim / ATTN_WMMA_N;
    const int n_kv_groups = ATTN_TILE_KV / ATTN_WMMA_N;

    fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, float> frag_out[ATTN_MAX_SLICES];
    for (int s = 0; s < n_slices; ++s) fill_fragment(frag_out[s], 0.f);

    __syncthreads();

    const int num_tiles = seq_len / ATTN_TILE_KV;

    for (int t = 0; t < num_tiles; ++t) {
        // Load K tile as INT8; load V tile dequantized to FP16 for attn×V WMMA.
        const size_t kv_base = ((size_t)bh * seq_len + t * ATTN_TILE_KV) * head_dim;
        for (int idx = tid; idx < ATTN_TILE_KV * head_dim; idx += ATTN_BDIM) {
            const int row = idx / head_dim, col = idx % head_dim;
            const size_t off = kv_base + row * head_dim + col;
            s_K_i8[row * ks_i8 + col]  = K[off];
            s_V_fp16[row * vs   + col]  = __float2half((float)V[off] * scale_V);
        }
        __syncthreads();

        // QK^T via INT8 WMMA → INT32 accumulator.
        // K stored row-major → loaded col_major gives K^T implicitly.
        fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int32_t>
            frag_qk[ATTN_TILE_KV / ATTN_WMMA_N];
        for (int g = 0; g < n_kv_groups; ++g) {
            fill_fragment(frag_qk[g], (int32_t)0);
            for (int k = 0; k < head_dim; k += ATTN_WMMA_K) {
                fragment<matrix_a, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int8_t, row_major> fq;
                fragment<matrix_b, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int8_t, col_major> fk;
                load_matrix_sync(fq, s_Q_i8 + warp_id * ATTN_WMMA_M * ks_i8 + k, ks_i8);
                load_matrix_sync(fk, s_K_i8 + g * ATTN_WMMA_N * ks_i8 + k,       ks_i8);
                mma_sync(frag_qk[g], fq, fk, frag_qk[g]);
            }
        }

        // Cast INT32 accumulators to float and apply qk_scale.
        // Store in sf[group][element] since we can't write float back to int32_t frag.
        float sf[ATTN_TILE_KV / ATTN_WMMA_N][8];
        float lmax0 = -1e38f, lmax1 = -1e38f;
        for (int g = 0; g < n_kv_groups; ++g) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) sf[g][i] = (float)frag_qk[g].x[i] * qk_scale;
            lmax0 = fmaxf(lmax0, fmaxf(fmaxf(sf[g][0],sf[g][1]), fmaxf(sf[g][4],sf[g][5])));
            lmax1 = fmaxf(lmax1, fmaxf(fmaxf(sf[g][2],sf[g][3]), fmaxf(sf[g][6],sf[g][7])));
        }
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 1));
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 2));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 1));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 2));

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

        // Softmax weights → s_scores (FP16, for attn×V WMMA).
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

        // attn × V via FP16 WMMA (s_V_fp16 is already FP16).
        __syncwarp();
        for (int s = 0; s < n_slices; ++s) {
            const int d_base = s * ATTN_WMMA_N;
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

    // Normalize and quantize to INT8.
    const float inv0 = 1.f / rsum0, inv1 = 1.f / rsum1;
    for (int s = 0; s < n_slices; ++s) {
        const int d    = s * ATTN_WMMA_N;
        const int qi0  = qi_base + warp_id * ATTN_WMMA_M + frow0;
        const int qi1  = qi_base + warp_id * ATTN_WMMA_M + frow1;
        const size_t b0 = ((size_t)bh * seq_len + qi0) * head_dim;
        const size_t b1 = ((size_t)bh * seq_len + qi1) * head_dim;
        if (qi0 < seq_len) {
            out[b0+d+fcol_lo]   = f32_to_i8(frag_out[s].x[0]*inv0, inv_out_scale);
            out[b0+d+fcol_lo+1] = f32_to_i8(frag_out[s].x[1]*inv0, inv_out_scale);
            out[b0+d+fcol_lo+8] = f32_to_i8(frag_out[s].x[4]*inv0, inv_out_scale);
            out[b0+d+fcol_lo+9] = f32_to_i8(frag_out[s].x[5]*inv0, inv_out_scale);
        }
        if (qi1 < seq_len) {
            out[b1+d+fcol_lo]   = f32_to_i8(frag_out[s].x[2]*inv1, inv_out_scale);
            out[b1+d+fcol_lo+1] = f32_to_i8(frag_out[s].x[3]*inv1, inv_out_scale);
            out[b1+d+fcol_lo+8] = f32_to_i8(frag_out[s].x[6]*inv1, inv_out_scale);
            out[b1+d+fcol_lo+9] = f32_to_i8(frag_out[s].x[7]*inv1, inv_out_scale);
        }
    }
}

#endif  // INT8_ATTN_WMMA

// ── Public interface ───────────────────────────────────────────────────
// scale_Q/K/V are device pointers — no cudaMemcpy in the hot path.
// attn_scale = 1/sqrt(head_dim) is host-computable and scale-independent.
void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, int8_t* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int batch, int heads, int seq_len, int head_dim
) {
    const float attn_scale = 1.f / sqrtf((float)head_dim);

#if INT8_ATTN_WMMA
    if (head_dim % ATTN_WMMA_K == 0 && seq_len % ATTN_TILE_KV == 0) {
        const int BH = batch * heads;
        dim3 grid((seq_len + ATTN_TILE_Q - 1) / ATTN_TILE_Q, BH);
        // Q + K in INT8, V in FP16 (dequant on load), scores in FP16.
        const size_t smem =
            (size_t)(ATTN_TILE_Q + ATTN_TILE_KV) * head_dim * sizeof(int8_t) +
            (size_t) ATTN_TILE_KV * (head_dim + ATTN_SMEM_PAD) * sizeof(half) +
            (size_t) ATTN_TILE_Q  * (ATTN_TILE_KV + ATTN_SMEM_PAD) * sizeof(half);
        cudaFuncSetAttribute(int8_wmma_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        int8_wmma_attention_kernel<<<grid, ATTN_BDIM, smem>>>(
            Q, K, V, out, scale_Q, scale_K, scale_V,
            seq_len, head_dim, attn_scale);
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
