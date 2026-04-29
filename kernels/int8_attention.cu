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
    float scale_Q, float scale_K, float scale_V,
    int seq_len, int head_dim,
    float attn_scale, float inv_out_scale   // inv_out_scale = 1/scale_V
) {
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
    float scale_Q, float scale_K, float scale_V,
    int seq_len, int head_dim,
    float attn_scale, float inv_out_scale
) {
    const int bh      = blockIdx.y;
    const int qi_base = blockIdx.x * ATTN_TILE_Q;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    extern __shared__ char smem_raw[];
    const int qs = head_dim + ATTN_SMEM_PAD;       // padded row stride (halves)
    const int ss = ATTN_TILE_KV + ATTN_SMEM_PAD;   // padded score row stride

    half* s_Q      = (half*) smem_raw;
    half* s_K0     = s_Q  + ATTN_TILE_Q  * qs;
    half* s_K1     = s_K0 + ATTN_TILE_KV * qs;
    half* s_V0     = s_K1 + ATTN_TILE_KV * qs;
    half* s_V1     = s_V0 + ATTN_TILE_KV * qs;
    half* s_scores = s_V1 + ATTN_TILE_KV * qs;

    // Load and dequantize Q tile into FP16 shared memory.
    for (int idx = tid; idx < ATTN_TILE_Q * head_dim; idx += ATTN_BDIM) {
        const int row = idx / head_dim, col = idx % head_dim;
        const int qi  = qi_base + row;
        s_Q[row * qs + col] = (qi < seq_len)
            ? __float2half((float)Q[((size_t)bh * seq_len + qi) * head_dim + col] * scale_Q)
            : __float2half(0.f);
    }

    const int frow0   = lane / 4;
    const int frow1   = lane / 4 + 8;
    const int fcol_lo = (lane % 4) * 2;

    float rmax0 = -1e38f, rmax1 = -1e38f;
    float rsum0 = 0.f,    rsum1 = 0.f;

    const int n_slices    = head_dim / ATTN_WMMA_N;   // e.g. 4 for head_dim=64
    const int n_kv_groups = ATTN_TILE_KV / ATTN_WMMA_N;

    fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, float> frag_out[ATTN_MAX_SLICES];
    for (int s = 0; s < n_slices; ++s) fill_fragment(frag_out[s], 0.f);

    __syncthreads();

    const int num_tiles = seq_len / ATTN_TILE_KV;

    for (int t = 0; t < num_tiles; ++t) {
        half* s_K_cur = (t & 1) ? s_K1 : s_K0;
        half* s_V_cur = (t & 1) ? s_V1 : s_V0;

        // Load K,V tile for tile t: dequantize INT8 → FP16 in smem.
        // No cp.async here — avoids needing a separate INT8 smem buffer.
        const size_t kv_base = ((size_t)bh * seq_len + t * ATTN_TILE_KV) * head_dim;
        for (int idx = tid; idx < ATTN_TILE_KV * head_dim; idx += ATTN_BDIM) {
            const int row = idx / head_dim, col = idx % head_dim;
            const size_t off = kv_base + row * head_dim + col;
            s_K_cur[row * qs + col] = __float2half((float)K[off] * scale_K);
            s_V_cur[row * qs + col] = __float2half((float)V[off] * scale_V);
        }
        __syncthreads();

        // QK^T via FP16 WMMA (identical to Jonathan's kernel).
        fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, float>
            frag_qk[ATTN_TILE_KV / ATTN_WMMA_N];
        for (int g = 0; g < n_kv_groups; ++g) {
            fill_fragment(frag_qk[g], 0.f);
            for (int k = 0; k < head_dim; k += ATTN_WMMA_K) {
                fragment<matrix_a, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fq;
                fragment<matrix_b, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, col_major> fk;
                load_matrix_sync(fq, s_Q     + warp_id * ATTN_WMMA_M * qs + k, qs);
                load_matrix_sync(fk, s_K_cur + g * ATTN_WMMA_N * qs + k,       qs);
                mma_sync(frag_qk[g], fq, fk, frag_qk[g]);
            }
        }

        // Per-row max over frag_qk (Jonathan's register-level online softmax).
        float lmax0 = -1e38f, lmax1 = -1e38f;
        for (int g = 0; g < n_kv_groups; ++g) {
            const float s0 = frag_qk[g].x[0] * attn_scale,  s1 = frag_qk[g].x[1] * attn_scale;
            const float s8 = frag_qk[g].x[4] * attn_scale,  s9 = frag_qk[g].x[5] * attn_scale;
            frag_qk[g].x[0]=s0; frag_qk[g].x[1]=s1; frag_qk[g].x[4]=s8; frag_qk[g].x[5]=s9;
            lmax0 = fmaxf(lmax0, fmaxf(fmaxf(s0,s1), fmaxf(s8,s9)));
            const float t0 = frag_qk[g].x[2] * attn_scale,  t1 = frag_qk[g].x[3] * attn_scale;
            const float t8 = frag_qk[g].x[6] * attn_scale,  t9 = frag_qk[g].x[7] * attn_scale;
            frag_qk[g].x[2]=t0; frag_qk[g].x[3]=t1; frag_qk[g].x[6]=t8; frag_qk[g].x[7]=t9;
            lmax1 = fmaxf(lmax1, fmaxf(fmaxf(t0,t1), fmaxf(t8,t9)));
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

        // Softmax weights → s_scores.
        float lsum0 = 0.f, lsum1 = 0.f;
        for (int g = 0; g < n_kv_groups; ++g) {
            const int qr0 = warp_id * ATTN_WMMA_M + frow0;
            const int qr1 = warp_id * ATTN_WMMA_M + frow1;
            const int kc  = g * ATTN_WMMA_N + fcol_lo;
            const float w0=__expf(frag_qk[g].x[0]-rmax0), w1=__expf(frag_qk[g].x[1]-rmax0);
            const float w8=__expf(frag_qk[g].x[4]-rmax0), w9=__expf(frag_qk[g].x[5]-rmax0);
            lsum0 += w0+w1+w8+w9;
            s_scores[qr0*ss+kc]=__float2half(w0); s_scores[qr0*ss+kc+1]=__float2half(w1);
            s_scores[qr0*ss+kc+8]=__float2half(w8); s_scores[qr0*ss+kc+9]=__float2half(w9);
            const float u0=__expf(frag_qk[g].x[2]-rmax1), u1=__expf(frag_qk[g].x[3]-rmax1);
            const float u8=__expf(frag_qk[g].x[6]-rmax1), u9=__expf(frag_qk[g].x[7]-rmax1);
            lsum1 += u0+u1+u8+u9;
            s_scores[qr1*ss+kc]=__float2half(u0); s_scores[qr1*ss+kc+1]=__float2half(u1);
            s_scores[qr1*ss+kc+8]=__float2half(u8); s_scores[qr1*ss+kc+9]=__float2half(u9);
        }
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 1);
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 2);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 1);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 2);
        rsum0 += lsum0;  rsum1 += lsum1;

        // attn × V via FP16 WMMA.
        __syncwarp();
        for (int s = 0; s < n_slices; ++s) {
            const int d_base = s * ATTN_WMMA_N;
            for (int k = 0; k < ATTN_TILE_KV; k += ATTN_WMMA_K) {
                fragment<matrix_a, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fw;
                fragment<matrix_b, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fv;
                load_matrix_sync(fw, s_scores + warp_id * ATTN_WMMA_M * ss + k, ss);
                load_matrix_sync(fv, s_V_cur  + k * qs + d_base,                qs);
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
void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, int8_t* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int batch, int heads, int seq_len, int head_dim
) {
    float h_sQ, h_sK, h_sV;
    cudaMemcpy(&h_sQ, scale_Q, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sK, scale_K, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sV, scale_V, sizeof(float), cudaMemcpyDeviceToHost);

    const float attn_scale    = 1.f / sqrtf((float)head_dim);
    const float inv_out_scale = 1.f / h_sV;

#if INT8_ATTN_WMMA
    if (head_dim % ATTN_WMMA_K == 0 && seq_len % ATTN_TILE_KV == 0) {
        const int BH = batch * heads;
        dim3 grid((seq_len + ATTN_TILE_Q - 1) / ATTN_TILE_Q, BH);
        const size_t smem =
            ((size_t)(ATTN_TILE_Q + 4 * ATTN_TILE_KV) * (head_dim + ATTN_SMEM_PAD) +
             (size_t) ATTN_TILE_Q * (ATTN_TILE_KV + ATTN_SMEM_PAD)) * sizeof(half);
        cudaFuncSetAttribute(int8_wmma_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        int8_wmma_attention_kernel<<<grid, ATTN_BDIM, smem>>>(
            Q, K, V, out, h_sQ, h_sK, h_sV,
            seq_len, head_dim, attn_scale, inv_out_scale);
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
        Q, K, V, out, h_sQ, h_sK, h_sV,
        seq_len, head_dim, attn_scale, inv_out_scale);
}
