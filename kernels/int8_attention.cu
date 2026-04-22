// int8_attention.cu — INT8 quantized attention CUDA kernel.
// Owner: Heling
//
// v1-v5: see history.
// v6 (this): warp-level tiling — each warp computes a 32×32 output tile
//            with 2×2 accumulator fragments, halving HBM fragment loads.

#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>
#include <mma.h>

using namespace nvcuda::wmma;

extern "C" void quantize_fp16_to_int8(
    const half* input, int8_t* output, float* scale_out, int n
);

// ── Fused QK^T + dequant + scale, 32×32 warp tile ──────────────────────
// K accessed via col_major fragment (physical (S,D) row-major = logical K^T).
static __global__ void int8_qk_dequant_wmma(
    const int8_t* Q, const int8_t* K, half* QK_fp16,
    int B, int H, int S, int D, float combined_scale
) {
    int bh  = blockIdx.z;
    int row = blockIdx.y * 32;
    int col = blockIdx.x * 32;
    if (row >= S || col >= S) return;

    const int8_t* Q_base = Q + (size_t)bh * S * D;
    const int8_t* K_base = K + (size_t)bh * S * D;
    half*         O_base = QK_fp16 + (size_t)bh * S * S;

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> q_frag[2];
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> k_frag[2];
    fragment<accumulator, 16, 16, 16, int32_t>        c_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0);

    for (int k = 0; k < D; k += 16) {
        load_matrix_sync(q_frag[0], Q_base + (row + 0)  * D + k, D);
        load_matrix_sync(q_frag[1], Q_base + (row + 16) * D + k, D);
        load_matrix_sync(k_frag[0], K_base + (col + 0)  * D + k, D);
        load_matrix_sync(k_frag[1], K_base + (col + 16) * D + k, D);

        mma_sync(c_frag[0][0], q_frag[0], k_frag[0], c_frag[0][0]);
        mma_sync(c_frag[0][1], q_frag[0], k_frag[1], c_frag[0][1]);
        mma_sync(c_frag[1][0], q_frag[1], k_frag[0], c_frag[1][0]);
        mma_sync(c_frag[1][1], q_frag[1], k_frag[1], c_frag[1][1]);
    }

    __shared__ int32_t c_smem[32 * 32];
    store_matrix_sync(c_smem +  0 * 32 +  0, c_frag[0][0], 32, mem_row_major);
    store_matrix_sync(c_smem +  0 * 32 + 16, c_frag[0][1], 32, mem_row_major);
    store_matrix_sync(c_smem + 16 * 32 +  0, c_frag[1][0], 32, mem_row_major);
    store_matrix_sync(c_smem + 16 * 32 + 16, c_frag[1][1], 32, mem_row_major);
    __syncwarp();

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = lane; i < 1024; i += 32) {
        int r = i >> 5;
        int c = i & 31;
        if (row + r < S && col + c < S) {
            float f = (float)c_smem[i] * combined_scale;
            O_base[(row + r) * S + (col + c)] = __float2half(f);
        }
    }
}

// ── Softmax (FP16 in place) ────────────────────────────────────────────
__global__ void softmax_kernel(half* QK, int S) {
    int bh_i = blockIdx.x;
    int tid  = threadIdx.x;
    int blk  = blockDim.x;

    half* row = QK + (size_t)bh_i * S;

    extern __shared__ float sdata[];

    float local_max = -1e30f;
    for (int j = tid; j < S; j += blk) {
        float v = __half2float(row[j]);
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blk / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < S; j += blk) {
        float v = __half2float(row[j]);
        float e = expf(v - row_max);
        row[j] = __float2half(e);
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blk / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int j = tid; j < S; j += blk) {
        float e = __half2float(row[j]);
        row[j] = __float2half(e * inv_sum);
    }
}

// ── Fused attn·V + dequant, 32×32 warp tile ────────────────────────────
// NOTE: attention's N dimension is head_dim=64, so only 2 column blocks;
// tiling is limited in that dimension but row dim still benefits.
static __global__ void int8_av_dequant_wmma(
    const int8_t* attn, const int8_t* V, half* out,
    int B, int H, int S, int D,
    const float* scale_A_ptr, float scale_B
) {
    int bh  = blockIdx.z;
    int row = blockIdx.y * 32;
    int col = blockIdx.x * 32;
    if (row >= S || col >= D) return;

    const int8_t* A_base = attn + (size_t)bh * S * S;
    const int8_t* V_base = V    + (size_t)bh * S * D;
    half*         O_base = out  + (size_t)bh * S * D;

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag[2];
    fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag[2];
    fragment<accumulator, 16, 16, 16, int32_t>        c_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0);

    for (int k = 0; k < S; k += 16) {
        load_matrix_sync(a_frag[0], A_base + (row + 0)  * S + k, S);
        load_matrix_sync(a_frag[1], A_base + (row + 16) * S + k, S);
        load_matrix_sync(b_frag[0], V_base + k * D + (col + 0),  D);
        load_matrix_sync(b_frag[1], V_base + k * D + (col + 16), D);

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
        if (row + r < S && col + c < D) {
            float f = (float)c_smem[i] * scale;
            O_base[(row + r) * D + (col + c)] = __float2half(f);
        }
    }
}

// ── Cached workspace ───────────────────────────────────────────────────
struct AttnWorkspace {
    half*    QK_fp16      = nullptr;
    int8_t*  attn_int8    = nullptr;
    float*   d_scale_attn = nullptr;
    size_t   qk_cap       = 0;

    void ensure(size_t qk_n) {
        if (qk_n > qk_cap) {
            if (QK_fp16)   cudaFree(QK_fp16);
            if (attn_int8) cudaFree(attn_int8);
            cudaMalloc(&QK_fp16,   qk_n * sizeof(half));
            cudaMalloc(&attn_int8, qk_n * sizeof(int8_t));
            qk_cap = qk_n;
        }
        if (!d_scale_attn) cudaMalloc(&d_scale_attn, sizeof(float));
    }
};

static AttnWorkspace g_attn_ws;

// ── Public API ─────────────────────────────────────────────────────────
extern "C" void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    float scale_Q, float scale_K, float scale_V,
    int batch, int heads, int seq_len, int head_dim
) {
    int B = batch, H = heads, S = seq_len, D = head_dim;
    size_t bh = (size_t)B * H;
    size_t qk_count  = bh * S * S;

    g_attn_ws.ensure(qk_count);

    // Step 1: fused QK^T + dequant + scale → FP16 QK (32×32 tile)
    {
        float combined_scale = scale_Q * scale_K / sqrtf((float)D);
        dim3 block(32);
        dim3 grid((S + 31) / 32, (S + 31) / 32, B * H);
        int8_qk_dequant_wmma<<<grid, block>>>(
            Q, K, g_attn_ws.QK_fp16, B, H, S, D, combined_scale
        );
    }

    // Step 2: softmax
    {
        int threads = 256;
        int blocks  = B * H * S;
        softmax_kernel<<<blocks, threads, threads * sizeof(float)>>>(
            g_attn_ws.QK_fp16, S
        );
    }

    // Step 3: quantize softmax output → INT8 (scale stays on device)
    quantize_fp16_to_int8(
        g_attn_ws.QK_fp16, g_attn_ws.attn_int8,
        g_attn_ws.d_scale_attn, (int)qk_count
    );

    // Step 4: fused attn·V + dequant (32×32 tile)
    {
        dim3 block(32);
        dim3 grid((D + 31) / 32, (S + 31) / 32, B * H);
        int8_av_dequant_wmma<<<grid, block>>>(
            g_attn_ws.attn_int8, V, out,
            B, H, S, D,
            g_attn_ws.d_scale_attn, scale_V
        );
    }
}
