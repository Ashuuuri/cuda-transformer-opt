// int8_attention.cu — INT8 quantized attention CUDA kernel.
// Owner: Heling
//
// Optimization log:
//   v1: cudaMalloc/cudaFree each forward.
//   v2: cached workspace.
//   v3 (this): WMMA INT8 tensor core GEMM; scale stays on GPU.

#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>
#include <mma.h>

using namespace nvcuda::wmma;

extern "C" void quantize_fp16_to_int8(
    const half* input, int8_t* output, float* scale_out, int n
);

// ── WMMA QK^T: C[b,h,i,j] = Σ_k Q[b,h,i,k] × K[b,h,j,k] ────────────────
// Trick: K is stored as (S, D) row-major; treat it as K^T (D, S) col_major
// so that wmma does C = Q × K^T directly without physically transposing.
__global__ void int8_qk_wmma(
    const int8_t* Q, const int8_t* K, int32_t* QK,
    int B, int H, int S, int D
) {
    int bh  = blockIdx.z;
    int row = blockIdx.y * 16;
    int col = blockIdx.x * 16;
    if (row >= S || col >= S) return;

    const int8_t* Q_base = Q  + (size_t)bh * S * D;
    const int8_t* K_base = K  + (size_t)bh * S * D;
    int32_t*      QK_base = QK + (size_t)bh * S * S;

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> q_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> k_frag;  // K^T via col_major
    fragment<accumulator, 16, 16, 16, int32_t>        c_frag;
    fill_fragment(c_frag, 0);

    for (int k = 0; k < D; k += 16) {
        load_matrix_sync(q_frag, Q_base + row * D + k, D);
        load_matrix_sync(k_frag, K_base + col * D + k, D);
        mma_sync(c_frag, q_frag, k_frag, c_frag);
    }

    store_matrix_sync(QK_base + row * S + col, c_frag, S, mem_row_major);
}

// ── Dequant + scale + softmax (per row) ───────────────────────────────
__global__ void dequant_softmax_kernel(
    const int32_t* QK_int32, half* QK_fp16,
    int S, float combined_scale
) {
    int bh_i = blockIdx.x;
    int tid  = threadIdx.x;
    int blk  = blockDim.x;

    const int32_t* row_in  = QK_int32 + (size_t)bh_i * S;
    half*          row_out = QK_fp16  + (size_t)bh_i * S;

    extern __shared__ float sdata[];

    // Pass 1: row max
    float local_max = -1e30f;
    for (int j = tid; j < S; j += blk) {
        float v = (float)row_in[j] * combined_scale;
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

    // Pass 2: exp(v - max), sum
    float local_sum = 0.0f;
    for (int j = tid; j < S; j += blk) {
        float v = (float)row_in[j] * combined_scale;
        float e = expf(v - row_max);
        row_out[j] = __float2half(e);
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

    // Pass 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int j = tid; j < S; j += blk) {
        float e = __half2float(row_out[j]);
        row_out[j] = __float2half(e * inv_sum);
    }
}

// ── WMMA attn @ V: out[b,h,i,d] = Σ_j attn[b,h,i,j] × V[b,h,j,d] ───────
__global__ void int8_av_wmma(
    const int8_t* attn, const int8_t* V, int32_t* out,
    int B, int H, int S, int D
) {
    int bh  = blockIdx.z;
    int row = blockIdx.y * 16;
    int col = blockIdx.x * 16;
    if (row >= S || col >= D) return;

    const int8_t* A_base = attn + (size_t)bh * S * S;
    const int8_t* V_base = V    + (size_t)bh * S * D;
    int32_t*      O_base = out  + (size_t)bh * S * D;

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t>        c_frag;
    fill_fragment(c_frag, 0);

    for (int k = 0; k < S; k += 16) {
        load_matrix_sync(a_frag, A_base + row * S + k, S);
        load_matrix_sync(b_frag, V_base + k   * D + col, D);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(O_base + row * D + col, c_frag, D, mem_row_major);
}

// ── Dequant INT32 → FP16 using device-side scale ───────────────────────
static __global__ void dequant_kernel_dev_attn(
    const int32_t* input, half* output,
    const float* scale_A_ptr, float scale_B, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float scale = (*scale_A_ptr) * scale_B;
    output[idx] = __float2half((float)input[idx] * scale);
}

// ── Cached workspace ───────────────────────────────────────────────────
struct AttnWorkspace {
    int32_t* QK_int32     = nullptr;
    half*    QK_softmax   = nullptr;
    int8_t*  attn_int8    = nullptr;
    int32_t* out_int32    = nullptr;
    float*   d_scale_attn = nullptr;
    size_t   qk_cap       = 0;
    size_t   out_cap      = 0;

    void ensure(size_t qk_n, size_t out_n) {
        if (qk_n > qk_cap) {
            if (QK_int32)   cudaFree(QK_int32);
            if (QK_softmax) cudaFree(QK_softmax);
            if (attn_int8)  cudaFree(attn_int8);
            cudaMalloc(&QK_int32,   qk_n * sizeof(int32_t));
            cudaMalloc(&QK_softmax, qk_n * sizeof(half));
            cudaMalloc(&attn_int8,  qk_n * sizeof(int8_t));
            qk_cap = qk_n;
        }
        if (out_n > out_cap) {
            if (out_int32) cudaFree(out_int32);
            cudaMalloc(&out_int32, out_n * sizeof(int32_t));
            out_cap = out_n;
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
    size_t out_count = bh * S * D;

    g_attn_ws.ensure(qk_count, out_count);

    // Step 1: QK^T (WMMA)
    {
        dim3 block(32);
        dim3 grid((S + 15) / 16, (S + 15) / 16, B * H);
        int8_qk_wmma<<<grid, block>>>(Q, K, g_attn_ws.QK_int32, B, H, S, D);
    }

    // Step 2: dequant + scale + softmax → FP16
    {
        float combined_scale = scale_Q * scale_K / sqrtf((float)D);
        int threads = 256;
        int blocks  = B * H * S;
        dequant_softmax_kernel<<<blocks, threads, threads * sizeof(float)>>>(
            g_attn_ws.QK_int32, g_attn_ws.QK_softmax, S, combined_scale
        );
    }

    // Step 3: quantize softmax output → INT8 (scale stays on device)
    quantize_fp16_to_int8(
        g_attn_ws.QK_softmax, g_attn_ws.attn_int8,
        g_attn_ws.d_scale_attn, (int)qk_count
    );

    // Step 4: attn @ V (WMMA)
    {
        dim3 block(32);
        dim3 grid((D + 15) / 16, (S + 15) / 16, B * H);
        int8_av_wmma<<<grid, block>>>(
            g_attn_ws.attn_int8, V, g_attn_ws.out_int32, B, H, S, D
        );
    }

    // Step 5: dequant → FP16 output using device-side scale
    {
        int threads = 256;
        int blocks  = (out_count + threads - 1) / threads;
        dequant_kernel_dev_attn<<<blocks, threads>>>(
            g_attn_ws.out_int32, out,
            g_attn_ws.d_scale_attn, scale_V, (int)out_count
        );
    }
}
