// int8_attention.cu — INT8 quantized attention CUDA kernel (NAIVE version).
// Owner: Heling
//
// Required interface:
//
//   void int8_attention_forward(
//       const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
//       float scale_Q, float scale_K, float scale_V,
//       int batch, int heads, int seq_len, int head_dim
//   );
//
// Inputs are row-major contiguous with shape (batch, heads, seq_len, head_dim).
// Per-tensor symmetric quantization: real_value = int8_value * scale.
// Output is FP16 with shape (batch, heads, seq_len, head_dim).
//
// Data flow:
//   Q_int8 @ K_int8^T  (INT32 acc)         [B, H, S, S]
//     → dequant × (scale_Q × scale_K / sqrt(D)), then softmax
//   requant softmax(FP16) → INT8                [B, H, S, S]
//   attn_int8 @ V_int8  (INT32 acc)         [B, H, S, D]
//     → dequant × (scale_attn × scale_V) → output (FP16)
//
// This is a NAIVE implementation: one thread per output element. No tensor
// cores, no tiling. Validates correctness before optimization.

#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>

// Forward declaration from quant_utils.cu
extern "C" void quantize_fp16_to_int8(
    const half* input, int8_t* output, float* scale_out, int n
);

// ── INT8 GEMM: QK^T ────────────────────────────────────────────────────
// Q: [B, H, S, D] (row-major), K: [B, H, S, D] (so we use K[j,k] not K^T[k,j])
// Output QK_int32: [B, H, S, S]
// QK[b,h,i,j] = sum_k Q[b,h,i,k] * K[b,h,j,k]
__global__ void int8_qk_naive(
    const int8_t* Q, const int8_t* K, int32_t* QK,
    int B, int H, int S, int D
) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S || j >= S) return;

    const int8_t* Q_row = Q + ((size_t)bh * S + i) * D;
    const int8_t* K_row = K + ((size_t)bh * S + j) * D;

    int32_t acc = 0;
    for (int k = 0; k < D; k++) {
        acc += (int32_t)Q_row[k] * (int32_t)K_row[k];
    }
    QK[((size_t)bh * S + i) * S + j] = acc;
}

// ── Dequant + scale + softmax (per row) ───────────────────────────────
// Each block handles one row of one (batch, head) — total B*H*S blocks.
// Block size = 256 threads, processes a row of length S in chunks.
// Combined scale = scale_Q * scale_K / sqrt(D), applied during dequant.
__global__ void dequant_softmax_kernel(
    const int32_t* QK_int32, half* QK_fp16,
    int S, float combined_scale
) {
    int bh_i = blockIdx.x;       // which (b, h, i) row
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const int32_t* row_in = QK_int32 + (size_t)bh_i * S;
    half* row_out = QK_fp16 + (size_t)bh_i * S;

    extern __shared__ float sdata[];

    // Pass 1: find row max (after applying combined_scale)
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

    // Pass 2: compute exp(v - max), store into row_out, accumulate sum
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

    // Pass 3: normalize by sum
    float inv_sum = 1.0f / row_sum;
    for (int j = tid; j < S; j += blk) {
        float e = __half2float(row_out[j]);
        row_out[j] = __float2half(e * inv_sum);
    }
}

// ── INT8 GEMM: attn @ V ────────────────────────────────────────────────
// attn: [B, H, S, S], V: [B, H, S, D], out_int32: [B, H, S, D]
// out[b,h,i,d] = sum_j attn[b,h,i,j] * V[b,h,j,d]
__global__ void int8_av_naive(
    const int8_t* attn, const int8_t* V, int32_t* out,
    int B, int H, int S, int D
) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S || d >= D) return;

    const int8_t* attn_row = attn + ((size_t)bh * S + i) * S;
    int32_t acc = 0;
    for (int j = 0; j < S; j++) {
        acc += (int32_t)attn_row[j] * (int32_t)V[((size_t)bh * S + j) * D + d];
    }
    out[((size_t)bh * S + i) * D + d] = acc;
}

// ── Dequantize INT32 → FP16 ────────────────────────────────────────────
__global__ void dequant_int32_kernel(
    const int32_t* input, half* output, float scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __float2half((float)input[idx] * scale);
}

// ── Public API ─────────────────────────────────────────────────────────
extern "C" void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    float scale_Q, float scale_K, float scale_V,
    int batch, int heads, int seq_len, int head_dim
) {
    int B = batch, H = heads, S = seq_len, D = head_dim;
    size_t bh = (size_t)B * H;
    size_t qk_count = bh * S * S;
    size_t out_count = bh * S * D;

    // ── Step 1: QK^T (INT32 accumulation) ──────────────────────────────
    int32_t* QK_int32;
    cudaMalloc(&QK_int32, qk_count * sizeof(int32_t));

    dim3 block_qk(16, 16);
    dim3 grid_qk((S + 15) / 16, (S + 15) / 16, B * H);
    int8_qk_naive<<<grid_qk, block_qk>>>(Q, K, QK_int32, B, H, S, D);

    // ── Step 2: dequant + scale (1/sqrt(D)) + softmax ──────────────────
    half* QK_softmax_fp16;
    cudaMalloc(&QK_softmax_fp16, qk_count * sizeof(half));

    float combined_scale = scale_Q * scale_K / sqrtf((float)D);

    int threads_sm = 256;
    int blocks_sm = B * H * S;
    dequant_softmax_kernel<<<blocks_sm, threads_sm, threads_sm * sizeof(float)>>>(
        QK_int32, QK_softmax_fp16, S, combined_scale
    );

    cudaFree(QK_int32);

    // ── Step 3: requantize softmax output to INT8 ──────────────────────
    int8_t* attn_int8;
    float* d_scale_attn;
    cudaMalloc(&attn_int8, qk_count * sizeof(int8_t));
    cudaMalloc(&d_scale_attn, sizeof(float));

    quantize_fp16_to_int8(QK_softmax_fp16, attn_int8, d_scale_attn, (int)qk_count);

    float scale_attn;
    cudaMemcpy(&scale_attn, d_scale_attn, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(QK_softmax_fp16);
    cudaFree(d_scale_attn);

    // ── Step 4: attn @ V (INT32 accumulation) ──────────────────────────
    int32_t* out_int32;
    cudaMalloc(&out_int32, out_count * sizeof(int32_t));

    dim3 block_av(16, 16);
    dim3 grid_av((D + 15) / 16, (S + 15) / 16, B * H);
    int8_av_naive<<<grid_av, block_av>>>(attn_int8, V, out_int32, B, H, S, D);

    cudaFree(attn_int8);

    // ── Step 5: dequantize → FP16 output ───────────────────────────────
    int threads_dq = 256;
    int blocks_dq = (out_count + threads_dq - 1) / threads_dq;
    dequant_int32_kernel<<<blocks_dq, threads_dq>>>(
        out_int32, out, scale_attn * scale_V, (int)out_count
    );

    cudaFree(out_int32);
}
