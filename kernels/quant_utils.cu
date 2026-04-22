// quant_utils.cu — Shared quantization utilities (FP16 <-> INT8 conversion).
// Owner: Heling
//
// Per-tensor symmetric quantization:
//   real_value = int8_value * scale
//   scale = max(|x|) / 127
//
// Public API:
//   void quantize_fp16_to_int8(const half*, int8_t*, float*, int);
//   void dequantize_int8_to_fp16(const int8_t*, half*, float, int);

#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

// ── Block reduction helper ──────────────────────────────────────────────
// Reduces values within a block to a single max, stored in shared mem [0].
__device__ void block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
}

// ── Step 1: Find absolute max across the whole tensor ──────────────────
// Each block reduces its chunk to a local max, then atomically updates
// the global max. global_max should be initialized to 0 before launch.
__global__ void abs_max_kernel(const half* input, float* global_max, int n) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Per-thread: take abs of own element (or 0 if out of bounds)
    float val = (idx < n) ? fabsf(__half2float(input[idx])) : 0.0f;

    // Reduce within block
    block_reduce_max(val, shared);

    // Block leader updates global max atomically
    if (tid == 0) {
        atomicMax((int*)global_max, __float_as_int(shared[0]));
    }
}

// ── Step 2: Quantize each element using the computed scale ─────────────
__global__ void quantize_kernel(const half* input, int8_t* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = __half2float(input[idx]);
    int q = __float2int_rn(x / scale);          // round to nearest int
    if (q > 127)  q = 127;                       // clamp
    if (q < -128) q = -128;
    output[idx] = (int8_t)q;
}

// ── Step 2b: Quantize reading scale from device memory (no CPU sync) ──
__global__ void quantize_kernel_dev(const half* input, int8_t* output,
                                      const float* scale_ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float scale = *scale_ptr;
    float x = __half2float(input[idx]);
    int q = __float2int_rn(x / scale);
    if (q > 127)  q = 127;
    if (q < -128) q = -128;
    output[idx] = (int8_t)q;
}

// ── Finalize scale: max / 127 → scale, on device ──────────────────────
__global__ void finalize_scale_kernel(float* scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *scale = (*scale) / 127.0f;
    }
}

// ── Step 3: Dequantize each element back to FP16 ───────────────────────
__global__ void dequantize_kernel(const int8_t* input, half* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = (float)input[idx] * scale;
    output[idx] = __float2half(x);
}

// ── Public API ─────────────────────────────────────────────────────────

// Quantize FP16 tensor to INT8. Computes scale = max(|x|) / 127.
// `scale_out` is a device pointer for the resulting scale (single float).
extern "C" void quantize_fp16_to_int8(
    const half* input, int8_t* output, float* scale_out, int n
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // 1. Find max(|x|) — async, no CPU sync
    cudaMemsetAsync(scale_out, 0, sizeof(float));
    abs_max_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, scale_out, n);

    // 2. Divide by 127 in place (device-side, no CPU involvement)
    finalize_scale_kernel<<<1, 1>>>(scale_out);

    // 3. Quantize reading scale from device memory
    quantize_kernel_dev<<<blocks, threads>>>(input, output, scale_out, n);
}

// Dequantize INT8 tensor back to FP16 using a known scale (host scalar).
extern "C" void dequantize_int8_to_fp16(
    const int8_t* input, half* output, float scale, int n
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    dequantize_kernel<<<blocks, threads>>>(input, output, scale, n);
}