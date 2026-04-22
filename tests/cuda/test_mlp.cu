// test_mlp.cu — Pure CUDA correctness test + timing for MLP kernel.
// Owner: Shengjing
//
// Usage:
//   make test_mlp && ./test_mlp
//
// Reads testdata/{small,large}/ .bin files, runs mlp_forward,
// compares output against pre-computed reference, then reports
// mean kernel latency using CUDA Events.

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "test_utils.h"

// Declared in kernels/mlp.cu
extern void mlp_forward(
    const half* x, const half* W1, const half* W2, half* out,
    int batch, int seq_len, int d_model, int d_ff
);

// ── Correctness test ───────────────────────────────────────────────────
bool run_test(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], S = cfg["seq_len"], M = cfg["d_model"], F = cfg["d_ff"];

    size_t x_count   = (size_t)B * S * M;
    size_t w1_count  = (size_t)M * F;
    size_t w2_count  = (size_t)F * M;
    size_t out_count = x_count;

    char path[512];

    snprintf(path, sizeof(path), "%s/mlp_x.bin",  data_dir);
    half* x  = load_bin<half>(path, x_count);

    snprintf(path, sizeof(path), "%s/mlp_W1.bin", data_dir);
    half* W1 = load_bin<half>(path, w1_count);

    snprintf(path, sizeof(path), "%s/mlp_W2.bin", data_dir);
    half* W2 = load_bin<half>(path, w2_count);

    half* out;
    cudaMalloc(&out, out_count * sizeof(half));

    // Run kernel
    mlp_forward(x, W1, W2, out, B, S, M, F);
    cudaDeviceSynchronize();

    // Check result
    snprintf(path, sizeof(path), "%s/mlp_ref.bin", data_dir);
    bool passed = check_result_fp16(out, path, out_count, 1e-2f, label);

    cudaFree(x);
    cudaFree(W1);
    cudaFree(W2);
    cudaFree(out);
    return passed;
}

// ── Timing benchmark ───────────────────────────────────────────────────
void run_timing(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], S = cfg["seq_len"], M = cfg["d_model"], F = cfg["d_ff"];

    size_t x_count   = (size_t)B * S * M;
    size_t w1_count  = (size_t)M * F;
    size_t w2_count  = (size_t)F * M;
    size_t out_count = x_count;

    char path[512];

    snprintf(path, sizeof(path), "%s/mlp_x.bin",  data_dir);
    half* x  = load_bin<half>(path, x_count);

    snprintf(path, sizeof(path), "%s/mlp_W1.bin", data_dir);
    half* W1 = load_bin<half>(path, w1_count);

    snprintf(path, sizeof(path), "%s/mlp_W2.bin", data_dir);
    half* W2 = load_bin<half>(path, w2_count);

    half* out;
    cudaMalloc(&out, out_count * sizeof(half));

    // Warmup (also pre-allocates the static hidden buffer in mlp.cu)
    const int N_WARMUP = 10;
    const int N_ITER   = 50;

    for (int i = 0; i < N_WARMUP; i++)
        mlp_forward(x, W1, W2, out, B, S, M, F);
    cudaDeviceSynchronize();

    // Timed iterations using CUDA Events
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);

    float total_ms = 0.f;
    for (int i = 0; i < N_ITER; i++) {
        cudaEventRecord(t_start);
        mlp_forward(x, W1, W2, out, B, S, M, F);
        cudaEventRecord(t_stop);
        cudaEventSynchronize(t_stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, t_start, t_stop);
        total_ms += ms;
    }
    float mean_ms = total_ms / N_ITER;

    // Compute TFLOPS
    // Two GEMMs: (T × d_model × d_ff) + (T × d_ff × d_model), each 2× for mul+add
    long long T = (long long)B * S;
    double flops = 2.0 * T * M * F + 2.0 * T * F * M;
    double tflops = flops / (mean_ms * 1e-3) / 1e12;

    printf("[Timing] %s  |  mean = %.3f ms  |  %.2f TFLOPS\n",
           label, mean_ms, tflops);

    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);
    cudaFree(x);
    cudaFree(W1);
    cudaFree(W2);
    cudaFree(out);
}

// ── Main ───────────────────────────────────────────────────────────────
int main() {
    printf("=== MLP Kernel Test ===\n\n");

    // Correctness
    printf("--- Correctness ---\n");
    bool all_passed = true;
    all_passed &= run_test("testdata/small", "small");
    all_passed &= run_test("testdata/large", "large");
    printf("\n%s\n\n", all_passed ? "All tests PASSED." : "Some tests FAILED.");

    // Timing (only meaningful on large config)
    printf("--- Timing (50 runs, 10 warmup) ---\n");
    run_timing("testdata/small", "small");
    run_timing("testdata/large", "large");

    return all_passed ? 0 : 1;
}
