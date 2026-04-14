// test_mlp.cu — Pure CUDA correctness test for MLP kernel.
// Owner: Shengjing
//
// Usage:
//   make test_mlp && ./test_mlp
//
// Reads testdata/{small,large}/ .bin files, runs mlp_forward,
// compares output against pre-computed reference.

#include <cstdio>
#include <cuda_fp16.h>
#include "test_utils.h"

// Declared in kernels/mlp.cu
extern void mlp_forward(
    const half* x, const half* W1, const half* W2, half* out,
    int batch, int seq_len, int d_model, int d_ff
);

bool run_test(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], S = cfg["seq_len"], M = cfg["d_model"], F = cfg["d_ff"];

    size_t x_count = (size_t)B * S * M;
    size_t w1_count = (size_t)M * F;
    size_t w2_count = (size_t)F * M;
    size_t out_count = x_count;

    char path[512];

    snprintf(path, sizeof(path), "%s/mlp_x.bin", data_dir);
    half* x = load_bin<half>(path, x_count);

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

int main() {
    printf("=== MLP Kernel Test ===\n");

    bool all_passed = true;
    all_passed &= run_test("testdata/small", "small");
    all_passed &= run_test("testdata/large", "large");

    printf("\n%s\n", all_passed ? "All tests PASSED." : "Some tests FAILED.");
    return all_passed ? 0 : 1;
}
