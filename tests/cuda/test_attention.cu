// test_attention.cu — Pure CUDA correctness test for attention kernel.
// Owner: Jonathan
//
// Usage:
//   make test_attention && ./test_attention
//
// Reads testdata/{small,large}/ .bin files, runs attention_forward,
// compares output against pre-computed reference.

#include <cstdio>
#include <cuda_fp16.h>
#include "test_utils.h"

// Declared in kernels/attention.cu
extern void attention_forward(
    const half* Q, const half* K, const half* V, half* out,
    int batch, int heads, int seq_len, int head_dim
);

bool run_test(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], H = cfg["heads"], S = cfg["seq_len"], D = cfg["head_dim"];
    size_t count = (size_t)B * H * S * D;

    char path[512];

    snprintf(path, sizeof(path), "%s/attn_Q.bin", data_dir);
    half* Q = load_bin<half>(path, count);

    snprintf(path, sizeof(path), "%s/attn_K.bin", data_dir);
    half* K = load_bin<half>(path, count);

    snprintf(path, sizeof(path), "%s/attn_V.bin", data_dir);
    half* V = load_bin<half>(path, count);

    half* out;
    cudaMalloc(&out, count * sizeof(half));

    // Run kernel
    attention_forward(Q, K, V, out, B, H, S, D);
    cudaDeviceSynchronize();

    // Check result
    snprintf(path, sizeof(path), "%s/attn_ref.bin", data_dir);
    bool passed = check_result_fp16(out, path, count, 1e-2f, label);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(out);
    return passed;
}

int main() {
    printf("=== Attention Kernel Test ===\n");

    bool all_passed = true;
    all_passed &= run_test("testdata/small", "small");
    all_passed &= run_test("testdata/large", "large");

    printf("\n%s\n", all_passed ? "All tests PASSED." : "Some tests FAILED.");
    return all_passed ? 0 : 1;
}
