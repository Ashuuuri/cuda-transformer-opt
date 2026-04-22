// test_int8.cu — Pure CUDA correctness test for INT8 attention + MLP kernels.
// Owner: Heling
//
// Usage:
//   make test_int8 && ./test_int8
//
// Reads testdata/{small,large}/ .bin files, runs int8_attention_forward
// and int8_mlp_forward, compares output against pre-computed reference.

#include <cstdio>
#include <cuda_fp16.h>
#include <cstdint>
#include "test_utils.h"

// Declared in kernels/quant_utils.cu
extern "C" void quantize_fp16_to_int8(
    const half* input, int8_t* output, float* scale_out, int n
);
extern "C" void dequantize_int8_to_fp16(
    const int8_t* input, half* output, float scale, int n
);

// Declared in kernels/int8_attention.cu
extern "C" void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    float scale_Q, float scale_K, float scale_V,
    int batch, int heads, int seq_len, int head_dim
);

// Declared in kernels/int8_mlp.cu
extern "C" void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, half* out,
    float scale_x, float scale_W1, float scale_W2,
    int batch, int seq_len, int d_model, int d_ff
);

bool run_attention_test(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], H = cfg["heads"], S = cfg["seq_len"], D = cfg["head_dim"];
    size_t count = (size_t)B * H * S * D;

    char path[512];

    snprintf(path, sizeof(path), "%s/int8_attn_Q.bin", data_dir);
    int8_t* Q = load_bin<int8_t>(path, count);

    snprintf(path, sizeof(path), "%s/int8_attn_K.bin", data_dir);
    int8_t* K = load_bin<int8_t>(path, count);

    snprintf(path, sizeof(path), "%s/int8_attn_V.bin", data_dir);
    int8_t* V = load_bin<int8_t>(path, count);

    // Load scales (host scalars)
    snprintf(path, sizeof(path), "%s/int8_attn_scale_Q.bin", data_dir);
    float h_scale_Q = load_scale(path);
    snprintf(path, sizeof(path), "%s/int8_attn_scale_K.bin", data_dir);
    float h_scale_K = load_scale(path);
    snprintf(path, sizeof(path), "%s/int8_attn_scale_V.bin", data_dir);
    float h_scale_V = load_scale(path);

    // Allocate FP16 output
    half* out_fp16;
    cudaMalloc(&out_fp16, count * sizeof(half));

    // Run kernel
    int8_attention_forward(Q, K, V, out_fp16, h_scale_Q, h_scale_K, h_scale_V, B, H, S, D);
    cudaDeviceSynchronize();

    // Compare against FP16 reference (attn_ref.bin from gen_testdata.py uses
    // dequantized inputs through the FP16 baseline — apples-to-apples).
    snprintf(path, sizeof(path), "%s/attn_ref.bin", data_dir);
    bool passed = check_result_fp16(out_fp16, path, count, 0.1f, label);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(out_fp16);
    return passed;
}

bool run_mlp_test(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], S = cfg["seq_len"], M = cfg["d_model"], F = cfg["d_ff"];
    size_t x_count = (size_t)B * S * M;
    size_t w1_count = (size_t)M * F;
    size_t w2_count = (size_t)F * M;
    size_t out_count = x_count;

    char path[512];

    snprintf(path, sizeof(path), "%s/int8_mlp_x.bin", data_dir);
    int8_t* x = load_bin<int8_t>(path, x_count);

    snprintf(path, sizeof(path), "%s/int8_mlp_W1.bin", data_dir);
    int8_t* W1 = load_bin<int8_t>(path, w1_count);

    snprintf(path, sizeof(path), "%s/int8_mlp_W2.bin", data_dir);
    int8_t* W2 = load_bin<int8_t>(path, w2_count);

    // Load scales (host scalars)
    snprintf(path, sizeof(path), "%s/int8_mlp_scale_x.bin", data_dir);
    float h_scale_x = load_scale(path);
    snprintf(path, sizeof(path), "%s/int8_mlp_scale_W1.bin", data_dir);
    float h_scale_W1 = load_scale(path);
    snprintf(path, sizeof(path), "%s/int8_mlp_scale_W2.bin", data_dir);
    float h_scale_W2 = load_scale(path);

    // Allocate FP16 output
    half* out_fp16;
    cudaMalloc(&out_fp16, out_count * sizeof(half));

    // Run kernel
    int8_mlp_forward(x, W1, W2, out_fp16, h_scale_x, h_scale_W1, h_scale_W2, B, S, M, F);
    cudaDeviceSynchronize();

    // Compare against FP16 reference (mlp_ref.bin from gen_testdata.py uses
    // dequantized inputs through the FP16 baseline — apples-to-apples).
    snprintf(path, sizeof(path), "%s/mlp_ref.bin", data_dir);
    bool passed = check_result_fp16(out_fp16, path, out_count, 0.1f, label);

    cudaFree(x);
    cudaFree(W1);
    cudaFree(W2);
    cudaFree(out_fp16);
    return passed;
}

// ── quant_utils round-trip test ────────────────────────────────────────
// FP16 → quantize → INT8 → dequantize → FP16, compare to original.
// Max error should be bounded by scale (one quantization step).
bool run_quant_roundtrip_test(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], H = cfg["heads"], S = cfg["seq_len"], D = cfg["head_dim"];
    size_t count = (size_t)B * H * S * D;

    // Use the FP16 attention Q tensor as input
    char path[512];
    snprintf(path, sizeof(path), "%s/attn_Q.bin", data_dir);
    half* d_input = load_bin<half>(path, count);

    // Allocate device buffers
    int8_t* d_quantized;
    half* d_dequantized;
    float* d_scale;
    cudaMalloc(&d_quantized, count * sizeof(int8_t));
    cudaMalloc(&d_dequantized, count * sizeof(half));
    cudaMalloc(&d_scale, sizeof(float));

    // Round trip
    quantize_fp16_to_int8(d_input, d_quantized, d_scale, (int)count);
    cudaDeviceSynchronize();

    float h_scale;
    cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost);

    dequantize_int8_to_fp16(d_quantized, d_dequantized, h_scale, (int)count);
    cudaDeviceSynchronize();

    // Compare against original — tolerance is one quantization step (scale)
    // plus a small margin for FP16 conversion noise.
    float atol = h_scale * 1.5f;
    char check_label[128];
    snprintf(check_label, sizeof(check_label), "quant_roundtrip_%s (scale=%.6f)", label, h_scale);

    // Save original to a temp .bin would be wasteful; instead use input directly
    // by writing a custom comparison here.
    half* h_orig = (half*)malloc(count * sizeof(half));
    half* h_recv = (half*)malloc(count * sizeof(half));
    cudaMemcpy(h_orig, d_input, count * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_recv, d_dequantized, count * sizeof(half), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    float sum_err = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(__half2float(h_orig[i]) - __half2float(h_recv[i]));
        sum_err += diff;
        if (diff > max_err) max_err = diff;
    }
    float mean_err = sum_err / count;
    bool passed = (max_err <= atol);

    printf("[%s] %s  |  max err = %.6f  |  mean err = %.6f  |  atol = %.6f\n",
           passed ? "PASSED" : "FAILED", check_label, max_err, mean_err, atol);

    free(h_orig);
    free(h_recv);
    cudaFree(d_input);
    cudaFree(d_quantized);
    cudaFree(d_dequantized);
    cudaFree(d_scale);
    return passed;
}

int main() {
    printf("=== INT8 Kernel Tests ===\n\n");

    printf("--- quant_utils round-trip ---\n");
    bool all_passed = true;
    all_passed &= run_quant_roundtrip_test("testdata/small", "small");
    all_passed &= run_quant_roundtrip_test("testdata/large", "large");

    printf("\n--- INT8 Attention ---\n");
    all_passed &= run_attention_test("testdata/small", "small_attn");
    all_passed &= run_attention_test("testdata/large", "large_attn");

    printf("\n--- INT8 MLP ---\n");
    all_passed &= run_mlp_test("testdata/small", "small_mlp");
    all_passed &= run_mlp_test("testdata/large", "large_mlp");

    printf("\n%s\n", all_passed ? "All tests PASSED." : "Some tests FAILED.");
    return all_passed ? 0 : 1;
}
