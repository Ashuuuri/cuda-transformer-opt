"""Test harness for INT8 attention + MLP kernels — Heling.

Usage:
    python tests/test_int8.py

Runs correctness checks against dequantized FP16 baseline, then benchmarks
against FP16 baseline and cuBLAS INT8 (torch._int_mm).
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline import attention_baseline, mlp_baseline, check_cuda
from correctness import check_correctness
from benchmark import benchmark

# ── A100 peak performance ───────────────────────────────────────────────
A100_FP16_TFLOPS = 312.0
A100_INT8_TOPS = 624.0


def quantize_to_int8(t):
    """Per-tensor symmetric quantization to INT8."""
    t_f = t.float()
    scale = t_f.abs().max() / 127.0
    t_int8 = (t_f / scale).round().clamp(-128, 127).to(torch.int8)
    return t_int8, scale


def compute_attention_flops(batch, heads, seq_len, head_dim):
    qk_flops = 2 * batch * heads * seq_len * head_dim * seq_len
    av_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    return qk_flops + av_flops


def compute_mlp_flops(batch, seq_len, d_model, d_ff):
    tokens = batch * seq_len
    return 2 * tokens * d_model * d_ff + 2 * tokens * d_ff * d_model


def main():
    check_cuda()

    device = "cuda"
    dtype = torch.float16
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    d_model, d_ff = 512, 2048

    torch.manual_seed(42)

    # ── Prepare FP16 inputs ─────────────────────────────────────────────
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.02

    # ── Quantize ────────────────────────────────────────────────────────
    Q_int8, scale_Q = quantize_to_int8(Q)
    K_int8, scale_K = quantize_to_int8(K)
    V_int8, scale_V = quantize_to_int8(V)

    x_int8, scale_x = quantize_to_int8(x)
    W1_int8, scale_W1 = quantize_to_int8(W1)
    W2_int8, scale_W2 = quantize_to_int8(W2)

    # ── INT8 reference: dequant → FP16 baseline ────────────────────────
    Q_deq = (Q_int8.float() * scale_Q).half()
    K_deq = (K_int8.float() * scale_K).half()
    V_deq = (V_int8.float() * scale_V).half()
    attn_ref = attention_baseline(Q_deq, K_deq, V_deq)

    x_deq = (x_int8.float() * scale_x).half()
    W1_deq = (W1_int8.float() * scale_W1).half()
    W2_deq = (W2_int8.float() * scale_W2).half()
    mlp_ref = mlp_baseline(x_deq, W1_deq, W2_deq)

    # ── Load INT8 CUDA kernels ───────────────────────────────────────────
    from torch.utils.cpp_extension import load
    int8_ext = load(
        name="int8_ext",
        sources=[
            "kernels/int8_attention.cu",
            "kernels/int8_mlp.cu",
            "kernels/quant_utils.cu",
            "kernels/int8_ext.cu",
        ],
        extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"],
        verbose=False,
    )

    # ── Correctness ─────────────────────────────────────────────────────
    print("\n=== Correctness ===")
    attn_out_int8 = int8_ext.int8_attention_forward(
        Q_int8, K_int8, V_int8, float(scale_Q), float(scale_K), float(scale_V))
    mlp_out_int8, mlp_out_scale = int8_ext.int8_mlp_forward(
        x_int8, W1_int8, W2_int8, float(scale_x), float(scale_W1), float(scale_W2))

    # Dequantize for comparison against FP16 reference.
    # Attention: output scale = scale_V (convex combo of V rows).
    # MLP: output scale returned by kernel (dynamically computed).
    attn_out = (attn_out_int8.float() * scale_V).half()
    mlp_out  = (mlp_out_int8.float() * mlp_out_scale).half()

    check_correctness(attn_ref, attn_out, label="int8_attention", mode="int8")
    check_correctness(mlp_ref,  mlp_out,  label="int8_mlp",       mode="int8")

    # ── Benchmark: INT8 Attention ───────────────────────────────────────
    print("\n=== Benchmark: INT8 Attention ===")

    fp16_attn_ms = benchmark(attention_baseline, Q, K, V)

    int8_attn_ms = benchmark(int8_ext.int8_attention_forward,
                             Q_int8, K_int8, V_int8,
                             float(scale_Q), float(scale_K), float(scale_V))

    attn_flops = compute_attention_flops(batch, heads, seq_len, head_dim)
    attn_tops = attn_flops / (int8_attn_ms * 1e-3) / 1e12

    print(f"  FP16 baseline:    {fp16_attn_ms:.3f} ms")
    print(f"  Your INT8 kernel: {int8_attn_ms:.3f} ms  ({fp16_attn_ms / int8_attn_ms:.2f}x vs FP16)")
    print(f"  Your TOPS:        {attn_tops:.1f}  |  Utilization: {attn_tops / A100_INT8_TOPS * 100:.1f}%")

    # ── Benchmark: INT8 MLP ─────────────────────────────────────────────
    print("\n=== Benchmark: INT8 MLP ===")

    # Fair FP16 reference for INT8: benchmark the same values that the INT8
    # kernel sees after quantize/dequantize, not the original unquantized input.
    fp16_mlp_ms = benchmark(mlp_baseline, x_deq, W1_deq, W2_deq)

    # benchmark unpacks tuple automatically; only latency matters here
    int8_mlp_ms = benchmark(lambda: int8_ext.int8_mlp_forward(
        x_int8, W1_int8, W2_int8, float(scale_x), float(scale_W1), float(scale_W2)))

    # cuBLAS INT8 comparison (torch._int_mm, 2D only)
    x_2d_int8 = x_int8.view(-1, d_model)

    def cublas_int8_gemm():
        torch._int_mm(x_2d_int8, W1_int8)

    cublas_int8_ms = benchmark(cublas_int8_gemm)

    mlp_flops = compute_mlp_flops(batch, seq_len, d_model, d_ff)
    mlp_tops = mlp_flops / (int8_mlp_ms * 1e-3) / 1e12

    print(f"  FP16 baseline:     {fp16_mlp_ms:.3f} ms")
    print(f"  Your INT8 kernel:  {int8_mlp_ms:.3f} ms  ({fp16_mlp_ms / int8_mlp_ms:.2f}x vs FP16)")
    print(f"  cuBLAS INT8 GEMM:  {cublas_int8_ms:.3f} ms  (single GEMM only)")
    print(f"  Your TOPS:         {mlp_tops:.1f}  |  Utilization: {mlp_tops / A100_INT8_TOPS * 100:.1f}%")


if __name__ == "__main__":
    main()
