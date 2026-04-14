"""Test harness for FP16 attention kernel — Jonathan.

Usage:
    python tests/test_attention.py

Runs correctness check against PyTorch baseline, then benchmarks against
naive baseline and FlashAttention-2 (PyTorch built-in).
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline import attention_baseline, check_cuda
from correctness import check_correctness
from benchmark import benchmark

# ── A100 FP16 Tensor Core peak ──────────────────────────────────────────
A100_FP16_TFLOPS = 312.0


def compute_attention_flops(batch, heads, seq_len, head_dim):
    """FLOPs for scaled dot-product attention: two batched matmuls + softmax."""
    # Q @ K^T: 2 * B*H * S * D * S
    qk_flops = 2 * batch * heads * seq_len * head_dim * seq_len
    # attn @ V: 2 * B*H * S * S * D
    av_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    # softmax is small relative to matmuls
    return qk_flops + av_flops


def main():
    check_cuda()

    device = "cuda"
    dtype = torch.float16
    batch, heads, seq_len, head_dim = 2, 8, 512, 64

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # ── Reference ───────────────────────────────────────────────────────
    ref = attention_baseline(Q, K, V)

    # ── Your kernel ─────────────────────────────────────────────────────
    # TODO: Replace this with your CUDA kernel once implemented.
    #   from torch.utils.cpp_extension import load
    #   attn_module = load(name="attention", sources=["kernels/attention.cu"], verbose=True)
    #   out = attn_module.attention_forward(Q, K, V)
    print("\n=== Correctness ===")
    print("[SKIP] attention kernel not yet implemented — using baseline as placeholder")
    out = attention_baseline(Q, K, V)  # placeholder
    check_correctness(ref, out, label="fp16_attention", mode="fp16")

    # ── Benchmark ───────────────────────────────────────────────────────
    print("\n=== Benchmark ===")

    naive_ms = benchmark(attention_baseline, Q, K, V)

    # Your kernel (placeholder for now)
    kernel_ms = benchmark(attention_baseline, Q, K, V)  # TODO: replace

    # FlashAttention-2 (PyTorch built-in)
    flash_ms = benchmark(F.scaled_dot_product_attention, Q, K, V)

    # TFLOPS calculation
    flops = compute_attention_flops(batch, heads, seq_len, head_dim)
    kernel_tflops = flops / (kernel_ms * 1e-3) / 1e12
    utilization = kernel_tflops / A100_FP16_TFLOPS * 100

    print(f"  Naive baseline:  {naive_ms:.3f} ms")
    print(f"  Your kernel:     {kernel_ms:.3f} ms  ({naive_ms / kernel_ms:.2f}x vs naive)")
    print(f"  FlashAttn-2:     {flash_ms:.3f} ms  ({naive_ms / flash_ms:.2f}x vs naive)")
    print(f"  Your TFLOPS:     {kernel_tflops:.1f}  |  Utilization: {utilization:.1f}%")


if __name__ == "__main__":
    main()
