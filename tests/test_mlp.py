"""Test harness for FP16 MLP kernel — Shengjing.

Usage:
    python tests/test_mlp.py

Runs correctness check against PyTorch baseline, then benchmarks against
naive baseline and cuBLAS (torch.mm).
"""

import os
import sys
import torch
from torch.utils.cpp_extension import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from baseline import mlp_baseline, check_cuda
from correctness import check_correctness
from benchmark import benchmark

# ── A100 FP16 Tensor Core peak ──────────────────────────────────────────
A100_FP16_TFLOPS = 312.0


def compute_mlp_flops(batch, seq_len, d_model, d_ff):
    """FLOPs for two-layer MLP: two GEMMs (GELU is negligible)."""
    tokens = batch * seq_len
    # x @ W1: 2 * tokens * d_model * d_ff
    gemm1 = 2 * tokens * d_model * d_ff
    # hidden @ W2: 2 * tokens * d_ff * d_model
    gemm2 = 2 * tokens * d_ff * d_model
    return gemm1 + gemm2


def load_mlp_ext():
    kernel_dir = os.path.join(ROOT, "kernels")
    return load(
        name="mlp_ext",
        sources=[
            os.path.join(kernel_dir, "mlp.cu"),
            os.path.join(kernel_dir, "mlp_ext.cu"),
        ],
        extra_cuda_cflags=[
            "-O3", "--std=c++17", "-arch=sm_80", "--use_fast_math",
            "-DMLP_STAGE_K=32",
        ],
        verbose=False,
    )


def main():
    check_cuda()

    device = "cuda"
    dtype = torch.float16
    batch, seq_len, d_model, d_ff = 2, 512, 512, 2048

    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.02

    # ── Reference ───────────────────────────────────────────────────────
    ref = mlp_baseline(x, W1, W2)

    print("Compiling MLP WMMA extension...")
    mlp_ext = load_mlp_ext()
    print("Done.")

    # ── Your kernel ─────────────────────────────────────────────────────
    print("\n=== Correctness ===")
    out = mlp_ext.mlp_forward(x, W1, W2)
    check_correctness(ref, out, label="fp16_mlp", mode="fp16")

    # ── Benchmark ───────────────────────────────────────────────────────
    print("\n=== Benchmark ===")

    naive_ms = benchmark(mlp_baseline, x, W1, W2)

    kernel_ms = benchmark(mlp_ext.mlp_forward, x, W1, W2)

    # cuBLAS comparison: just the two GEMMs (no GELU), as upper bound
    x_2d = x.view(-1, d_model)

    def cublas_gemms():
        h = torch.mm(x_2d, W1)
        torch.mm(h, W2)

    cublas_ms = benchmark(cublas_gemms)

    # TFLOPS calculation
    flops = compute_mlp_flops(batch, seq_len, d_model, d_ff)
    kernel_tflops = flops / (kernel_ms * 1e-3) / 1e12
    utilization = kernel_tflops / A100_FP16_TFLOPS * 100

    print(f"  Naive baseline:  {naive_ms:.3f} ms")
    print(f"  Your kernel:     {kernel_ms:.3f} ms  ({naive_ms / kernel_ms:.2f}x vs naive)")
    print(f"  cuBLAS (GEMMs):  {cublas_ms:.3f} ms  ({naive_ms / cublas_ms:.2f}x vs naive)")
    print(f"  Your TFLOPS:     {kernel_tflops:.1f}  |  Utilization: {utilization:.1f}%")


if __name__ == "__main__":
    main()
