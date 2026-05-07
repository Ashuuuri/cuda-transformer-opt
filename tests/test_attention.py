"""Test harness for FP16 attention kernel — Jonathan.
Tuned tile sizes (TILE_Q=64, TILE_KV=128/64, 4 warps) by Heling.

Usage:
    python tests/test_attention.py              # full test (correctness + benchmark)
    python tests/test_attention.py --quick      # correctness only, all head_dims
    python tests/test_attention.py --head-dim 256  # test specific head_dim only

Runs correctness check against PyTorch baseline, then benchmarks against
naive baseline and FlashAttention-2 (PyTorch built-in).
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load as _load

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline import attention_baseline, check_cuda
from correctness import check_correctness
from benchmark import benchmark

_root = os.path.join(os.path.dirname(__file__), "..")
print("Compiling FP16 attention kernel ...")
_attn_ext = _load(
    name="attention_ext",
    sources=[
        os.path.join(_root, "kernels", "attention.cu"),
        os.path.join(_root, "kernels", "attention_ext.cu"),
    ],
    extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O2"],
    verbose=False,
)
print("Done.\n")

def run_kernel(Q, K, V):
    return _attn_ext.attention_forward(Q, K, V)

# ── A100 FP16 Tensor Core peak ──────────────────────────────────────────
A100_FP16_TFLOPS = 312.0


def compute_attention_flops(batch, heads, seq_len, head_dim):
    """FLOPs for scaled dot-product attention: two batched matmuls + softmax."""
    qk_flops = 2 * batch * heads * seq_len * head_dim * seq_len
    av_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    return qk_flops + av_flops


# ── Multi-config correctness (matches INT8 test configs) ─────────────────
ATTN_CONFIGS = [
    # (batch, heads, seq_len, head_dim, description)
    (2, 8, 512,  64,  "head_dim=64  (TILE_KV=64, double-buf)"),
    (2, 8, 512,  128, "head_dim=128 (TILE_KV=32, double-buf)"),
    (2, 8, 512,  256, "head_dim=256 (TILE_KV=32, double-buf)"),
    (2, 8, 1024, 64,  "head_dim=64  seq=1024 (TILE_KV=64)"),
    (2, 8, 1024, 128, "head_dim=128 seq=1024 (TILE_KV=32)"),
    (2, 8, 1024, 256, "head_dim=256 seq=1024 (TILE_KV=32)"),
]


def test_attention_correctness(configs=None):
    """Run attention correctness across multiple configs. Returns (n_pass, n_total)."""
    if configs is None:
        configs = ATTN_CONFIGS

    device = "cuda"
    dtype = torch.float16
    n_pass = 0

    for batch, heads, seq_len, head_dim, desc in configs:
        torch.manual_seed(42)
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

        ref = attention_baseline(Q, K, V)
        out = run_kernel(Q, K, V)

        passed = check_correctness(ref, out, label=f"fp16_attn [{desc}]", mode="fp16")
        if passed:
            n_pass += 1

    return n_pass, len(configs)


def main():
    parser = argparse.ArgumentParser(description="FP16 attention correctness & benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Correctness only (skip benchmark)")
    parser.add_argument("--head-dim", type=int, default=None,
                        help="Test only this head_dim (64, 128, or 256)")
    args = parser.parse_args()

    check_cuda()

    # ── Correctness ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  CORRECTNESS: FP16 Attention (multi head_dim)")
    print("=" * 60)

    if args.head_dim:
        configs = [(b, h, s, hd, desc) for b, h, s, hd, desc in ATTN_CONFIGS
                   if hd == args.head_dim]
        if not configs:
            print(f"No configs with head_dim={args.head_dim}")
            sys.exit(1)
    else:
        configs = ATTN_CONFIGS

    n_pass, n_total = test_attention_correctness(configs)
    print(f"\nAttention: {n_pass}/{n_total} configs passed")

    all_pass = (n_pass == n_total)
    print("\n" + "=" * 60)
    if all_pass:
        print("  ALL CORRECTNESS CHECKS PASSED")
    else:
        print(f"  {n_total - n_pass} CONFIG(S) FAILED")
    print("=" * 60)

    if args.quick:
        sys.exit(0 if all_pass else 1)

    if not all_pass:
        print("\nSkipping benchmark due to correctness failures.")
        sys.exit(1)

    # ── Benchmark: head_dim=64 (TILE_KV=128) ────────────────────────────
    device = "cuda"
    dtype = torch.float16
    batch, heads, seq_len, head_dim = 2, 8, 512, 64

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    print("\n=== Benchmark: FP16 Attention (head_dim=64, TILE_KV=64 double-buf) ===")
    naive_ms = benchmark(attention_baseline, Q, K, V)
    kernel_ms = benchmark(run_kernel, Q, K, V)
    flash_ms = benchmark(F.scaled_dot_product_attention, Q, K, V)

    flops = compute_attention_flops(batch, heads, seq_len, head_dim)
    kernel_tflops = flops / (kernel_ms * 1e-3) / 1e12
    utilization = kernel_tflops / A100_FP16_TFLOPS * 100

    print(f"  Naive baseline:  {naive_ms:.3f} ms")
    print(f"  Your kernel:     {kernel_ms:.3f} ms  ({naive_ms / kernel_ms:.2f}x vs naive)")
    print(f"  FlashAttn-2:     {flash_ms:.3f} ms  ({naive_ms / flash_ms:.2f}x vs naive)")
    print(f"  Your TFLOPS:     {kernel_tflops:.1f}  |  Utilization: {utilization:.1f}%")

    # ── Benchmark: head_dim=256 (TILE_KV=64) ────────────────────────────
    head_dim_256 = 256
    print("\n=== Benchmark: FP16 Attention (head_dim=256, TILE_KV=32 double-buf) ===")
    torch.manual_seed(42)
    Q2 = torch.randn(batch, heads, seq_len, head_dim_256, device=device, dtype=dtype)
    K2 = torch.randn(batch, heads, seq_len, head_dim_256, device=device, dtype=dtype)
    V2 = torch.randn(batch, heads, seq_len, head_dim_256, device=device, dtype=dtype)

    naive256_ms = benchmark(attention_baseline, Q2, K2, V2)
    kernel256_ms = benchmark(run_kernel, Q2, K2, V2)
    flash256_ms = benchmark(F.scaled_dot_product_attention, Q2, K2, V2)

    flops256 = compute_attention_flops(batch, heads, seq_len, head_dim_256)
    tflops256 = flops256 / (kernel256_ms * 1e-3) / 1e12
    util256 = tflops256 / A100_FP16_TFLOPS * 100

    print(f"  Naive baseline:  {naive256_ms:.3f} ms")
    print(f"  Your kernel:     {kernel256_ms:.3f} ms  ({naive256_ms / kernel256_ms:.2f}x vs naive)")
    print(f"  FlashAttn-2:     {flash256_ms:.3f} ms  ({naive256_ms / flash256_ms:.2f}x vs naive)")
    print(f"  Your TFLOPS:     {tflops256:.1f}  |  Utilization: {util256:.1f}%")


if __name__ == "__main__":
    main()
