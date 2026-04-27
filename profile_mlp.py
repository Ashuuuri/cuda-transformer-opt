"""profile_mlp.py — Profile the fused MLP kernel using torch.profiler.

No ncu, no nsys, no shell scripts needed.
Just run:  python profile_mlp.py

Prints a table of GPU kernel times directly in the terminal.
"""

import os
import sys
import torch
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from baseline import mlp_baseline, check_cuda


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
    dtype  = torch.float16
    batch, seq_len, d_model, d_ff = 2, 512, 512, 2048

    torch.manual_seed(42)
    x  = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff,           device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff,    d_model,         device=device, dtype=dtype) * 0.02

    print("Compiling kernel...")
    mlp_ext = load_mlp_ext()
    print("Done.\n")

    # Warmup
    for _ in range(10):
        mlp_ext.mlp_forward(x, W1, W2)
        mlp_baseline(x, W1, W2)
    torch.cuda.synchronize()

    # ── Profile your fused kernel ────────────────────────────────────────
    print("=" * 60)
    print("Fused kernel profile")
    print("=" * 60)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(20):
            with record_function("mlp_fused"):
                mlp_ext.mlp_forward(x, W1, W2)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10
    ))

    # ── Profile naive baseline ───────────────────────────────────────────
    print("=" * 60)
    print("Naive PyTorch baseline profile")
    print("=" * 60)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(20):
            with record_function("mlp_naive"):
                mlp_baseline(x, W1, W2)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10
    ))


if __name__ == "__main__":
    main()
