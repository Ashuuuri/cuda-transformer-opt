"""sweep.py — Parameter sweep for MLP kernel benchmarking.

Sweeps:
  seq_len  : 512, 1024, 2048, 4096
  d_model  : 512, 1024, 2048   (d_ff = 4 × d_model, following standard ratio)
  batch    : fixed at 8
  heads    : fixed at 8 (not used for MLP but kept for consistency)

For each configuration, measures:
  - Latency (ms): fused kernel, naive PyTorch baseline, cuBLAS GEMMs
  - TFLOPS: derived from latency
  - Memory bandwidth utilization (analytical estimate)
  - Speedup vs naive

Results are saved to results/mlp_sweep.csv and plots to results/figures/.

Usage:
    python sweep.py
"""

import os
import sys
import csv
import time
import torch
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from baseline import mlp_baseline, check_cuda
from benchmark import benchmark

# ── A100 constants ──────────────────────────────────────────────────────
A100_FP16_TFLOPS  = 312.0          # Tensor Core FP16 peak, TFLOPS
A100_HBM_BW_TBps  = 2.0            # HBM peak bandwidth, TB/s  (2000 GB/s)

# ── Sweep grid ──────────────────────────────────────────────────────────
SEQ_LENS  = [512, 1024, 2048, 4096]
D_MODELS  = [512, 1024, 2048]
BATCH     = 8

# ── Output directory ────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "mlp_sweep.csv")


# ── Load CUDA extension ─────────────────────────────────────────────────
def load_mlp_ext():
    kernel_dir = os.path.join(ROOT, "kernels")
    return load(
        name="mlp_ext",
        sources=[
            os.path.join(kernel_dir, "mlp.cu"),
            os.path.join(kernel_dir, "mlp_ext.cu"),
        ],
        extra_cuda_cflags=["-O2", "--std=c++17", "-arch=sm_80"],
        verbose=False,
    )


# ── FLOPs & bandwidth helpers ────────────────────────────────────────────
def mlp_flops(batch, seq_len, d_model, d_ff):
    """Multiply-add FLOPs for two GEMMs."""
    T = batch * seq_len
    return 2 * T * d_model * d_ff + 2 * T * d_ff * d_model


def mlp_hbm_bytes_fused(batch, seq_len, d_model, d_ff):
    """
    Analytical HBM bytes for the fused kernel (FP16 = 2 bytes/element).
    Fused kernel avoids writing the hidden buffer to HBM.
      Reads : x (T×d_model) + W1 (d_model×d_ff) + W2 (d_ff×d_model)
      Writes: out (T×d_model)
    """
    T     = batch * seq_len
    reads  = (T * d_model + d_model * d_ff + d_ff * d_model) * 2
    writes = T * d_model * 2
    return reads + writes


def mlp_hbm_bytes_unfused(batch, seq_len, d_model, d_ff):
    """
    Analytical HBM bytes for naive unfused execution.
    Extra: hidden (T×d_ff) written after GEMM1, read before GEMM2.
    """
    T      = batch * seq_len
    base   = mlp_hbm_bytes_fused(batch, seq_len, d_model, d_ff)
    extra  = 2 * T * d_ff * 2    # write + read hidden
    return base + extra


# ── Single configuration benchmark ──────────────────────────────────────
def run_one(mlp_ext, batch, seq_len, d_model, d_ff):
    device = "cuda"
    dtype  = torch.float16

    torch.manual_seed(42)
    x  = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff,            device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff,    d_model,          device=device, dtype=dtype) * 0.02

    # ── Kernel latencies ──────────────────────────────────────────────
    naive_ms  = benchmark(mlp_baseline, x, W1, W2)
    kernel_ms = benchmark(mlp_ext.mlp_forward, x, W1, W2)

    x_2d = x.view(-1, d_model)
    def cublas_gemms():
        h = torch.mm(x_2d, W1)
        torch.mm(h, W2)
    cublas_ms = benchmark(cublas_gemms)

    # ── Derived metrics ───────────────────────────────────────────────
    flops   = mlp_flops(batch, seq_len, d_model, d_ff)

    def tflops(ms):
        return flops / (ms * 1e-3) / 1e12

    def bw_util(ms, hbm_bytes):
        # Achieved BW (TB/s) / A100 peak BW (TB/s) × 100
        achieved = hbm_bytes / (ms * 1e-3) / 1e12
        return achieved / A100_HBM_BW_TBps * 100

    kernel_tflops  = tflops(kernel_ms)
    naive_tflops   = tflops(naive_ms)
    cublas_tflops  = tflops(cublas_ms)

    bw_kernel = bw_util(kernel_ms,  mlp_hbm_bytes_fused(batch, seq_len, d_model, d_ff))
    bw_naive  = bw_util(naive_ms,   mlp_hbm_bytes_unfused(batch, seq_len, d_model, d_ff))

    return {
        "batch":          batch,
        "seq_len":        seq_len,
        "d_model":        d_model,
        "d_ff":           d_ff,
        # latencies
        "kernel_ms":      kernel_ms,
        "naive_ms":       naive_ms,
        "cublas_ms":      cublas_ms,
        # TFLOPS
        "kernel_tflops":  kernel_tflops,
        "naive_tflops":   naive_tflops,
        "cublas_tflops":  cublas_tflops,
        # utilisation
        "kernel_a100_util_pct": kernel_tflops / A100_FP16_TFLOPS * 100,
        "naive_a100_util_pct":  naive_tflops  / A100_FP16_TFLOPS * 100,
        # HBM bandwidth utilisation
        "kernel_bw_util_pct":  bw_kernel,
        "naive_bw_util_pct":   bw_naive,
        # speedups
        "speedup_vs_naive":    naive_ms  / kernel_ms,
        "speedup_vs_cublas":   cublas_ms / kernel_ms,
    }


# ── CSV writer ───────────────────────────────────────────────────────────
FIELDNAMES = [
    "batch", "seq_len", "d_model", "d_ff",
    "kernel_ms", "naive_ms", "cublas_ms",
    "kernel_tflops", "naive_tflops", "cublas_tflops",
    "kernel_a100_util_pct", "naive_a100_util_pct",
    "kernel_bw_util_pct", "naive_bw_util_pct",
    "speedup_vs_naive", "speedup_vs_cublas",
]

def save_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[CSV] Saved {len(rows)} rows → {CSV_PATH}")


# ── Plots ─────────────────────────────────────────────────────────────────
def make_plots(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not found — skipping plots. pip install matplotlib")
        return

    import collections

    # ── Helper: group rows ────────────────────────────────────────────
    def group_by(rows, key):
        d = collections.defaultdict(list)
        for r in rows:
            d[r[key]].append(r)
        return d

    colors = {
        "Fused kernel":  "#2563EB",
        "Naive PyTorch": "#DC2626",
        "cuBLAS GEMMs":  "#16A34A",
    }
    markers = {"Fused kernel": "o", "Naive PyTorch": "s", "cuBLAS GEMMs": "^"}

    # ── Figure 1: Latency vs seq_len (one subplot per d_model) ────────
    fig, axes = plt.subplots(1, len(D_MODELS), figsize=(5 * len(D_MODELS), 4),
                             sharey=False)
    if len(D_MODELS) == 1:
        axes = [axes]

    by_dmodel = group_by(rows, "d_model")

    for ax, d_model in zip(axes, D_MODELS):
        subset = sorted(by_dmodel[d_model], key=lambda r: r["seq_len"])
        seqs   = [r["seq_len"]   for r in subset]
        for label, col in [("Fused kernel",  "kernel_ms"),
                            ("Naive PyTorch", "naive_ms"),
                            ("cuBLAS GEMMs",  "cublas_ms")]:
            vals = [r[col] for r in subset]
            ax.plot(seqs, vals, marker=markers[label], label=label,
                    color=colors[label], linewidth=2, markersize=6)

        ax.set_title(f"d_model = {d_model}  (d_ff = {d_model * 4})", fontsize=11)
        ax.set_xlabel("Sequence length", fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=10)
        ax.set_xticks(seqs)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("MLP Fused Kernel: Latency vs Sequence Length\n"
                 f"batch={BATCH}, FP16, A100", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "mlp_latency_vs_seqlen.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")

    # ── Figure 2: Bar chart — latency per d_model at largest seq_len ──
    max_seq = max(SEQ_LENS)
    by_dmodel_maxseq = {
        dm: next(r for r in rows if r["d_model"] == dm and r["seq_len"] == max_seq)
        for dm in D_MODELS
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos    = range(len(D_MODELS))
    bar_w    = 0.25
    methods  = [("Fused kernel",  "kernel_ms"),
                ("Naive PyTorch", "naive_ms"),
                ("cuBLAS GEMMs",  "cublas_ms")]

    for i, (label, col) in enumerate(methods):
        vals = [by_dmodel_maxseq[dm][col] for dm in D_MODELS]
        offsets = [x + (i - 1) * bar_w for x in x_pos]
        bars = ax.bar(offsets, vals, width=bar_w, label=label,
                      color=colors[label], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([f"d_model={dm}\nd_ff={dm*4}" for dm in D_MODELS])
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_title(f"MLP Kernel Latency by Hidden Size  "
                 f"(seq_len={max_seq}, batch={BATCH}, FP16, A100)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "mlp_latency_bar_by_dmodel.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")

    # ── Figure 3: TFLOPS vs seq_len (kernel only, one line per d_model)
    fig, ax = plt.subplots(figsize=(6, 4))
    for d_model in D_MODELS:
        subset = sorted(by_dmodel[d_model], key=lambda r: r["seq_len"])
        seqs   = [r["seq_len"]        for r in subset]
        tflops = [r["kernel_tflops"]  for r in subset]
        ax.plot(seqs, tflops, marker="o", label=f"d_model={d_model}",
                linewidth=2, markersize=6)

    ax.axhline(A100_FP16_TFLOPS, color="gray", linestyle="--",
               linewidth=1.2, label=f"A100 FP16 peak ({A100_FP16_TFLOPS} TFLOPS)")
    ax.set_xlabel("Sequence length", fontsize=10)
    ax.set_ylabel("TFLOPS", fontsize=10)
    ax.set_xticks(SEQ_LENS)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_title(f"Fused MLP Kernel Throughput  (batch={BATCH}, FP16)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "mlp_tflops_vs_seqlen.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")

    # ── Figure 4: Speedup vs naive, line plot ─────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for d_model in D_MODELS:
        subset  = sorted(by_dmodel[d_model], key=lambda r: r["seq_len"])
        seqs    = [r["seq_len"]         for r in subset]
        speedup = [r["speedup_vs_naive"] for r in subset]
        ax.plot(seqs, speedup, marker="o", label=f"d_model={d_model}",
                linewidth=2, markersize=6)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="Baseline (1×)")
    ax.set_xlabel("Sequence length", fontsize=10)
    ax.set_ylabel("Speedup over naive PyTorch (×)", fontsize=10)
    ax.set_xticks(SEQ_LENS)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_title(f"Fused MLP Speedup vs Naive  (batch={BATCH}, FP16)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "mlp_speedup_vs_seqlen.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")

    # ── Figure 5: HBM bandwidth utilisation bar chart ─────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    methods_bw = [("Fused kernel",  "kernel_bw_util_pct"),
                  ("Naive PyTorch", "naive_bw_util_pct")]

    for i, (label, col) in enumerate(methods_bw):
        vals    = [by_dmodel_maxseq[dm][col] for dm in D_MODELS]
        offsets = [x + (i - 0.5) * 0.35 for x in range(len(D_MODELS))]
        bars    = ax.bar(offsets, vals, width=0.32, label=label,
                         color=colors[label], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(range(len(D_MODELS))))
    ax.set_xticklabels([f"d_model={dm}" for dm in D_MODELS])
    ax.set_ylabel("HBM Bandwidth Utilisation (%)", fontsize=10)
    ax.set_title(f"HBM Bandwidth Utilisation  "
                 f"(seq_len={max_seq}, batch={BATCH}, A100 peak = 2 TB/s)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "mlp_hbm_bw_utilisation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    check_cuda()

    print("Compiling CUDA kernel …")
    mlp_ext = load_mlp_ext()
    print("Done.\n")

    total = len(SEQ_LENS) * len(D_MODELS)
    rows  = []
    done  = 0

    for d_model in D_MODELS:
        d_ff = d_model * 4          # standard FFN ratio
        for seq_len in SEQ_LENS:
            done += 1
            tag = f"[{done:2d}/{total}] batch={BATCH}  seq={seq_len:4d}  " \
                  f"d_model={d_model}  d_ff={d_ff}"
            print(tag, end="  ", flush=True)
            t0  = time.perf_counter()
            row = run_one(mlp_ext, BATCH, seq_len, d_model, d_ff)
            elapsed = time.perf_counter() - t0
            print(f"kernel={row['kernel_ms']:.2f} ms  "
                  f"naive={row['naive_ms']:.2f} ms  "
                  f"speedup={row['speedup_vs_naive']:.2f}×  "
                  f"({elapsed:.1f}s)")
            rows.append(row)

    save_csv(rows)
    make_plots(rows)
    print("\nSweep complete.")


if __name__ == "__main__":
    main()
