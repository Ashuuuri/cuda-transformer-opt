"""sweep.py — Unified parameter sweep for all three CUDA kernels.

Sweeps (from proposal §4):
  seq_len  : 512, 1024, 2048, 4096
  d_model  : 512, 1024, 2048   (d_ff = 4 × d_model)
  head_dim : fixed at 64       (standard; heads fixed at 8)
  batch    : fixed at 8

Usage:
    python sweep.py --kernel mlp        # Shengjing
    python sweep.py --kernel attention  # Jonathan
    python sweep.py --kernel int8       # Heling

Each run produces:
    results/<kernel>_sweep.csv
    results/figures/<kernel>_latency_vs_seqlen.png
    results/figures/<kernel>_latency_bar_by_dmodel.png
    results/figures/<kernel>_tflops_vs_seqlen.png
    results/figures/<kernel>_speedup_vs_seqlen.png
    results/figures/<kernel>_hbm_bw_utilisation.png

All plots use the same colour scheme and layout so every kernel's
results look identical in style on the poster.
"""

import argparse
import collections
import csv
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from baseline import attention_baseline, mlp_baseline, check_cuda
from benchmark import benchmark

# ══════════════════════════════════════════════════════════════════════════
#  A100 hardware constants
# ══════════════════════════════════════════════════════════════════════════
A100_FP16_TFLOPS = 312.0   # Tensor Core FP16 peak (TFLOPS)
A100_INT8_TOPS   = 624.0   # Tensor Core INT8 peak (TOPS)
A100_HBM_BW_TBps = 2.0    # HBM bandwidth peak (TB/s)

# ══════════════════════════════════════════════════════════════════════════
#  Sweep grid  (§4 of proposal)
# ══════════════════════════════════════════════════════════════════════════
SEQ_LENS  = [512, 1024, 2048, 4096]
D_MODELS  = [512, 1024, 2048]
BATCH     = 8
HEADS     = 8
HEAD_DIM  = 64   # d_model // heads for attention

# ══════════════════════════════════════════════════════════════════════════
#  Shared plot colours/markers  (uniform across all three kernels)
# ══════════════════════════════════════════════════════════════════════════
COLORS = {
    "Fused kernel":      "#2563EB",
    "Naive PyTorch":     "#DC2626",
    "FP16 WMMA (Sherry)": "#DC2626",
    "FP16 WMMA (Jonathan)": "#DC2626",
    "FP16 WMMA baseline": "#DC2626",
    "cuBLAS / Flash":    "#16A34A",
    "Naive PyTorch (cuBLAS)": "#F59E0B",
}
MARKERS = {
    "Fused kernel":      "o",
    "Naive PyTorch":     "s",
    "FP16 WMMA (Sherry)": "s",
    "FP16 WMMA (Jonathan)": "s",
    "FP16 WMMA baseline": "s",
    "cuBLAS / Flash":    "^",
    "Naive PyTorch (cuBLAS)": "D",
}

# ══════════════════════════════════════════════════════════════════════════
#  Per-kernel loaders
# ══════════════════════════════════════════════════════════════════════════
def load_mlp_ext():
    kdir = os.path.join(ROOT, "kernels")
    return load(
        name="mlp_ext",
        sources=[
            os.path.join(kdir, "mlp.cu"),
            os.path.join(kdir, "mlp_ext.cu"),
        ],
        extra_cuda_cflags=[
            "-O3", "--std=c++17", "-arch=sm_80", "--use_fast_math",
            "-DMLP_STAGE_K=32",
        ],
        verbose=False,
    )


def load_attention_ext():
    kdir = os.path.join(ROOT, "kernels")
    return load(
        name="attention_ext",
        sources=[
            os.path.join(kdir, "attention.cu"),
            os.path.join(kdir, "attention_ext.cu"),
        ],
        extra_cuda_cflags=["-O2", "--std=c++17", "-arch=sm_80"],
        verbose=False,
    )


def load_int8_ext():
    kdir = os.path.join(ROOT, "kernels")
    return load(
        name="int8_ext",
        sources=[
            os.path.join(kdir, "int8_attention.cu"),
            os.path.join(kdir, "int8_mlp.cu"),
            os.path.join(kdir, "quant_utils.cu"),
            os.path.join(kdir, "int8_ext.cu"),
        ],
        extra_cuda_cflags=["-O2", "--std=c++17", "-arch=sm_80"],
        verbose=False,
    )


# ══════════════════════════════════════════════════════════════════════════
#  FLOPs counters
# ══════════════════════════════════════════════════════════════════════════
def mlp_flops(batch, seq_len, d_model, d_ff):
    T = batch * seq_len
    return 2 * T * d_model * d_ff + 2 * T * d_ff * d_model


def attention_flops(batch, heads, seq_len, head_dim):
    # QK^T + softmax(·)V  — two batched matmuls dominate
    return 2 * batch * heads * seq_len * head_dim * seq_len * 2


def int8_mlp_flops(batch, seq_len, d_model, d_ff):
    return mlp_flops(batch, seq_len, d_model, d_ff)  # same op count


# ══════════════════════════════════════════════════════════════════════════
#  Analytical HBM bytes
# ══════════════════════════════════════════════════════════════════════════
def mlp_hbm_fused(batch, seq_len, d_model, d_ff):
    """Fused: hidden never written to HBM."""
    T = batch * seq_len
    reads  = (T * d_model + d_model * d_ff + d_ff * d_model) * 2   # FP16 = 2 B
    writes = T * d_model * 2
    return reads + writes


def mlp_hbm_unfused(batch, seq_len, d_model, d_ff):
    """Unfused: hidden written after GEMM1, read before GEMM2."""
    T     = batch * seq_len
    extra = 2 * T * d_ff * 2
    return mlp_hbm_fused(batch, seq_len, d_model, d_ff) + extra


def attn_hbm_fused(batch, heads, seq_len, head_dim):
    """Fused attention: attn matrix (S×S) stays in SRAM."""
    T  = batch * heads
    qkv = T * seq_len * head_dim * 2 * 3
    out = T * seq_len * head_dim * 2
    return qkv + out


def attn_hbm_unfused(batch, heads, seq_len, head_dim):
    """Unfused: full S×S attention matrix written + read."""
    T     = batch * heads
    extra = 2 * T * seq_len * seq_len * 2   # write + read attn matrix
    return attn_hbm_fused(batch, heads, seq_len, head_dim) + extra


# ══════════════════════════════════════════════════════════════════════════
#  INT8 quantisation helper
# ══════════════════════════════════════════════════════════════════════════
def quantize_to_int8(t):
    """Per-tensor symmetric quantization."""
    t_f   = t.float()
    scale = t_f.abs().max() / 127.0
    t_i8  = (t_f / scale).round().clamp(-128, 127).to(torch.int8)
    return t_i8, scale


def quantize_per_token(t):
    """Per-token symmetric quantization for attention Q/K/V.

    Input:  (batch, heads, seq_len, head_dim)
    Returns:
        t_int8: (batch, heads, seq_len, head_dim) torch.int8
        scales: (batch, heads, seq_len) torch.float32
    """
    t_f = t.float()
    abs_max = t_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = abs_max / 127.0
    t_i8 = (t_f / scales).round().clamp(-128, 127).to(torch.int8)
    scales = scales.squeeze(-1)
    return t_i8, scales


# ══════════════════════════════════════════════════════════════════════════
#  Per-kernel benchmark functions
#  Each returns a dict with the SAME keys so the CSV/plot code is shared.
# ══════════════════════════════════════════════════════════════════════════

# ── MLP ──────────────────────────────────────────────────────────────────
def bench_mlp(ext, batch, seq_len, d_model):
    d_ff   = d_model * 4
    device = "cuda"
    dtype  = torch.float16
    torch.manual_seed(42)

    x  = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff,           device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff,    d_model,         device=device, dtype=dtype) * 0.02

    kernel_ms = benchmark(ext.mlp_forward, x, W1, W2)
    naive_ms  = benchmark(mlp_baseline, x, W1, W2)

    x_2d = x.view(-1, d_model)
    def ref2():
        h = torch.mm(x_2d, W1)
        torch.mm(h, W2)
    ref2_ms = benchmark(ref2)

    flops       = mlp_flops(batch, seq_len, d_model, d_ff)
    peak_tops   = A100_FP16_TFLOPS

    return _make_row(
        batch, seq_len, d_model,
        kernel_ms, naive_ms, ref2_ms,
        flops, peak_tops,
        mlp_hbm_fused(batch, seq_len, d_model, d_ff),
        mlp_hbm_unfused(batch, seq_len, d_model, d_ff),
        ref2_label="cuBLAS GEMMs",
        ref2_is_fused=False,
    )


# ── Attention ─────────────────────────────────────────────────────────────
def bench_attention(ext, batch, seq_len, d_model):
    import torch.nn.functional as F
    heads    = HEADS
    head_dim = d_model // heads
    device   = "cuda"
    dtype    = torch.float16
    torch.manual_seed(42)

    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    kernel_ms = benchmark(ext.attention_forward, Q, K, V)
    naive_ms  = benchmark(attention_baseline, Q, K, V)
    flash_ms  = benchmark(F.scaled_dot_product_attention, Q, K, V)

    flops     = attention_flops(batch, heads, seq_len, head_dim)
    peak_tops = A100_FP16_TFLOPS

    return _make_row(
        batch, seq_len, d_model,
        kernel_ms, naive_ms, flash_ms,
        flops, peak_tops,
        attn_hbm_fused(batch, heads, seq_len, head_dim),
        attn_hbm_unfused(batch, heads, seq_len, head_dim),
        ref2_label="FlashAttn-2",
    )


# ── INT8 Attention ────────────────────────────────────────────────────────
def bench_int8_attn(ext, batch, seq_len, d_model):
    import torch.nn.functional as F
    heads    = HEADS
    head_dim = d_model // heads
    device   = "cuda"
    dtype    = torch.float16
    torch.manual_seed(42)

    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # Per-token quantization for attention
    Q_i8, sQ = quantize_per_token(Q)
    K_i8, sK = quantize_per_token(K)
    V_i8, sV = quantize_per_token(V)
    Q_deq = (Q_i8.float() * sQ.unsqueeze(-1)).half()
    K_deq = (K_i8.float() * sK.unsqueeze(-1)).half()
    V_deq = (V_i8.float() * sV.unsqueeze(-1)).half()

    attn_ext = load_attention_ext()

    sQ_c = sQ.contiguous().cuda()
    sK_c = sK.contiguous().cuda()
    sV_c = sV.contiguous().cuda()

    def run_int8():
        ext.int8_attention_forward(Q_i8, K_i8, V_i8, sQ_c, sK_c, sV_c)

    kernel_ms  = benchmark(run_int8)
    fp16_wm_ms = benchmark(attn_ext.attention_forward, Q_deq, K_deq, V_deq)
    naive_ms   = benchmark(attention_baseline, Q, K, V)
    flash_ms   = benchmark(F.scaled_dot_product_attention, Q_deq, K_deq, V_deq)

    flops     = attention_flops(batch, heads, seq_len, head_dim)
    peak_tops = A100_INT8_TOPS

    hbm_fused_bytes   = attn_hbm_fused(batch, heads, seq_len, head_dim)
    hbm_unfused_bytes = attn_hbm_unfused(batch, heads, seq_len, head_dim)

    row = _make_row(
        batch, seq_len, d_model,
        kernel_ms, fp16_wm_ms, flash_ms,
        flops, peak_tops,
        hbm_fused_bytes, hbm_unfused_bytes,
        naive_label="FP16 WMMA baseline",
        ref2_label="FlashAttn-2",
        naive_is_fused=True,   # FP16 WMMA is fused attention
        ref2_is_fused=True,    # FlashAttn-2 is fused
    )
    # Extra baseline: Naive PyTorch (cuBLAS matmul attention) — unfused
    row["naive_pytorch_ms"] = naive_ms
    achieved = hbm_unfused_bytes / (naive_ms * 1e-3) / 1e12
    row["naive_pytorch_bw_util_pct"] = achieved / A100_HBM_BW_TBps * 100
    return row


# ── INT8 MLP ──────────────────────────────────────────────────────────────
def bench_int8(ext, batch, seq_len, d_model):
    d_ff   = d_model * 4
    device = "cuda"
    dtype  = torch.float16
    torch.manual_seed(42)

    x  = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff,           device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff,    d_model,         device=device, dtype=dtype) * 0.02

    x_i8,  sx  = quantize_to_int8(x)
    W1_i8, sW1 = quantize_to_int8(W1)
    W2_i8, sW2 = quantize_to_int8(W2)
    x_deq  = x_i8.float().mul(sx).half()
    W1_deq = W1_i8.float().mul(sW1).half()
    W2_deq = W2_i8.float().mul(sW2).half()

    mlp_ext = load_mlp_ext()

    def run_int8():
        ext.int8_mlp_forward(x_i8, W1_i8, W2_i8, sx, sW1, sW2)

    kernel_ms  = benchmark(run_int8)
    fp16_wm_ms = benchmark(mlp_ext.mlp_forward, x_deq, W1_deq, W2_deq)

    x_2d = x_i8.view(-1, d_model)
    def cublas_int8():
        torch._int_mm(x_2d, W1_i8)
    ref2_ms = benchmark(cublas_int8)

    flops     = int8_mlp_flops(batch, seq_len, d_model, d_ff)
    peak_tops = A100_INT8_TOPS

    return _make_row(
        batch, seq_len, d_model,
        kernel_ms, fp16_wm_ms, ref2_ms,
        flops, peak_tops,
        mlp_hbm_fused(batch, seq_len, d_model, d_ff),
        mlp_hbm_unfused(batch, seq_len, d_model, d_ff),
        naive_label="FP16 WMMA baseline",
        ref2_label="cuBLAS INT8",
        naive_is_fused=True,   # FP16 WMMA is fused MLP
        ref2_is_fused=False,   # cuBLAS INT8 is unfused
    )


# ══════════════════════════════════════════════════════════════════════════
#  Shared row builder  (all bench_* functions return this dict schema)
# ══════════════════════════════════════════════════════════════════════════
def _make_row(
    batch, seq_len, d_model,
    kernel_ms, naive_ms, ref2_ms,
    flops, peak_tops,
    hbm_fused, hbm_unfused,
    ref2_label,
    naive_label="Naive PyTorch",
    naive_is_fused=False,
    ref2_is_fused=True,
):
    def to_tops(ms):
        return flops / (ms * 1e-3) / 1e12

    def bw_util(ms, hbm_bytes):
        achieved = hbm_bytes / (ms * 1e-3) / 1e12
        return achieved / A100_HBM_BW_TBps * 100

    kernel_tops = to_tops(kernel_ms)
    naive_tops  = to_tops(naive_ms)

    return {
        "batch":               batch,
        "seq_len":             seq_len,
        "d_model":             d_model,
        # latencies
        "kernel_ms":           kernel_ms,
        "naive_ms":            naive_ms,
        "ref2_ms":             ref2_ms,
        "naive_label":         naive_label,
        "ref2_label":          ref2_label,
        # throughput
        "kernel_tops":         kernel_tops,
        "naive_tops":          naive_tops,
        "ref2_tops":           to_tops(ref2_ms),
        # A100 compute utilisation
        "kernel_util_pct":     kernel_tops / peak_tops * 100,
        "naive_util_pct":      naive_tops  / peak_tops * 100,
        # HBM bandwidth utilisation (analytical)
        "kernel_bw_util_pct":  bw_util(kernel_ms, hbm_fused),
        "naive_bw_util_pct":   bw_util(naive_ms, hbm_fused if naive_is_fused else hbm_unfused),
        "ref2_bw_util_pct":    bw_util(ref2_ms, hbm_fused if ref2_is_fused else hbm_unfused),
        # speedups
        "speedup_vs_naive":    naive_ms  / kernel_ms,
        "speedup_vs_ref2":     ref2_ms   / kernel_ms,
    }


# ══════════════════════════════════════════════════════════════════════════
#  CSV
# ══════════════════════════════════════════════════════════════════════════
FIELDNAMES = [
    "batch", "seq_len", "d_model",
    "kernel_ms", "naive_ms", "ref2_ms", "naive_label", "ref2_label",
    "naive_pytorch_ms",
    "kernel_tops", "naive_tops", "ref2_tops",
    "kernel_util_pct", "naive_util_pct",
    "kernel_bw_util_pct", "naive_bw_util_pct", "ref2_bw_util_pct",
    "naive_pytorch_bw_util_pct",
    "speedup_vs_naive", "speedup_vs_ref2",
]

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[CSV] {len(rows)} rows → {path}")


# ══════════════════════════════════════════════════════════════════════════
#  Plots  (uniform style for all kernels)
# ══════════════════════════════════════════════════════════════════════════
def make_plots(rows, kernel_name, figures_dir, peak_tops):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not found — skipping plots.")
        return

    naive_label = rows[0].get("naive_label", "Naive PyTorch")
    ref2_label  = rows[0]["ref2_label"]
    KNAME_DISPLAY = {
        "int8_attn": "INT8 Attention Kernel",
        "int8":      "INT8 MLP Kernel",
        "attention": "FP16 Attention Kernel",
        "mlp":       "FP16 MLP Kernel",
    }
    kname = KNAME_DISPLAY.get(kernel_name, kernel_name.upper())
    kernel_label = "INT8 kernel" if kernel_name.startswith("int8") else "Fused kernel"

    # Map our internal keys to display names for the legend
    method_cols = [
        (kernel_label, "kernel_ms"),
        (naive_label,  "naive_ms"),
        (ref2_label,   "ref2_ms"),
    ]
    # If Naive PyTorch (cuBLAS) data is present, add as 4th line
    has_naive_pytorch = "naive_pytorch_ms" in rows[0] and rows[0]["naive_pytorch_ms"] is not None
    if has_naive_pytorch:
        method_cols.append(("Naive PyTorch (cuBLAS)", "naive_pytorch_ms"))
    # Reuse shared colours; map labels to colour slots
    col_map = {
        kernel_label: COLORS["Fused kernel"],
        naive_label:  COLORS.get(naive_label, COLORS["Naive PyTorch"]),
        ref2_label:   COLORS["cuBLAS / Flash"],
        "Naive PyTorch (cuBLAS)": COLORS["Naive PyTorch (cuBLAS)"],
    }
    mrk_map = {
        kernel_label: MARKERS["Fused kernel"],
        naive_label:  MARKERS.get(naive_label, MARKERS["Naive PyTorch"]),
        ref2_label:   MARKERS["cuBLAS / Flash"],
        "Naive PyTorch (cuBLAS)": MARKERS["Naive PyTorch (cuBLAS)"],
    }

    def group_by(rows, key):
        d = collections.defaultdict(list)
        for r in rows:
            d[r[key]].append(r)
        return d

    by_dm   = group_by(rows, "d_model")
    max_seq = max(SEQ_LENS)

    # ── Fig 1: Latency vs seq_len (one subplot per d_model) ───────────
    fig, axes = plt.subplots(1, len(D_MODELS),
                             figsize=(5 * len(D_MODELS), 4), sharey=False)
    if len(D_MODELS) == 1:
        axes = [axes]

    for ax, dm in zip(axes, D_MODELS):
        sub  = sorted(by_dm[dm], key=lambda r: r["seq_len"])
        seqs = [r["seq_len"] for r in sub]
        for lbl, col in method_cols:
            ax.plot(seqs, [r[col] for r in sub],
                    marker=mrk_map[lbl], color=col_map[lbl],
                    label=lbl, linewidth=2, markersize=6)
        ax.set_title(f"d_model={dm}  d_ff={dm*4}", fontsize=11)
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Latency (ms)")
        ax.set_xticks(seqs)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{kname} Kernel: Latency vs Sequence Length  "
                 f"(batch={BATCH}, FP16, A100)", fontsize=13, y=1.02)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"{kernel_name}_latency_vs_seqlen.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")

    # ── Fig 2: Bar chart at largest seq_len ───────────────────────────
    maxseq_rows = {
        dm: next(r for r in rows
                 if r["d_model"] == dm and r["seq_len"] == max_seq)
        for dm in D_MODELS
    }
    n_methods = len(method_cols)
    fig, ax = plt.subplots(figsize=(max(8, 3 * len(D_MODELS) + 2), 4.5))
    bw = min(0.25, 0.8 / n_methods)
    for i, (lbl, col) in enumerate(method_cols):
        vals = [maxseq_rows[dm][col] for dm in D_MODELS]
        offs = [x + (i - (n_methods - 1) / 2) * bw for x in range(len(D_MODELS))]
        bars = ax.bar(offs, vals, width=bw, label=lbl,
                      color=col_map[lbl], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01 + 0.2,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(range(len(D_MODELS))))
    ax.set_xticklabels([f"d_model={dm}\nd_ff={dm*4}" for dm in D_MODELS])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{kname} Latency by Hidden Size  "
                 f"(seq_len={max_seq}, batch={BATCH}, FP16, A100)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"{kernel_name}_latency_bar_by_dmodel.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")

    # ── Fig 3: TFLOPS vs seq_len ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for dm in D_MODELS:
        sub = sorted(by_dm[dm], key=lambda r: r["seq_len"])
        ax.plot([r["seq_len"] for r in sub],
                [r["kernel_tops"] for r in sub],
                marker="o", label=f"d_model={dm}", linewidth=2, markersize=6)
    ax.axhline(peak_tops, color="gray", linestyle="--", linewidth=1.2,
               label=f"A100 peak ({peak_tops:.0f} TOPS)")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("TFLOPS / TOPS")
    ax.set_xticks(SEQ_LENS)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_title(f"{kname} Kernel Throughput  (batch={BATCH}, FP16)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"{kernel_name}_tflops_vs_seqlen.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")

    # ── Fig 4: Speedup vs naive ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for dm in D_MODELS:
        sub = sorted(by_dm[dm], key=lambda r: r["seq_len"])
        ax.plot([r["seq_len"] for r in sub],
                [r["speedup_vs_naive"] for r in sub],
                marker="o", label=f"d_model={dm}", linewidth=2, markersize=6)
    ax.axhline(1.0, color="gray", linestyle="--",
               linewidth=1.0, label="Baseline (1×)")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(f"Speedup vs {naive_label} (×)")
    ax.set_xticks(SEQ_LENS)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_title(f"{kname} Speedup vs FP16 WMMA Baseline  (batch={BATCH}, A100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"{kernel_name}_speedup_vs_seqlen.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")

    # ── Fig 5: HBM bandwidth utilisation ─────────────────────────────
    bw_methods = [
        (kernel_label, "kernel_bw_util_pct"),
        (naive_label,  "naive_bw_util_pct"),
    ]
    if "ref2_bw_util_pct" in rows[0]:
        bw_methods.append((ref2_label, "ref2_bw_util_pct"))
    if has_naive_pytorch and "naive_pytorch_bw_util_pct" in rows[0]:
        bw_methods.append(("Naive PyTorch (cuBLAS)", "naive_pytorch_bw_util_pct"))

    n_bw = len(bw_methods)
    bw_w = min(0.30, 0.8 / n_bw)
    fig, ax = plt.subplots(figsize=(max(8, 3 * len(D_MODELS) + 2), 4.5))
    for i, (lbl, col) in enumerate(bw_methods):
        vals = [maxseq_rows[dm][col] for dm in D_MODELS]
        offs = [x + (i - (n_bw - 1) / 2) * bw_w for x in range(len(D_MODELS))]
        bars = ax.bar(offs, vals, width=bw_w, label=lbl,
                      color=col_map[lbl], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01 + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(range(len(D_MODELS))))
    ax.set_xticklabels([f"d_model={dm}" for dm in D_MODELS])
    ax.set_ylabel("HBM Bandwidth Utilisation (%)")
    ax.set_title(f"{kname} HBM Bandwidth Utilisation  "
                 f"(seq_len={max_seq}, batch={BATCH}, A100 peak=2TB/s)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir,
                     f"{kernel_name}_hbm_bw_utilisation.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")


# ══════════════════════════════════════════════════════════════════════════
#  Kernel registry  — add a new kernel here, nothing else needs to change
# ══════════════════════════════════════════════════════════════════════════
KERNELS = {
    "mlp": {
        "loader":    load_mlp_ext,
        "bench_fn":  bench_mlp,
        "peak_tops": A100_FP16_TFLOPS,
        "owner":     "Shengjing",
    },
    "attention": {
        "loader":    load_attention_ext,
        "bench_fn":  bench_attention,
        "peak_tops": A100_FP16_TFLOPS,
        "owner":     "Jonathan",
    },
    "int8": {
        "loader":    load_int8_ext,
        "bench_fn":  bench_int8,
        "peak_tops": A100_INT8_TOPS,
        "owner":     "Heling",
    },
    "int8_attn": {
        "loader":    load_int8_ext,
        "bench_fn":  bench_int8_attn,
        "peak_tops": A100_INT8_TOPS,
        "owner":     "Heling",
    },
}


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for CUDA Transformer kernels."
    )
    parser.add_argument(
        "--kernel",
        choices=list(KERNELS.keys()),
        required=True,
        help="Which kernel to sweep: mlp | attention | int8 | int8_attn",
    )
    args = parser.parse_args()

    check_cuda()
    cfg = KERNELS[args.kernel]

    results_dir = os.path.join(ROOT, "results")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{args.kernel}_sweep.csv")

    print(f"Kernel  : {args.kernel}  (owner: {cfg['owner']})")
    print(f"Grid    : seq_lens={SEQ_LENS}  d_models={D_MODELS}  batch={BATCH}")
    print(f"Output  : {csv_path}\n")

    print("Compiling CUDA kernel …")
    ext = cfg["loader"]()
    print("Done.\n")

    total, done, rows = len(SEQ_LENS) * len(D_MODELS), 0, []

    for d_model in D_MODELS:
        for seq_len in SEQ_LENS:
            done += 1
            print(f"[{done:2d}/{total}] seq={seq_len:4d}  d_model={d_model}",
                  end="  ", flush=True)
            t0  = time.perf_counter()
            row = cfg["bench_fn"](ext, BATCH, seq_len, d_model)
            elapsed = time.perf_counter() - t0
            extra = ""
            if "naive_pytorch_ms" in row and row["naive_pytorch_ms"] is not None:
                cublas_ms = row["naive_pytorch_ms"]
                extra = f"  cuBLAS={cublas_ms:.2f}ms  vs_cuBLAS={cublas_ms/row['kernel_ms']:.2f}×"
            print(f"kernel={row['kernel_ms']:.2f}ms  "
                  f"naive={row['naive_ms']:.2f}ms  "
                  f"speedup={row['speedup_vs_naive']:.2f}×"
                  f"{extra}  "
                  f"({elapsed:.1f}s)")
            rows.append(row)

    save_csv(rows, csv_path)
    make_plots(rows, args.kernel, figures_dir, cfg["peak_tops"])
    print("\nSweep complete.")


if __name__ == "__main__":
    main()
