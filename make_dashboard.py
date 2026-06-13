#!/usr/bin/env python3
"""Consolidated INT8 performance + profiling dashboard.

Produces ONE big figure, results/figures/int8_dashboard.png, replacing the
20 scattered per-kernel PNGs with a single readable sheet:

  Row 1  INT8 MLP   — latency / throughput / speedup vs 3 clearly-labelled refs
  Row 2  INT8 attn  — latency / throughput / speedup vs FlashAttn-2 & FP16 WMMA
  Row 3  Profiling  — ncu stall breakdown · kernel-time split (nsys-style) ·
                      MLP optimisation journey (iter 7→8→10) from ncu metrics

Data sources (all under results/):
  *_sweep.csv               sweep.py latency/throughput/speedup
  kernel_time_breakdown.csv collect_profile.py torch.profiler per-kernel time
  ncu_{mlp,attn}_raw.csv    raw `ncu --csv` stall metrics (preamble tolerated)
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RES, "figures")
os.makedirs(FIGDIR, exist_ok=True)

C_OURS  = "#2563EB"   # our INT8 fused kernel
C_FP16  = "#DC2626"   # FP16 cuBLAS / manual reference
C_SOTA  = "#16A34A"   # cuBLAS INT8 pipeline / FlashAttn-2
C_LOWER = "#F59E0B"   # bare 2xGEMM / FP16 WMMA kernel
REP_DM  = 1024        # representative d_model for the line panels


def read_sweep(name):
    rows = []
    with open(os.path.join(RES, name)) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if _isnum(v) else v) for k, v in r.items()})
    return rows


def _isnum(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def at_dm(rows, dm):
    sub = [r for r in rows if int(r["d_model"]) == dm]
    sub.sort(key=lambda r: r["seq_len"])
    return sub


# ────────────────────────────────────────────────────────────────────────
#  ncu stall parsing
# ────────────────────────────────────────────────────────────────────────
STALL_KEYS = [
    ("wait",            "smsp__warp_issue_stalled_wait_per_warp_active.pct"),
    ("long_scoreboard", "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
    ("short_scoreboard","smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"),
    ("mio_throttle",    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"),
    ("barrier",         "smsp__warp_issue_stalled_barrier_per_warp_active.pct"),
    ("math_throttle",   "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"),
    ("not_selected",    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct"),
    ("lg_throttle",     "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"),
]
IMMA_KEY = "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"


def read_ncu(name):
    """Return {kernel_label: {metric_short: value}} from a raw `ncu --csv` dump.

    Tolerates the `==PROF==` / banner preamble ncu prints before the CSV: we
    skip everything up to the real header row (ncu quotes every field, and the
    header contains "Kernel Name").
    """
    out = {}
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return out
    with open(path) as f:
        lines = [ln for ln in f if ln.startswith('"')]
    if not lines:
        return out
    for r in csv.DictReader(lines):
        kn = r.get("Kernel Name", "")
        try:
            val = float(str(r.get("Metric Value", "")).replace(",", ""))
        except ValueError:
            continue
        mn = r.get("Metric Name", "")
        out.setdefault(kn, {})[mn] = val
    return out


def label_kernel(kn):
    if "gemm_int8_wmma_f16_kernel<1, 0, 0>" in kn:
        return "MLP GEMM1\n(x·W1+GELU)"
    if "gemm_int8_wmma_f16_kernel<0, 1, 0>" in kn:
        return "MLP GEMM2\n(h·W2)"
    if "attention" in kn:
        return "Attn QK^T\n(INT8 part)"
    return kn[:18]


# ────────────────────────────────────────────────────────────────────────
def panel_latency(ax, rows, series, title, ylabel="latency (ms)"):
    sub = at_dm(rows, REP_DM)
    x = [r["seq_len"] for r in sub]
    for col, lbl, color in series:
        y = [r[col] for r in sub]
        ax.plot(x, y, marker="o", color=color, label=lbl, linewidth=2, markersize=5)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
    ax.set_xticks(x); ax.set_xticklabels([int(s) for s in x])
    ax.set_xlabel("seq_len"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8, loc="upper left")


def panel_tops(ax, rows, series, peak, title):
    sub = at_dm(rows, REP_DM)
    x = [r["seq_len"] for r in sub]
    for col, lbl, color in series:
        y = [r[col] for r in sub]
        ax.plot(x, y, marker="o", color=color, label=lbl, linewidth=2, markersize=5)
    if peak:
        ax.axhline(peak, color="gray", ls="--", lw=1.2, label=f"A100 peak ≈{peak:.0f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.set_xticklabels([int(s) for s in x])
    ax.set_xlabel("seq_len"); ax.set_ylabel("throughput (TOPS / TFLOPS)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="lower right")


def panel_speedup(ax, rows, series, title):
    sub = at_dm(rows, REP_DM)
    x = np.arange(len(sub)); seqs = [int(r["seq_len"]) for r in sub]
    n = len(series); w = 0.8 / n
    for i, (col, lbl, color) in enumerate(series):
        y = [r[col] for r in sub]
        ax.bar(x + (i - (n - 1) / 2) * w, y, w, label=lbl, color=color, alpha=0.9)
    ax.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(seqs)
    ax.set_xlabel("seq_len"); ax.set_ylabel("speedup (× ref, >1 = ours faster)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8, loc="upper left")


def panel_stalls(ax, ncu_mlp, ncu_attn):
    kernels = []
    for kn, m in ncu_mlp.items():
        kernels.append((label_kernel(kn), m))
    for kn, m in ncu_attn.items():
        kernels.append((label_kernel(kn), m))
    if not kernels:
        ax.text(0.5, 0.5, "no ncu data", ha="center"); return
    labels = [k for k, _ in kernels]
    x = np.arange(len(labels))
    cmap = plt.get_cmap("tab10")
    bottoms = np.zeros(len(labels))
    for i, (short, full) in enumerate(STALL_KEYS):
        vals = np.array([m.get(full, 0.0) for _, m in kernels])
        ax.bar(x, vals, 0.55, bottom=bottoms, label=short, color=cmap(i % 10))
        bottoms += vals
    # tensor pipe utilisation overlay (markers)
    imma = [m.get(IMMA_KEY, np.nan) for _, m in kernels]
    ax.plot(x, imma, "D", color="black", markersize=8, label="tensor_op_imma %", zorder=5)
    for xi, v in zip(x, imma):
        if not np.isnan(v):
            ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points",
                        xytext=(7, 0), fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% of warp-active cycles (stacked stalls)")
    ax.set_title("ncu stall breakdown — wait = MMA-dependency ceiling",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)


def panel_kerneltime(ax):
    path = os.path.join(RES, "kernel_time_breakdown.csv")
    if not os.path.exists(path):
        ax.text(0.5, 0.5, "no breakdown data", ha="center"); return
    from collections import defaultdict
    data = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            data[r["forward"]].append((r["category"], float(r["pct"])))
    # Show MLP all-in-one vs MLP prepacked (transpose hoisted out) vs attn.
    fwds   = [f for f in ["int8_mlp", "int8_mlp_prepacked", "int8_attn"] if f in data]
    labels = {"int8_mlp": "MLP forward\n(transpose/call)",
              "int8_mlp_prepacked": "MLP forward\n(prepacked W)",
              "int8_attn": "attn forward"}
    x = np.arange(len(fwds))
    cats = []
    for fw in fwds:
        for c, _ in data[fw]:
            if c not in cats: cats.append(c)
    cmap = plt.get_cmap("Set2")
    bottoms = np.zeros(len(fwds))
    for i, cat in enumerate(cats):
        vals = np.array([dict(data[fw]).get(cat, 0.0) for fw in fwds])
        ax.bar(x, vals, 0.5, bottom=bottoms, label=cat, color=cmap(i % 8))
        for xi, (b, v) in enumerate(zip(bottoms, vals)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold")
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels([labels.get(f, f) for f in fwds], fontsize=8)
    ax.set_ylabel("% of forward CUDA time")
    ax.set_title("Where the GPU time goes (torch.profiler, nsys-style)\n"
                 "prepacking W removes the per-call transpose",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.25)


def panel_journey(ax):
    # Documented graded-shape GEMM2 ncu metrics across the MLP iterations
    # (CLAUDE.md / README iter 7→8; iter 10 measured this session, reverted).
    iters = ["iter 7\n(bank-conflict\nfix)", "iter 8\n(STAGE_K 64)",
             "iter 10\n(denser-mma\nREVERTED)"]
    imma = [30.9, 38.2, 38.24]
    wait = [24.3, 28.2, 28.18]
    longsb = [27.4, 7.7, 7.65]
    x = np.arange(len(iters)); w = 0.25
    b1 = ax.bar(x - w, imma, w, label="tensor_op_imma", color=C_OURS)
    b2 = ax.bar(x,     wait, w, label="wait (MMA dep)", color=C_FP16)
    b3 = ax.bar(x + w, longsb, w, label="long_scoreboard", color=C_LOWER)
    # mark iter 10 bars as reverted (hatch)
    for b in (b1, b2, b3):
        b[-1].set_hatch("//"); b[-1].set_alpha(0.55)
    ax.axhline(60, color="green", ls="--", lw=1.3)
    ax.text(0.02, 61, "cuBLAS tensor-pipe target ≈60%+", color="green",
            fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(iters, fontsize=8)
    ax.set_ylabel("% (GEMM2, graded shape)")
    ax.set_title("MLP optimisation journey — iter 10 moved nothing (dead end)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 70)


def main():
    mlp = read_sweep("int8_mlp_sweep.csv")
    attn = read_sweep("int8_attn_sweep.csv")
    ncu_mlp = read_ncu("ncu_mlp_raw.csv")
    ncu_attn = read_ncu("ncu_attn_raw.csv")

    fig = plt.figure(figsize=(19, 17))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.26,
                  top=0.90, bottom=0.05, left=0.10, right=0.98)

    # Row 1 — INT8 MLP
    mlp_lat = [("kernel_ms", "INT8 fused (ours)", C_OURS),
               ("naive_ms",  "FP16 cuBLAS (torch)", C_FP16),
               ("ref2_ms",   "cuBLAS INT8 pipeline", C_SOTA),
               ("ref3_ms",   "cuBLAS INT8 2×GEMM (floor)", C_LOWER)]
    panel_latency(fig.add_subplot(gs[0, 0]), mlp, mlp_lat,
                  "INT8 MLP — latency (d_model=1024, batch=8)")
    mlp_tops = [("kernel_tops", "INT8 fused (ours)", C_OURS),
                ("naive_tops",  "FP16 cuBLAS", C_FP16),
                ("ref2_tops",   "cuBLAS INT8 pipeline", C_SOTA),
                ("ref3_tops",   "cuBLAS INT8 2×GEMM", C_LOWER)]
    panel_tops(fig.add_subplot(gs[0, 1]), mlp, mlp_tops, None,
               "INT8 MLP — throughput (d_model=1024)")
    mlp_sp = [("speedup_vs_naive", "vs FP16 cuBLAS", C_FP16),
              ("speedup_vs_ref2",  "vs cuBLAS INT8 pipeline", C_SOTA),
              ("speedup_vs_ref3",  "vs cuBLAS INT8 2×GEMM", C_LOWER)]
    panel_speedup(fig.add_subplot(gs[0, 2]), mlp, mlp_sp,
                  "INT8 MLP — speedup vs each reference")

    # Row 2 — INT8 attention
    at_lat = [("kernel_ms", "INT8 fused (ours)", C_OURS),
              ("naive_ms",  "FP16 manual attn", C_FP16),
              ("ref2_ms",   "FlashAttn-2 (SDPA)", C_SOTA),
              ("ref3_ms",   "FP16 WMMA kernel (team)", C_LOWER)]
    panel_latency(fig.add_subplot(gs[1, 0]), attn, at_lat,
                  "INT8 attention — latency (d_model=1024, batch=8)")
    at_tops = [("kernel_tops", "INT8 fused (ours)", C_OURS),
               ("naive_tops",  "FP16 manual", C_FP16),
               ("ref2_tops",   "FlashAttn-2", C_SOTA),
               ("ref3_tops",   "FP16 WMMA kernel", C_LOWER)]
    panel_tops(fig.add_subplot(gs[1, 1]), attn, at_tops, None,
               "INT8 attention — throughput (d_model=1024)")
    at_sp = [("speedup_vs_naive", "vs FP16 manual", C_FP16),
             ("speedup_vs_ref2",  "vs FlashAttn-2", C_SOTA),
             ("speedup_vs_ref3",  "vs FP16 WMMA kernel", C_LOWER)]
    panel_speedup(fig.add_subplot(gs[1, 2]), attn, at_sp,
                  "INT8 attention — speedup vs each reference")

    # Row 3 — profiling
    panel_stalls(fig.add_subplot(gs[2, 0]), ncu_mlp, ncu_attn)
    panel_kerneltime(fig.add_subplot(gs[2, 1]))
    panel_journey(fig.add_subplot(gs[2, 2]))

    fig.suptitle("INT8 Transformer Kernels — Performance & Profiling Dashboard "
                 "(A100-SXM4-40GB, sm_80)\n"
                 "graded sweep: batch=8, head_dim=64; profiling shape b=8 s=512 "
                 "d_model=1024 d_ff=4096",
                 fontsize=15, fontweight="bold")
    # row band labels in the left margin (rotated, centred on each row band)
    for y, txt in [(0.755, "①  INT8 MLP benchmark"),
                   (0.470, "②  INT8 attention benchmark"),
                   (0.185, "③  Profiling: ncu stalls · kernel-time · opt journey")]:
        fig.text(0.018, y, txt, fontsize=13, fontweight="bold", color="#374151",
                 rotation=90, va="center", ha="center")

    out = os.path.join(FIGDIR, "int8_dashboard.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
