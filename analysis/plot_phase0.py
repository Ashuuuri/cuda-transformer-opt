# -*- coding: utf-8 -*-
"""Generate publication-quality charts for Phase 0 precision analysis.

Creates two figures for the report:
  1. Per-tensor vs per-token attention output cosine similarity across layers
  2. V-tensor distribution analysis (kurtosis & range utilization)

Usage:
    # From cached activations (if analysis/results/activations_*.pt exist):
    python analysis/plot_phase0.py

    # Use hardcoded data from experiment log (no GPU / no HuggingFace needed):
    python analysis/plot_phase0.py --hardcoded

Output:
    results/phase0_precision.png     (main chart: cosine similarity comparison)
    results/phase0_distribution.png  (supplementary: V-tensor outlier analysis)
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ═════════════════════════════════════════════════════════════════════
#  Hardcoded data from experiment log (fallback when no cached .pt)
# ═════════════════════════════════════════════════════════════════════

# Per-tensor attention output cosine similarity
PERTENSOR_ATTN_OUTPUT = {
    "GPT-2": {
        "layer_0":  1.0000,
        "layer_5":  0.9999,
        "layer_11": 0.9998,
    },
    "OPT-6.7B": {
        "layer_0": 0.9998,
        "layer_4": 0.9949,
        "layer_8": 0.9840,
    },
}

# Per-tensor V-tensor cosine similarity (shows the root cause)
PERTENSOR_V_COSSIM = {
    "GPT-2": {
        "layer_0":  1.0000,
        "layer_5":  1.0000,
        "layer_11": 0.9999,
    },
    "OPT-6.7B": {
        "layer_0": 0.9997,
        "layer_4": 0.9984,
        "layer_8": 0.9984,
    },
}

# Per-token attention output cosine similarity
# (From the experiment log: worst-case GPT-2=0.9998, OPT-6.7B=0.9999)
PERTOKEN_ATTN_OUTPUT = {
    "GPT-2": {
        "layer_0":  1.0000,
        "layer_5":  0.9999,
        "layer_11": 0.9998,
    },
    "OPT-6.7B": {
        "layer_0": 1.0000,
        "layer_4": 0.9999,
        "layer_8": 0.9999,
    },
}

# V-tensor distribution stats (kurtosis and range utilization)
V_DISTRIBUTION = {
    "GPT-2": {
        # (kurtosis, range_utilization)
        "layer_0":  (3.1, 0.25),
        "layer_5":  (3.2, 0.22),
        "layer_11": (3.5, 0.18),
    },
    "OPT-6.7B": {
        "layer_0": (4.2, 0.12),
        "layer_4": (7.8, 0.055),
        "layer_8": (11.8, 0.035),
    },
}


# ═════════════════════════════════════════════════════════════════════
#  Recompute from cached activations (when available)
# ═════════════════════════════════════════════════════════════════════

def try_load_from_cache():
    """Try to load cached activations and recompute all metrics.
    Returns (pertensor, pertoken, v_cossim, v_dist) dicts or None if cache unavailable.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None

    cache_dir = os.path.join(os.path.dirname(__file__), "results")
    model_map = {
        "gpt2": "GPT-2",
        "opt-6.7b": "OPT-6.7B",
    }

    pertensor_attn = {}
    pertoken_attn = {}
    v_cossim = {}
    v_dist = {}

    any_found = False
    for file_key, display_name in model_map.items():
        cache_path = os.path.join(cache_dir, f"activations_{file_key}.pt")
        if not os.path.exists(cache_path):
            continue
        any_found = True

        print(f"  Loading cached activations: {cache_path}")
        activations = torch.load(cache_path, map_location="cpu")

        pertensor_attn[display_name] = {}
        pertoken_attn[display_name] = {}
        v_cossim[display_name] = {}
        v_dist[display_name] = {}

        for layer_name, data in sorted(activations.items()):
            Q = data["Q"].float()
            K = data["K"].float()
            V = data["V"].float()
            head_dim = Q.shape[-1]
            scale = head_dim ** -0.5

            # Ground truth
            scores_fp = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_fp = F.softmax(scores_fp, dim=-1)
            out_fp = torch.matmul(attn_fp, V)

            # --- Per-tensor path ---
            def quant_pt(t):
                t_f = t.float()
                s = t_f.abs().max() / 127.0
                t_i8 = (t_f / s).round().clamp(-128, 127).to(torch.int8)
                return t_i8, s

            Qi, sQ = quant_pt(Q)
            Ki, sK = quant_pt(K)
            Vi, sV = quant_pt(V)

            qk_i32 = torch.matmul(Qi.int(), Ki.int().transpose(-2, -1))
            scores_i8 = qk_i32.float() * (sQ * sK * scale)
            attn_i8 = F.softmax(scores_i8, dim=-1)
            V_deq = Vi.float() * sV
            out_i8 = torch.matmul(attn_i8, V_deq)

            cs_pt = F.cosine_similarity(
                out_fp.flatten().unsqueeze(0),
                out_i8.flatten().unsqueeze(0)).item()
            pertensor_attn[display_name][layer_name] = cs_pt

            # V cosine sim (per-tensor)
            V_deq_direct = Vi.float() * sV
            cs_v = F.cosine_similarity(
                V.flatten().unsqueeze(0),
                V_deq_direct.flatten().unsqueeze(0)).item()
            v_cossim[display_name][layer_name] = cs_v

            # --- Per-token path ---
            def quant_ptoken(t):
                t_f = t.float()
                am = t_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                s = am / 127.0
                t_i8 = (t_f / s).round().clamp(-128, 127).to(torch.int8)
                return t_i8, s

            Qi2, sQ2 = quant_ptoken(Q)
            Ki2, sK2 = quant_ptoken(K)
            Vi2, sV2 = quant_ptoken(V)

            qk_i32_2 = torch.matmul(Qi2.int(), Ki2.int().transpose(-2, -1))
            scale_mat = sQ2 * sK2.transpose(-2, -1) * scale
            scores_i8_2 = qk_i32_2.float() * scale_mat
            attn_i8_2 = F.softmax(scores_i8_2, dim=-1)
            V_deq_2 = Vi2.float() * sV2
            out_i8_2 = torch.matmul(attn_i8_2, V_deq_2)

            cs_tok = F.cosine_similarity(
                out_fp.flatten().unsqueeze(0),
                out_i8_2.flatten().unsqueeze(0)).item()
            pertoken_attn[display_name][layer_name] = cs_tok

            # V distribution stats
            Vf = V.float()
            std = Vf.std().item()
            mean = Vf.mean().item()
            abs_max = Vf.abs().max().item()
            kurtosis = ((Vf - mean) / std).pow(4).mean().item() if std > 0 else 0.0
            range_util = (Vf.abs().mean() / abs_max).item() if abs_max > 0 else 0
            v_dist[display_name][layer_name] = (kurtosis, range_util)

    if not any_found:
        return None

    return pertensor_attn, pertoken_attn, v_cossim, v_dist


# ═════════════════════════════════════════════════════════════════════
#  Chart 1: Cosine Similarity Comparison (main chart)
# ═════════════════════════════════════════════════════════════════════

def plot_cosine_comparison(pertensor, pertoken, save_path):
    """Grouped bar chart: per-tensor vs per-token cosine similarity per layer."""

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Build x-axis labels and data
    labels = []
    pt_vals = []   # per-tensor
    tok_vals = []  # per-token

    for model in ["GPT-2", "OPT-6.7B"]:
        for layer in sorted(pertensor[model].keys(),
                            key=lambda x: int(x.split("_")[1])):
            labels.append(f"{model}\n{layer}")
            pt_vals.append(pertensor[model][layer])
            tok_vals.append(pertoken[model][layer])

    x = np.arange(len(labels))
    width = 0.35

    # Colors — muted academic palette
    c_pt  = "#D97B6B"  # warm salmon (per-tensor, the problematic one)
    c_tok = "#5A9EAF"  # teal blue (per-token, the fix)
    c_pt_dark  = "#B8554A"  # darker salmon for badge
    c_tok_dark = "#3A7A8A"  # darker teal for badge

    bars1 = ax.bar(x - width/2, pt_vals, width, label="Per-tensor",
                   color=c_pt, edgecolor="white", linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + width/2, tok_vals, width, label="Per-token",
                   color=c_tok, edgecolor="white", linewidth=0.8, zorder=3)

    # PASS threshold line
    ax.axhline(y=0.999, color="#888888", linestyle="--", linewidth=1.2,
               label="PASS threshold (0.999)", zorder=2)

    # Y-axis: focus on [0.98, 1.001] to show the differences
    ax.set_ylim(0.975, 1.004)
    ax.set_ylabel("Attention Output Cosine Similarity", fontsize=12)
    ax.set_xlabel("Model / Layer", fontsize=12)
    ax.set_title("INT8 Quantization Precision: Per-Tensor vs Per-Token",
                 fontsize=14, fontweight="bold", pad=12)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Annotate the OPT-6.7B layer_8 per-tensor failure
    fail_idx = None
    for i, (model_label, pt_v) in enumerate(zip(labels, pt_vals)):
        if pt_v < 0.99:
            fail_idx = i
            break

    if fail_idx is not None:
        fail_val = pt_vals[fail_idx]
        fix_val  = tok_vals[fail_idx]
        # Badge on the failing salmon bar
        ax.text(fail_idx - width/2, fail_val + 0.0004,
                f"{fail_val:.4f}",
                fontsize=7.5, fontweight="bold", color="white",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=c_pt_dark,
                          edgecolor="none", alpha=0.95))
        # Badge on the teal bar
        ax.text(fail_idx + width/2, fix_val + 0.0004,
                f"{fix_val:.4f}",
                fontsize=7.5, fontweight="bold", color="white",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=c_tok_dark,
                          edgecolor="none", alpha=0.95))

    # Add a vertical separator between models
    n_gpt2 = len(PERTENSOR_ATTN_OUTPUT.get("GPT-2", {}))
    if n_gpt2 > 0 and n_gpt2 < len(labels):
        ax.axvline(x=n_gpt2 - 0.5, color="#BDC3C7", linestyle="-",
                   linewidth=1, zorder=1)

    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  [Saved] {save_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
#  Chart 2: V-tensor Distribution Analysis (supplementary)
# ═════════════════════════════════════════════════════════════════════

def plot_distribution_analysis(v_dist, v_cossim, save_path):
    """Dual-axis chart: V-tensor kurtosis (bar) + cosine similarity (line)."""

    fig, ax1 = plt.subplots(figsize=(10, 5))

    labels = []
    kurtosis_vals = []
    range_util_vals = []
    cossim_vals = []

    for model in ["GPT-2", "OPT-6.7B"]:
        for layer in sorted(v_dist[model].keys(),
                            key=lambda x: int(x.split("_")[1])):
            labels.append(f"{model}\n{layer}")
            k, ru = v_dist[model][layer]
            kurtosis_vals.append(k)
            range_util_vals.append(ru * 100)  # percent
            cossim_vals.append(v_cossim[model][layer])

    x = np.arange(len(labels))

    # Kurtosis bars
    c_kurt = "#3498DB"
    bars = ax1.bar(x, kurtosis_vals, 0.5, color=c_kurt,
                   edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3,
                   label="V-tensor kurtosis")
    ax1.set_ylabel("Kurtosis", fontsize=12, color=c_kurt)
    ax1.tick_params(axis="y", labelcolor=c_kurt)
    ax1.set_ylim(0, max(kurtosis_vals) * 1.3)

    # Normal kurtosis reference line
    ax1.axhline(y=3.0, color=c_kurt, linestyle=":", linewidth=1,
                alpha=0.6, zorder=2)
    ax1.text(len(labels) - 0.5, 3.2, "Normal (κ=3)", fontsize=8,
             color=c_kurt, alpha=0.7, ha="right")

    # Annotate the worst kurtosis
    worst_idx = np.argmax(kurtosis_vals)
    ax1.annotate(
        f"κ={kurtosis_vals[worst_idx]:.1f}\nRange util: {range_util_vals[worst_idx]:.1f}%",
        xy=(worst_idx, kurtosis_vals[worst_idx]),
        xytext=(worst_idx - 0.8, kurtosis_vals[worst_idx] + 1.5),
        fontsize=9, fontweight="bold", color="#C0392B",
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
        ha="center",
    )

    # Second y-axis: V cosine similarity
    ax2 = ax1.twinx()
    c_cs = "#E67E22"
    ax2.plot(x, cossim_vals, "o-", color=c_cs, linewidth=2, markersize=7,
             zorder=4, label="V cosine similarity (per-tensor)")
    ax2.set_ylabel("V Cosine Similarity", fontsize=12, color=c_cs)
    ax2.tick_params(axis="y", labelcolor=c_cs)
    ax2.set_ylim(0.996, 1.001)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_xlabel("Model / Layer", fontsize=12)
    ax1.set_title("V-Tensor Distribution: Why Per-Tensor Quantization Fails",
                  fontsize=14, fontweight="bold", pad=12)

    # Model separator
    n_gpt2 = len(V_DISTRIBUTION.get("GPT-2", {}))
    if n_gpt2 > 0 and n_gpt2 < len(labels):
        ax1.axvline(x=n_gpt2 - 0.5, color="#BDC3C7", linestyle="-",
                    linewidth=1, zorder=1)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=10, framealpha=0.9)

    ax1.grid(axis="y", alpha=0.2, zorder=0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  [Saved] {save_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot Phase 0 precision analysis for report")
    parser.add_argument("--hardcoded", action="store_true",
                        help="Use hardcoded data (no cached activations needed)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 0: Precision Analysis Plots")
    print("=" * 60)

    if args.hardcoded:
        print("\n  Using hardcoded data from experiment log")
        pertensor = PERTENSOR_ATTN_OUTPUT
        pertoken  = PERTOKEN_ATTN_OUTPUT
        v_cs      = PERTENSOR_V_COSSIM
        v_d       = V_DISTRIBUTION
    else:
        print("\n  Trying to load cached activations...")
        result = try_load_from_cache()
        if result is not None:
            pertensor, pertoken, v_cs, v_d = result
            print("  Loaded from cache — recomputed all metrics.")
        else:
            print("  No cached activations found. Using hardcoded data.")
            print("  (Run phase0_real_activations.py first to generate cache,")
            print("   or use --hardcoded to skip.)")
            pertensor = PERTENSOR_ATTN_OUTPUT
            pertoken  = PERTOKEN_ATTN_OUTPUT
            v_cs      = PERTENSOR_V_COSSIM
            v_d       = V_DISTRIBUTION

    # Generate charts
    chart1_path = os.path.join(RESULTS_DIR, "phase0_precision.png")
    chart2_path = os.path.join(RESULTS_DIR, "phase0_distribution.png")

    print()
    plot_cosine_comparison(pertensor, pertoken, chart1_path)
    plot_distribution_analysis(v_d, v_cs, chart2_path)

    print(f"\n  Charts ready for report:")
    print(f"    {chart1_path}")
    print(f"    {chart2_path}")
    print()


if __name__ == "__main__":
    main()
