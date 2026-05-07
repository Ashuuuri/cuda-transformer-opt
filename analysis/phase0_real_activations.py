# -*- coding: utf-8 -*-
"""Phase 0: INT8 quantization precision analysis using real model activations.

Compares a small model (GPT-2, 124M) vs a large model (OPT-6.7B) to check
whether per-tensor INT8 quantization is precise enough for attention.

Key question: Does the "emergent outlier" problem (LLM.int8(), Dettmers 2022)
affect our INT8 kernel's precision?

Usage:
    # Small model only (login node, no GPU needed):
    python analysis/phase0_real_activations.py --model gpt2

    # Large model (needs GPU node, ~13GB VRAM):
    python analysis/phase0_real_activations.py --model opt-6.7b

    # Both models side-by-side comparison:
    python analysis/phase0_real_activations.py --model all

Output:
    analysis/results/phase0_report.txt
    analysis/results/activations_<model>.pt  (cached for reuse)
"""

import argparse
import os
import sys
import math
import torch
import torch.nn.functional as F

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ══════════════════════════════════════════════════════════════════════════
#  Model configs
# ══════════════════════════════════════════════════════════════════════════
MODELS = {
    "gpt2": {
        "hf_name": "gpt2",
        "desc": "GPT-2 124M (12 layers, 12 heads, d=768, head_dim=64)",
        "layers": [0, 3, 6, 11],
        "dtype": torch.float32,  # GPT-2 is distributed in FP32
        "device": "cpu",         # small enough for CPU
    },
    "opt-6.7b": {
        "hf_name": "facebook/opt-6.7b",
        "desc": "OPT-6.7B (32 layers, 32 heads, d=4096, head_dim=128)",
        "layers": [0, 8, 16, 31],
        "dtype": torch.float16,  # load in FP16 to save memory
        "device": "cpu",         # CPU is fine, only one forward pass needed
    },
}

# Diverse test texts
TEST_TEXTS = [
    "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow the model to attend to all positions in the input sequence simultaneously.",
    "In distributed computing, consistency and availability are fundamentally at odds when network partitions occur, as demonstrated by the CAP theorem.",
    "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    "The eigenvalues of a symmetric positive definite matrix are all positive, and its Cholesky decomposition exists and is unique.",
]

report_lines = []


def report(s=""):
    print(s)
    report_lines.append(s)


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-1: Extract real Q, K, V
# ══════════════════════════════════════════════════════════════════════════

def extract_activations(model_key):
    """Load a model and intercept Q, K, V projections at multiple layers."""
    cfg = MODELS[model_key]
    cache_path = os.path.join(RESULTS_DIR, f"activations_{model_key}.pt")

    report(f"\n  Model: {cfg['desc']}")
    report(f"  HuggingFace ID: {cfg['hf_name']}")
    report(f"  Layers to capture: {cfg['layers']}")
    report(f"  Device: {cfg['device']}, dtype: {cfg['dtype']}")

    # Check cache
    if os.path.exists(cache_path):
        report(f"  [Cache] Loading: {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        report(f"  Layers found: {list(data.keys())}")
        return data

    from transformers import AutoModel, AutoTokenizer

    report(f"  Loading model (this may take a minute for large models)...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model_kwargs = {}
    if cfg["dtype"] == torch.float16:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(cfg["hf_name"], **model_kwargs).eval()
    if cfg["device"] == "cuda":
        model = model.to("cuda")

    # Tokenize
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(TEST_TEXTS, return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
    if cfg["device"] == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    activations = {}

    # --- GPT-2 style (model.h[i].attn with c_attn) ---
    if "gpt2" in cfg["hf_name"]:
        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs, output):
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                if hidden_states is None:
                    return
                qkv = module.c_attn(hidden_states)
                bs, seq, _ = qkv.shape
                d = module.embed_dim
                nh = module.num_heads
                hd = d // nh
                q, k, v = qkv.split(d, dim=2)
                q = q.view(bs, seq, nh, hd).transpose(1, 2)
                k = k.view(bs, seq, nh, hd).transpose(1, 2)
                v = v.view(bs, seq, nh, hd).transpose(1, 2)
                activations[f"layer_{layer_idx}"] = {
                    "Q": q.detach().cpu().clone(),
                    "K": k.detach().cpu().clone(),
                    "V": v.detach().cpu().clone(),
                    "shape": f"batch={bs}, heads={nh}, seq={seq}, head_dim={hd}",
                }
            return hook_fn

        hooks = []
        for li in cfg["layers"]:
            h = model.h[li].attn.register_forward_hook(make_hook(li), with_kwargs=True)
            hooks.append(h)

    # --- OPT style (model.decoder.layers[i].self_attn with q/k/v_proj) ---
    elif "opt" in cfg["hf_name"]:
        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs, output):
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                if hidden_states is None:
                    return
                bs, seq, _ = hidden_states.shape
                nh = module.num_heads
                hd = module.head_dim
                q = module.q_proj(hidden_states).view(bs, seq, nh, hd).transpose(1, 2)
                k = module.k_proj(hidden_states).view(bs, seq, nh, hd).transpose(1, 2)
                v = module.v_proj(hidden_states).view(bs, seq, nh, hd).transpose(1, 2)
                activations[f"layer_{layer_idx}"] = {
                    "Q": q.detach().cpu().clone(),
                    "K": k.detach().cpu().clone(),
                    "V": v.detach().cpu().clone(),
                    "shape": f"batch={bs}, heads={nh}, seq={seq}, head_dim={hd}",
                }
            return hook_fn

        hooks = []
        for li in cfg["layers"]:
            h = model.decoder.layers[li].self_attn.register_forward_hook(
                make_hook(li), with_kwargs=True)
            hooks.append(h)
    else:
        raise ValueError(f"Unsupported model: {cfg['hf_name']}")

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Free GPU memory
    del model
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()

    # Report
    for key, val in activations.items():
        q = val["Q"]
        report(f"  {key}: {val['shape']}, Q range=[{q.min():.3f}, {q.max():.3f}]")

    # Save cache
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(activations, cache_path)
    report(f"  [Saved] {cache_path}")

    return activations


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-2: Analyze distribution characteristics
# ══════════════════════════════════════════════════════════════════════════

def analyze_distributions(activations, model_key):
    report(f"\n  --- Distribution stats ({model_key}) ---\n")
    report(f"  {'Layer':<12} {'Tensor':<4} {'Mean':>8} {'Std':>8} "
           f"{'AbsMax':>8} {'Kurtosis':>9} {'|x|>3s%':>8} {'RangeUtil':>9}")
    report(f"  {'-'*12} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*9}")

    all_stats = {}

    for layer_name, data in activations.items():
        for tensor_name in ["Q", "K", "V"]:
            t = data[tensor_name].float()
            mean = t.mean().item()
            std = t.std().item()
            abs_max = t.abs().max().item()

            kurtosis = ((t - mean) / std).pow(4).mean().item() if std > 0 else 0.0
            outlier_pct = (t.abs() > 3 * std).float().mean().item() * 100
            range_util = (t.abs().mean() / abs_max).item() if abs_max > 0 else 0

            report(f"  {layer_name:<12} {tensor_name:<4} {mean:>7.4f} {std:>7.4f} "
                   f"{abs_max:>7.3f} {kurtosis:>8.1f}  {outlier_pct:>6.2f}% {range_util:>8.3f}")

            all_stats[f"{layer_name}/{tensor_name}"] = {
                "mean": mean, "std": std, "abs_max": abs_max,
                "kurtosis": kurtosis, "outlier_pct": outlier_pct,
                "range_util": range_util,
            }

    return all_stats


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-3: Per-tensor INT8 quantization precision
# ══════════════════════════════════════════════════════════════════════════

def quantize_per_tensor(t):
    """Per-tensor symmetric INT8 quantization (identical to our CUDA kernel)."""
    t_f = t.float()
    abs_max = t_f.abs().max()
    if abs_max == 0:
        return torch.zeros_like(t_f, dtype=torch.int8), torch.tensor(1.0)
    scale = abs_max / 127.0
    t_int8 = (t_f / scale).round().clamp(-128, 127).to(torch.int8)
    return t_int8, scale


def quantize_per_token(t):
    """Per-token symmetric INT8 quantization.

    For tensor of shape [batch, heads, seq, head_dim], each token (last-dim vector)
    gets its own scale. This avoids one outlier token ruining precision for all others.

    Returns:
        t_int8: same shape, dtype=int8
        scales: shape [batch, heads, seq, 1]
    """
    t_f = t.float()
    # Scale per token: max along head_dim axis
    abs_max = t_f.abs().amax(dim=-1, keepdim=True)  # [batch, heads, seq, 1]
    abs_max = abs_max.clamp(min=1e-8)  # avoid division by zero
    scales = abs_max / 127.0
    t_int8 = (t_f / scales).round().clamp(-128, 127).to(torch.int8)
    return t_int8, scales


def dequantize(t_int8, scale):
    return t_int8.float() * scale


def cosine_sim(a, b):
    """Cosine similarity between two tensors (flattened)."""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def analyze_quantization(activations, model_key):
    report(f"\n  --- Quantization precision ({model_key}) ---")
    report(f"  Method: scale = max|tensor| / 127 (per-tensor symmetric)\n")

    report(f"  {'Layer':<12} {'Tensor':<4} {'CosSim':>8} {'RMSE':>10} "
           f"{'MaxErr':>8} {'SNR(dB)':>8} {'Grade':>6}")
    report(f"  {'-'*12} {'-'*4} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")

    results = {}

    for layer_name, data in activations.items():
        for tensor_name in ["Q", "K", "V"]:
            t = data[tensor_name].float()
            t_i8, scale = quantize_per_tensor(t)
            t_deq = dequantize(t_i8, scale)

            cs = cosine_sim(t, t_deq)
            diff = (t - t_deq).abs()
            rmse = diff.pow(2).mean().sqrt().item()
            max_err = diff.max().item()
            snr = (10 * torch.log10(
                t.pow(2).mean() / diff.pow(2).mean().clamp(min=1e-20)
            )).item()

            if cs > 0.9999:
                verdict = "A"
            elif cs > 0.999:
                verdict = "B"
            elif cs > 0.99:
                verdict = "C"
            else:
                verdict = "F"

            report(f"  {layer_name:<12} {tensor_name:<4} {cs:>7.5f} {rmse:>9.6f} "
                   f"{max_err:>7.4f} {snr:>7.1f}  {verdict}")

            results[f"{layer_name}/{tensor_name}"] = {
                "cosine_sim": cs, "rmse": rmse, "max_err": max_err,
                "snr_db": snr, "scale": scale.item(),
            }

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-4: Full attention path error propagation
# ══════════════════════════════════════════════════════════════════════════

def analyze_attention_propagation(activations, model_key):
    report(f"\n  --- Error propagation through attention ({model_key}) ---")
    report(f"  Path: Q,K,V -> INT8 -> QK^T(INT32) -> float -> softmax -> xV -> output\n")

    report(f"  {'Layer':<12} {'QKT CS':>8} {'Softmax CS':>10} "
           f"{'Output CS':>10} {'Out SNR(dB)':>11} {'OK':>4}")
    report(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*11} {'-'*4}")

    results = {}

    for layer_name, data in activations.items():
        Q = data["Q"].float()
        K = data["K"].float()
        V = data["V"].float()
        head_dim = Q.shape[-1]
        scale = head_dim ** -0.5

        # FP32 ground truth
        scores_fp = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_fp = F.softmax(scores_fp, dim=-1)
        out_fp = torch.matmul(attn_fp, V)

        # INT8 path (simulates our kernel)
        Q_i8, sQ = quantize_per_tensor(Q)
        K_i8, sK = quantize_per_tensor(K)
        V_i8, sV = quantize_per_tensor(V)

        # INT8 QK^T: integer matmul -> INT32 -> float -> apply combined scale
        qk_int32 = torch.matmul(Q_i8.int(), K_i8.int().transpose(-2, -1))
        scores_int8 = qk_int32.float() * (sQ * sK * scale)

        # Softmax in FP32 (same as kernel)
        attn_int8 = F.softmax(scores_int8, dim=-1)

        # Multiply with dequantized V
        V_deq = dequantize(V_i8, sV)
        out_int8 = torch.matmul(attn_int8, V_deq)

        # Metrics
        cs_qkt = cosine_sim(scores_fp, scores_int8)
        cs_softmax = cosine_sim(attn_fp, attn_int8)
        cs_output = cosine_sim(out_fp, out_int8)
        out_snr = (10 * torch.log10(
            out_fp.pow(2).mean() / (out_fp - out_int8).pow(2).mean().clamp(min=1e-20)
        )).item()

        verdict = "Y" if cs_output > 0.999 else "~" if cs_output > 0.99 else "N"

        report(f"  {layer_name:<12} {cs_qkt:>7.5f}  {cs_softmax:>9.5f}  "
               f"{cs_output:>9.5f}  {out_snr:>9.1f}dB  {verdict}")

        results[layer_name] = {
            "cs_qkt": cs_qkt, "cs_softmax": cs_softmax,
            "cs_output": cs_output, "out_snr_db": out_snr,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-4b: Per-token quantization (the fix)
# ══════════════════════════════════════════════════════════════════════════

def analyze_per_token_attention(activations, model_key):
    """Same as Step 0-4, but using per-token quantization instead of per-tensor.

    Per-token means each token (row) in Q/K/V gets its own scale factor.
    This prevents one outlier token from destroying precision for all others.

    In the INT8 QK^T computation:
      score[i][j] = (Q_int8[i] @ K_int8[j]) * scale_Q[i] * scale_K[j] * attn_scale

    Each element of QK^T gets a different combined scale (product of row i and col j scales).
    """
    report(f"\n  --- Per-token quantization: error propagation ({model_key}) ---")
    report(f"  Path: Q,K,V -> INT8(per-token) -> QK^T(INT32) -> float -> softmax -> xV -> out")
    report(f"  Each token has its own scale (no single outlier can ruin global precision)\n")

    report(f"  {'Layer':<12} {'QKT CS':>8} {'Softmax CS':>10} "
           f"{'Output CS':>10} {'Out SNR(dB)':>11} {'OK':>4}")
    report(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*11} {'-'*4}")

    results = {}

    for layer_name, data in activations.items():
        Q = data["Q"].float()
        K = data["K"].float()
        V = data["V"].float()
        head_dim = Q.shape[-1]
        attn_scale = head_dim ** -0.5

        # FP32 ground truth
        scores_fp = torch.matmul(Q, K.transpose(-2, -1)) * attn_scale
        attn_fp = F.softmax(scores_fp, dim=-1)
        out_fp = torch.matmul(attn_fp, V)

        # Per-token INT8 path
        Q_i8, sQ = quantize_per_token(Q)  # sQ: [batch, heads, seq, 1]
        K_i8, sK = quantize_per_token(K)  # sK: [batch, heads, seq, 1]
        V_i8, sV = quantize_per_token(V)  # sV: [batch, heads, seq, 1]

        # INT8 QK^T with per-token scales:
        # result[b,h,i,j] = sum_d(Q_i8[b,h,i,d] * K_i8[b,h,j,d]) * sQ[b,h,i] * sK[b,h,j] * attn_scale
        qk_int32 = torch.matmul(Q_i8.int(), K_i8.int().transpose(-2, -1))
        # sQ is [b,h,seq,1], sK is [b,h,seq,1] -> sK^T is [b,h,1,seq]
        # Broadcasting: [b,h,seq,1] * [b,h,1,seq] -> [b,h,seq,seq]
        scale_matrix = sQ * sK.transpose(-2, -1) * attn_scale
        scores_int8 = qk_int32.float() * scale_matrix

        # Softmax in FP32
        attn_int8 = F.softmax(scores_int8, dim=-1)

        # attn x V(dequantized): V_deq = V_i8 * sV (per-token broadcast)
        V_deq = V_i8.float() * sV
        out_int8 = torch.matmul(attn_int8, V_deq)

        # Metrics
        cs_qkt = cosine_sim(scores_fp, scores_int8)
        cs_softmax = cosine_sim(attn_fp, attn_int8)
        cs_output = cosine_sim(out_fp, out_int8)
        out_snr = (10 * torch.log10(
            out_fp.pow(2).mean() / (out_fp - out_int8).pow(2).mean().clamp(min=1e-20)
        )).item()

        verdict = "Y" if cs_output > 0.999 else "~" if cs_output > 0.99 else "N"

        report(f"  {layer_name:<12} {cs_qkt:>7.5f}  {cs_softmax:>9.5f}  "
               f"{cs_output:>9.5f}  {out_snr:>9.1f}dB  {verdict}")

        results[layer_name] = {
            "cs_qkt": cs_qkt, "cs_softmax": cs_softmax,
            "cs_output": cs_output, "out_snr_db": out_snr,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Side-by-side comparison
# ══════════════════════════════════════════════════════════════════════════

def compare_models(all_results, all_pertoken_results):
    """Print side-by-side comparison: per-tensor vs per-token, small vs large."""
    report("\n" + "=" * 70)
    report("  COMPARISON: per-tensor vs per-token quantization")
    report("=" * 70)

    report(f"\n  Attention output cosine similarity (the key metric):\n")
    report(f"  {'Model':<16} {'Method':<12} {'Worst CS':>10} {'Avg CS':>10} {'Verdict':>8}")
    report(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

    for model_key in all_results.keys():
        # Per-tensor
        cs_list = [r["cs_output"] for r in all_results[model_key].values()]
        worst = min(cs_list)
        avg = sum(cs_list) / len(cs_list)
        verdict = "PASS" if worst > 0.999 else "MARGINAL" if worst > 0.99 else "FAIL"
        report(f"  {model_key:<16} {'per-tensor':<12} {worst:>9.6f}  {avg:>9.6f}  {verdict:>8}")

        # Per-token
        if model_key in all_pertoken_results:
            cs_list_pt = [r["cs_output"] for r in all_pertoken_results[model_key].values()]
            worst_pt = min(cs_list_pt)
            avg_pt = sum(cs_list_pt) / len(cs_list_pt)
            verdict_pt = "PASS" if worst_pt > 0.999 else "MARGINAL" if worst_pt > 0.99 else "FAIL"
            report(f"  {'':<16} {'per-token':<12} {worst_pt:>9.6f}  {avg_pt:>9.6f}  {verdict_pt:>8}")

    report(f"\n  Interpretation:")
    report(f"    per-tensor: one scale for entire Q/K/V tensor (current kernel)")
    report(f"    per-token:  one scale per token row (proposed improvement)")
    report(f"    PASS = cosine sim > 0.999, MARGINAL = > 0.99, FAIL = < 0.99")


# ══════════════════════════════════════════════════════════════════════════
#  Conclusion
# ══════════════════════════════════════════════════════════════════════════

def write_conclusion(all_results, all_pertoken_results):
    report("\n" + "=" * 70)
    report("  CONCLUSION")
    report("=" * 70)

    # Per-tensor results
    tensor_cs = []
    for attn_results in all_results.values():
        tensor_cs.extend(r["cs_output"] for r in attn_results.values())
    tensor_worst = min(tensor_cs)

    # Per-token results
    token_cs = []
    for attn_results in all_pertoken_results.values():
        token_cs.extend(r["cs_output"] for r in attn_results.values())
    token_worst = min(token_cs) if token_cs else 0.0

    report(f"\n  Per-tensor quantization:")
    report(f"    Worst cosine sim: {tensor_worst:.6f}  {'PASS' if tensor_worst > 0.999 else 'FAIL'}")
    report(f"\n  Per-token quantization:")
    report(f"    Worst cosine sim: {token_worst:.6f}  {'PASS' if token_worst > 0.999 else 'FAIL'}")

    if tensor_worst < 0.999 and token_worst > 0.999:
        report(f"\n  FINDING: Per-tensor fails on large models, per-token fixes it.")
        report(f"  -> Per-token quantization eliminates the outlier problem.")
        report(f"  -> Kernel change needed: scale becomes [batch*heads*seq] array")
        report(f"     instead of single float.")
        report(f"  -> QK^T element [i][j] scaled by: scale_Q[i] * scale_K[j] * attn_scale")
        report(f"  -> This is what SageAttention does (ICLR 2025).")
    elif tensor_worst > 0.999:
        report(f"\n  Per-tensor is already sufficient for all tested models.")
        report(f"  -> No need for per-token in this regime.")
    else:
        report(f"\n  Both methods struggle. Consider per-channel or mixed-precision.")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: INT8 quantization precision analysis")
    parser.add_argument("--model", default="all", choices=["gpt2", "opt-6.7b", "all"],
                        help="Which model(s) to analyze")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.model == "all":
        model_keys = ["gpt2", "opt-6.7b"]
    else:
        model_keys = [args.model]

    report("=" * 70)
    report("  Phase 0: INT8 Quantization Precision Analysis")
    report("  Do real transformer activations survive per-tensor INT8?")
    report("=" * 70)
    report(f"\n  Models to test: {model_keys}")
    report(f"  Quantization: per-tensor symmetric (scale = max|x| / 127)")
    report(f"  Key metric: attention output cosine similarity (>0.999 = pass)\n")

    all_dist_stats = {}
    all_quant_results = {}
    all_attn_results = {}
    all_pertoken_results = {}

    for model_key in model_keys:
        report("\n" + "=" * 70)
        report(f"  MODEL: {model_key}")
        report("=" * 70)

        activations = extract_activations(model_key)
        all_dist_stats[model_key] = analyze_distributions(activations, model_key)
        all_quant_results[model_key] = analyze_quantization(activations, model_key)
        all_attn_results[model_key] = analyze_attention_propagation(activations, model_key)
        all_pertoken_results[model_key] = analyze_per_token_attention(activations, model_key)

    # Side-by-side comparison
    compare_models(all_attn_results, all_pertoken_results)
    write_conclusion(all_attn_results, all_pertoken_results)

    # Grading reference
    report(f"\n  --- Grading reference ---")
    report(f"  CosSim > 0.9999: A (near-lossless, SageAttention reports ~1.0)")
    report(f"  CosSim > 0.999:  B (acceptable for inference)")
    report(f"  CosSim > 0.99:   C (noticeable degradation)")
    report(f"  CosSim < 0.99:   F (unacceptable)")
    report(f"\n  Kurtosis = 3: normal (torch.randn)")
    report(f"  Kurtosis > 5: heavy tails / outliers present")
    report(f"  RangeUtil < 0.1: outliers waste most of INT8 dynamic range")

    # Save report
    report_path = os.path.join(RESULTS_DIR, "phase0_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    report(f"\n[Saved] {report_path}")


if __name__ == "__main__":
    main()
