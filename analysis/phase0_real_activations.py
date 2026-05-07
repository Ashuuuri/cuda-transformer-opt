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
        "dtype": torch.float16,  # load in FP16 to fit in 40GB
        "device": "cuda",        # needs GPU
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
#  Side-by-side comparison
# ══════════════════════════════════════════════════════════════════════════

def compare_models(all_results):
    """Print side-by-side comparison of attention output cosine similarity."""
    report("\n" + "=" * 70)
    report("  COMPARISON: Small model vs Large model")
    report("=" * 70)

    report(f"\n  Attention output cosine similarity (the key metric):\n")
    report(f"  {'Model':<20} {'Worst CS':>10} {'Average CS':>10} {'Verdict':>8}")
    report(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")

    for model_key, attn_results in all_results.items():
        all_cs = [r["cs_output"] for r in attn_results.values()]
        worst = min(all_cs)
        avg = sum(all_cs) / len(all_cs)
        if worst > 0.999:
            verdict = "PASS"
        elif worst > 0.99:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"
        report(f"  {model_key:<20} {worst:>9.6f}  {avg:>9.6f}  {verdict:>8}")

    report(f"\n  Interpretation:")
    report(f"    PASS:     per-tensor INT8 is sufficient, proceed with kernel optimization")
    report(f"    MARGINAL: works but precision loss is noticeable, document in poster")
    report(f"    FAIL:     need per-token quantization or K-smoothing")

    # Check if outlier problem appears
    model_keys = list(all_results.keys())
    if len(model_keys) >= 2:
        cs_small = min(r["cs_output"] for r in all_results[model_keys[0]].values())
        cs_large = min(r["cs_output"] for r in all_results[model_keys[1]].values())
        gap = cs_small - cs_large
        report(f"\n  Precision gap (small - large): {gap:+.6f}")
        if gap > 0.01:
            report(f"  -> Large model is significantly harder to quantize (outlier effect)")
            report(f"  -> This confirms LLM.int8() findings: outliers emerge with scale")
        elif gap > 0.001:
            report(f"  -> Slight degradation at larger scale, but still manageable")
        else:
            report(f"  -> No significant difference: per-tensor INT8 works at both scales")


# ══════════════════════════════════════════════════════════════════════════
#  Conclusion
# ══════════════════════════════════════════════════════════════════════════

def write_conclusion(all_results):
    report("\n" + "=" * 70)
    report("  CONCLUSION")
    report("=" * 70)

    # Collect all cosine similarities across all models
    all_cs = []
    for attn_results in all_results.values():
        all_cs.extend(r["cs_output"] for r in attn_results.values())

    worst = min(all_cs)
    avg = sum(all_cs) / len(all_cs)

    report(f"\n  Overall attention output cosine similarity:")
    report(f"    Worst across all models/layers: {worst:.6f}")
    report(f"    Average:                        {avg:.6f}")

    if worst > 0.999:
        report(f"\n  PASS: Per-tensor INT8 quantization is precise enough.")
        report(f"  -> Our INT8 kernel's quantization scheme is validated.")
        report(f"  -> Proceed to Phase 1 (correctness) and Phase 2 (performance).")
        report(f"  -> Poster claim: 'INT8 quantization error is negligible for attention'")
    elif worst > 0.99:
        report(f"\n  MARGINAL: Noticeable precision loss but within usable range.")
        report(f"  -> Report exact numbers. Note which layers/models degrade.")
        report(f"  -> Consider: per-token quant or K-smoothing (ref: SageAttention)")
        report(f"  -> Poster: 'INT8 works for small/medium models; larger models")
        report(f"     benefit from per-token quantization'")
    else:
        report(f"\n  FAIL: Per-tensor INT8 precision is insufficient for large models.")
        report(f"  -> Per-tensor quant breaks down due to outlier features.")
        report(f"  -> Must implement per-token quantization for real-world use.")
        report(f"  -> Poster: document this as a finding (matches LLM.int8() results)")


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

    for model_key in model_keys:
        report("\n" + "=" * 70)
        report(f"  MODEL: {model_key}")
        report("=" * 70)

        activations = extract_activations(model_key)
        all_dist_stats[model_key] = analyze_distributions(activations, model_key)
        all_quant_results[model_key] = analyze_quantization(activations, model_key)
        all_attn_results[model_key] = analyze_attention_propagation(activations, model_key)

    # Side-by-side comparison (if multiple models)
    if len(model_keys) > 1:
        compare_models(all_attn_results)

    write_conclusion(all_attn_results)

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
