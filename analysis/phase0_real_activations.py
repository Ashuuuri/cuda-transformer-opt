# -*- coding: utf-8 -*-
"""Phase 0: INT8 quantization precision analysis using real GPT-2 activations.

Runs all Phase 0 steps in one shot:
  0-1: Extract real Q, K, V from GPT-2 (multiple layers)
  0-2: Analyze distribution characteristics (outlier, kurtosis, range utilization)
  0-3: Per-tensor INT8 quantization, compute cosine similarity
  0-4: Simulate full attention path error propagation
  0-5: Conclusion

Usage:
    module load pytorch
    pip install --user transformers  # one-time setup
    python analysis/phase0_real_activations.py

Output:
    analysis/results/phase0_report.txt
    analysis/results/real_activations.pt  (cached for reuse)
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════
MODEL_NAME = "gpt2"  # 124M params, 12 layers, 12 heads, d_model=768, head_dim=64
LAYERS_TO_CAPTURE = [0, 3, 6, 11]  # capture layers at different depths
CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "real_activations.pt")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "results", "phase0_report.txt")

# Diverse test texts (English prose, code-like, math-like)
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
#  Step 0-1: Extract real Q, K, V from GPT-2
# ══════════════════════════════════════════════════════════════════════════

def extract_activations():
    """Load GPT-2 and intercept Q, K, V projections at multiple layers via hooks."""
    report("=" * 70)
    report("  Step 0-1: Extract real Q, K, V activations from GPT-2")
    report("=" * 70)

    # Check cache
    if os.path.exists(CACHE_PATH):
        report(f"\n  [Cache] Loading saved activations: {CACHE_PATH}")
        data = torch.load(CACHE_PATH, map_location="cpu")
        report(f"  Layers captured: {list(data.keys())}")
        return data

    from transformers import AutoModel, AutoTokenizer

    report(f"\n  Model: {MODEL_NAME}")
    report(f"  Layers to capture: {LAYERS_TO_CAPTURE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()

    # GPT-2 uses float32; we keep it as-is for extraction
    device = "cpu"  # login node is sufficient

    # Tokenize all texts together for a longer sequence
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(TEST_TEXTS, return_tensors="pt", padding=True, truncation=True,
                       max_length=512)

    activations = {}

    def make_attn_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # GPT2Attention: c_attn projects hidden_states to [batch, seq, 3*d_model]
            # We recompute Q/K/V split here since GPT-2 doesn't expose them separately.
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return

            qkv = module.c_attn(hidden_states)
            batch_size, seq_len, _ = qkv.shape
            d_model = module.embed_dim
            num_heads = module.num_heads
            head_dim = d_model // num_heads

            # Split and reshape to [batch, heads, seq, head_dim]
            q, k, v = qkv.split(d_model, dim=2)
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            activations[f"layer_{layer_idx}"] = {
                "Q": q.detach().clone(),
                "K": k.detach().clone(),
                "V": v.detach().clone(),
                "shape": f"batch={batch_size}, heads={num_heads}, seq={seq_len}, head_dim={head_dim}",
            }
        return hook_fn

    hooks = []
    for layer_idx in LAYERS_TO_CAPTURE:
        h = model.h[layer_idx].attn.register_forward_hook(
            make_attn_hook(layer_idx), with_kwargs=True)
        hooks.append(h)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Report
    for key, val in activations.items():
        report(f"  {key}: {val['shape']}, Q range=[{val['Q'].min():.3f}, {val['Q'].max():.3f}]")

    # Save cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    torch.save(activations, CACHE_PATH)
    report(f"\n  [Saved] {CACHE_PATH}")

    return activations


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-2: Analyze distribution characteristics
# ══════════════════════════════════════════════════════════════════════════

def analyze_distributions(activations):
    report("\n" + "=" * 70)
    report("  Step 0-2: Real activation distribution characteristics")
    report("=" * 70)

    report(f"\n  {'Layer':<12} {'Tensor':<4} {'Mean':>8} {'Std':>8} "
           f"{'AbsMax':>8} {'Kurtosis':>9} {'|x|>3s':>7} {'RangeUtil':>9}")
    report(f"  {'---'*4:<12} {'---':<4} {'---':>8} {'---':>8} "
           f"{'---':>8} {'---':>9} {'---':>7} {'---':>9}")

    all_stats = {}

    for layer_name, data in activations.items():
        for tensor_name in ["Q", "K", "V"]:
            t = data[tensor_name].float()
            mean = t.mean().item()
            std = t.std().item()
            abs_max = t.abs().max().item()

            # Kurtosis (normal distribution = 3)
            if std > 0:
                kurtosis = ((t - mean) / std).pow(4).mean().item()
            else:
                kurtosis = 0.0

            # Outlier percentage (|x| > 3 sigma)
            outlier_pct = (t.abs() > 3 * std).float().mean().item() * 100

            # Range utilization: mean|x| / max|x|
            # Low value means outliers dominate the scale, wasting most INT8 bins.
            range_util = (t.abs().mean() / abs_max).item() if abs_max > 0 else 0

            report(f"  {layer_name:<12} {tensor_name:<4} {mean:>7.4f} {std:>7.4f} "
                   f"{abs_max:>7.3f} {kurtosis:>8.1f}  {outlier_pct:>5.2f}% {range_util:>8.3f}")

            all_stats[f"{layer_name}/{tensor_name}"] = {
                "mean": mean, "std": std, "abs_max": abs_max,
                "kurtosis": kurtosis, "outlier_pct": outlier_pct,
                "range_util": range_util,
            }

    report(f"\n  Interpretation:")
    report(f"    Kurtosis = 3: normal distribution (torch.randn)")
    report(f"    Kurtosis > 5: significant outliers, per-tensor quant wastes range")
    report(f"    RangeUtil < 0.1: extreme values dominate scale, most values use few INT8 bins")
    report(f"\n  Reference (torch.randn N(0,1)): kurtosis=3.0, outlier=0.27%, range_util~0.20")

    return all_stats


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-3: Per-tensor INT8 quantization, compute cosine similarity
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


def analyze_quantization(activations):
    report("\n" + "=" * 70)
    report("  Step 0-3: Per-tensor INT8 quantization precision")
    report("=" * 70)
    report(f"\n  Method: scale = max|tensor| / 127 (per-tensor symmetric)")
    report(f"  Identical to our CUDA kernel's quantization scheme\n")

    report(f"  {'Layer':<12} {'Tensor':<4} {'CosSim':>8} {'RMSE':>10} "
           f"{'MaxErr':>8} {'SNR(dB)':>8} {'Grade':>6}")
    report(f"  {'---'*4:<12} {'---':<4} {'---':>8} {'---':>10} "
           f"{'---':>8} {'---':>8} {'---':>6}")

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

    report(f"\n  Grading (ref: SageAttention, ICLR 2025):")
    report(f"    A: CosSim > 0.9999 (near-lossless, SageAttention reports ~1.0)")
    report(f"    B: CosSim > 0.999  (acceptable for inference)")
    report(f"    C: CosSim > 0.99   (noticeable degradation)")
    report(f"    F: CosSim < 0.99   (unacceptable, need better quant strategy)")

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-4: Full attention path error propagation
# ══════════════════════════════════════════════════════════════════════════

def analyze_attention_propagation(activations):
    report("\n" + "=" * 70)
    report("  Step 0-4: Error propagation through attention")
    report("=" * 70)
    report(f"\n  Path: Q,K,V -> INT8 quant -> QK^T -> softmax -> xV -> output")
    report(f"  Comparison: FP32 full-precision vs INT8 quantized path\n")

    report(f"  {'Layer':<12} {'QKT CosSim':>10} {'Softmax CS':>10} "
           f"{'Output CS':>10} {'Out SNR(dB)':>11} {'OK':>4}")
    report(f"  {'---'*4:<12} {'---':>10} {'---':>10} "
           f"{'---':>10} {'---':>11} {'---':>4}")

    results = {}

    for layer_name, data in activations.items():
        Q = data["Q"].float()
        K = data["K"].float()
        V = data["V"].float()
        head_dim = Q.shape[-1]
        scale = head_dim ** -0.5

        # -- FP32 ground truth --
        scores_fp = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_fp = F.softmax(scores_fp, dim=-1)
        out_fp = torch.matmul(attn_fp, V)

        # -- INT8 path (simulates our kernel's computation) --
        Q_i8, sQ = quantize_per_tensor(Q)
        K_i8, sK = quantize_per_tensor(K)
        V_i8, sV = quantize_per_tensor(V)

        # INT8 QK^T: integer matmul -> INT32 -> float -> multiply (sQ * sK * attn_scale)
        Q_i32 = Q_i8.int()
        K_i32 = K_i8.int()
        qk_int32 = torch.matmul(Q_i32, K_i32.transpose(-2, -1))
        scores_int8 = qk_int32.float() * (sQ * sK * scale)

        # Softmax stays in FP32 (same as kernel)
        attn_int8 = F.softmax(scores_int8, dim=-1)

        # Multiply with dequantized V
        V_deq = dequantize(V_i8, sV)
        out_int8 = torch.matmul(attn_int8, V_deq)

        # -- Metrics --
        cs_qkt = cosine_sim(scores_fp, scores_int8)
        cs_softmax = cosine_sim(attn_fp, attn_int8)
        cs_output = cosine_sim(out_fp, out_int8)
        out_diff = (out_fp - out_int8).abs()
        out_snr = (10 * torch.log10(
            out_fp.pow(2).mean() / out_diff.pow(2).mean().clamp(min=1e-20)
        )).item()

        verdict = "Y" if cs_output > 0.999 else "~" if cs_output > 0.99 else "N"

        report(f"  {layer_name:<12} {cs_qkt:>9.5f}  {cs_softmax:>9.5f}  "
               f"{cs_output:>9.5f}  {out_snr:>9.1f}dB  {verdict}")

        results[layer_name] = {
            "cs_qkt": cs_qkt, "cs_softmax": cs_softmax,
            "cs_output": cs_output, "out_snr_db": out_snr,
        }

    report(f"\n  What each column means:")
    report(f"    QKT CosSim:     quantization impact on the matmul itself")
    report(f"    Softmax CS:     nonlinear amplification of error through exp()")
    report(f"    Output CS:      final precision after full attention (the key number)")
    report(f"    Out SNR(dB):    signal-to-noise ratio (>40dB = excellent)")

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-5: Compare with torch.randn (our current test setup)
# ══════════════════════════════════════════════════════════════════════════

def compare_with_randn(activations):
    report("\n" + "=" * 70)
    report("  Step 0-5: Compare with torch.randn (our existing test)")
    report("=" * 70)

    # Use one layer's shape to generate random inputs of the same size
    sample_layer = list(activations.values())[0]
    shape = sample_layer["Q"].shape
    head_dim = shape[-1]
    scale = head_dim ** -0.5

    report(f"\n  Shape: {list(shape)}, head_dim={head_dim}")
    report(f"  Comparing: real GPT-2 activations vs torch.randn\n")

    torch.manual_seed(42)
    Q_rand = torch.randn(shape)
    K_rand = torch.randn(shape)
    V_rand = torch.randn(shape)

    # randn attention path
    scores_fp = torch.matmul(Q_rand, K_rand.transpose(-2, -1)) * scale
    attn_fp = F.softmax(scores_fp, dim=-1)
    out_fp = torch.matmul(attn_fp, V_rand)

    Q_i8, sQ = quantize_per_tensor(Q_rand)
    K_i8, sK = quantize_per_tensor(K_rand)
    V_i8, sV = quantize_per_tensor(V_rand)
    qk_int32 = torch.matmul(Q_i8.int(), K_i8.int().transpose(-2, -1))
    scores_int8 = qk_int32.float() * (sQ * sK * scale)
    attn_int8 = F.softmax(scores_int8, dim=-1)
    V_deq = dequantize(V_i8, sV)
    out_int8 = torch.matmul(attn_int8, V_deq)

    cs_randn = cosine_sim(out_fp, out_int8)

    # Real activation (use first captured layer)
    first_layer = list(activations.keys())[0]
    Q_real = activations[first_layer]["Q"].float()
    K_real = activations[first_layer]["K"].float()
    V_real = activations[first_layer]["V"].float()

    scores_fp2 = torch.matmul(Q_real, K_real.transpose(-2, -1)) * scale
    attn_fp2 = F.softmax(scores_fp2, dim=-1)
    out_fp2 = torch.matmul(attn_fp2, V_real)

    Q_i8r, sQr = quantize_per_tensor(Q_real)
    K_i8r, sKr = quantize_per_tensor(K_real)
    V_i8r, sVr = quantize_per_tensor(V_real)
    qk_int32r = torch.matmul(Q_i8r.int(), K_i8r.int().transpose(-2, -1))
    scores_int8r = qk_int32r.float() * (sQr * sKr * scale)
    attn_int8r = F.softmax(scores_int8r, dim=-1)
    V_deqr = dequantize(V_i8r, sVr)
    out_int8r = torch.matmul(attn_int8r, V_deqr)

    cs_real = cosine_sim(out_fp2, out_int8r)

    report(f"  {'Input type':<25} {'Output CosSim':>12} {'OK':>4}")
    report(f"  {'---'*8:<25} {'---':>12} {'---':>4}")
    report(f"  {'torch.randn (N(0,1))':<25} {cs_randn:>11.6f}  {'Y' if cs_randn > 0.999 else 'N'}")
    report(f"  {'GPT-2 real activation':<25} {cs_real:>11.6f}  {'Y' if cs_real > 0.999 else 'N'}")

    diff = cs_randn - cs_real
    report(f"\n  Gap: {diff:+.6f}")
    if abs(diff) < 0.001:
        report(f"  -> Minimal difference: torch.randn is a reasonable test proxy")
    elif diff > 0:
        report(f"  -> Real distribution is harder to quantize (likely has outliers)")
    else:
        report(f"  -> Real distribution is easier to quantize (values more concentrated)")


# ══════════════════════════════════════════════════════════════════════════
#  Final conclusion
# ══════════════════════════════════════════════════════════════════════════

def write_conclusion(quant_results, attn_results):
    report("\n" + "=" * 70)
    report("  Conclusion")
    report("=" * 70)

    # Check if all outputs pass
    all_cs = [r["cs_output"] for r in attn_results.values()]
    min_cs = min(all_cs)
    avg_cs = sum(all_cs) / len(all_cs)

    report(f"\n  Attention output cosine similarity:")
    report(f"    Worst:   {min_cs:.6f}")
    report(f"    Average: {avg_cs:.6f}")

    if min_cs > 0.999:
        report(f"\n  PASS: Per-tensor INT8 quantization precision is acceptable")
        report(f"        on real GPT-2 activations.")
        report(f"  -> Proceed to Phase 1 (kernel correctness) and Phase 2 (perf sweep).")
        report(f"  -> Poster claim: INT8 quantization error is negligible for attention.")
    elif min_cs > 0.99:
        report(f"\n  MARGINAL: Per-tensor INT8 has noticeable precision loss,")
        report(f"            but still within usable range.")
        report(f"  -> Report exact numbers on poster.")
        report(f"  -> Consider per-token quantization or K-smoothing for improvement.")
    else:
        report(f"\n  FAIL: Per-tensor INT8 precision is insufficient.")
        report(f"  -> Need better quantization strategy before claiming speedup.")
        report(f"  -> Try: per-token quantization (independent scale per token)")
        report(f"  -> Or: K-smoothing (K -= K.mean(dim=seq), ref: SageAttention)")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    report("=" * 70)
    report("  Phase 0: INT8 Quantization Precision Analysis")
    report("  Using real GPT-2 activations")
    report("=" * 70)
    report(f"\n  Model: {MODEL_NAME} (12 layers, 12 heads, d=768, head_dim=64)")
    report(f"  Quantization: per-tensor symmetric (scale = max|x| / 127)")
    report(f"  Question: Is this quantization precise enough for our INT8 kernel?\n")

    # Run all steps
    activations = extract_activations()
    dist_stats = analyze_distributions(activations)
    quant_results = analyze_quantization(activations)
    attn_results = analyze_attention_propagation(activations)
    compare_with_randn(activations)
    write_conclusion(quant_results, attn_results)

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    report(f"\n[Saved] {REPORT_PATH}")


if __name__ == "__main__":
    main()
