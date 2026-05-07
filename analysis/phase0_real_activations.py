# -*- coding: utf-8 -*-
"""Phase 0: INT8 量化精度分析 — 使用 GPT-2 真實 activation。

一次跑完 Phase 0 所有步驟：
  0-1: 從 GPT-2 抓真實 Q, K, V（多層）
  0-2: 分析分布特性（outlier, kurtosis, range utilization）
  0-3: Per-tensor INT8 量化，算 cosine similarity
  0-4: 模擬完整 attention 路徑誤差傳播
  0-5: 結論

Usage:
    # 在 Perlmutter login node 或 GPU node 上：
    module load pytorch
    pip install --user transformers  # 只需裝一次
    python analysis/phase0_real_activations.py

Output:
    analysis/results/phase0_report.txt
    analysis/results/real_activations.pt  (cached, 後續可重用)
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
LAYERS_TO_CAPTURE = [0, 3, 6, 11]  # 抓不同深度的 layer
CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "real_activations.pt")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "results", "phase0_report.txt")

# 測試用的文本（多樣性：英文、程式碼風格、長句）
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
#  Step 0-1: 從 GPT-2 抓真實 Q, K, V
# ══════════════════════════════════════════════════════════════════════════

def extract_activations():
    """載入 GPT-2，用 hook 攔截多層的 Q, K, V projection 輸出。"""
    report("=" * 70)
    report("  Step 0-1: 從 GPT-2 抽取真實 Q, K, V activations")
    report("=" * 70)

    # Check cache
    if os.path.exists(CACHE_PATH):
        report(f"\n  [Cache] 載入已存的 activations: {CACHE_PATH}")
        data = torch.load(CACHE_PATH, map_location="cpu")
        report(f"  Layers captured: {list(data.keys())}")
        return data

    from transformers import AutoModel, AutoTokenizer

    report(f"\n  Model: {MODEL_NAME}")
    report(f"  Layers to capture: {LAYERS_TO_CAPTURE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()

    # GPT-2 用 float32，我們保持原樣抓取
    device = "cpu"  # login node 夠用

    # Tokenize all texts together for longer sequence
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(TEST_TEXTS, return_tensors="pt", padding=True, truncation=True,
                       max_length=512)

    # Register hooks to capture Q, K, V
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            # GPT-2 的 attention: c_attn 是 Q/K/V 合併的 linear projection
            # output of c_attn = [batch, seq, 3 * d_model]
            # 我們要在 attention forward 之前抓，所以 hook c_attn
            pass
        return hook_fn

    # 更好的方式：直接 hook attention module 的 forward，手動算 Q/K/V
    activations = {}

    def make_attn_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # GPT2Attention forward: 先過 c_attn 得到 qkv
            # 我們重新算一次來抓 Q, K, V（因為 GPT-2 沒有分開暴露）
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return

            # c_attn projects to [batch, seq, 3*d_model]
            qkv = module.c_attn(hidden_states)
            batch_size, seq_len, _ = qkv.shape
            d_model = module.embed_dim
            num_heads = module.num_heads
            head_dim = d_model // num_heads

            # Split into Q, K, V and reshape to [batch, heads, seq, head_dim]
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
#  Step 0-2: 分析分布特性
# ══════════════════════════════════════════════════════════════════════════

def analyze_distributions(activations):
    report("\n" + "=" * 70)
    report("  Step 0-2: 真實 activation 分布特性")
    report("=" * 70)

    report(f"\n  {'Layer':<12} {'Tensor':<4} {'Mean':>8} {'Std':>8} "
           f"{'AbsMax':>8} {'Kurtosis':>9} {'|x|>3σ':>7} {'RangeUtil':>9}")
    report(f"  {'─'*12} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*7} {'─'*9}")

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

            # Outlier percentage (|x| > 3σ)
            outlier_pct = (t.abs() > 3 * std).float().mean().item() * 100

            # Range utilization (mean|x| / max|x|)
            # 如果這個值很低，代表大部分值很小但有極端值把 range 撐開
            range_util = (t.abs().mean() / abs_max).item() if abs_max > 0 else 0

            report(f"  {layer_name:<12} {tensor_name:<4} {mean:>7.4f} {std:>7.4f} "
                   f"{abs_max:>7.3f} {kurtosis:>8.1f}  {outlier_pct:>5.2f}% {range_util:>8.3f}")

            all_stats[f"{layer_name}/{tensor_name}"] = {
                "mean": mean, "std": std, "abs_max": abs_max,
                "kurtosis": kurtosis, "outlier_pct": outlier_pct,
                "range_util": range_util,
            }

    report(f"\n  解讀：")
    report(f"    Kurtosis = 3: 正態分布（torch.randn）")
    report(f"    Kurtosis > 5: 有顯著 outlier，per-tensor quant 會浪費 range")
    report(f"    RangeUtil < 0.1: 極端值主導了 scale，大部分值只用到少數 INT8 bin")

    # Compare with torch.randn
    report(f"\n  對比 torch.randn (N(0,1)):  kurtosis=3.0, outlier=0.27%, range_util≈0.20")

    return all_stats


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-3: Per-tensor INT8 量化，算 cosine similarity
# ══════════════════════════════════════════════════════════════════════════

def quantize_per_tensor(t):
    """Per-tensor symmetric INT8 quantization（跟我們 kernel 一模一樣的做法）。"""
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
    """Flatten 後算 cosine similarity。"""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def analyze_quantization(activations):
    report("\n" + "=" * 70)
    report("  Step 0-3: Per-tensor INT8 量化精度")
    report("=" * 70)
    report(f"\n  量化方式: scale = max|tensor| / 127 (per-tensor symmetric)")
    report(f"  跟我們 CUDA kernel 完全一致\n")

    report(f"  {'Layer':<12} {'Tensor':<4} {'CosSim':>8} {'RMSE':>10} "
           f"{'MaxErr':>8} {'SNR(dB)':>8} {'判定':>6}")
    report(f"  {'─'*12} {'─'*4} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*6}")

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

            # 判定
            if cs > 0.9999:
                verdict = "優"
            elif cs > 0.999:
                verdict = "良"
            elif cs > 0.99:
                verdict = "可"
            else:
                verdict = "差"

            report(f"  {layer_name:<12} {tensor_name:<4} {cs:>7.5f} {rmse:>9.6f} "
                   f"{max_err:>7.4f} {snr:>7.1f}  {verdict}")

            results[f"{layer_name}/{tensor_name}"] = {
                "cosine_sim": cs, "rmse": rmse, "max_err": max_err,
                "snr_db": snr, "scale": scale.item(),
            }

    report(f"\n  標準（參考 SageAttention）：")
    report(f"    CosSim > 0.9999: 幾乎無損（SageAttention 報告 ≈1.0）")
    report(f"    CosSim > 0.999:  可接受")
    report(f"    CosSim < 0.999:  需要改進量化策略")

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-4: 完整 attention 路徑誤差傳播
# ══════════════════════════════════════════════════════════════════════════

def analyze_attention_propagation(activations):
    report("\n" + "=" * 70)
    report("  Step 0-4: 誤差在 Attention 中的傳播")
    report("=" * 70)
    report(f"\n  路徑: Q,K,V → INT8 quant → QK^T → softmax → ×V → output")
    report(f"  比較: FP32 完整精度 vs INT8 量化路徑\n")

    report(f"  {'Layer':<12} {'QKT CosSim':>10} {'Softmax CS':>10} "
           f"{'Output CS':>10} {'Out SNR(dB)':>11} {'判定':>4}")
    report(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*11} {'─'*4}")

    results = {}

    for layer_name, data in activations.items():
        Q = data["Q"].float()
        K = data["K"].float()
        V = data["V"].float()
        head_dim = Q.shape[-1]
        scale = head_dim ** -0.5

        # ── FP32 ground truth ──
        scores_fp = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_fp = F.softmax(scores_fp, dim=-1)
        out_fp = torch.matmul(attn_fp, V)

        # ── INT8 path (模擬我們 kernel 的做法) ──
        Q_i8, sQ = quantize_per_tensor(Q)
        K_i8, sK = quantize_per_tensor(K)
        V_i8, sV = quantize_per_tensor(V)

        # INT8 QK^T: 整數乘法 → INT32 → float → ×(sQ * sK * attn_scale)
        Q_i32 = Q_i8.int()
        K_i32 = K_i8.int()
        qk_int32 = torch.matmul(Q_i32, K_i32.transpose(-2, -1))
        scores_int8 = qk_int32.float() * (sQ * sK * scale)

        # Softmax in FP32 (跟 kernel 一樣)
        attn_int8 = F.softmax(scores_int8, dim=-1)

        # × V (dequantized)
        V_deq = dequantize(V_i8, sV)
        out_int8 = torch.matmul(attn_int8, V_deq)

        # ── Metrics ──
        cs_qkt = cosine_sim(scores_fp, scores_int8)
        cs_softmax = cosine_sim(attn_fp, attn_int8)
        cs_output = cosine_sim(out_fp, out_int8)
        out_diff = (out_fp - out_int8).abs()
        out_snr = (10 * torch.log10(
            out_fp.pow(2).mean() / out_diff.pow(2).mean().clamp(min=1e-20)
        )).item()

        verdict = "✓" if cs_output > 0.999 else "△" if cs_output > 0.99 else "✗"

        report(f"  {layer_name:<12} {cs_qkt:>9.5f}  {cs_softmax:>9.5f}  "
               f"{cs_output:>9.5f}  {out_snr:>9.1f}dB  {verdict}")

        results[layer_name] = {
            "cs_qkt": cs_qkt, "cs_softmax": cs_softmax,
            "cs_output": cs_output, "out_snr_db": out_snr,
        }

    report(f"\n  觀察：")
    report(f"    - QK^T cosine similarity: 量化對矩陣乘法的影響")
    report(f"    - Softmax cosine similarity: 非線性放大效果")
    report(f"    - Output cosine similarity: 最終精度（這是最重要的數字）")

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Step 0-5: 結論 + 對比 torch.randn
# ══════════════════════════════════════════════════════════════════════════

def compare_with_randn(activations):
    report("\n" + "=" * 70)
    report("  Step 0-5: 對比 torch.randn（我們現有的測試）")
    report("=" * 70)

    # 取一個 layer 的 shape 來生成相同大小的隨機輸入
    sample_layer = list(activations.values())[0]
    shape = sample_layer["Q"].shape
    head_dim = shape[-1]
    scale = head_dim ** -0.5

    report(f"\n  Shape: {list(shape)}, head_dim={head_dim}")
    report(f"  比較真實 GPT-2 activation vs torch.randn 的量化精度差異\n")

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

    # real activation (use first captured layer)
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

    report(f"  {'Input type':<25} {'Output CosSim':>12} {'判定':>6}")
    report(f"  {'─'*25} {'─'*12} {'─'*6}")
    report(f"  {'torch.randn (N(0,1))':<25} {cs_randn:>11.6f}  {'✓' if cs_randn > 0.999 else '✗'}")
    report(f"  {'GPT-2 real activation':<25} {cs_real:>11.6f}  {'✓' if cs_real > 0.999 else '✗'}")

    diff = cs_randn - cs_real
    report(f"\n  差距: {diff:+.6f}")
    if abs(diff) < 0.001:
        report(f"  → torch.randn 跟真實分布的精度差異很小，現有測試設計合理")
    elif diff > 0:
        report(f"  → 真實分布比 randn 更難量化（可能有 outlier），需注意")
    else:
        report(f"  → 真實分布反而更好量化（值比較集中）")


# ══════════════════════════════════════════════════════════════════════════
#  Final conclusion
# ══════════════════════════════════════════════════════════════════════════

def write_conclusion(quant_results, attn_results):
    report("\n" + "=" * 70)
    report("  結論")
    report("=" * 70)

    # Check if all outputs pass
    all_cs = [r["cs_output"] for r in attn_results.values()]
    min_cs = min(all_cs)
    avg_cs = sum(all_cs) / len(all_cs)

    report(f"\n  Attention output cosine similarity:")
    report(f"    最差: {min_cs:.6f}")
    report(f"    平均: {avg_cs:.6f}")

    if min_cs > 0.999:
        report(f"\n  ✅ 結論：Per-tensor INT8 量化在 GPT-2 真實分布下精度可接受。")
        report(f"     可以繼續進行 Phase 1（kernel 正確性驗證）和 Phase 2（效能測量）。")
        report(f"     Poster 可以宣稱：INT8 量化對 attention 的精度影響可忽略。")
    elif min_cs > 0.99:
        report(f"\n  △ 結論：Per-tensor INT8 有一定精度損失，但仍在可用範圍。")
        report(f"     建議在 poster 上明確標註精度數字和適用條件。")
        report(f"     如果要改善：考慮 per-token 量化或 K smoothing。")
    else:
        report(f"\n  ✗ 結論：Per-tensor INT8 精度不足，需要改進量化策略。")
        report(f"     建議：嘗試 per-token 量化（每個 token 獨立 scale）")
        report(f"     或 K smoothing（K -= K.mean(dim=seq)，參考 SageAttention）")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    report("╔══════════════════════════════════════════════════════════════════════╗")
    report("║  Phase 0: INT8 Quantization Precision Analysis                     ║")
    report("║  Using real GPT-2 activations                                      ║")
    report("╚══════════════════════════════════════════════════════════════════════╝")
    report(f"\n  Model: {MODEL_NAME} (12 layers, 12 heads, d=768, head_dim=64)")
    report(f"  Quantization: per-tensor symmetric (scale = max|x| / 127)")
    report(f"  Question: 這個量化精度足以支持我們的 INT8 kernel 嗎？\n")

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
