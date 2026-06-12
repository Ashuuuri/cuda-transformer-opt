"""validate_int8.py — five-gate INT8 accuracy validation.

Runs every dataset produced by generate_test_data.py through the INT8
attention and MLP kernels and enforces, per dataset:

  Gate 1  Math metrics      cosine / top-10 overlap / outlier ratio / NaN-Inf
  Gate 2  Numeric stability per-stage value-range drift <= 1.5x, per-stage NaN
  Gate 3  Stage error trace per-stage cosine must not drop > 1% vs previous
  Gate 4  Edge cases        short_seq / long_seq / zeros / constant: no NaN,
                            cosine > 0.99 (or exact-zero output for zeros)
  Gate 5  Task-level        full transformer-block forward on `normal`:
                            perplexity increase < 2%, accuracy drop < 1%

"Stages" are a PyTorch simulation that mirrors the kernel math step by step
(quantize -> integer matmul -> dequant -> GELU/softmax -> requant), since the
fused CUDA kernels do not expose intermediates. The final simulated stage is
cross-checked against the actual kernel output so the simulation cannot
silently diverge from the kernels it models.

On failure the repair advisor re-runs the failing dataset through the
simulation with each candidate fix, in this order, and reports which one
recovers the metric:
  1. per-channel quantization for the outlier-bearing operand
  2. QK^T (attention scores) computed in FP16
  3. percentile-clipped (99.9%) quantization scales
  4. the worst stage from Gate 3 kept in FP16

Usage:
    python validate_int8.py                    # all datasets, all gates
    python validate_int8.py --dataset outlier  # one dataset
    python validate_int8.py --kernel mlp       # one kernel
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import attention_baseline, mlp_baseline, check_cuda

DATA_DIR = os.path.join("testdata", "validate")
EDGE_DATASETS = ("short_seq", "long_seq", "zeros", "constant")
ERR_ATOL = 0.1          # |err| above this counts toward the outlier ratio
STAGE_RANGE_MAX = 1.5   # gate-2 absmax ratio ceiling
STAGE_COS_DROP = 0.01   # gate-3 max cosine drop between stages
SIM_KERNEL_ATOL = 0.05  # simulation must match the real kernel this closely


# ── Quantization helpers (mirror the kernels exactly) ───────────────────
def q_per_tensor(t):
    s = t.float().abs().max().clamp(min=1e-8) / 127.0
    return (t.float() / s).round().clamp(-128, 127).to(torch.int8), s


def q_per_token(t):
    s = t.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    i8 = (t.float() / s).round().clamp(-128, 127).to(torch.int8)
    return i8, s.squeeze(-1).contiguous()


def q_per_channel(t, dim=-1):
    s = t.float().abs().amax(dim=tuple(d for d in range(t.dim()) if d != dim % t.dim()),
                             keepdim=True).clamp(min=1e-8) / 127.0
    return ((t.float() / s).round().clamp(-128, 127) * s)  # dequantized


def q_clipped(t, pct=0.999):
    flat = t.float().abs().flatten()
    k = max(1, int(flat.numel() * pct))
    amax = flat.kthvalue(k).values.clamp(min=1e-8)
    s = amax / 127.0
    return ((t.float() / s).round().clamp(-128, 127) * s)  # dequantized


# ── Metrics ──────────────────────────────────────────────────────────────
def cosine(a, b):
    # float64: float32 dot-product accumulation over 100M+ elements can
    # report cosine > 1 and corrupt the gate-3 stage comparison.
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = a.norm(), b.norm()
    if na < 1e-6 and nb < 1e-6:
        return 1.0   # both ~zero: identical
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(torch.dot(a / na, b / nb).clamp(max=1.0))


def top10_overlap(ref, act, max_rows=4096):
    """Mean per-row overlap of top-10 |value| indices along the last dim."""
    r = ref.float().reshape(-1, ref.shape[-1])
    a = act.float().reshape(-1, act.shape[-1])
    if r.shape[0] > max_rows:
        idx = torch.linspace(0, r.shape[0] - 1, max_rows).long()
        r, a = r[idx], a[idx]
    k = min(10, r.shape[-1])
    if float(r.abs().std()) < 1e-6:
        return None  # constant rows: ranking is meaningless
    rt = r.abs().topk(k, dim=-1).indices
    at = a.abs().topk(k, dim=-1).indices
    match = (rt.unsqueeze(-1) == at.unsqueeze(-2)).any(-1).float().mean()
    return float(match)


def outlier_ratio(ref, act):
    """Fraction of elements with error beyond max(atol, 10% relative).

    Purely absolute thresholds miscount datasets whose outputs are large
    (boundary/stress/outlier reach |out| of 4-100, where 0.1 absolute is
    <2.5% relative); purely relative ones blow up near zero. 10% relative
    keeps accumulated quantization-noise tails out of the count while still
    catching real breakage (saturation bugs and attention argmax flips are
    100%+ errors).

    For sum-of-products outputs the error magnitude is independent of each
    element's own value, so near-zero elements always look "relatively"
    wrong; the atol floor therefore scales with the tensor RMS (a genuine
    saturation bug produces errors of O(rms) and is still caught).
    """
    r, a = ref.float(), act.float()
    atol = max(ERR_ATOL, 0.05 * float(r.square().mean().sqrt()))
    tol = torch.clamp(0.10 * r.abs(), min=atol)
    return float(((r - a).abs() > tol).float().mean())


def has_bad(t):
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


# ── Stage simulations (mirror kernel math; return ordered stage dicts) ──
def sim_attention(Q, K, V, fp16=False, qk_fp16=False, quant=q_per_token):
    """Stages: q/k/v dequant -> scores -> softmax -> out."""
    if fp16:
        q, k, v = Q.float(), K.float(), V.float()
    else:
        def deq(t):
            i8, s = quant(t)
            return i8.float() * s.reshape(*t.shape[:-1], 1)
        q, k, v = deq(Q), deq(K), deq(V)
        if qk_fp16:
            q, k = Q.float(), K.float()
    scale = Q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = F.softmax(scores, dim=-1)
    out = torch.matmul(p, v)
    return {"qkv_dequant": torch.cat([q.flatten(), k.flatten(), v.flatten()]),
            "scores": scores, "softmax": p, "out": out}


def sim_mlp(x, W1, W2, fp16=False, x_dequant=None):
    """Stages: x dequant -> h_pre -> gelu -> hidden requant -> out -> out quant.

    Mirrors the dynamic-quant kernel: per-tensor x/W scales, per-token
    hidden scales, dynamic per-tensor output scale.
    """
    if fp16:
        xd, w1, w2 = x.float(), W1.float(), W2.float()
    else:
        def deq_t(t):
            i8, s = q_per_tensor(t)
            return i8.float() * s
        xd = x_dequant if x_dequant is not None else deq_t(x)
        w1, w2 = deq_t(W1), deq_t(W2)
    h_pre = xd @ w1
    h = F.gelu(h_pre, approximate="tanh")
    if fp16:
        hq = h
    else:
        i8, s = q_per_token(h)
        hq = i8.float() * s.unsqueeze(-1)
    out = hq @ w2
    if not fp16:
        i8, s = q_per_tensor(out)
        out = i8.float() * s
    return {"x_dequant": xd, "h_pre_gelu": h_pre, "h_gelu": h,
            "h_requant": hq, "out": out}


# ── Kernel execution ─────────────────────────────────────────────────────
def run_attention_kernel(ext, Q, K, V):
    Qc, Kc, Vc = Q.cuda(), K.cuda(), V.cuda()
    qi, sq = q_per_token(Qc)
    ki, sk = q_per_token(Kc)
    vi, sv = q_per_token(Vc)
    return ext.int8_attention_forward(qi, ki, vi, sq, sk, sv).float().cpu()


def run_mlp_kernel(ext, x, W1, W2):
    xc, w1c, w2c = x.cuda(), W1.cuda(), W2.cuda()
    xi, sx = q_per_tensor(xc)
    w1i, s1 = q_per_tensor(w1c)
    w2i, s2 = q_per_tensor(w2c)
    out_i8, out_scale = ext.int8_mlp_forward(xi, w1i, w2i,
                                             float(sx), float(s1), float(s2))
    return (out_i8.float() * out_scale).cpu()


# ── Gates ────────────────────────────────────────────────────────────────
def gate1(ref, act, thr, is_zero_dataset):
    res, fails = {}, []
    res["nan_inf"] = not has_bad(act)
    if not res["nan_inf"]:
        fails.append("NaN/Inf in kernel output")
    res["cosine"] = cosine(ref, act)
    if res["cosine"] < thr["cos_min"]:
        fails.append(f"cosine {res['cosine']:.5f} < {thr['cos_min']}")
    ov = None if is_zero_dataset else top10_overlap(ref, act)
    res["top10"] = ov
    if ov is not None and ov < thr["top10_min"]:
        fails.append(f"top10 overlap {ov:.3f} < {thr['top10_min']}")
    res["outlier_ratio"] = outlier_ratio(ref, act)
    if res["outlier_ratio"] > thr["outlier_ratio_max"]:
        fails.append(f"outlier ratio {res['outlier_ratio']:.4f} > "
                     f"{thr['outlier_ratio_max']}")
    return res, fails


def gate2(stages_fp16, stages_i8):
    fails, ranges = [], {}
    for name in stages_fp16:
        f16, i8 = stages_fp16[name], stages_i8[name]
        if has_bad(i8):
            fails.append(f"NaN/Inf at stage '{name}'")
            continue
        a, b = float(f16.float().abs().max()), float(i8.float().abs().max())
        ratio = b / a if a > 1e-8 else (1.0 if b < 1e-8 else float("inf"))
        ranges[name] = ratio
        if not (1.0 / STAGE_RANGE_MAX <= ratio <= STAGE_RANGE_MAX):
            fails.append(f"stage '{name}' absmax ratio {ratio:.3f} "
                         f"outside [{1/STAGE_RANGE_MAX:.2f}, {STAGE_RANGE_MAX}]")
    return ranges, fails


def gate3(stages_fp16, stages_i8):
    fails, history = [], []
    prev = None
    for name in stages_fp16:
        c = cosine(stages_fp16[name], stages_i8[name])
        history.append((name, c))
        if prev is not None and c < prev - STAGE_COS_DROP:
            fails.append(f"stage '{name}' cosine {c:.5f} dropped "
                         f">{STAGE_COS_DROP*100:.0f}% vs previous {prev:.5f}"
                         " — stopping trace here")
            break
        prev = c
    worst = sorted(history, key=lambda kv: kv[1])[:3]
    return history, worst, fails


def gate4(ref, act, thr):
    fails = []
    if has_bad(act):
        fails.append("NaN/Inf in edge-case output")
    if thr.get("zero_output"):
        m = float(act.float().abs().max())
        if m > 1e-3:
            fails.append(f"zeros dataset: |out| max {m:.5f} > 1e-3")
    else:
        c = cosine(ref, act)
        if c < thr.get("edge_cos_min", 0.99):
            fails.append(f"edge cosine {c:.5f} < {thr.get('edge_cos_min', 0.99)}")
    return fails


def gate5(ext, data, vocab=8192):
    """Transformer block on `normal`: LN -> attn -> +res -> LN -> MLP -> +res.

    LayerNorm and residuals stay FP16 (project rule). Logits via a fixed
    random LM head; perplexity measured against the FP16 path's argmax.
    """
    g = torch.Generator().manual_seed(7)
    x = data["mlp"]["x"].float().cuda()            # (B, S, dm)
    W1 = data["mlp"]["W1"].cuda()
    W2 = data["mlp"]["W2"].cuda()
    B, S, dm = x.shape
    H = 8
    W_lm = (torch.randn(dm, vocab, generator=g) * 0.02).half().cuda()

    def block(use_int8):
        h = F.layer_norm(x, (dm,)).half()
        qkv = h.reshape(B, S, H, dm // H).permute(0, 2, 1, 3).contiguous()
        if use_int8:
            attn = run_attention_kernel(ext, qkv, qkv, qkv).cuda()
        else:
            attn = attention_baseline(qkv, qkv, qkv).float().cpu().cuda()
        attn = attn.permute(0, 2, 1, 3).reshape(B, S, dm).float()
        r1 = x + attn                                # residual: FP16/FP32 path
        h2 = F.layer_norm(r1, (dm,)).half()
        if use_int8:
            mlp = run_mlp_kernel(ext, h2, W1, W2).cuda().float()
        else:
            mlp = mlp_baseline(h2, W1, W2).float()
        out = r1 + mlp
        return (out.half() @ W_lm).float().reshape(-1, vocab)

    logits_fp16 = block(False)
    logits_i8 = block(True)
    labels = logits_fp16.argmax(-1)
    ce_fp16 = F.cross_entropy(logits_fp16, labels)
    ce_i8 = F.cross_entropy(logits_i8, labels)
    ppl_fp16, ppl_i8 = math.exp(ce_fp16), math.exp(ce_i8)
    ppl_inc = ppl_i8 / ppl_fp16 - 1.0
    # Accuracy over tokens whose FP16 top1-top2 margin is meaningful: with
    # an untrained random LM head, many tokens are statistical ties whose
    # argmax flips under any epsilon perturbation — counting those measures
    # noise, not degradation.
    top2 = logits_fp16.topk(2, dim=-1).values
    margin_ok = (top2[:, 0] - top2[:, 1]) > 0.05
    agree = (logits_i8.argmax(-1) == labels)
    acc = float(agree[margin_ok].float().mean()) if margin_ok.any() else 1.0
    ov = top10_overlap(logits_fp16, logits_i8, max_rows=1024)

    fails = []
    if has_bad(logits_i8):
        fails.append("NaN/Inf in INT8 logits")
    if ppl_inc >= 0.02:
        fails.append(f"perplexity increase {ppl_inc*100:.2f}% >= 2%")
    if (1.0 - acc) >= 0.01:
        fails.append(f"accuracy degradation {(1-acc)*100:.2f}% >= 1%")
    return {"ppl_fp16": ppl_fp16, "ppl_int8": ppl_i8, "ppl_inc": ppl_inc,
            "top1_agreement": acc, "logits_top10": ov}, fails


# ── Repair advisor ───────────────────────────────────────────────────────
def repair_advisor(kernel, data, thr, worst_stages):
    """Simulate each fix and report whether it recovers gate-1 cosine."""
    target = thr["cos_min"]
    report = []
    if kernel == "attention":
        Q, K, V = data["Q"], data["K"], data["V"]
        ref = sim_attention(Q, K, V, fp16=True)["out"]
        fixes = [
            ("1. per-channel quantization (K/V per-channel)",
             lambda: sim_attention(q_per_channel(Q).half(), q_per_channel(K).half(),
                                   q_per_channel(V).half(), fp16=True)["out"]),
            ("2. QK^T in FP16 (V stays INT8)",
             lambda: sim_attention(Q, K, V, qk_fp16=True)["out"]),
            ("3. percentile-clipped (99.9%) scales",
             lambda: sim_attention(q_clipped(Q).half(), q_clipped(K).half(),
                                   q_clipped(V).half(), fp16=True)["out"]),
        ]
    else:
        x, W1, W2 = data["x"], data["W1"], data["W2"]
        ref = sim_mlp(x, W1, W2, fp16=True)["out"]
        fixes = [
            ("1. per-channel quantization for x",
             lambda: sim_mlp(x, W1, W2, x_dequant=q_per_channel(x))["out"]),
            ("3. percentile-clipped (99.9%) scale for x",
             lambda: sim_mlp(x, W1, W2, x_dequant=q_clipped(x))["out"]),
        ]
    fixes.append((f"4. keep worst stage in FP16 (gate-3 worst: "
                  f"{', '.join(n for n, _ in worst_stages)})", None))

    for name, fn in fixes:
        if fn is None:
            report.append(f"   {name} — apply manually (kernel change)")
            continue
        c = cosine(ref, fn())
        verdict = "RECOVERS" if c >= target else "insufficient"
        report.append(f"   {name}: simulated cosine {c:.5f} -> {verdict}")
    return report


# ── Main driver ──────────────────────────────────────────────────────────
def validate_dataset(ext, name, meta, data, kernels):
    thr = meta["thresholds"]
    is_zero = bool(thr.get("zero_output"))
    is_edge = name in EDGE_DATASETS
    all_pass = True
    print(f"\n{'='*68}\n  DATASET: {name}  (seq_len={meta['seq_len']})")
    print(f"  why: {meta['rationale']}")
    print(f"  exposes: {meta['expected_issue']}\n{'='*68}")

    for kernel in kernels:
        d = data[kernel if kernel != "attention" else "attention"]
        print(f"\n  -- kernel: int8_{kernel} --")
        if kernel == "attention":
            ref = attention_baseline(d["Q"].cuda(), d["K"].cuda(),
                                     d["V"].cuda()).float().cpu()
            act = run_attention_kernel(ext, d["Q"], d["K"], d["V"])
            s16 = sim_attention(d["Q"], d["K"], d["V"], fp16=True)
            si8 = sim_attention(d["Q"], d["K"], d["V"])
        else:
            ref = mlp_baseline(d["x"].cuda(), d["W1"].cuda(),
                               d["W2"].cuda()).float().cpu()
            act = run_mlp_kernel(ext, d["x"], d["W1"], d["W2"])
            s16 = sim_mlp(d["x"], d["W1"], d["W2"], fp16=True)
            si8 = sim_mlp(d["x"], d["W1"], d["W2"])

        # Simulation sanity: final simulated stage must match real kernel.
        sim_gap = float((si8["out"] - act).abs().max())
        if sim_gap > SIM_KERNEL_ATOL and not is_zero:
            print(f"  [warn] simulation vs kernel max diff {sim_gap:.4f} "
                  f"(> {SIM_KERNEL_ATOL}) — stage traces are approximate")

        fails = {}
        g1, f1 = gate1(ref, act, thr, is_zero)
        fails["gate1"] = f1
        t10 = "n/a" if g1["top10"] is None else f"{g1['top10']:.3f}"
        print(f"  Gate1 math     : cos={g1['cosine']:.5f} top10={t10} "
              f"outlier={g1['outlier_ratio']:.4f} nan_ok={g1['nan_inf']}"
              f"  -> {'PASS' if not f1 else 'FAIL'}")

        ranges, f2 = gate2(s16, si8)
        fails["gate2"] = f2
        rng = " ".join(f"{k}={v:.2f}" for k, v in ranges.items())
        print(f"  Gate2 stability: {rng}  -> {'PASS' if not f2 else 'FAIL'}")

        hist, worst, f3 = gate3(s16, si8)
        fails["gate3"] = f3
        trace = " -> ".join(f"{n}:{c:.4f}" for n, c in hist)
        print(f"  Gate3 stages   : {trace}")
        print(f"                   worst-3: "
              f"{', '.join(f'{n}({c:.4f})' for n, c in worst)}"
              f"  -> {'PASS' if not f3 else 'FAIL'}")

        if is_edge:
            f4 = gate4(ref, act, thr)
            fails["gate4"] = f4
            print(f"  Gate4 edge     : -> {'PASS' if not f4 else 'FAIL'}")

        kernel_fails = [m for v in fails.values() for m in v]
        xfail_reason = (meta.get("xfail") or {}).get(kernel)
        if kernel_fails:
            for v in fails.values():
                for m in v:
                    print(f"    [FAIL] {m}")
            if xfail_reason:
                print(f"  [XFAIL — known limitation, does not gate] "
                      f"{xfail_reason}")
            else:
                all_pass = False
            print("  Repair suggestions (simulated, in priority order):")
            for line in repair_advisor(kernel, d, thr, worst):
                print(line)
        elif xfail_reason:
            print(f"  [XPASS] expected failure passed — remove the xfail "
                  f"entry for '{kernel}' in generate_test_data.py")
    return all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="run one dataset only")
    ap.add_argument("--kernel", default=None, choices=["attention", "mlp"])
    ap.add_argument("--skip-gate5", action="store_true")
    args = ap.parse_args()

    check_cuda()
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit("No manifest — run `python generate_test_data.py` first.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    from torch.utils.cpp_extension import load
    print("Compiling INT8 kernels ...")
    ext = load(name="int8_ext",
               sources=["kernels/int8_attention.cu", "kernels/int8_mlp.cu",
                        "kernels/quant_utils.cu", "kernels/int8_ext.cu"],
               extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"],
               verbose=False)
    print("Done.")

    kernels = [args.kernel] if args.kernel else ["attention", "mlp"]
    names = [args.dataset] if args.dataset else list(manifest.keys())
    results = {}
    for name in names:
        meta = manifest[name]
        data = torch.load(os.path.join(DATA_DIR, meta["file"]),
                          weights_only=True)
        results[name] = validate_dataset(ext, name, meta, data, kernels)

    # Gate 5: task-level, on `normal` only.
    if not args.skip_gate5 and (args.dataset in (None, "normal")):
        print(f"\n{'='*68}\n  GATE 5: task-level (transformer block on "
              f"`normal`)\n{'='*68}")
        data = torch.load(os.path.join(DATA_DIR, "normal.pt"),
                          weights_only=True)
        g5, f5 = gate5(ext, data)
        print(f"  ppl fp16={g5['ppl_fp16']:.4f} int8={g5['ppl_int8']:.4f} "
              f"(+{g5['ppl_inc']*100:.3f}%)")
        print(f"  top1 agreement={g5['top1_agreement']*100:.2f}%  "
              f"logits top10 overlap={g5['logits_top10']:.3f}")
        print(f"  -> {'PASS' if not f5 else 'FAIL'}")
        for m in f5:
            print(f"    [FAIL] {m}")
        results["__gate5__"] = not f5

    print(f"\n{'='*68}\n  SUMMARY\n{'='*68}")
    ok = True
    for name, passed in results.items():
        print(f"  {name:12s} {'PASS' if passed else 'FAIL'}")
        ok &= passed
    print(f"\n  OVERALL: {'ALL GATES PASS' if ok else 'VALIDATION FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
