"""Decode-regime benchmark: INT8 KV-cache attention vs the FP16 decode path.

The decode regime (seq_q == 1, long KV) is where INT8 attention's real value
lives: each generated token re-reads the entire KV cache, so decode is
**bandwidth-bound on the KV cache**, and an INT8 cache is half the bytes of FP16.
The square WMMA kernel cannot run this shape (seq_q==1 wastes its Q-tiling);
int8_decode_attention_forward (kernels/int8_decode_attention.cu) is the
flash-decoding entry point built for it.

Peers:
  - ours : int8_decode_attention_forward  (INT8 Q + INT8 KV cache)
  - sdpa : F.scaled_dot_product_attention, seq_q==1  (FP16, PyTorch's decode path)
SageAttention is a *prefill* (block) kernel and is not a decode peer, so it is
not included here (see bench_attn_sota.py for the prefill SOTA comparison).

Also reports the KV-cache footprint: INT8 stores K and V at 1 byte/elem vs FP16's
2, the structural memory win a serving system cares about (more context / bigger
batch per card, half the bytes streamed per token).

Usage: python3 bench_attn_decode.py
"""
import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from benchmark import benchmark

CSV_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "attn_decode.csv")

# (batch, heads, head_dim) — small to serving-scale; one warp per (b,h), so
# batch*heads is the parallelism (small b*h is grid-starved on 108 SMs).
CONFIGS = [(8, 8, 64), (32, 8, 64), (64, 16, 64), (128, 16, 64), (64, 16, 128)]
SEQS = [1024, 2048, 4096]


def _load():
    return load(name="int8_ext",
        sources=["kernels/int8_attention.cu", "kernels/int8_decode_attention.cu",
                 "kernels/int8_mlp.cu", "kernels/quant_utils.cu", "kernels/int8_ext.cu"],
        extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"], verbose=False)


def _qpt(t):
    am = t.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    sc = am / 127.0
    return (t / sc).round().clamp(-128, 127).to(torch.int8), sc.squeeze(-1).contiguous().float()


def run(ext, B, H, D, S):
    torch.manual_seed(0)
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    Qi, sQ = _qpt(Q); Ki, sK = _qpt(K); Vi, sV = _qpt(V)

    ours_ms = benchmark(ext.int8_decode_attention_forward, Qi, Ki, Vi, sQ, sK, sV)

    Q4 = Q.unsqueeze(2)  # (B,H,1,D) for SDPA decode
    sdpa_ms = benchmark(F.scaled_dot_product_attention, Q4, K, V)

    # accuracy vs fp32 ground truth on the original inputs
    out = ext.int8_decode_attention_forward(Qi, Ki, Vi, sQ, sK, sV)
    ref = F.scaled_dot_product_attention(Q.float().unsqueeze(2), K.float(), V.float()).squeeze(2)
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()

    kv_int8_mb = B * H * S * D * 2 / 1e6        # K+V, 1 byte each
    kv_fp16_mb = kv_int8_mb * 2
    # bytes our kernel streams from the KV cache per decode step (≈ the whole cache)
    ours_gbps = (B * H * S * D * 2) / (ours_ms * 1e-3) / 1e9
    return dict(B=B, H=H, D=D, S=S, ours_ms=ours_ms, sdpa_ms=sdpa_ms, cos=cos,
                kv_int8_mb=kv_int8_mb, kv_fp16_mb=kv_fp16_mb, ours_gbps=ours_gbps)


def main():
    ext = _load()
    print("\nDecode regime (seq_q == 1).  Latency = mean ms / decode step over 50 iters.")
    print("ours = INT8 Q + INT8 KV cache;  sdpa = FP16 SDPA decode path.\n")
    hdr = (f"{'B':>4} {'H':>3} {'D':>4} {'S':>5} | {'ours ms':>8} {'sdpa ms':>8} "
           f"{'ours/sdpa':>9} | {'cos':>8} | {'KV int8':>9} {'KV fp16':>9} "
           f"{'ours GB/s':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for (B, H, D) in CONFIGS:
        for S in SEQS:
            r = run(ext, B, H, D, S)
            spd = r["sdpa_ms"] / r["ours_ms"]
            print(f"{B:>4} {H:>3} {D:>4} {S:>5} | {r['ours_ms']:8.4f} {r['sdpa_ms']:8.4f} "
                  f"{spd:8.2f}x | {r['cos']:8.5f} | {r['kv_int8_mb']:8.1f}M {r['kv_fp16_mb']:8.1f}M "
                  f"{r['ours_gbps']:9.1f}")
            r["speedup_vs_sdpa"] = spd
            rows.append(r)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[CSV] {len(rows)} rows -> {CSV_OUT}")
    print("ours/sdpa > 1.0 = INT8 decode kernel faster than the FP16 path.")
    print("INT8 KV cache is half the FP16 bytes (more context / bigger batch per card).")
    print("A100-SXM4 HBM peak ~1555 GB/s — ours GB/s shows how bandwidth-bound we are.\n")


if __name__ == "__main__":
    main()
