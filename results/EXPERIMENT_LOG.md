# Experiment Log — INT8 CUDA Transformer (Heling)

Hardware: NVIDIA A100-SXM4-40GB (Perlmutter, NERSC)
PyTorch: 2.8.0+cu129
Date: 2026-05-06

---

## 1. Correctness: Per-token INT8 Attention (Opt #3)

Commit: `beba07a` — per-token symmetric quantization, INT8 WMMA QK^T, FP16 output.

```
python tests/test_int8.py --quick
```

| Config | max abs error | mean abs error | atol | Status |
|--------|--------------|----------------|------|--------|
| head_dim=64, seq=512 | 0.000977 | 0.000033 | 0.1 | PASS |
| head_dim=128, seq=512 | 0.000977 | 0.000041 | 0.1 | PASS |
| head_dim=256, seq=512 | 0.000732 | 0.000034 | 0.1 | PASS |
| head_dim=64, seq=1024 | 0.000610 | 0.000024 | 0.1 | PASS |
| head_dim=128, seq=1024 | 0.000977 | 0.000030 | 0.1 | PASS |
| head_dim=256, seq=1024 | 0.000488 | 0.000024 | 0.1 | PASS |
| MLP (per-tensor, d=512) | 0.038498 | 0.006258 | 0.1 | PASS |

Note: per-token attention error is ~100x below tolerance. MLP (still per-tensor) is ~2.5x below.

---

## 2. Benchmark: INT8 Attention vs FP16 Baseline (test_int8.py)

Baseline = PyTorch naive attention (`attention_baseline`), NOT Jonathan's kernel.
batch=2, heads=8, seq=512.

| Config | FP16 baseline | INT8 kernel | Speedup | Utilization |
|--------|--------------|-------------|---------|-------------|
| head_dim=64 | 0.107 ms | 0.151 ms | 0.71x | 1.1% |
| head_dim=256 | 0.117 ms | 0.568 ms | 0.21x | 1.2% |

MLP: 0.085 ms (FP16) vs 0.248 ms (INT8) = 0.34x, cuBLAS INT8 = 0.088 ms.

Note: small batch (=2) underutilizes A100. Sweep below uses batch=8.

---

## 3. Sweep: INT8 Attention vs Jonathan's FP16 WMMA (sweep.py --kernel int8_attn)

Commit: `beba07a` (Opt #3 + per-token quantization)
Baseline: Jonathan's FP16 WMMA attention kernel (`attention_ext`)
batch=8, heads=8, head_dim = d_model/8

### Results Table

| d_model | seq_len | INT8 (ms) | FP16 WMMA (ms) | FlashAttn-2 (ms) | Speedup vs FP16 |
|---------|---------|-----------|----------------|-------------------|-----------------|
| 512 | 512 | 0.35 | 0.24 | 0.07 | 0.70x |
| 512 | 1024 | 1.09 | 0.81 | 0.16 | 0.75x |
| 512 | 2048 | 3.11 | 2.36 | 0.53 | 0.76x |
| 512 | 4096 | 11.76 | 8.97 | 1.91 | 0.76x |
| 1024 | 512 | 0.62 | 0.39 | 0.10 | 0.63x |
| 1024 | 1024 | 1.87 | 1.27 | 0.26 | 0.68x |
| 1024 | 2048 | 6.72 | 4.69 | 0.90 | 0.70x |
| 1024 | 4096 | 25.81 | 17.98 | 3.28 | 0.70x |
| 2048 | 512 | 1.62 | 0.95 | 0.18 | 0.59x |
| 2048 | 1024 | 5.09 | 3.10 | 0.51 | 0.61x |
| 2048 | 2048 | 19.69 | 11.35 | 1.86 | 0.58x |
| 2048 | 4096 | 74.63 | 43.40 | 6.86 | 0.58x |

### Summary by d_model (head_dim)

| head_dim | Speedup range | Trend |
|----------|--------------|-------|
| 64 (d=512) | 0.70–0.76x | Improves slightly with seq_len |
| 128 (d=1024) | 0.63–0.70x | Improves slightly with seq_len |
| 256 (d=2048) | 0.58–0.61x | Flat |

---

## 3b. Sweep: INT8 Attention after Opt #4 (4 warps, pre-fold attn_scale)

Commit: `98ce569` — ATTN_BDIM 64→128, ATTN_TILE_Q 32→64, pre-fold attn_scale into Q scales.
Baseline: Jonathan's FP16 WMMA attention kernel (`attention_ext`)
batch=8, heads=8, head_dim = d_model/8

### Results Table

| d_model | seq_len | INT8 (ms) | FP16 WMMA (ms) | FlashAttn-2 (ms) | Speedup vs FP16 |
|---------|---------|-----------|----------------|-------------------|-----------------|
| 512 | 512 | 0.27 | 0.24 | — | 0.89x |
| 512 | 1024 | 1.00 | 0.82 | — | 0.83x |
| 512 | 2048 | 2.94 | 2.36 | — | 0.80x |
| 512 | 4096 | 10.33 | 8.97 | — | 0.87x |
| 1024 | 512 | 0.53 | 0.39 | — | 0.74x |
| 1024 | 1024 | 1.70 | 1.28 | — | 0.75x |
| 1024 | 2048 | 5.90 | 4.70 | — | 0.80x |
| 1024 | 4096 | 22.53 | 18.01 | — | 0.80x |
| 2048 | 512 | 1.08 | 0.95 | — | 0.88x |
| 2048 | 1024 | 3.50 | 3.10 | — | 0.89x |
| 2048 | 2048 | 12.34 | 11.35 | — | 0.92x |
| 2048 | 4096 | 47.00 | 43.39 | — | 0.92x |

### Summary by d_model (head_dim)

| head_dim | Speedup range | vs Opt #3 |
|----------|--------------|-----------|
| 64 (d=512) | 0.80–0.89x | was 0.70–0.76x (+10-13%) |
| 128 (d=1024) | 0.74–0.80x | was 0.63–0.70x (+10-11%) |
| 256 (d=2048) | **0.88–0.92x** | was 0.58–0.61x (**+30%**) |

Key: 4 warps (128 threads) dramatically improved latency hiding, especially at head_dim=256.

---

## 4. Sweep: INT8 MLP vs Sherry's FP16 WMMA (sweep.py --kernel int8)

From previous sweep (Opt #2, per-tensor MLP, same commit branch).
Baseline: Sherry's FP16 WMMA MLP kernel (`mlp_ext`)
batch=8, head_dim = d_model/8

| d_model | seq_len | INT8 (ms) | FP16 WMMA (ms) | cuBLAS INT8 (ms) | Speedup vs FP16 |
|---------|---------|-----------|----------------|-------------------|-----------------|
| 512 | 512 | 0.52 | 0.30 | 0.18 | 0.58x |
| 512 | 1024 | 0.93 | 0.57 | 0.34 | 0.61x |
| 512 | 2048 | 1.70 | 1.05 | 0.63 | 0.62x |
| 512 | 4096 | 3.29 | 2.03 | 1.23 | 0.62x |
| 1024 | 512 | 1.68 | 1.04 | 0.65 | 0.62x |
| 1024 | 1024 | 3.16 | 1.94 | 1.21 | 0.61x |
| 1024 | 2048 | 6.18 | 3.76 | 2.39 | 0.61x |
| 1024 | 4096 | 12.21 | 7.43 | 4.76 | 0.61x |
| 2048 | 512 | 6.07 | 4.19 | 2.37 | 0.69x |
| 2048 | 1024 | 11.97 | 8.11 | 4.71 | 0.68x |
| 2048 | 2048 | 23.76 | 16.10 | 9.40 | 0.68x |
| 2048 | 4096 | 47.32 | 31.94 | 18.76 | 0.68x |

### Summary by d_model

| d_model | Speedup range |
|---------|--------------|
| 512 | 0.58–0.62x |
| 1024 | 0.61–0.62x |
| 2048 | 0.68–0.69x |

---

## 3c. Sweep: INT8 Attention after Opt #5 (smem bank conflict fix)

Commit: `d3a04d7` — ATTN_I8_PAD=16 breaks bank conflicts on INT8 Q/K rows in smem.
Root cause: head_dim=256 → stride=256 bytes → 256%128=0 → all rows map to same 32 bank set.
Fix: pad stride to 272 bytes → 272%128=16 → rows cycle through different bank sets.
Baseline: Jonathan's FP16 WMMA attention kernel (`attention_ext`)
batch=8, heads=8, head_dim = d_model/8

### Results Table

| d_model | seq_len | INT8 (ms) | FP16 WMMA (ms) | Speedup vs FP16 |
|---------|---------|-----------|----------------|-----------------|
| 512 | 512 | 0.26 | 0.24 | 0.93x |
| 512 | 1024 | 0.94 | 0.73 | 0.77x |
| 512 | 2048 | 2.62 | 2.35 | 0.90x |
| 512 | 4096 | 9.67 | 8.99 | 0.93x |
| 1024 | 512 | 0.42 | 0.40 | 0.95x |
| 1024 | 1024 | 1.42 | 1.28 | 0.90x |
| 1024 | 2048 | 4.84 | 4.72 | 0.98x |
| 1024 | 4096 | 18.31 | 18.09 | 0.99x |
| 2048 | 512 | 0.85 | 0.96 | **1.13x** |
| 2048 | 1024 | 2.97 | 3.10 | **1.04x** |
| 2048 | 2048 | 10.41 | 11.35 | **1.09x** |
| 2048 | 4096 | 40.17 | 43.39 | **1.08x** |

### Summary by d_model (head_dim)

| head_dim | Speedup range | vs Opt #4 |
|----------|--------------|-----------|
| 64 (d=512) | 0.77–0.93x | was 0.80–0.89x (mixed) |
| 128 (d=1024) | 0.90–0.99x | was 0.74–0.80x (**+15-20%**) |
| 256 (d=2048) | **1.04–1.13x** | was 0.88–0.92x (**beats FP16!**) |

Analysis:
- Bank conflicts were the dominant bottleneck for head_dim=128,256 (stride 128,256 bytes both multiples of 128)
- head_dim=64: stride=64 bytes, 64%128≠0, so bank conflicts were already mild → less improvement
- head_dim=256: biggest win (+21% at seq=512) because stride=256 was the worst case (exactly 2×128)
- INT8 now **surpasses FP16** at head_dim=256: 2x INT8 TOPS + no bank conflicts > FP16 compute

---

## 5. Optimization History (Attention)

All vs Jonathan's FP16 WMMA kernel.

| Optimization | head_dim=64 | head_dim=128 | head_dim=256 |
|-------------|-------------|--------------|--------------|
| Baseline (no opt) | ~0.57x | ~0.35x | ~0.15x |
| Opt #1: remove cudaMemcpy | ~0.65x | ~0.40x | ~0.20x |
| Opt #2: static scale, no requant | 0.63–0.67x | 0.41–0.48x | 0.20–0.23x |
| Opt #3: INT8 WMMA QK^T + per-token | 0.70–0.76x | 0.63–0.70x | 0.58–0.61x |
| Opt #4: 4 warps + pre-fold scale | 0.80–0.89x | 0.74–0.80x | 0.88–0.92x |
| **Opt #5: INT8 smem bank conflict fix** | **0.77–0.93x** | **0.90–0.99x** | **1.04–1.13x** |

Key improvements:
- Opt #3: head_dim=256 from 0.20x to 0.58x (smem 87KB → 36KB, occupancy 1 → 4 blocks/SM)
- Opt #4: head_dim=256 from 0.58x to 0.92x (2→4 warps, 2x latency hiding)
- Opt #5: head_dim=256 from 0.92x to **1.13x** (pad INT8 stride to break bank conflicts)

---

## 6. Phase 0: Quantization Precision Analysis

Script: `analysis/phase0_real_activations.py`
Models: GPT-2 (124M), OPT-6.7B

### Per-tensor quantization — cosine similarity of attention output

| Model | Layer | Q | K | V | Attn output |
|-------|-------|---|---|---|-------------|
| GPT-2 | layer_0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| GPT-2 | layer_5 | 1.0000 | 1.0000 | 1.0000 | 0.9999 |
| GPT-2 | layer_11 | 1.0000 | 1.0000 | 0.9999 | 0.9998 |
| OPT-6.7B | layer_0 | 1.0000 | 1.0000 | 0.9997 | 0.9998 |
| OPT-6.7B | layer_4 | 1.0000 | 1.0000 | 0.9984 | 0.9949 |
| OPT-6.7B | layer_8 | 1.0000 | 1.0000 | 0.9984 | 0.9840 ← FAIL |

Root cause: V tensor outliers (kurtosis=11.8, range utilization=3.5%).

### Per-token quantization — same test

| Model | Worst layer cosine sim | Status |
|-------|----------------------|--------|
| GPT-2 | 0.9998 | PASS |
| OPT-6.7B | 0.9999 | PASS |

Per-token fixes OPT-6.7B precision: 0.984 → 0.9999.

---

## 7. Key Takeaways for Poster

1. INT8 has 2x theoretical TOPS but naive substitution makes things slower (overhead > compute savings)
2. Removing overhead (Opt #1, #2) helps but large head_dim still bottlenecked by smem/occupancy
3. Keeping Q/K as INT8 in smem (Opt #3) gives 3x improvement at head_dim=256 (0.20x → 0.58x)
4. Per-token quantization essential for large models (OPT-6.7B) — negligible perf cost, fixes precision
5. Increasing parallelism (Opt #4, 4 warps) pushes head_dim=256 to 0.92x — nearly matching FP16
6. Smem bank conflicts were the final bottleneck (Opt #5): padding INT8 rows → **1.13x, beating FP16**
7. INT8 advantage grows with d_model for MLP (0.58x → 0.69x) — better arithmetic intensity amortizes overhead
8. Full optimization journey: 0.15x → 1.13x at head_dim=256 (**7.5x improvement** through 5 optimization steps)
