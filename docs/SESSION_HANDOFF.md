# Session Handoff — CS 5220 INT8 CUDA Kernel Project

**Date written:** 2026-04-22 (session 1 ending)
**Project deadline:** 2026-04-29
**For:** the next Claude session continuing this work

---

## 0. Read this first — context

You are helping **Ashley** (English name; Chinese name Heling) with a CS 5220 Cornell course project. The project is a 3-person team building optimized CUDA kernels for transformer attention and MLP on NVIDIA A100 GPUs on the **Perlmutter** supercomputer at NERSC.

**Team:**
- **Ashley (me)** — INT8 quantized attention + MLP kernels
- **Jonathan** — FP16 attention kernel
- **Sherry** (a.k.a. Shengjing) — FP16 MLP kernel

**Repo:** `git@github.com-ashley:little-xiaohe/cuda-transformer-opt.git`
**Local path:** `/Users/ashley/Desktop/Cornell/CS5220-project/cuda-transformer-opt/`
**Working branch:** `feat/int8-on-team` (pushed)

Ashley speaks Chinese natively but is comfortable with English. Default to Chinese unless context suggests otherwise. Ashley is a MediaTek kernel engineer — strong systems background, comfortable with C/CUDA, but new to CUDA tensor cores / wmma / cp.async (we covered the concepts in this session).

---

## 1. What's done (project state)

### Infrastructure (all merged to `main`)
- `baseline.py` — PyTorch FP16 reference implementations for attention and MLP (uses GELU tanh approximation)
- `benchmark.py` — `torch.cuda.Event`-based timing harness (50 iter, 10 warmup)
- `correctness.py` — `torch.allclose` + diagnostic output on failure
- `tests/gen_testdata.py` — generates `.bin` test data with fixed seed (small + large configs)
- `tests/cuda/test_*.cu` — pure CUDA correctness tests (read .bin, compare reference)
- `tests/test_*.py` — Python benchmark scripts
- `Makefile` — compiles `test_attention`, `test_mlp`, `test_int8` (with optional `DFLAGS` for ablation)
- `sweep.py` — generalized sweep tool, accepts `--kernel {mlp,attention,int8}`, writes CSV + 5 PNGs to `results/`
- `gitignore` excludes `.DS_Store`, `__pycache__`, `testdata/`, compiled binaries

### Teammate kernels (in current branch `feat/int8-on-team`)

**Jonathan's `kernels/attention.cu` (FP16 WMMA)**:
- Two paths: `fused_attention_kernel` (scalar, fallback) and `wmma_attention_kernel`
- WMMA path: 32 Q rows per block, 2 warps, 16×16×16 fragments
- cp.async double buffer for K/V tiles
- Online softmax in register (per warp: rmax0/rmax1/rsum0/rsum1)
- Direct fragment access (`frag.x[0..7]`) — assumes FP32 accumulator layout
- Ablation flags: `ATTN_WMMA`, `ATTN_FAST_MATH`, `ATTN_SWIZZLE` (all default 1)
- `kernels/attention_ext.cu` — pybind11 binding
- Has `tests/test_attention.py` and `Makefile ablation_attention` target

**Sherry's `kernels/mlp.cu` (FP16 WMMA)**:
- Two paths: `gemm_fp16_scalar_kernel` and `gemm_fp16_wmma_kernel<bool apply_gelu>`
- WMMA path: 64×64 block tile, 8 warps, 16×16×16 fragments
- cp.async double buffer for A/B tiles, STAGE_K=32
- SMEM_SKEW=8 padding for bank conflicts
- GELU fused into accumulator epilogue (template parameter)
- Static workspace cache for hidden buffer
- `kernels/mlp_ext.cu` — pybind11 binding
- Has `tests/test_mlp.py`

### INT8 work (status: starting fresh)

- **Ashley's previous v1→v6 INT8 implementation is on branch `feat/int8` (deprecated).**
- `feat/int8-on-team` has the INT8 files as **original stubs** (no v6 code).
- The plan is to rebuild INT8 on top of the teammates' WMMA architecture.

The v6 approach was Ashley's standalone INT8 (16×16 wmma, no cp.async, no smem tiling). It got to 0.181ms attention / 0.231ms MLP, which was 10× faster than naive but 2-4× slower than FP16 baseline. We decided that going forward we should use the teammates' WMMA architecture (with cp.async + swizzle) and add INT8 on top, for fair team-internal comparison.

### Documentation (in `docs/`, NOT tracked in git — kept local)
- `transformer_shapes.html` — general intro to transformer attention/MLP shapes, includes INT8 explanation. Aimed at Ashley's understanding.
- `team_status.html` — current team status, agreements, timeline, action items. **Most recently updated.**
- `int8_integration_plan.html` — storyboard-style walkthrough of Sherry's MLP and Jonathan's attention architectures, then how to port each to INT8. Has detailed jargon explanations (Thread/Warp/Block, WMMA, cp.async, etc.) and step-by-step matrix visualizations. **Most detailed working doc.**

These are personal docs; they don't need to be pushed.

---

## 2. Team agreements (from 2026-04-22 meeting)

| Owner | When | Task |
|---|---|---|
| Sherry | by 4/26 | Generalize `sweep.py` to merge into main (✅ done) and port MLP to WMMA (✅ done) |
| Jonathan | by 4/26 | Port attention to WMMA + push `attention_ext.cu` (✅ done) |
| Ashley | 4/27–4/28 | Pull their WMMA branches, run INT8 on top, identify integration issues, optimize |
| Jonathan + Sherry | 4/27–4/28 | Start poster writing |
| Ashley | 4/28 night | Lock final INT8 version (no more code changes) |
| All | 4/29 | Review + submit (deadline) |

### Cross-cutting agreements

1. **Same hardware**: Perlmutter has both A100-PCIe and A100-SXM4. `salloc -C "gpu&hbm40g"` doesn't distinguish. Verify with `nvidia-smi --query-gpu=name --format=csv`. Aim for SXM4 if possible. **Whoever runs benchmarks first should announce which variant they got, so others can match.**
2. **Canonical benchmark config**: `batch=2, seq_len=512, heads=8, head_dim=64, d_model=512, d_ff=2048`
3. **Poster structure** (per person): ablation table (each step → % gain/loss) + insight prose (bottleneck analysis → next decision) + scaling plot. Ashley adds precision/tolerance analysis.
4. **Use Nsys/NCU profile data as evidence** for bottleneck claims in the poster prose (not just intuition).
5. **Don't use `Co-Authored-By` lines in commits** (Ashley doesn't want Claude in git history).

---

## 3. Plan for the next two days (4/27–4/28)

### Day 1 (4/27): Port + first benchmarks

**Morning (~2 hrs):**
1. On Perlmutter:
   ```bash
   module load pytorch
   salloc -A m4341_g -C "gpu&hbm40g" -N 1 -t 02:00:00 -q interactive
   nvidia-smi --query-gpu=name --format=csv  # note PCIe or SXM4
   git fetch && git checkout feat/int8-on-team
   ```
2. Verify teammates' FP16 kernels work:
   ```bash
   make test_attention && ./test_attention
   make test_mlp && ./test_mlp
   python tests/test_attention.py
   python tests/test_mlp.py
   python sweep.py --kernel attention
   python sweep.py --kernel mlp
   ```
3. **Record FP16 baseline numbers** at canonical config (these are Ashley's reference for INT8 comparison).

**Afternoon (~4 hrs): Port MLP to INT8**

Recommended approach: **start from `kernels/mlp.cu`, copy structure, swap dtype**.

Key changes (covered in detail in `docs/int8_integration_plan.html`):
- Fragment dtype: `<half, ..., float>` → `<int8_t, ..., int32_t>`
- cp.async loop bound: `STAGE_K / 8` → `STAGE_K / 16` (INT8 = 16 elements per 16-byte copy)
- Add scale parameters: `scale_x`, `scale_W1`, `scale_W2`
- GELU epilogue: `INT32 acc → ×scale_x×scale_W1 → FP32 → GELU → write FP16 hidden`
- Between GEMMs: separate `quantize_fp16_to_int8` kernel (per-tensor scale, simplest)
- GEMM 2 epilogue: `INT32 acc → ×scale_hidden×scale_W2 → FP16 out`
- Re-tune SMEM_PAD for INT8 (PAD=8 in halves was for FP16; INT8 may need different)

Test: `make test_int8 && ./test_int8` then `python tests/test_int8.py`.

**Evening (~3 hrs): Port attention to INT8**

Higher risk because Jonathan reads `frag_qk.x[0..7]` directly. **For first port, use the safer `store_matrix_sync` path** (write fragment to smem, read smem to do softmax). Loses ~1% to extra smem round-trip but doesn't depend on FP32-accumulator-layout assumption.

Steps:
- Copy `attention.cu` → `int8_attention_jonathan.cu`
- QK^T fragments → INT8/INT32
- After mma, store to smem instead of reading frag.x[i]
- Softmax in FP32 (read from smem)
- Per-row quantize the softmax output (warp shuffle for max within warp)
- attn × V fragments → INT8/INT32
- Output normalization includes per-row attn scale × scale_V

Test correctness, run `python sweep.py --kernel int8`.

### Day 2 (4/28): Optimize + freeze

**Morning:** Run `ncu` on the INT8 kernels. Identify dominant cost (smem bank conflict? quant overhead? cp.async stall?). Optimize accordingly.

**Afternoon:** Add ablation flags (`INT8_WMMA`, `INT8_PER_ROW_QUANT`, etc.) so the poster has an ablation table.

**Evening (BEFORE BED):** Lock final version. Commit final benchmark numbers + figures. Hand off to team for poster review.

### Day 3 (4/29): Deadline

- Review the full poster (everyone)
- Fix any remaining issues
- **Submit**

---

## 4. Risks Ashley should keep in mind

1. **WMMA INT8 fragment layout** — Jonathan assumes FP32 accumulator layout for direct `frag.x[i]` access. INT32 accumulator may differ. Don't assume; use safer `store_matrix_sync` path unless you've verified with a diagnostic kernel.
2. **Bank conflict re-tuning** — SMEM_PAD=8 (16 bytes for FP16) may not avoid INT8 bank conflicts (8 bytes). May need PAD=16 or different swizzle.
3. **Quantization granularity decision** — For MLP, per-tensor (Ashley's v6 approach) is simplest and works. For attention, per-row works naturally because each warp owns 16 Q rows.
4. **Hardware variance** — PCIe vs SXM4 changes timings 30%. Final benchmarks should be on one variant.
5. **Don't get distracted by "beating cuBLAS"** — the realistic story is "INT8 vs hand-written FP16 at same optimization level". Compare against teammates' kernels, not cuBLAS.

---

## 5. Conventions Ashley uses

- **Language**: Chinese for explanations, English for messages to teammates
- **Code commits**: Don't add `Co-Authored-By: Claude` trailer
- **Communication style**: Doesn't want to come across as too dominant; soften messages when proposing ideas. Tends to share observations rather than demand changes.
- **Naming**: `feat/<task>` for branches, kebab-case for files
- **Documentation**: Heavy on visualizations (HTML with SVG/CSS), structured tables, color-coded by owner (Ashley=blue, Jonathan=purple, Sherry=green)

---

## 6. What the previous session got stuck on or kept revisiting

1. **Fragment direct access risk** — kept coming back to whether INT32 layout matches FP32. Ashley correctly intuited that for safety, port should fall back to store_matrix_sync (don't inherit Jonathan's micro-optimization).
2. **Storyboard visualizations** — Ashley wanted MLP and attention explanations broken into many small step-by-step diagrams (not one big dense diagram). The HTML in `docs/int8_integration_plan.html` reflects this.
3. **Don't re-explain v1-v6** — Ashley explicitly asked to forget v6 and start fresh on teammates' branch. Don't bring up v6 unless explicitly asked.

---

## 7. Quick reference

### Branches
- `main` — has shared infrastructure + Sherry's sweep.py
- `feat/int8-on-team` — **CURRENT WORKING BRANCH**, has both teammates' WMMA + INT8 stubs ready to fill
- `origin/feat/attention` — Jonathan's standalone attention branch (already merged in)
- `origin/mlp` — Sherry's standalone MLP branch (already merged in)
- `feat/int8` — Ashley's old v1→v6 INT8 work (deprecated, ignore)

### Useful files
- `sweep.py` — `python sweep.py --kernel {mlp,attention,int8}`
- `Makefile` — `make {test_attention, test_mlp, test_int8}`, optional `DFLAGS="-DATTN_WMMA=0 ..."`
- `tests/gen_testdata.py` — run once to generate `testdata/{small,large}/*.bin`
- `kernels/attention.cu` (Jonathan) — reference architecture for INT8 attention port
- `kernels/mlp.cu` (Sherry) — reference architecture for INT8 MLP port

### Local-only files (in `docs/`, not tracked)
- `transformer_shapes.html` — basic transformer concepts
- `team_status.html` — meeting consensus, timeline, action items
- `int8_integration_plan.html` — detailed storyboard for porting INT8 onto teammates' kernels

---

## 8. If something goes wrong

- **Compile error in INT8 port**: most likely fragment dtype mismatch. Check that `<int8_t, ..., int32_t>` is consistent everywhere.
- **Correctness fail**: most likely scale handling. Check that `scale_x × scale_W1` is being applied to the INT32 accumulator correctly.
- **Slow performance**: run `ncu`. Don't guess. Sherry has `results/ncu_mlp_report.txt` as a template for the report format.
- **Hardware mismatch**: check `nvidia-smi --query-gpu=name --format=csv` and re-`salloc` if you got the wrong variant.

---

## 9. End-state for this session

- All three FP16 kernels (attention, MLP) are pushed and merged into `feat/int8-on-team`.
- INT8 files are stubs, ready to fill in.
- Local docs (3 HTML files in `docs/`) explain everything Ashley needs to start coding.
- Ashley plans to do the actual code changes on Perlmutter starting tomorrow (4/27 in project timeline).

**Good luck. Ashley is sharp; trust her engineering instincts. The plan above is solid; main risk is fragment layout — handle with the safer path.**
