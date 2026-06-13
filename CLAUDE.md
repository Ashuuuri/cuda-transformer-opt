# CLAUDE.md — cuda-transformer-opt

CUDA Transformer kernel optimization project (A100 / sm_80). Originally a
3-person CS 5220 course project, now continued solo by the repo owner on the
`opt-dev` branch with **INT8 quantized kernels as the optimization focus**.

## 1. Project Architecture

### File map

| File | Purpose |
|---|---|
| `kernels/attention.cu` | FP16 fused attention (WMMA split-Q + online softmax + cp.async) |
| `kernels/mlp.cu` | FP16 MLP: two WMMA GEMMs, GELU fused into GEMM1 |
| `kernels/int8_attention.cu` | **INT8 attention** (primary target): INT8 WMMA QK^T + FP16 WMMA PV, per-token scales, templated on `(TILE_KV, HEAD_DIM)` |
| `kernels/int8_mlp.cu` | **INT8 MLP** (primary target): 128×128-tile INT8 WMMA ×2, dynamic quantization (per-token hidden scales). Two entry points: `int8_mlp_forward` (per-tensor scalar scales, legacy/unchanged) and `int8_mlp_forward_per_channel` (per-token act + per-channel weight + per-token output — the real-model accuracy path) |
| `kernels/int8_common.cuh` | Shared device helpers: GELU, f32→i8, cp.async |
| `kernels/quant_utils.cu` | Per-tensor quantize/dequantize utility kernels |
| `kernels/*_ext.cu` | pybind11 bindings (for torch `cpp_extension.load`) |
| `tests/cuda/test_*.cu` + `tests/gen_testdata.py` | Pure-CUDA correctness smoke tests (read `.bin` files) |
| `tests/test_*.py` | Python correctness + benchmarks (vs FA2 / cuBLAS) |
| `generate_test_data.py` | **8-distribution INT8 validation datasets** (see §6) |
| `validate_int8.py` | **Five-gate accuracy validation** (see §5); Gate 5 = real GPT-2 perplexity on `testdata/real_corpus.txt` (WikiText-2) |
| `sweep.py` | Parameter sweep → `results/*.csv` + figures; `--kernel int8_attn|int8_mlp|attention|mlp` |
| `baseline.py` / `benchmark.py` / `correctness.py` | PyTorch references / timing / error checking |

### Kernel data flow

```
INT8 attention:
  Q,K,V fp16 → [Python: per-token quantize] → int8 + scales[B*H*S]
  → kernel: Q,K int8 in smem → INT8 WMMA QK^T (INT32 accum)
  → ×sQ[row]×sK[col]×1/√d → online softmax (fp32 registers)
  → P fp16 in smem → FP16 WMMA P@V (V dequantized per-row) → out fp16

INT8 MLP (dynamic-quant path, INT8_MLP_DYNAMIC=1 default):
  x int8 → GEMM1 (INT8 WMMA) → epilogue: ×sx·sW1 → GELU → fp16 hidden
           + per-token row absmax (atomicMax)
  → quantize_rows_kernel: hidden → int8 (per-token scales)
  → GEMM2 (INT8 WMMA, per-row A scales) → fp16 out + global absmax
  → quantize_tensor_kernel: out → int8 + dynamic per-tensor scale
```

## 2. Build & Run

### 2.0 Environment setup — run FIRST in a fresh environment

The system PyTorch 2.7 (`/usr/lib/python3/dist-packages/torch`) JIT-builds the
INT8 extension with `cpp_extension.load`, which needs **ninja** and the
**pybind11 C++ headers on the system include path**. A fresh box (or a new
container) usually ships with neither, and the failures are non-obvious
(`RuntimeError: Ninja is required...`, then `fatal error: pybind11/pybind11.h:
No such file or directory`). Before running anything else — `evolve.sh`,
`validate_int8.py`, `sweep.py`, `tests/test_int8.py` — verify/install these **in
this order**:

```bash
# 1. ninja (provides ~/.local/bin/ninja + the python module). Required by
#    torch cpp_extension.load. Check first; pip-install only if missing.
command -v ninja || pip3 install --user ninja

# 2. pybind11 C++ headers ON THE SYSTEM INCLUDE PATH (/usr/include/pybind11).
#    The system torch does NOT add the pip pybind11 package's include dir to
#    the nvcc command line, so `pip install pybind11` alone does NOT fix the
#    build — you need the apt -dev package (passwordless sudo is available).
ls /usr/include/pybind11/pybind11.h 2>/dev/null || sudo apt-get install -y pybind11-dev

# 3. Sanity-check the toolchain end to end (JIT-compiles both kernels, ~2 min):
rm -rf ~/.cache/torch_extensions/py312_cu128/int8_ext
python3 -c "from torch.utils.cpp_extension import load; \
  ext=load(name='int8_ext', sources=['kernels/int8_attention.cu','kernels/int8_mlp.cu',\
  'kernels/quant_utils.cu','kernels/int8_ext.cu'], \
  extra_cuda_cflags=['-arch=sm_80','--std=c++17','-O3']); print('BUILD OK')"
```

Only once "BUILD OK" prints is the environment ready. (`ncu`/`nsys` for §3 are
apt-installed separately; profiling counters also need sudo — see §3.)

### 2.1 Commands

```bash
# Environment: local A100-SXM4-40GB, CUDA 12.8, system PyTorch 2.7
# Deps: ninja + pybind11-dev — see §2.0; install before first build

# Pure-CUDA smoke test (run `python tests/gen_testdata.py` once for .bin data)
make test_int8 && ./test_int8

# Python correctness + benchmark (JIT compile, ~2 min first time, cached)
python tests/test_int8.py --quick     # correctness only
python tests/test_int8.py             # + FA2 / FP16 kernel / cuBLAS baselines

# INT8 five-gate accuracy validation (the gate for every optimization)
python generate_test_data.py          # one-time: generate the 8 datasets
python prepare_real_corpus.py         # one-time: fetch WikiText-2 for Gate 5
                                      #   (needs `pip install --user datasets`;
                                      #    Gate 5 SKIPs gracefully if absent)
python validate_int8.py               # all datasets × both kernels × 5 gates
python validate_int8.py --dataset outlier --kernel mlp   # filters

# Performance sweeps (refresh results/*.csv; figures via the dashboard below)
python sweep.py --kernel int8_attn
python sweep.py --kernel int8_mlp

# Consolidated figure: ONE dashboard (results/figures/int8_dashboard.png)
# replaced the old 20 scattered per-kernel PNGs. Build it after the sweeps:
python collect_profile.py     # torch.profiler per-kernel time split (no sudo)
# (optional) ncu stall metrics for the profiling row — see §3 for the sudo cmd;
#   save raw `ncu --csv` output to results/ncu_{mlp,attn}_raw.csv
python make_dashboard.py      # -> results/figures/int8_dashboard.png
#   (sweep.py --legacy-figs still emits the old per-kernel PNGs if needed)
```

The torch extension rebuilds automatically after `.cu` edits; if the cache
misbehaves: `rm -rf ~/.cache/torch_extensions/py312_cu128/int8_ext`.

## 3. Profiling Workflow

`ncu` (Nsight Compute 2025.1) and `nsys` are installed via apt
(`nsight-compute`, `nsight-systems`). **GPU performance counters require
sudo** (`ERR_NVGPUCTRPERM` otherwise): run `sudo ncu ...`, or enable for all
users permanently with
`echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf`
(takes effect after reboot). For quick per-kernel timings without counters,
`torch.profiler` needs no privileges; registers/smem come from the ptxas
report.

**CRITICAL — profile the shape sweep.py grades, not `test_int8.py --quick`.**
`--quick` is batch=2, seq=512, head_dim 64/128/256: a GRID-STARVED toy (128
blocks / 108 SMs ≈ 1.18 blocks/SM → `warps_active` ~7.4%). sweep.py grades
batch=8, **head_dim=64 only**, seq 512–4096: 512–4096 blocks → occupancy MAXED
(3 blocks/SM, `warps_active` ~18% ≈ `sm__maximum_warps_per_active_cycle_pct`
18.75%). They are different regimes; profiling the toy made earlier iterations
chase an occupancy/register lever that does not exist in the graded config. Use
`profile_kernel.py` (sweep-matched shapes) for any perf-relevant diagnosis.

```bash
# The ncu metrics that matter most for this project's bottlenecks.
# profile_kernel.py runs batch=8 head_dim=64 (the graded shape), NOT --quick.
sudo env "PATH=$PATH" HOME="$HOME" \
    ncu --kernel-name regex:int8_wmma --launch-count 2 \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,sm__maximum_warps_per_active_cycle_pct \
    python3 profile_kernel.py attn 2>&1 | grep -A2 -E "Metric|int8_"
# ^ ALWAYS filter with --kernel-name + grep — full ncu output floods context.
# env PATH/HOME keeps the user JIT cache + ninja visible under sudo.
```

Priorities and how to read them, measured on the **graded** shape (b=8,
head_dim=64, s=2048):
1. **Attention occupancy is a DEAD END — do not chase it.** The default
   `<128,64>` runs at 3 blocks/SM (`warps_active` ~18% = the 18.75% / 12-warp
   ceiling, co-capped by `occupancy_limit_registers`=3 AND
   `occupancy_limit_shared_mem`=3, 164 regs). Forcing the `<64,64>` config
   (122 regs, 19.5 KB smem) DOES reach 4 blocks/SM (limits 3→4, max-warps→25%,
   warps_active→23.5%) but produced **no speedup** (latency +0–1% seq≥1024,
   +8% seq=512; `tensor_pipe` stayed ~28% with 16 warps as with 12). A −24%
   smem-traffic cut also gave no speedup. **Both occupancy levers are
   empirically exhausted** (README, 2026-06-13) — the bottleneck is the
   per-warp online-softmax dependency chain, not occupancy or smem traffic.
2. **Tensor pipe active %** — ~28% on the graded attention shape vs cuBLAS 60%+,
   pinned by that dependency chain (`ldmatrix → mma → exp/MUFU → pack → mma` +
   cross-tile rescale). More warps / less smem traffic do not move it; only
   restructuring the softmax dependency does (deep, high-risk).
3. **The MLP GEMMs are the live target instead** — compute-bound, multi-stage
   cp.async pipelining unexplored (§4 #1). Check spills/regs with:
   `nvcc -arch=sm_80 -O3 --std=c++17 --ptxas-options=-v -c kernels/int8_mlp.cu -o /dev/null 2>&1 | grep -E "registers|spill"`

Without ncu:

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    ...  # run forward a few times
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

Profile first: `int8_wmma_attention_kernel` (largest share of time), then
`gemm_int8_wmma_f16_kernel` (GEMM1/GEMM2). The `quantize_*` kernels are
~2.5% of forward time — not worth touching first.

## 4. Optimization Targets & Rules

**Optimize first** (by expected return). The before/after ncu numbers behind
every claim below are in `OPTIMIZATION_LOG.md` — this section is the distilled,
still-actionable conclusion only.

> **STATUS 2026-06-13: the INT8 MLP GEMM kernels are perf-EXHAUSTED.** Every
> internal ceiling is closed or proven structurally dead — do not re-open the
> GEMM inner loop. The only remaining *kernel-internal* perf lever in the project
> is target #2 (attention online-softmax dependency chain), deep and high-risk.
> The MLP *forward* yielded a non-kernel win (iter 11): the per-forward W
> transpose (~10% of forward time) is amortized via a prepacked entry point
> (`int8_mlp_forward_prepacked` + `transpose_int8_weights`, transpose-once at
> load; sweep grades this path). **Forward orchestration is now ALSO exhausted:**
> the only other non-GEMM slice was `quantize_rows` (8% of the graded forward),
> and fusing it into GEMM2's load (iter 13, `INT8_MLP_FUSE_QUANT`) REGRESSED
> +45–63% — GEMM2 is wait-bound and the serial on-load convert per K-stage more
> than doubles it; the separate memory-bound quantize_rows is cheaper. Do not
> re-attempt without first decoupling the convert from the mma critical path.

1. ~~**MLP `int8_mlp.cu` GEMMs**~~ — **CLOSED; do not re-attempt any of these**
   (graded shape b=8, s=512, d_model=1024, d_ff=4096; numbers in the log):
   - **smem bank conflicts** — FIXED iter 7 (hand-rolled k-contiguous
     `mma.sync m16n8k32` load, conflict-free bank map; NOT XOR swizzle).
   - **global-load latency (`long_scoreboard`)** — FIXED iter 8 (`STAGE_K` 32→64).
     A true 3-stage cp.async pipeline is a dead end (latency already hidden; a 3rd
     buffer breaks the smem ceiling). cp.async double-buffer is also a negative on
     attention (`INT8_ATTN_DB`).
   - **`wait` (MMA-result dependency, top stall; `tensor_op_imma` 30/38% vs
     cuBLAS 60%+)** — the denser-mma reschedule (iter 10) regressed +1.5–3% and
     left GEMM2 *byte-identical* (ptxas already at its scheduling optimum). Hiding
     `wait` needs more independent accumulator chains, but `acc[2][4][8]`=64 regs
     maxes the register budget.
   - **occupancy — dead from BOTH directions.** Can't raise it at 256 threads
     (iters 8/10, reg-capped at 2 blocks/SM); and raising it via a finer
     4×4/512-thread tile (iter 12, `INT8_MLP_FINE_WARP`) DOES double `warps_active`
     and halve `wait`, but the smaller `acc[2][2][8]` halves operand reuse so
     `tensor_op_imma` net falls and latency regresses +15–28%. Occupancy and
     operand reuse are coupled through the acc-tile register cost.
2. **Attention `int8_attention.cu` WMMA main loop — occupancy well is DRY; only a
   deep algorithmic change remains.** Done, do not re-attempt: the
   `INT8_ATTN_REGPV=1` path (P in registers, `mma.sync` m16n8k32 QK^T + m16n8k16
   P@V; iters 1–6). Both occupancy levers are empirically dead — forcing 4
   blocks/SM (`<64,64>`) gave no speedup, and a −24% smem-traffic cut gave none
   either; `tensor_pipe` stays ~28% regardless. That ceiling is set by the
   **per-warp serial dependency chain** (`ldmatrix → mma → exp/MUFU → pack → mma`
   + the cross-KV-tile online-softmax rescale), which neither more warps nor less
   smem traffic relieves. The ONLY remaining attention lever is breaking that
   softmax dependency chain itself (cheaper/approximate exp, decoupling the
   per-tile rescale) — deeper, higher-risk; attempt only after the MLP is
   exhausted.

**Do not touch:**
- LayerNorm and residual paths stay FP16 (in the `validate_int8.py` block
  harness and any future full-layer integration — never quantize them)
- Existing public interface signatures stay backward-compatible. The
  no-interface-change rule was **relaxed by the owner (2026-06-13)** for the
  accuracy track: new quantization granularities are added as *new* entry
  points / overloads, never by breaking the old ones. Concretely: the per-tensor
  `int8_mlp_forward` (scalar scales, scalar out_scale) is UNCHANGED; per-channel
  work lives in `int8_mlp_forward_per_channel` (C++) and a tensor-scale
  `int8_mlp_forward` pybind overload (returns a per-token out_scale tensor).
  `int8_attention_forward` is untouched.
- The `tests/cuda/` .bin flow (teammates' smoke-test compatibility) — guaranteed
  by keeping the per-tensor `int8_mlp_forward` device signature intact
- Do not retry known negative results: cp.async double-buffering
  (`INT8_ATTN_DB`) and V register prefetch (`INT8_ATTN_VPREFETCH`) both
  measured slower on A100 — reasons documented in README

**Iteration discipline:**
- **One change per iteration = one COHERENT structural change to one kernel.**
  This is NOT a size limit. A single change may rewrite an entire main loop,
  introduce a new smem/register layout, and touch `int8_common.cuh`, as long as
  it is *one idea*. Do not downgrade an ambitious idea into a one-line reorder
  to play it safe — the validation gates + perf gate are the safety net, so a
  failed bold attempt that gets reverted is a better iteration than a committed
  no-op.
- **A change only counts as progress if it MOVES an ncu bottleneck metric**
  beyond run-to-run noise — `sm__warps_active`,
  `sm__pipe_tensor_cycles_active`, or `l1tex__data_pipe_lsu_wavefronts_mem_shared`.
  A latency-neutral micro-opt (no metric moved) is a no-op: report it as such
  and revert it rather than committing, even though it passes the gates.
  **Exception — documented negative results are KEPT, not reverted:** if a bold
  idea did not help but is worth recording (so it is not re-attempted), keep it
  *fully behind an OFF-by-default ablation flag* (default path unchanged) and
  document it in README. Under `evolve.sh` this is signalled by emitting a
  `NEGATIVE_RESULT:` line, which makes the otherwise-neutral change commit
  (preserving the flag + note) instead of being wiped by `git reset --hard`.
  This is the ONLY way a latency-neutral change should be committed.
  **`evolve.sh` also offers this proactively:** when a change is
  latency-neutral but the post-change `ncu` re-profile shows a bottleneck
  metric *moved* beyond noise (mem_shared rel |Δ|≥5%, warps_active |Δ|≥1.0pp,
  or tensor_pipe |Δ|≥2.0pp — e.g. "−24% smem traffic that did not speed
  anything up"), the gate asks the inner claude to gate it behind an OFF flag
  and record what the moved-but-no-speedup metric *rules out*, before falling
  back to reverting. Such empirical dead-ends are exactly what must not be
  silently lost.
- Treat sweep latency as the **median of ≥5 runs**; run-to-run noise on this
  box is ~2%, so any |delta| < 2% is noise, not a result. Never claim a speedup
  from a single sweep.
- Before changing anything, run `python validate_int8.py` and
  `python sweep.py --kernel int8_*` to establish the baseline; record numbers
- After the change: correctness (validate) → performance (sweep) → commit
  only when both pass
- Every commit carries before/after numbers; record negative results too
  (keep them behind ablation flags)

## 5. INT8 Accuracy Validation Process

Run `python validate_int8.py` after **every** optimization. All five gates
must pass:

| Gate | Checks | Criteria |
|---|---|---|
| 1 Math metrics | cosine, top-10 overlap, outlier ratio, NaN/Inf | cos>0.999 (outlier dataset: >0.995), top10>0.8, outlier<0.01 |
| 2 Numeric stability | per-stage absmax ratio, per-stage NaN | ratio ∈ [1/1.5, 1.5]; offending stage names reported |
| 3 Stage error trace | per-stage cosine must not drop >1% vs previous stage | on violation: stop and report the stage; worst-3 stages listed |
| 4 Edge cases | short_seq / long_seq / zeros / constant | no NaN; cos>0.99 (zeros: output must be ≈0) |
| 5 Task-level | **real GPT-2 perplexity on real WikiText-2** (`testdata/real_corpus.txt`), MLP blocks patched with the kernel-faithful per-channel/per-token quant sim | perplexity increase <2% (SKIPs gracefully if `transformers`/corpus missing) |

- **Any gate failure stops the optimization immediately.** Apply the repair
  suggestions printed by validate (in order: per-channel quantization →
  QK^T back to fp16 → adjust scale computation → worst stage back to fp16),
  then re-run from Gate 1. **No skipping gates.**
- A "stage" is a PyTorch simulation mirroring the kernel math step by step
  (the fused kernels expose no intermediates); the simulation's final stage
  is cross-checked against the real kernel output (warns if max diff >
  0.05), so the simulation cannot silently diverge from the kernels.
- **Gate 5 is the realistic gate (since iteration 8).** It measures real GPT-2
  perplexity on WikiText-2 with each MLP block running `_kernel_faithful_mlp`
  (the kernel's exact per-channel weight + per-token activation + **per-token
  output** quant, with GPT-2's b1-before-GELU / b2-after that the fused epilogue
  cannot host). The old random-weight self-consistency Gate 5 falsely reported
  +0.011%; on real weights, per-tensor MLP output quant gives **+64%** perplexity
  (it crushes output channel outliers) — only per-token output quant recovers it
  (−0.07%). Do NOT revert Gate 5 to random weights.

## 6. Test Data

`generate_test_data.py` writes 8 datasets to `testdata/validate/` (plus
`manifest.json` carrying each dataset's rationale and thresholds; summary):

| Dataset | Problem it targets | Key thresholds |
|---|---|---|
| `normal` | Regression baseline: healthy post-LayerNorm distribution; a failure here is a plain bug | cos>0.999, top10>0.8, outlier<0.01 |
| `outlier` | LLM.int8()-style outlier channels (10-100x): inflates per-tensor scales, crushes a token's other dims under per-token scales | cos>0.995, outlier<0.02 |
| `boundary` | Quantized codes piled at ±126/127: clamp asymmetry, rounding mode, scale off-by-one | cos>0.999 |
| `stress` | Heavy tails + 3-decade channel magnitudes + extremely peaky softmax: online-softmax rescale underflow | cos>0.995, top10>0.7, outlier<0.05 |
| `short_seq` | seq=4: scalar fallback path (the MLP fallback still uses static scales — lower precision there is a known state) | cos>0.99, no NaN |
| `long_seq` | seq=4096: 64 KV tiles of online-softmax accumulation depth | cos>0.99, no NaN |
| `zeros` | amax=0 divide-by-zero guard; softmax over all-equal scores | no NaN, output ≈ 0 |
| `constant` | All-ones: softmax must be exactly uniform; single-code quantization must be exact | cos>0.99, no NaN |

The old `tests/gen_testdata.py` is Gaussian-only, its attention scale
interface is stale (saves one per-tensor scale; the kernel takes per-token
arrays), and it has no edge cases — it stays only for the pure-CUDA smoke
tests. All INT8 accuracy validation uses the new flow.
## 7. Iteration Log

The full chronological iteration journey — every optimization's before/after
ncu numbers, validation traces, and negative results — lives in
**`OPTIMIZATION_LOG.md`**, not here. After each optimization append a new entry
there (newest at the bottom) in this format:

```markdown
### Iteration [N] - [YYYY-MM-DD]
- **Change**: which file, which kernel, what structural change
- **Target metric**: which ncu metric (or torch.profiler time) should improve
- **Profiling results**: before/after numbers (latency / TOPS / regs / occupancy)
- **Accuracy validation**: Gate 1-5 pass/fail
- **Conclusion**: pass / fail, failure cause, next direction
```

Keep the *distilled, still-actionable* conclusions (what is exhausted, what not
to re-attempt) in §3–§4 above; keep the blow-by-blow evidence in the log.
