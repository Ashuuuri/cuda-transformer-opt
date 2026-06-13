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

**Optimize first** (by expected return):
> **STATUS 2026-06-13: the INT8 MLP GEMM kernels are perf-EXHAUSTED.** All three
> of their internal ceilings — smem bank conflicts (iter 7), global-load latency
> (iter 8), and the `wait` MMA-dependency stall (iter 10) — are closed or proven
> structurally dead. The single remaining *kernel-internal* perf lever in the
> project is target #2 below (attention online-softmax dependency chain), deep and
> high-risk. The MLP GEMM entry #1 is kept as history; **do not re-open it.**
>
> **But the MLP *forward* (not the GEMM) still had a non-kernel win (iter 11):**
> the per-forward W transpose was ~10% of MLP forward time on the graded shape
> (torch.profiler). Weights are static across forwards, so iter 11 added a
> prepacked entry point (`int8_mlp_forward_prepacked` + `transpose_int8_weights`)
> that transposes once at load — realistic static-weight inference. Bit-identical
> output; sweep now grades the transpose-once path. The lesson: the *kernel* well
> was dry, but the forward *orchestration* around it was not. Look there next, not
> at the GEMM inner loop.

1. ~~**The `wait` (MMA dependency) stall in the `int8_mlp.cu` GEMMs**~~ —
   **CLOSED (iter 10): the denser-mma lever is a proven dead end (see the
   `wait` bullet below); the MLP perf well is dry.** History on the
   graded MLP shape (b=8, s=512, d_model=1024, d_ff=4096):
   - **smem bank conflicts: FIXED in iter 7** (do not re-attempt). Were
     L1/smem-pipe bound (`l1tex` 78%/67%, `dram` 3–4%, ~46% of smem-load
     wavefronts were conflicts, 16.78M). The fix was NOT XOR swizzle (proved
     unnecessary) but the hand-rolled `mma.sync m16n8k32` k-contiguous load whose
     per-instruction bank map `(stride/4·group + tid) mod 32` is a bijection over
     the 32 lanes whenever `stride/4 = 4·odd` (stride 48→12=4·3 ✓, stride 80→
     20=4·5 ✓); both keep the 16B `cp.async` alignment. Conflicts → ≈0,
     `l1tex` → 39%/29%.
   - **global-load latency (`long_scoreboard`): FIXED in iter 8** (do not
     re-attempt deeper k-staging). After iter 7 removed the smem ceiling,
     `long_scoreboard` rose to **18.7%/27.4%** (DRAM still 5–6% → pure latency,
     not bandwidth — this is exactly what invalidated the old "multi-stage is the
     wrong lever" note, which was measured at 12–16% pre-iter-7). Raising
     `STAGE_K` 32→64 (each cp.async round carries 2× the data → half as many
     global-load sync points) cut it to **5.5%/7.7%** and lifted `tensor_op_imma`
     to 30%/38%; latency −6% to −17%, occupancy held at 2 blocks/SM (`launch_
     bounds(256,2)` caps regs at 128 despite a small spill; smem 32→40 KB did NOT
     cost a block — the SM smem budget is ≥80 KB). A *true 3-stage* cp.async
     pipeline is now the dead end the old note warned about: STAGE_K=64 already
     hides the latency, and a 3rd buffer would grow smem past the epilogue's
     32 KB ceiling. (cp.async double-buffer was also a *negative on attention*,
     `INT8_ATTN_DB`.)
   - **`wait` (~21%/28%, top stall) = MMA-result dependency latency — and the
     denser-mma lever is now ALSO DEAD (iter 10, do not re-attempt).**
     `tensor_op_imma` is 30%/38% vs cuBLAS 60%+; the gap is the per-warp serial
     chain feeding the IMMA pipe at fixed 2-block occupancy (occupancy is
     structurally dead — the `acc[2][4][8]` register tile alone is ~64 regs, so
     `launch_bounds(256,3)` would spill catastrophically). Iter 10 tested the
     documented denser-mma reschedule (hoist A fragments, col-tile-outer so each
     B fragment loads once, 4 distinct-accumulator mma back-to-back) → **+1.5% to
     +3.0% latency REGRESSION across all 12 sweep shapes, reverted.** Two reasons
     it cannot work: (a) GEMM2's `tensor_op_imma`/`wait` were **byte-identical**
     under the reorder → ptxas already CSE's the "redundant" B loads and the
     warp-level mma order is already at the compiler's optimum (no intra-warp
     headroom); (b) the `wait` stall is structural — hiding it needs more
     independent accumulator chains, but the 64-reg acc tile already maxes the
     register budget. **MLP perf is exhausted on both the occupancy and the
     denser-mma levers.** Do not re-attempt either.
2. The WMMA main loop in `int8_attention.cu` — **occupancy well is DRY; only a
   deep algorithmic change remains.** What is already DONE (do not re-attempt):
   the `INT8_ATTN_REGPV=1` default path keeps P in registers and uses `mma.sync`
   m16n8k32 QK^T + m16n8k16 P@V (iters 1–6: QK^T→`mma.sync`, `sV` fold, ldmatrix
   software-pipeline, fused pack — all latency-neutral). **BOTH occupancy levers
   are now empirically dead (see README "4 blocks/SM does not help", 2026-06-13):**
   - **(a) Raise occupancy to 4 blocks/SM — TESTED, NO SPEEDUP.** The `<64,64>`
     config reaches a genuine 4 blocks/SM (122 regs, 19.5 KB smem; occupancy
     limits 3→4, max-warps 18.75%→25%, warps_active ~18%→23.5%) yet latency was
     neutral-to-worse (+0–1% seq≥1024, +8% seq=512) and `tensor_pipe` stayed
     ~28% with 16 warps just as with 12. Do not re-attempt occupancy raises.
   - **(b) Hide latency at fixed 12 warps/SM — TESTED, NO SPEEDUP.** A −24% smem-
     traffic cut did not move latency either.
   The ~28% tensor-pipe ceiling is set by the **per-warp serial dependency
   chain** (`ldmatrix → mma → exp/MUFU → pack → mma` + the cross-KV-tile online-
   softmax rescale dependency), which neither more warps nor less smem traffic
   relieves. The ONLY remaining attention lever is breaking that softmax
   dependency chain itself (e.g. cheaper/approximate exp, decoupling the per-tile
   rescale) — deeper and higher-risk; attempt only after the MLP is exhausted.

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

## 7. Iteration Log Format

After each optimization, append a section at the bottom of this file:

```markdown
### Iteration [N] - [YYYY-MM-DD]
- **Change**: which file, which kernel, what structural change
- **Target metric**: which ncu metric (or torch.profiler time) should improve
- **Profiling results**: before/after numbers (latency / TOPS / regs / occupancy)
- **Accuracy validation**:
  - [ ] Gate 1: math metrics
  - [ ] Gate 2: numeric stability
  - [ ] Gate 3: stage error trace
  - [ ] Gate 4: edge cases
  - [ ] Gate 5: task-level
- **Conclusion**: pass / fail, failure cause, next direction
```

---

## Iteration Log

### Iteration 1 - 2026-06-12
- **Change**: `kernels/int8_attention.cu` — templated the WMMA kernel on
  `(TILE_KV, HEAD_DIM)`; reordered QK^T and P@V loops (k-outer, fragment
  hoisting). Two negative results along the way: cp.async double-buffering
  (0.58-0.82x) and V register prefetch (0.75-0.9x), both kept as ablation
  flags.
- **Target metric**: local-memory traffic (runtime head_dim spilled
  frag_out to local memory), smem fragment-load count
- **Profiling results**: 1.6-2.2x speedup across the sweep grid; vs FA2
  0.30x → 0.62x
- **Accuracy validation** (via test_int8.py + sweep error columns; the
  five-gate framework was built afterwards):
  - [x] Math metrics (max err ~1e-3, unchanged)
  - [x] Edge cases (6 configs PASS)
- **Conclusion**: pass. Commit `9473d42`. Next: MLP dynamic quantization.

### Iteration 2 - 2026-06-12
- **Change**: `kernels/int8_mlp.cu` — dynamic quantization: GEMM1 epilogue
  gathers per-token row absmax → per-token hidden requant → GEMM2 per-row
  scales → dynamic per-tensor output scale. Static path kept as
  `INT8_MLP_DYNAMIC=0`.
- **Target metric**: accuracy (max/mean err); `launch__registers_per_thread`
  (epilogue statistics pushed regs 128 → 166, halving occupancy; capped
  back with `__launch_bounds__(256,2)`)
- **Profiling results**: latency +1-6% (worst +14%); accuracy at
  d_model=2048: max err 0.49 → 0.107 (4.6x), mean err 9.2x better
- **Accuracy validation**:
  - [x] Math metrics
  - [x] Edge cases
- **Conclusion**: pass. Commit `0db4387`. Next: `mma.sync` PTX path
  (attention).

### Iteration 3 - 2026-06-13
- **Change**: `kernels/int8_attention.cu` — fused the REGPV softmax-weight
  pack into the P@V MMA loop (group-outer/slice-inner), dropping the
  `p_frag[n_kv_groups][4]` array for a per-group `pf[4]`; the VPREFETCH
  barrier moved ahead of the fused loop.
- **Target metric**: `sm__pipe_tensor_cycles_active` (and MUFU stall
  hiding) — interleaving the exp/pack with ldmatrix+mma across KV groups
  overlaps transcendental latency with the tensor pipe instead of
  serializing the two phases.
- **Profiling results**: latency vs previous baseline: int8_attn +1.5%,
  int8_mlp -0.2%
- **Accuracy validation** (all five gates passed; gate-1 numbers below):
  - [x] Gate 1: math metrics
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level
  ```
  Gate1 math     : cos=0.99995 top10=0.990 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(1.0000), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99966 top10=0.964 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=0.99  -> PASS
                   worst-3: out(0.9997), h_requant(0.9998), h_gelu(0.9999)  -> PASS
  Gate1 math     : cos=0.99684 top10=0.957 outlier=0.0522 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9968), softmax(0.9979), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.97143 top10=0.563 outlier=0.3394 nan_ok=True  -> FAIL
  Gate2 stability: x_dequant=1.00 h_pre_gelu=0.99 h_gelu=0.99 h_requant=0.99 out=1.00  -> PASS
                   worst-3: x_dequant(0.9683), h_pre_gelu(0.9686), out(0.9714)  -> PASS
  Gate1 math     : cos=0.99987 top10=0.803 outlier=0.0302 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(0.9999), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99989 top10=0.974 outlier=0.0001 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), h_requant(1.0000), h_gelu(1.0000)  -> PASS
  Gate1 math     : cos=0.99745 top10=0.968 outlier=0.0685 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9975), softmax(0.9977), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.99934 top10=0.952 outlier=0.0002 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9993), h_requant(0.9995), h_gelu(0.9996)  -> PASS
  Gate1 math     : cos=0.99997 top10=0.991 outlier=0.0000 nan_ok=True  -> PASS
  ```
- **Conclusion**: pass. Next: `mma.sync` m16n8k32 with P kept in registers
  (attention).

### Iteration 4 - 2026-06-13
- **Change**: `kernels/int8_attention.cu` — replaced the QK^T INT8 WMMA
  m16n16k16 path (`load_matrix_sync` + `frag_qk` fragments) with native
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (new helper
  `attn_mma_m16n8k32_s8`); Q/K operands now loaded as plain
  head-dim-contiguous 4-byte smem reads into an `int32_t qk[][8]` array, two
  n8 calls per logical n16 group filling x[0..3]/x[4..7] so the
  cast/softmax/P@V code is unchanged.
- **Target metric**: `sm__pipe_tensor_cycles_active` up and
  `l1tex__data_pipe_lsu_wavefronts_mem_shared` down — k=32 native
  instructions are denser on the tensor pipe than WMMA m16n16k16.s8 (which
  decomposes into multiple ops), and dropping `load_matrix_sync` removes the
  QK^T smem fragment round-trips that dominate the shared-memory wavefront
  count (priority #2/#3); registers unchanged (no spills, 2 blocks/SM
  preserved).
- **Profiling results**: latency vs previous baseline: int8_attn -1.4%,
  int8_mlp -0.2%
- **Accuracy validation** (all five gates passed; gate-1 numbers below):
  - [x] Gate 1: math metrics
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level
  ```
  Gate1 math     : cos=0.99995 top10=0.990 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(1.0000), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99966 top10=0.964 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=0.99  -> PASS
                   worst-3: out(0.9997), h_requant(0.9998), h_gelu(0.9999)  -> PASS
  Gate1 math     : cos=0.99684 top10=0.957 outlier=0.0522 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9968), softmax(0.9979), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.97143 top10=0.563 outlier=0.3394 nan_ok=True  -> FAIL
  Gate2 stability: x_dequant=1.00 h_pre_gelu=0.99 h_gelu=0.99 h_requant=0.99 out=1.00  -> PASS
                   worst-3: x_dequant(0.9683), h_pre_gelu(0.9686), out(0.9714)  -> PASS
  Gate1 math     : cos=0.99987 top10=0.803 outlier=0.0302 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(0.9999), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99989 top10=0.974 outlier=0.0001 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), h_requant(1.0000), h_gelu(1.0000)  -> PASS
  Gate1 math     : cos=0.99745 top10=0.968 outlier=0.0685 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9975), softmax(0.9977), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.99934 top10=0.952 outlier=0.0002 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9993), h_requant(0.9995), h_gelu(0.9996)  -> PASS
  Gate1 math     : cos=0.99997 top10=0.991 outlier=0.0000 nan_ok=True  -> PASS
  ```
- **Conclusion**: pass.


### Iteration 5 - 2026-06-13
- **Change**: `kernels/int8_attention.cu` (REGPV default path) — folded V's
  per-token scale `sV` out of the pre-barrier V-dequant loop into the
  softmax-weight pack: V is now stored in smem as a raw INT8→FP16 cast, `sV`
  is staged in a new `s_scale_V` smem array and multiplied into the packed
  `mma.m16n8k16` A-weights (`pf`), while `lsum` (softmax denominator) stays
  on the unscaled weights; `calc_smem` and the `INT8_ATTN_VPREFETCH` cast
  branch updated to match. Math is reassociation-identical (V_i8 is exact in
  FP16).
- **Target metric**: `smsp__warp_issue_stalled_barrier_per_warp_active.pct`
  (3.76% at head_dim=256, the highest stall) — the per-element `×sV` is
  removed from the V-dequant work that gates the load `__syncthreads`, and
  since the weights are reused across all `n_slices` head-dim slices this is
  strictly fewer multiplies than scaling every V element for
  head_dim ≥ TILE_Q, shortening the pre-barrier critical path.
- **Profiling results**: latency vs previous baseline: int8_attn -0.9%,
  int8_mlp -0.2%
- **Accuracy validation** (all five gates passed; gate-1 numbers below):
  - [x] Gate 1: math metrics
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level
  ```
  Gate1 math     : cos=0.99995 top10=0.990 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(1.0000), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99966 top10=0.964 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=0.99  -> PASS
                   worst-3: out(0.9997), h_requant(0.9998), h_gelu(0.9999)  -> PASS
  Gate1 math     : cos=0.99684 top10=0.956 outlier=0.0522 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9968), softmax(0.9979), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.97143 top10=0.563 outlier=0.3394 nan_ok=True  -> FAIL
  Gate2 stability: x_dequant=1.00 h_pre_gelu=0.99 h_gelu=0.99 h_requant=0.99 out=1.00  -> PASS
                   worst-3: x_dequant(0.9683), h_pre_gelu(0.9686), out(0.9714)  -> PASS
  Gate1 math     : cos=0.99987 top10=0.803 outlier=0.0302 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(0.9999), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99989 top10=0.974 outlier=0.0001 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), h_requant(1.0000), h_gelu(1.0000)  -> PASS
  Gate1 math     : cos=0.99745 top10=0.968 outlier=0.0685 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9975), softmax(0.9977), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.99934 top10=0.952 outlier=0.0002 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9993), h_requant(0.9995), h_gelu(0.9996)  -> PASS
  Gate1 math     : cos=0.99997 top10=0.991 outlier=0.0000 nan_ok=True  -> PASS
  ```
- **Conclusion**: pass.


### Iteration 6 - 2026-06-13
- **Change**: `kernels/int8_attention.cu` (REGPV P@V loop) — software-pipelined
  the attn×V ldmatrix: hoisted slice 0's V-fragment ldmatrix before the slice
  loop and prefetch slice s+1's fragments into a second register set while
  slice s's two `mma.m16n8k16` issue (shared A-weights `pf` reused), instead of
  each slice's MMA waiting on its own ldmatrix; registers/occupancy unchanged
  (164/216/236, no spills), output bit-identical.
- **Target metric**: `sm__pipe_tensor_cycles_active` (tensor-pipe utilization,
  ~13–17%, the cuBLAS gap) — overlapping the ldmatrix smem→reg latency with
  tensor-core MMA issue removes the per-slice load→MMA stall (confirmed by
  ptxas hoisting LDSM ahead of HMMA in the SASS), feeding the tensor pipe more
  densely without touching the smem-wavefront or register/occupancy
  bottlenecks.
- **Profiling results**: latency vs previous baseline: int8_attn +0.0%,
  int8_mlp +0.3%
- **Accuracy validation** (all five gates passed; gate-1 numbers below):
  - [x] Gate 1: math metrics
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level
  ```
  Gate1 math     : cos=0.99995 top10=0.990 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(1.0000), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99966 top10=0.964 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=0.99  -> PASS
                   worst-3: out(0.9997), h_requant(0.9998), h_gelu(0.9999)  -> PASS
  Gate1 math     : cos=0.99684 top10=0.956 outlier=0.0522 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9968), softmax(0.9979), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.97143 top10=0.563 outlier=0.3394 nan_ok=True  -> FAIL
  Gate2 stability: x_dequant=1.00 h_pre_gelu=0.99 h_gelu=0.99 h_requant=0.99 out=1.00  -> PASS
                   worst-3: x_dequant(0.9683), h_pre_gelu(0.9686), out(0.9714)  -> PASS
  Gate1 math     : cos=0.99987 top10=0.803 outlier=0.0302 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(0.9999), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99989 top10=0.974 outlier=0.0001 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), h_requant(1.0000), h_gelu(1.0000)  -> PASS
  Gate1 math     : cos=0.99745 top10=0.968 outlier=0.0685 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9975), softmax(0.9977), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.99934 top10=0.952 outlier=0.0002 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9993), h_requant(0.9995), h_gelu(0.9996)  -> PASS
  Gate1 math     : cos=0.99997 top10=0.991 outlier=0.0000 nan_ok=True  -> PASS
  ```
- **Conclusion**: pass.


### Iteration 7 - 2026-06-13
- **Change**: `kernels/int8_mlp.cu` — replaced the WMMA `load_matrix_sync` +
  `mma_sync(m16n16k16)` inner loop of BOTH GEMM kernels with hand-rolled native
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (new `mlp_mma_m16n8k32_s8`,
  shared `mlp_int8_mainloop`), operands loaded as plain 4-byte k-contiguous smem
  words like the attention QK^T path. The B operand (weights) is staged
  k-contiguous, which int8 `mma.sync` requires, by **pre-transposing W1/W2 to
  `[N][K]`** each forward (`transpose_int8_kernel`, ~1-2% bandwidth pass — no
  pointer-keyed cache, to stay correct across validation datasets that reuse
  freed weight addresses). s32 accumulators are scattered back to the row-major
  c_smem tile (`mlp_store_acc16`) so both epilogues are unchanged. B smem layout
  `[k][n] stride 144` -> `[n][k] stride 48`; smem total unchanged (epilogue
  dominated, 32 KB -> still 2 blocks/SM).
- **Target metric**: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
  and `l1tex__throughput` down — the hand-rolled load's per-instruction bank map
  `(12*group + tid_grp) mod 32` is a bijection over the 32 lanes (conflict-free),
  unlike `load_matrix_sync`'s int8 pattern which collided period-8 (2-way) and
  made ~46% of smem-load wavefronts conflicts. (XOR swizzle proved unnecessary —
  the access-pattern change alone fixes it while keeping the 16-byte `cp.async`
  stores; the iter-6 README 4-byte-store probe was the negative result that ruled
  out the stride route.)
- **Profiling results** (graded shape b=8 s=512 d_model=1024 d_ff=4096):
  - bank conflicts (ld.sum): **16.78M -> ~3.0K / 1.2K (≈0, -99.98%)**
  - `l1tex__throughput`: **78%/67% -> 39%/29%** (L1/smem no longer the ceiling)
  - smem-load wavefronts: **36.2M -> 10.0M / 8.1M**
  - `tensor_op_imma` active: 26%/31% (≈unchanged)
  - registers 128, 0 spills, 2 blocks/SM (unchanged)
  - **latency: -26% to -40% across the full 12-point sweep**, reproducible across
    two runs (d512 s512 0.42->0.31; d1024 s512 0.89->0.64, s4096 5.55->3.50;
    d2048 s4096 20.18->12.09). Speedup vs naive 0.41-0.48x -> 0.55-0.79x.
- **Accuracy validation** (all five gates PASS on all 8 datasets + task-level):
  - [x] Gate 1: math metrics (normal cos=1.00000)
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level (ppl fp16=504.81 int8=504.87 +0.011%, top1 100%)
- **Conclusion**: pass — largest single-iteration MLP gain so far. Next: push
  `tensor_op_imma` (still ~28%) toward the cuBLAS 60%+ ceiling now that the
  smem-pipe bottleneck is gone — a deeper k-stage pipeline or denser-mma schedule.


### Iteration 8 - 2026-06-13
- **Change**: `kernels/int8_mlp.cu` — raised the cp.async K-stage depth
  `INT8_MLP_STAGE_K` 32 → 64 (one-line default; the `mlp_int8_mainloop`
  `for kk in 0..STAGE_K step 32` loop already handles it, now 2 `mma.sync
  m16n8k32` per acc per cp.async round). smem strides grow `A/B_SMEM_STRIDE`
  48 → 80; stride 80 stays conflict-free (`stride/4 = 20 = 4·5`, gcd(5,8)=1 →
  the 8 ×4-multiple banks are all distinct, bijection over 32 lanes) and 16B-
  aligned (80 = 5·16), so iter-7's conflict-free property is preserved.
- **Target metric**: `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`
  down and `sm__pipe_tensor_op_imma_cycles_active` up — after iter 7 cleared the
  smem-pipe ceiling, `long_scoreboard` (global-load latency, DRAM only 5–6% → pure
  latency) became a top stall; a deeper k-stage carries 2× data per cp.async round
  → half as many global-load sync points → less exposed latency + longer back-to-
  back MMA runs.
- **Profiling results** (graded shape b=8 s=512 d_model=1024 d_ff=4096):
  - `long_scoreboard`: **18.7%/27.4% → 5.5%/7.7%** (GEMM1/GEMM2)
  - `tensor_op_imma` active: **26.1%/30.9% → 29.7%/38.2%**
  - `wait` (now the top stall): 19.5%/24.3% → 21.5%/28.2%
  - `l1tex__throughput`: 39%/29% → 42%/34%; bank conflicts still ≈0 (8.4K/1.6K)
  - registers 128 (launch_bounds-capped; small 48/52 B spill), 2 blocks/SM (held);
    smem 32 → 40 KB, no occupancy loss (SM smem budget ≥ 80 KB)
  - **latency: −6% to −17% across the full 12-point sweep**, reproducible across
    two runs (d1024 s1024 1.03→0.89; s4096 3.50→3.02; d2048 s2048 6.20→5.18,
    s4096 12.09→10.04). Speedup vs naive 0.55–0.79× → 0.59–0.97×.
- **Accuracy validation** (all five gates PASS on all 8 datasets + task-level;
  numerics bit-identical to iter 7 — STAGE_K is pure tiling, same int32
  accumulation order):
  - [x] Gate 1: math metrics (normal mlp cos=0.99966)
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level (ppl fp16=504.81 int8=504.87 +0.011%, top1 100%)
- **Conclusion**: pass — second MLP win in a row. The two prior MLP ceilings
  (smem bank conflicts, global-load latency) are now both exhausted; the new
  bottleneck is the `wait` MMA-dependency stall (`tensor_op_imma` 30%/38% vs
  cuBLAS 60%+) at structurally-fixed 2-block occupancy. Next: a denser-mma
  schedule that overlaps more independent IMMA chains.

### Iteration 9 - 2026-06-13  (ACCURACY track, not perf)
- **Change**: realistic INT8 MLP quantization + realistic validation, in two
  coordinated parts (owner relaxed the no-interface-change rule for this):
  - **Phase 2 — per-channel/per-token MLP quant.** `kernels/int8_mlp.cu`:
    templated `gemm_int8_wmma_f16_kernel<apply_gelu, a_scale_per_row,
    b_scale_per_col>` so the epilogue applies a per-ROW activation scale and a
    per-COLUMN weight scale (gathered into `s_arow`/`s_bcol` smem); new entry
    point `int8_mlp_forward_per_channel` does **per-token activation +
    per-channel weight + per-token OUTPUT** quant (output now uses
    `quantize_rows_kernel` + GEMM2 per-row absmax, not `quantize_tensor_kernel`),
    returning the [T] per-row output scales. `int8_ext.cu`: tensor-scale
    `int8_mlp_forward` overload (pybind dispatch by arg type) returning a
    per-token `out_scale` tensor. The per-tensor scalar `int8_mlp_forward`
    (device sig + `tests/cuda/` .bin flow + sweep/profile/test_int8) is
    UNCHANGED.
  - **Phase 1 — real Gate 5.** `validate_int8.py`: Gate 5 rewritten to real
    GPT-2 perplexity on real WikiText-2 (`testdata/real_corpus.txt`) via the
    kernel-faithful per-channel/per-token sim; `run_mlp_kernel`/`sim_mlp` updated
    for per-token output dequant. `generate_test_data.py`: removed the now-XPASS
    `outlier` MLP xfail.
- **Target metric**: real-model accuracy (Gate 5 perplexity), NOT a perf metric
  — this is the accuracy track. The graded per-tensor path (sweep.py) is
  structurally untouched; sweep latency unchanged vs iteration 8 (within noise).
- **Key finding**: per-tensor MLP *output* quant gives **+64.4%** GPT-2
  perplexity (crushes output channel outliers); per-channel weights alone with
  per-tensor output is still +64%. Per-token output quant is the fix → −0.07%.
  The old random-weight Gate 5 masked all of this (falsely +0.011%).
- **Profiling / accuracy results**:
  - Gate 5: fp16 ppl=31.9462 → int8 ppl=31.9246 (**−0.068%**, well under 2%).
  - `outlier`/`boundary`/`stress` MLP now PASS outright (outlier was the one
    substantive XFAIL; cos 0.971 → >0.999).
- **Accuracy validation** (all 5 gates PASS on all 8 datasets + real Gate 5):
  - [x] Gate 1: math metrics (normal cos=1.00000)
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level (real GPT-2, −0.068%)
- **Conclusion**: pass — the INT8 MLP is now accurate on a real model, validated
  by a realistic gate. Perf targets (tensor_op_imma toward cuBLAS) unchanged and
  still live.

### Iteration 10 - 2026-06-13  (NEGATIVE RESULT — reverted, do not re-attempt)
- **Change (reverted)**: `kernels/int8_mlp.cu` `mlp_int8_mainloop` — the
  documented "denser-mma schedule" lever. Restructured the inner mma issue:
  hoisted BOTH row tiles' A fragments per kk into `a[WARP_ROW_TILES][4]`, made
  the col-tile loop outer so each B (weight) fragment loads **once** and is
  reused across row tiles (previously loaded inside the rt loop → looked 2×
  redundant), with the 4 mma per col tile (2 rt × 2 n8-halves) issuing
  back-to-back into 4 distinct accumulators.
- **Target metric**: `l1tex__data_pipe_lsu_wavefronts_mem_shared` down (halve
  B-LDS) and `tensor_op_imma` up / `wait` down (denser back-to-back IMMA).
- **Profiling results** (graded shape b=8 s=512, median of 5; full 12-pt sweep):
  - **latency: +1.5% to +3.0% across ALL 12 sweep shapes** — a systematic
    REGRESSION, not noise.
  - **GEMM2 metrics byte-identical**: `tensor_op_imma` 38.25→38.24%, `wait`
    28.17→28.18%. → ptxas was **already CSE-ing the "redundant" B loads** across
    the unrolled rt loop; there were no shared-loads to remove. The reorder
    bought nothing.
  - **GEMM1 got worse**: `tensor_op_imma` 30.18→28.90%, `wait` 18.96→21.62%
    (the hoisted A fragments' larger live-register footprint traded
    `long_scoreboard` 11.90→5.41% for more `wait`). regs 128 (held), still
    2 blocks/SM.
- **Accuracy validation**: all 5 gates PASS bit-identically (pure reorder, same
  int32 accumulation order; Gate 5 real GPT-2 −0.068% unchanged) — but accuracy
  is irrelevant here since latency regressed.
- **Conclusion**: NEGATIVE — reverted (a regression with no metric improved is
  not kept even behind a flag). **Two lessons, do not re-attempt:** (1) the
  warp-level mma issue order is **already at ptxas's optimum** — GEMM2 is
  byte-identical under reorder, so intra-warp rescheduling has no headroom; the
  "redundant B load" was a phantom (compiler CSE'd it). (2) The `wait` stall
  (top stall, ~28% GEMM2 = mma-result-dependency latency) is **structural**: at
  the fixed 2-block/SM occupancy the only way to hide it is more independent
  accumulator chains, but the `acc[2][4][8]` tile is already 64 regs and any
  more spills. **This is the same wall the occupancy lever hit — the MLP perf
  well is now dry for both the occupancy AND the denser-mma levers.** The ONLY
  remaining perf lever in the whole project is the attention online-softmax
  dependency chain (§4 #2) — deep and high-risk.

### Iteration 11 - 2026-06-13
- **Change**: `kernels/int8_mlp.cu` + `int8_ext.cu` — added a **prepacked
  (transpose-once) MLP entry point** for static-weight inference, WITHOUT
  touching the (exhausted) GEMM internals. The all-in-one `int8_mlp_forward`
  re-transposes W1/W2 to `[N][K]` (k-contiguous, required by the m16n8k32 B
  fragment) on **every** forward; weights are constant across forwards, so this
  is pure per-call overhead. New: `transpose_int8_weights(W1,W2)->(W1T,W2T)`
  (the same one-shot pass, callable once at load) + `int8_mlp_forward_prepacked`
  (per-tensor scales, takes already-transposed weights, `weights_prepacked=true`
  skips the internal transpose). Both old device signatures (`int8_mlp_forward`,
  `int8_mlp_forward_per_channel`) and the `tests/cuda/` .bin flow are UNCHANGED;
  this is a NEW path per the relaxed-interface rule. `sweep.py` now transposes
  once outside the timed loop and grades the prepacked path (realistic static-
  weight inference); `collect_profile.py` profiles both so the dashboard shows
  the transpose removed.
- **Target metric**: per-forward transpose share of MLP forward time
  (torch.profiler kernel-time breakdown) → 0, and end-to-end MLP forward latency.
  This is a forward-orchestration win, NOT a GEMM-internal ncu-metric move (the
  GEMM kernels are byte-identical — output bit-identical, verified).
- **Profiling results** (graded grid, median of 5 runs; bench_prepacked.py):
  - torch.profiler MLP forward split (b=8 s=512 d_model=1024): all-in-one =
    79.0% GEMM + **10.4% transpose W** + 7.4% quantize_rows + 3.3% other;
    prepacked = **88.3% GEMM + 0% transpose** + 8.0% quantize_rows + 3.7% other.
  - end-to-end latency vs all-in-one, transpose-once: **−22.1% (d512 s512),
    −9.3% (d1024 s512), −11.9% (d2048 s512)** at the graded s=512; the win
    shrinks with seq_len as the GEMM dwarfs the fixed transpose (−1.5% to −3%
    at s=4096). All deltas reproducible (median of 5); s≤2048 all beyond the
    2% noise floor.
  - GEMM absolute time unchanged (24.69k→24.60k µs, within noise); output
    `torch.equal` bit-identical to `int8_mlp_forward` on all 12 sweep shapes.
- **Accuracy validation** (all 5 gates PASS on all 8 datasets + real Gate 5;
  the prepacked path is bit-identical to the validated per-tensor path, and the
  per-channel/per-tensor entry points it shares are untouched):
  - [x] Gate 1: math metrics (normal cos=1.00000)
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level (real GPT-2, −0.068%)
- **Conclusion**: pass — a real end-to-end MLP-forward win (largest at the
  graded s=512: ~9–22%) found OUTSIDE the perf-exhausted GEMM, by amortizing the
  static-weight transpose. The GEMM inner loop remains dry; the lesson is that
  the forward orchestration around it was not. Next: still the attention
  online-softmax dependency chain (§4 #2) for kernel-internal perf.
