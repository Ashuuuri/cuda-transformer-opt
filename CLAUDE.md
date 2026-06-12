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
| `kernels/int8_mlp.cu` | **INT8 MLP** (primary target): 128×128-tile INT8 WMMA ×2, dynamic quantization (per-token hidden scales) |
| `kernels/int8_common.cuh` | Shared device helpers: GELU, f32→i8, cp.async |
| `kernels/quant_utils.cu` | Per-tensor quantize/dequantize utility kernels |
| `kernels/*_ext.cu` | pybind11 bindings (for torch `cpp_extension.load`) |
| `tests/cuda/test_*.cu` + `tests/gen_testdata.py` | Pure-CUDA correctness smoke tests (read `.bin` files) |
| `tests/test_*.py` | Python correctness + benchmarks (vs FA2 / cuBLAS) |
| `generate_test_data.py` | **8-distribution INT8 validation datasets** (see §6) |
| `validate_int8.py` | **Five-gate accuracy validation** (see §5) |
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

```bash
# Environment: local A100-SXM4-40GB, CUDA 12.8, system PyTorch 2.7
# Deps: ninja, pybind11-dev (both installed); no module load (not Perlmutter)

# Pure-CUDA smoke test (run `python tests/gen_testdata.py` once for .bin data)
make test_int8 && ./test_int8

# Python correctness + benchmark (JIT compile, ~2 min first time, cached)
python tests/test_int8.py --quick     # correctness only
python tests/test_int8.py             # + FA2 / FP16 kernel / cuBLAS baselines

# INT8 five-gate accuracy validation (the gate for every optimization)
python generate_test_data.py          # one-time: generate the 8 datasets
python validate_int8.py               # all datasets × both kernels × 5 gates
python validate_int8.py --dataset outlier --kernel mlp   # filters

# Performance sweeps (refresh results/*.csv + figures/)
python sweep.py --kernel int8_attn
python sweep.py --kernel int8_mlp
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

```bash
# The ncu metrics that matter most for this project's bottlenecks:
sudo ncu --kernel-name regex:int8_wmma --launch-count 3 \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
launch__registers_per_thread,launch__occupancy_limit_registers \
    python tests/test_int8.py --quick 2>&1 | grep -A2 -E "Metric|int8_"
# ^ ALWAYS filter with --kernel-name + grep — full ncu output floods context.
```

Priorities and how to read them:
1. **`launch__registers_per_thread`** — this project's recurring silent
   killer: >128 regs (at 256 threads/block) drops 2 blocks/SM to 1, an
   instant -35%. Check after every epilogue change:
   `nvcc -arch=sm_80 -O3 --std=c++17 --ptxas-options=-v -c kernels/int8_mlp.cu -o /dev/null 2>&1 | grep -E "registers|spill"`
2. **MIO/smem stalls** (`stalled_mio_throttle`, `mem_shared wavefronts`) —
   the wmma scores round-trip through smem is the structural bottleneck.
3. **Tensor pipe active %** — currently ~10-15%; cuBLAS reaches 60%+.

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
1. The WMMA main loop in `int8_attention.cu` (~1.6x gap to FA2 remains;
   next big lever: `mma.sync` m16n8k32 with P kept in registers)
2. GEMM tiling/pipeline in `int8_mlp.cu` (already matches bare cuBLAS INT8
   GEMMs; beating FP16 cuBLAS needs a deeper pipeline)

**Do not touch:**
- LayerNorm and residual paths stay FP16 (in the `validate_int8.py` block
  harness and any future full-layer integration — never quantize them)
- Public interface signatures (`int8_attention_forward` / `int8_mlp_forward`)
- The `tests/cuda/` .bin flow (teammates' smoke-test compatibility)
- Do not retry known negative results: cp.async double-buffering
  (`INT8_ATTN_DB`) and V register prefetch (`INT8_ATTN_VPREFETCH`) both
  measured slower on A100 — reasons documented in README

**Iteration discipline:**
- **One change per iteration** (one structural change to one kernel)
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
| 5 Task-level | full transformer block on `normal` | perplexity increase <2%, accuracy degradation <1% |

- **Any gate failure stops the optimization immediately.** Apply the repair
  suggestions printed by validate (in order: per-channel quantization →
  QK^T back to fp16 → adjust scale computation → worst stage back to fp16),
  then re-run from Gate 1. **No skipping gates.**
- A "stage" is a PyTorch simulation mirroring the kernel math step by step
  (the fused kernels expose no intermediates); the simulation's final stage
  is cross-checked against the real kernel output (warns if max diff >
  0.05), so the simulation cannot silently diverge from the kernels.

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
