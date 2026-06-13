# cuda-transformer-opt

CUDA-optimized Transformer kernels for CS 5220 (Spring 2025). We implement fused FP16 and INT8-quantized attention and MLP kernels, then benchmark them against PyTorch baselines and industry references (FlashAttention-2, cuBLAS).

## Team

| Member    | Kernel files | Test files |
|-----------|-------------|------------|
| Jonathan  | `kernels/attention.cu` | `tests/cuda/test_attention.cu`, `tests/test_attention.py` |
| Shengjing | `kernels/mlp.cu` | `tests/cuda/test_mlp.cu`, `tests/test_mlp.py` |
| Heling    | `kernels/int8_attention.cu`, `kernels/int8_mlp.cu`, `kernels/quant_utils.cu` | `tests/cuda/test_int8.cu`, `tests/test_int8.py` |

## Project Structure

```
cuda-transformer-opt/
├── kernels/                    # CUDA kernel source code
│   ├── attention.cu            # FP16 fused attention
│   ├── mlp.cu                  # FP16 fused MLP
│   ├── int8_attention.cu       # INT8 quantized attention
│   ├── int8_mlp.cu             # INT8 quantized MLP
│   └── quant_utils.cu          # Shared quantization helpers
├── tests/
│   ├── cuda/                   # Pure CUDA correctness tests
│   │   ├── test_utils.h        # Shared: load_bin, check_result, parse_config
│   │   ├── test_attention.cu
│   │   ├── test_mlp.cu
│   │   └── test_int8.cu
│   ├── gen_testdata.py         # Generate test inputs + reference answers
│   ├── test_attention.py       # Python benchmark + industry comparison
│   ├── test_mlp.py
│   └── test_int8.py
├── testdata/                   # Generated .bin files (git-ignored)
├── baseline.py                 # PyTorch FP16 reference implementations
├── benchmark.py                # GPU timing harness
├── correctness.py              # Numerical correctness checker
├── Makefile                    # CUDA compilation targets
├── sweep.py                    # Parameter sweep (TBD)
└── README.md
```

## Kernel Interfaces

All kernels use row-major contiguous layout. Do not change these signatures.

```cpp
// attention.cu — Jonathan
void attention_forward(
    const half* Q, const half* K, const half* V, half* out,
    int batch, int heads, int seq_len, int head_dim
);  // scale = 1/sqrt(head_dim), computed internally

// mlp.cu — Shengjing
void mlp_forward(
    const half* x, const half* W1, const half* W2, half* out,
    int batch, int seq_len, int d_model, int d_ff
);  // GELU uses tanh approximation

// int8_attention.cu — Heling
void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, int8_t* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int batch, int heads, int seq_len, int head_dim
);

// int8_mlp.cu — Heling
void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, int8_t* out,
    const float* scale_x, const float* scale_W1, const float* scale_W2,
    int batch, int seq_len, int d_model, int d_ff
);
```

## Setup on Perlmutter (NERSC)

### 1. Load the environment

```bash
module load pytorch
```

This provides Python 3.12, PyTorch 2.8.0, and CUDA 12.9. Do **not** load the `python` module separately — it will conflict.

### 2. Get a GPU node

```bash
salloc -A m4341_g -C "gpu&hbm40g" -N 1 -t 00:30:00 -q interactive
```

### 3. Generate test data (first time only)

```bash
python tests/gen_testdata.py
```

This creates `testdata/small/` (debug) and `testdata/large/` (validation) with inputs and PyTorch-computed reference answers. Uses a fixed random seed (42) so everyone gets identical test data.

**Note:** `testdata/` is git-ignored because `.bin` files are large binaries. Each person runs `gen_testdata.py` locally after cloning — the fixed seed ensures identical results.

## Development Workflow

Each person works independently on their `.cu` file. Nobody needs to touch shared files.

### Step 1: Compile + correctness check (pure CUDA, fast iteration)

```bash
make test_attention && ./test_attention    # Jonathan
make test_mlp && ./test_mlp                # Shengjing
make test_int8 && ./test_int8              # Heling
```

Reads `.bin` test data from `testdata/`, runs your kernel, compares output against pre-computed reference. Reports PASS/FAIL + max error. Uses `tests/cuda/test_utils.h` for shared utilities.

### Step 2: Benchmark + industry comparison (Python)

```bash
python tests/test_attention.py    # Jonathan
python tests/test_mlp.py          # Shengjing
python tests/test_int8.py         # Heling
```

Each script reports:
- Correctness vs PyTorch baseline
- Latency comparison (your kernel vs naive baseline vs industry reference)
- TFLOPS and A100 utilization %

## Benchmark Comparisons

| Kernel | Comparison baselines |
|--------|---------------------|
| FP16 Attention (Jonathan) | Naive PyTorch, FlashAttention-2 (`F.scaled_dot_product_attention`) |
| FP16 MLP (Shengjing) | Naive PyTorch, cuBLAS (`torch.mm`) |
| INT8 Attention (Heling) | FP16 baseline, FP16 kernel (Jonathan's) |
| INT8 MLP (Heling) | FP16 baseline, cuBLAS INT8 (`torch._int_mm`) |

### A100 theoretical peaks

| Precision | Peak (without sparsity) |
|-----------|------------------------|
| FP16 Tensor Core | 312 TFLOPS |
| INT8 Tensor Core | 624 TOPS |

Source: [NVIDIA A100 Data Sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) (applies to both PCIe and SXM4 variants).

### Industry comparison references

Benchmark scripts automatically measure these on the same GPU for apples-to-apples comparison.

**Important:** Perlmutter has both A100-PCIe-40GB and A100-SXM4-40GB nodes. The `salloc` constraint `gpu&hbm40g` does not distinguish between them, so you may get either variant across sessions. SXM4 has ~30% higher memory bandwidth than PCIe, which affects latency. **All benchmark numbers for the final report must be collected in a single `salloc` session** to ensure they are measured on the same node.

- **FlashAttention-2**: `F.scaled_dot_product_attention` — PyTorch 2.x built-in, backed by [Dao (2023)](https://arxiv.org/abs/2307.08691). Reports 50-73% A100 utilization in the paper.
- **cuBLAS FP16**: `torch.mm` — PyTorch's matrix multiply, backed by NVIDIA cuBLAS.
- **cuBLAS INT8**: `torch._int_mm` — PyTorch's INT8 integer matrix multiply, backed by cuBLAS `cublasLtMatmul`.

## Correctness Tolerances

- **FP16 kernels**: `atol = 1e-2`
- **INT8 kernels**: `atol = 0.1`

When a test fails, the checker prints diagnostics: worst error location, first 8 elements comparison, and percentage of elements exceeding tolerance.

---

## Continued INT8 Optimization (post-course, `opt-dev`)

After the course wrapped, the INT8 kernels were further optimized on a
dedicated A100-SXM4-40GB (CUDA 12.8). All numbers below are from that
machine (batch=8 sweeps; see `results/*.csv` for full data).

### Benchmark hardening

- INT8 attention is now compared against **FlashAttention-2** and the
  **FP16 WMMA kernel** (same engineering level, isolates the INT8 effect),
  not just naive PyTorch.
- INT8 MLP is compared against the **full cuBLAS INT8 pipeline**
  (`_int_mm` + dequant/GELU/requant — the same work our kernel does), with
  bare `2x _int_mm` kept as a GEMM-only lower bound.
- Sweeps emit `kernel_max_err` / `kernel_mean_err` columns (accuracy vs the
  FP16 reference) and use a blended INT8/FP16 peak (416 TOPS) for attention
  utilization. New entries: `python sweep.py --kernel int8_attn | int8_mlp`.

### INT8 attention: 1.6–2.2x (commit 9473d42)

The WMMA kernel is now templated on `(TILE_KV, HEAD_DIM)`. With a runtime
`head_dim`, the fragment arrays are dynamically indexed and nvcc spills them
to local memory — every MMA and every online-softmax rescale paid a local
load+store. Compile-time `HEAD_DIM` plus k-outer loop ordering (each Q /
scores fragment loaded once per k-step instead of once per KV-group/slice)
gives 1.6x at head_dim 64/128 and 2.0–2.2x at head_dim 256, closing the
FlashAttention-2 gap from 0.30x to ~0.62x. Accuracy unchanged (~1e-3).

Negative results (kept as ablation flags, both measured slower on A100):

- `INT8_ATTN_DB=1` — cp.async double buffering. Fitting two K/V buffers
  forces TILE_KV down (128→64); the doubled per-tile softmax/barrier
  overhead outweighs the overlap (0.58–0.82x).
- `INT8_ATTN_VPREFETCH=1` — V register prefetch. The extra block-wide
  barrier before attn×V costs more than the hidden global latency
  (0.75–0.9x). Warp parallelism already covers the loads.

**Negative result — raising occupancy to 4 blocks/SM does not help (2026-06-13).**
On the graded shape (b=8, head_dim=64, the only head_dim sweep.py grades) the
default `<TILE_KV=128, HEAD_DIM=64>` kernel runs at **3 blocks/SM**, co-capped
by *both* registers (164 regs → `occupancy_limit_registers`=3) and shared
memory (`occupancy_limit_shared_mem`=3 at the 132 KB carveout). The fallback
`<TILE_KV=64, HEAD_DIM=64>` config compiles to only **122 registers** and
19.5 KB smem, so forcing the graded shape onto it reaches a genuine **4
blocks/SM** (`occupancy_limit_registers`=4, `occupancy_limit_shared_mem`=4,
`sm__maximum_warps_per_active_cycle_pct` 18.75%→25%, `sm__warps_active`
~18%→23.5%). **Latency did not improve:** −0% to +1% across seq 1024–4096
(within noise), +8% at seq=512 (more tiles → more loop/softmax overhead at
small grids). `sm__pipe_tensor_cycles_active` stayed pinned at ~28% with 16
warps, exactly as with 12. Conclusion: this kernel is **neither
occupancy-bound nor smem-traffic-bound** — an earlier −24%-smem-traffic
experiment also gave no speedup. The ~28% tensor-pipe ceiling is set by the
**per-warp serial dependency chain** (`ldmatrix → mma → exp/MUFU → pack →
mma`, plus the cross-KV-tile online-softmax rescale dependency), which more
warps cannot hide and less smem traffic cannot relieve. Both occupancy levers
are therefore exhausted for attention; the remaining angle is breaking the
softmax dependency chain itself (deeper, higher-risk). Optimization effort
moves to the MLP GEMMs (compute-bound, multi-stage cp.async pipelining
unexplored). Probe was launch-policy only (no code path kept; the 64-tile
config still serves seq % 128 ≠ 0).

### INT8 MLP: dynamic quantization (commit 0db4387)

The static hidden scale (`sx·sW1·d_model`) is a worst-case bound that is
~sqrt(d_model) too conservative — max error grew 0.04 → 0.49 over d_model
512 → 2048. Now: the GEMM1 epilogue gathers per-token row absmax (register
accumulation + half-warp shuffle reduction, one atomic per row), hidden is
requantized per-token, GEMM2 applies per-row scales from smem, and the
output uses a dynamic per-tensor scale. `__launch_bounds__(256, 2)` is
required to keep the epilogue at 128 registers (2 blocks/SM).

| d_model | max err (static → dynamic) | mean err | latency |
|---------|---------------------------|----------|---------|
| 512     | 0.039 → 0.022             | 2.0x better | +5–14% |
| 1024    | 0.124 → 0.052             | 4.3x better | +5–6%  |
| 2048    | 0.49 → 0.107              | 9.2x better | +1%    |

Static scales remain available via `INT8_MLP_DYNAMIC=0`.

**MLP GEMM bottleneck = smem-read bank conflicts (profiled 2026-06-13).** On
the graded MLP shape (b=8, s=512, d_model=1024, d_ff=4096) both GEMMs are
**L1/smem-pipe bound** (`l1tex__throughput` 78%/67%), not DRAM (3–4%) or
compute (`tensor_op_imma` 16–18%) bound, and **~46% of smem-load wavefronts are
bank conflicts** (16.78M of 36.2M). The 16-byte `cp.async` loader forces every
smem row stride to be a multiple of 16, which makes `A_SMEM_STRIDE/4` even and
collides the 16 WMMA fragment rows period-8 (2-way). (So multi-stage cp.async
is the *wrong* lever here — there is almost no DRAM latency to hide.)

**Negative result — conflict-free stride via 4-byte cp.async is slower.** Setting
`SMEM_SKEW` 16→4 (strides 36/132, `stride/4` odd → 16 distinct banks) and
switching the loader to 4-byte `cp.async` cut bank conflicts **16.78M → 2.10M
(−87%)** and dropped `l1tex__throughput` 78%→48% — but latency got **~30% WORSE**
(d_model=1024: 0.86→1.09 ms) because the 4× more cp.async store instructions
flooded the issue pipe (`smsp__inst_executed` →55%, `lg_throttle` →10%). So the
read-conflict relief is real, but it must NOT come at the cost of 16-byte stores.
Reverted.

**Resolution — hand-rolled `mma.sync m16n8k32` over a k-contiguous layout
(2026-06-13, iter 7): 1.35–1.67× faster.** The conflicts come from
`load_matrix_sync`'s internal int8 access pattern, not from the stride per se.
Replacing both GEMM inner loops with native
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (the attention QK^T path,
operands loaded as plain 4-byte smem words) makes the per-instruction bank map
`(12·group + tid) mod 32` a *bijection* over the 32 lanes — **conflict-free with
the original 16-byte `cp.async` stores, no XOR swizzle needed**. The one
prerequisite: int8 `mma.sync` needs its B operand k-contiguous, but the weights
are `[K][N]` (n-contiguous) and there is no int8 `trans`-`ldmatrix`, so W1/W2 are
**pre-transposed to `[N][K]`** once per forward (a ~1–2% bandwidth pass).
Measured on the graded shape: bank conflicts **16.78M → ~3K (≈0)**,
`l1tex__throughput` **78%/67% → 39%/29%**, and end-to-end latency **−26% to −40%
across the full sweep** (d_model=1024 s=512: 0.89→0.64 ms; s=4096: 5.55→3.50 ms),
reproducible across runs. Registers/occupancy unchanged (128 regs, 0 spills, 2
blocks/SM). All five accuracy gates pass on all 8 datasets (task-level ppl
+0.011%). The XOR swizzle turned out to be unnecessary — the access-pattern
change alone removes the conflicts while keeping `load_matrix_sync`'s 16-byte
store efficiency.

**Follow-up — deeper K-stage (2026-06-13, iter 8): a further −6% to −17%.** Once
iter 7 cleared the smem-pipe ceiling, the dominant stall shifted to
`long_scoreboard` (global-load latency: **18.7%/27.4%** on the two GEMMs, with
DRAM still only 5–6% → pure latency, not bandwidth). Raising the cp.async K-stage
depth `INT8_MLP_STAGE_K` 32 → 64 (each round carries 2× the data → half as many
global-load sync points) cut `long_scoreboard` to **5.5%/7.7%** and lifted
`tensor_op_imma` to **30%/38%**. The wider smem stride (48 → 80 B) stays
conflict-free — `stride/4 = 20 = 4·5`, gcd(5,8)=1, so the bank map is still a
bijection — and 16-byte aligned. Latency **−6% to −17% across the full sweep**
(d_model=1024 s=4096: 3.50→3.02 ms; d_model=2048 s=4096: 12.09→10.04 ms),
reproducible across two runs; occupancy held at 2 blocks/SM (128 regs, smem
32→40 KB, no block lost). All five gates pass (numerics bit-identical — STAGE_K
is pure tiling). The new ceiling is the `wait` MMA-dependency stall
(`tensor_op_imma` 30%/38% vs cuBLAS 60%+) at structurally-fixed 2-block
occupancy.

### Future work

- `mma.sync` PTX path (m16n8k32 for INT8 QK^T, register-resident softmax
  weights) — the structural ceiling of the `nvcuda::wmma` API is the
  scores smem round-trip.
- Per-token output scales for the MLP (needs a small interface extension).
