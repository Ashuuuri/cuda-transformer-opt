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

After the course, the INT8 kernels were optimized further on a dedicated
A100-SXM4-40GB (CUDA 12.8). Headlines below; the **full iteration-by-iteration
journey — before/after ncu numbers, validation traces, and negative results —
is in [`OPTIMIZATION_LOG.md`](OPTIMIZATION_LOG.md)**, and the distilled
"what's exhausted / do-not-retry" rules are in `CLAUDE.md` §3–§4.

### Benchmark hardening

- INT8 attention vs **FlashAttention-2** and the **FP16 WMMA kernel** (same
  engineering level, isolates the INT8 effect), not just naive PyTorch.
- INT8 MLP vs the **full cuBLAS INT8 pipeline** (`_int_mm` + dequant/GELU/requant
  — the same work our kernel does), with bare `2× _int_mm` as a GEMM-only lower
  bound.
- Sweeps emit `kernel_max_err` / `kernel_mean_err` columns and use a blended
  INT8/FP16 (416 TOPS) attention peak. `python sweep.py --kernel int8_attn | int8_mlp`.

### Performance results (graded sweep, batch=8; see `results/*.csv`)

| Kernel | Result | Iter |
|---|---|---|
| INT8 attention | **1.6–2.2×** (templated `(TILE_KV, HEAD_DIM)`, k-outer loop, fragments compile-time-indexed); FA2 gap 0.30×→0.62× | 1 |
| INT8 MLP GEMMs | **−26% to −40%** (k-contiguous `mma.sync m16n8k32`, conflict-free bank map) then a further **−6% to −17%** (`STAGE_K` 32→64) | 7–8 |
| INT8 MLP forward | **−9% to −22%** at the graded s=512 (prepacked transpose-once for static weights: `int8_mlp_forward_prepacked`) | 11 |

Both kernels are now **perf-exhausted on every explored lever** — the MLP GEMM
(smem bank conflicts, global-load latency, the `wait` stall, and occupancy from
*both* directions) and attention occupancy. The only remaining kernel-internal
lever is the attention online-softmax dependency chain (deep, high-risk). The
proofs are in the log.

### Accuracy — real-model, per-channel/per-token quant (iter 9)

Gate 5 was switched from random weights to **real GPT-2 on real WikiText-2**,
which exposed that per-tensor MLP *output* quant costs **+64%** perplexity (it
crushes emergent output-channel outliers). Per-channel weights + per-token
activations + **per-token output** quant — added as a *new* entry point
`int8_mlp_forward_per_channel` (the per-tensor `int8_mlp_forward` and the
`tests/cuda/` .bin flow are unchanged) — recovers it: real perplexity
**31.946 → 31.925 (−0.068%)**, gate bar <2%. The `outlier`/`boundary`/`stress`
MLP datasets now pass outright.

### Ablation flags (negative results, OFF by default — do not re-attempt)

| Flag | Idea | Result |
|---|---|---|
| `INT8_ATTN_DB` | cp.async double-buffer (attention) | 0.58–0.82× (forces TILE_KV down) |
| `INT8_ATTN_VPREFETCH` | V register prefetch (attention) | 0.75–0.9× (extra barrier) |
| `INT8_MLP_FINE_WARP` | 4×4 / 512-thread finer tile (MLP) | +15–28% — doubles occupancy & halves `wait`, but halves operand reuse so `tensor_op_imma` net falls |
| `INT8_MLP_DYNAMIC=0` | static (non-dynamic) MLP scales | lower accuracy; kept for the .bin flow |

The denser-mma reschedule (iter 10) and the attention 4-blocks/SM raise were
reverted outright (byte-identical / no speedup); see the log.

### Future work

- **Attention online-softmax dependency chain** — the only remaining perf lever.
  The ~28% `tensor_pipe` ceiling is pinned by the per-warp
  `ldmatrix → mma → exp/MUFU → pack → mma` + cross-KV-tile rescale chain; breaking
  it (cheaper/approximate exp, decoupling the per-tile rescale) is deep and
  high-risk. Both attention occupancy levers are already proven dead.
- Per-channel / smoothing for **attention** K/V (the remaining outlier XFAIL is
  attention-only; the MLP per-token output already shipped, iter 9).
