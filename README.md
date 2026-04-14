# cuda-transformer-opt

CUDA-optimized Transformer kernels for CS 5220 (Spring 2025). We implement fused FP16 and INT8-quantized attention and MLP kernels, then benchmark them against PyTorch baselines and industry references (FlashAttention-2, cuBLAS).

## Team

| Member    | Responsibility                              |
|-----------|---------------------------------------------|
| Jonathan  | FP16 attention kernel (`kernels/attention.cu`) |
| Shengjing | FP16 MLP kernel (`kernels/mlp.cu`)           |
| Heling    | INT8 kernels & quantization utilities (`kernels/int8_attention.cu`, `kernels/int8_mlp.cu`, `kernels/quant_utils.cu`) |

## Project Structure

```
cuda-transformer-opt/
├── kernels/
│   ├── attention.cu        # FP16 fused attention kernel (Jonathan)
│   ├── mlp.cu              # FP16 fused MLP kernel (Shengjing)
│   ├── int8_attention.cu   # INT8 quantized attention kernel (Heling)
│   ├── int8_mlp.cu         # INT8 quantized MLP kernel (Heling)
│   └── quant_utils.cu      # Shared quantization helpers (Heling)
├── tests/
│   ├── gen_testdata.py     # Generate test inputs + reference answers
│   ├── test_attention.py   # Jonathan: correctness + benchmark
│   ├── test_mlp.py         # Shengjing: correctness + benchmark
│   └── test_int8.py        # Heling: correctness + benchmark
├── testdata/               # Generated .bin files (git-ignored)
├── baseline.py             # PyTorch FP16 reference implementations
├── benchmark.py            # GPU timing harness
├── correctness.py          # Numerical correctness checker
├── Makefile                # CUDA compilation targets
├── sweep.py                # Parameter sweep (TBD)
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

### Step 1: Compile + test (pure CUDA)

```bash
make test_attention && ./test_attention    # Jonathan
make test_mlp && ./test_mlp                # Shengjing
make test_int8 && ./test_int8              # Heling
```

Reads `.bin` test data, compares kernel output against reference. Reports PASS/FAIL + max error.

### Step 2: Benchmark against industry references (Python)

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

Benchmark scripts automatically measure these on the same GPU for apples-to-apples comparison:

- **FlashAttention-2**: `F.scaled_dot_product_attention` — PyTorch 2.x built-in, backed by [Dao (2023)](https://arxiv.org/abs/2307.08691). Reports 50-73% A100 utilization in the paper.
- **cuBLAS FP16**: `torch.mm` — PyTorch's matrix multiply, backed by NVIDIA cuBLAS.
- **cuBLAS INT8**: `torch._int_mm` — PyTorch's INT8 integer matrix multiply, backed by cuBLAS `cublasLtMatmul`.

## Correctness Tolerances

- **FP16 kernels**: `atol = 1e-2`
- **INT8 kernels**: `atol = 0.1`

When a test fails, the checker prints diagnostics: worst error location, first 8 elements comparison, and percentage of elements exceeding tolerance.
