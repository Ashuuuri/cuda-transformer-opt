# cuda-transformer-opt

CUDA-optimized Transformer kernels for CS 5220 (Spring 2025). We implement fused FP16 and INT8-quantized attention and MLP kernels, then benchmark them against a PyTorch baseline.

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
│   ├── attention.cu        # FP16 fused attention kernel
│   ├── mlp.cu              # FP16 fused MLP kernel
│   ├── int8_attention.cu   # INT8 quantized attention kernel
│   ├── int8_mlp.cu         # INT8 quantized MLP kernel
│   └── quant_utils.cu      # Shared quantization helpers
├── baseline.py             # PyTorch FP16 reference implementations
├── benchmark.py            # GPU timing harness
├── correctness.py          # Numerical correctness checker
├── sweep.py                # Parameter sweep (TBD)
└── README.md
```

## Setup on Perlmutter (NERSC)

### 1. Load the environment

```bash
module load pytorch
```

This provides Python 3, PyTorch 2.8.0, and CUDA 12.9. Do **not** load the `python` module separately — it will conflict.

### 2. Get a GPU node

```bash
salloc -A m4341_g -C "gpu&hbm40g" -N 1 -t 00:30:00 -q interactive
```

### 3. Verify CUDA is available

Every script prints a CUDA confirmation line at startup. You can also check manually:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
NVIDIA A100-SXM4-40GB
```

## Running

### Baseline (sanity check)

```bash
python baseline.py
```

Prints output shapes of attention and MLP to confirm the environment works.

### Correctness checks

```bash
python correctness.py
```

Compares kernel outputs against the PyTorch baseline using `torch.allclose`:
- **FP16 kernels**: `atol = 1e-2`
- **INT8 kernels**: `atol = 0.1`

Reports max absolute error and PASSED/FAILED for each test.

### Benchmarks

```bash
python benchmark.py
```

Times each kernel using `torch.cuda.Event`:
- 10 warmup iterations (discarded)
- 50 timed iterations
- Reports mean latency in milliseconds
