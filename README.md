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

## Requirements

- Python 3.8+
- PyTorch with CUDA support (`torch.cuda.is_available()` must be `True`)
- NVIDIA GPU with compute capability >= 7.0 (for FP16/INT8 tensor cores)

## Quick Start

### Run the baseline (sanity check)

```bash
python baseline.py
```

This prints the output shapes of the attention and MLP baselines to confirm everything loads correctly.

### Run correctness checks

```bash
python correctness.py
```

Compares kernel outputs against the PyTorch baseline. Uses `torch.allclose` with:
- **FP16 kernels**: `atol = 1e-2`
- **INT8 kernels**: `atol = 0.1`

Reports max absolute error and PASSED/FAILED for each test.

### Run benchmarks

```bash
python benchmark.py
```

Times each kernel using `torch.cuda.Event`:
- 10 warmup iterations (discarded)
- 50 timed iterations
- Reports mean latency in milliseconds
