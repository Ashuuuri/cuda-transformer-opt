# Nsight profile reports

Pre-captured Nsight reports for the **graded shapes** (the configs `sweep.py`
grades), so they can be downloaded and opened in the Nsight desktop GUIs on your
local machine. Captured 2026-06-13 on the A100-SXM4-40GB, CUDA 12.8.

## Files

| File | Tool | Open with | Shape |
|---|---|---|---|
| `int8_mlp_timeline.nsys-rep` | Nsight **Systems** | `nsys-ui` | MLP forward, b=8 s=512 d_model=1024 d_ff=4096 |
| `int8_attn_timeline.nsys-rep` | Nsight **Systems** | `nsys-ui` | attention forward, b=8 h=8 s=2048 d=64 |
| `int8_mlp_gemm.ncu-rep` | Nsight **Compute** | `ncu-ui` | `gemm_int8_wmma_f16_kernel` (GEMM1+GEMM2), full metric set |
| `int8_attn_kernel.ncu-rep` | Nsight **Compute** | `ncu-ui` | `int8_wmma_attention_kernel`, full metric set |

- **`.nsys-rep`** = timeline view: per-kernel durations, the order/overlap of the
  GEMMs, `quantize_*`, and memcpys across the whole forward. Use this to see the
  *time split* between kernels.
- **`.ncu-rep`** = single-kernel deep dive: occupancy, tensor-pipe %, stall
  reasons (the `wait`/`long_scoreboard` story), smem traffic, source/SASS. Use
  this for the *why* behind a single kernel's efficiency. Captured with
  `--set full`, so every section is populated.

## How to open

Download the folder, then either launch the GUI and `File → Open` the report,
or from a terminal that has Nsight installed:

```bash
nsys-ui results/nsight/int8_mlp_timeline.nsys-rep     # timeline
ncu-ui  results/nsight/int8_mlp_gemm.ncu-rep          # kernel detail
```

The desktop tool version should be >= the CLI that captured these
(Nsight Systems 2026.1, Nsight Compute 2025.1). Older GUIs may refuse to open a
newer report.

## How they were regenerated

```bash
# Nsight Systems (no sudo). TMPDIR must be writable (default /tmp/nvidia is not).
export TMPDIR=$PWD/results/nsight/.tmp && mkdir -p "$TMPDIR"
nsys profile -o results/nsight/int8_mlp_timeline --force-overwrite true \
  --trace=cuda,nvtx --sample=none --cuda-memory-usage=true \
  python3 profile_kernel.py mlp

# Nsight Compute (needs sudo for GPU counters; full metric set).
sudo env "PATH=$PATH" HOME="$HOME" TMPDIR="$TMPDIR" \
  ncu --set full -f -o results/nsight/int8_mlp_gemm \
  --kernel-name regex:gemm_int8 --launch-count 2 \
  python3 profile_kernel.py mlp
```

Swap `mlp`→`attn` and the `--kernel-name` regex to `int8_wmma_attention` for the
attention reports.
