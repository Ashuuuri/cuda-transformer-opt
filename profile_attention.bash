#!/bin/bash
# Profile the attention kernel with nsys and ncu.
# Run on a GPU node after building with: make test_attention
#

#SBATCH -A m4341_g
#SBATCH -t 00:20:00
#SBATCH -C "gpu&hbm40g"
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -o results/slurm_%j.out
#SBATCH -e results/slurm_%j.err

BINARY=./test_attention
OUTDIR=results
mkdir -p "$OUTDIR"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found — run 'make test_attention' first"
    exit 1
fi

echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== nsys ==="
nsys profile \
    --stats=true \
    --output=/tmp/nsys_attn \
    --force-overwrite=true \
    "$BINARY" 2>&1 | tee "$OUTDIR/nsys_attn_stats.txt"
cp /tmp/nsys_attn.qdstrm "$OUTDIR/nsys_attn.qdstrm" 2>/dev/null || true

echo ""
echo "=== ncu ==="
ncu \
    --set full \
    --target-processes all \
    -o /tmp/ncu_attn \
    -f \
    "$BINARY" 2>&1 | tee "$OUTDIR/ncu_attn_report.txt"
cp /tmp/ncu_attn.ncu-rep "$OUTDIR/ncu_attn.ncu-rep" 2>/dev/null || true

echo ""
echo "Done. Results in $OUTDIR/"
