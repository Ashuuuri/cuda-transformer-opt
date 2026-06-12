#!/usr/bin/env bash
# evolve.sh — autonomous INT8 kernel optimization loop.
#
# Each iteration: profile (ncu) -> let claude make ONE rule-compliant change
# -> compile-check -> 5-gate accuracy validation -> perf sweep vs baseline
# -> commit + append the CLAUDE.md iteration record. Any failed check
# restores the tree and moves on.
#
# Usage:
#   ./evolve.sh                 # 5 iterations
#   MAX_ITER=10 ./evolve.sh     # override iteration count
#
# Requirements: clean git tree, passwordless sudo for ncu (falls back to
# torch.profiler), `claude` CLI on PATH.

set -uo pipefail
cd "$(dirname "$0")"

MAX_ITER="${MAX_ITER:-5}"
LOG_DIR="${LOG_DIR:-/tmp/evolve_logs}"
CLAUDE_TIMEOUT="${CLAUDE_TIMEOUT:-1800}"
mkdir -p "$LOG_DIR"

# CLAUDE.md §3 metrics, verbatim.
NCU_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
launch__registers_per_thread,launch__occupancy_limit_registers"

PASS_COUNT=0
declare -a ITER_RESULTS

log() { echo "[evolve $(date +%H:%M:%S)] $*"; }

run_claude() {  # run_claude <logfile>  (prompt on stdin)
    timeout "$CLAUDE_TIMEOUT" claude --print --dangerously-skip-permissions \
        < /dev/stdin 2>&1 | tee "$1"
}

restore_tree() {
    git reset --hard HEAD >/dev/null
    log "tree restored to HEAD"
}

compile_check() {  # syntax + ptxas check of both INT8 kernels
    local out
    for f in kernels/int8_attention.cu kernels/int8_mlp.cu; do
        out=$(nvcc -arch=sm_80 -O3 --std=c++17 -I kernels \
                   --ptxas-options=-v -c "$f" -o /dev/null 2>&1) || {
            echo "$out"; return 1; }
        echo "$out" | grep -E "spill|registers" | head -4
    done
    return 0
}

# Mean kernel_ms ratio: new vs baseline CSV. Prints "+X.X%"; exit 1 if >5% worse.
compare_sweep() {  # compare_sweep <baseline.csv> <new.csv>
    python3 - "$1" "$2" <<'EOF'
import csv, sys
def ms(p):
    with open(p) as f:
        return {(r["seq_len"], r["d_model"]): float(r["kernel_ms"])
                for r in csv.DictReader(f)}
old, new = ms(sys.argv[1]), ms(sys.argv[2])
ratios = [new[k] / old[k] for k in old if k in new]
mean = sum(ratios) / len(ratios)
print(f"{(mean-1)*100:+.1f}%")
sys.exit(1 if mean > 1.05 else 0)
EOF
}

# ════════════════════════════════════════════════════════════════════════
# 1. Initialization
# ════════════════════════════════════════════════════════════════════════
if [ -n "$(git status --porcelain)" ]; then
    log "ERROR: git tree not clean — commit or stash first."; exit 1
fi

if [ ! -d testdata/validate ]; then
    log "generating validation datasets ..."
    python3 generate_test_data.py || exit 1
fi

log "establishing accuracy baseline -> /tmp/baseline.txt"
python3 validate_int8.py > /tmp/baseline.txt 2>&1 || {
    log "ERROR: baseline validation FAILED — fix before evolving."
    tail -20 /tmp/baseline.txt; exit 1; }

log "establishing perf baseline (sweep int8_attn + int8_mlp) ..."
python3 sweep.py --kernel int8_attn > "$LOG_DIR/sweep_attn_base.log" 2>&1 || exit 1
python3 sweep.py --kernel int8_mlp  > "$LOG_DIR/sweep_mlp_base.log"  2>&1 || exit 1
cp results/int8_attn_sweep.csv /tmp/baseline_attn.csv
cp results/int8_mlp_sweep.csv  /tmp/baseline_mlp.csv
git checkout -- results/ 2>/dev/null   # keep committed CSVs canonical
log "baselines ready."

# ════════════════════════════════════════════════════════════════════════
# 2. Main loop
# ════════════════════════════════════════════════════════════════════════
for i in $(seq 1 "$MAX_ITER"); do
    echo; log "════════ ITERATION $i / $MAX_ITER ════════"
    NCU_OUT="$LOG_DIR/ncu_$i.txt"

    # a. Profile. Pre-warm the JIT cache so ncu doesn't profile the build.
    python3 tests/test_int8.py --quick > /dev/null 2>&1
    if sudo -n true 2>/dev/null && command -v ncu >/dev/null; then
        log "profiling with ncu ..."
        # env PATH/HOME: keep the user's JIT cache + ninja visible under sudo,
        # otherwise root rebuilds the extension inside the profiler (or fails).
        sudo env "PATH=$PATH" HOME="$HOME" \
            ncu --kernel-name regex:int8_wmma --launch-count 3 \
            --metrics "$NCU_METRICS" \
            python3 tests/test_int8.py --quick 2>&1 \
            | grep -E "int8_|Metric Name|----|pct|registers|wavefronts|occupancy" \
            > "$NCU_OUT" || true
    fi
    if ! grep -q "registers" "$NCU_OUT" 2>/dev/null; then
        log "ncu unavailable/empty — falling back to torch.profiler"
        python3 - > "$NCU_OUT" 2>&1 <<'EOF'
import torch, sys; sys.path.insert(0, ".")
from torch.utils.cpp_extension import load
ext = load(name="int8_ext", sources=["kernels/int8_attention.cu",
    "kernels/int8_mlp.cu", "kernels/quant_utils.cu", "kernels/int8_ext.cu"],
    extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"], verbose=False)
def q(t): s=t.float().abs().max()/127; return (t.float()/s).round().clamp(-128,127).to(torch.int8), float(s)
def qt(t):
    s=t.float().abs().amax(-1,keepdim=True).clamp(min=1e-8)/127
    return (t.float()/s).round().clamp(-128,127).to(torch.int8), s.squeeze(-1).contiguous()
Q=torch.randn(8,8,2048,64,device="cuda").half(); qi,sq=qt(Q)
x=torch.randn(8,512,1024,device="cuda").half(); W1=torch.randn(1024,4096,device="cuda").half()*0.02
W2=torch.randn(4096,1024,device="cuda").half()*0.02
xi,sx=q(x); w1,s1=q(W1); w2,s2=q(W2)
for _ in range(3):
    ext.int8_attention_forward(qi,qi,qi,sq,sq,sq)
    ext.int8_mlp_forward(xi,w1,w2,sx,s1,s2)
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(5):
        ext.int8_attention_forward(qi,qi,qi,sq,sq,sq)
        ext.int8_mlp_forward(xi,w1,w2,sx,s1,s2)
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=12))
EOF
    fi

    # b. One rule-compliant change.
    log "asking claude for ONE optimization ..."
    run_claude "$LOG_DIR/claude_change_$i.log" <<EOF
You are optimizing the INT8 CUDA kernels in this repo. Read CLAUDE.md fully
and follow §4 strictly. Profiling output for this iteration:

$(cat "$NCU_OUT")

Make exactly ONE structural change to kernels/int8_attention.cu or
kernels/int8_mlp.cu (you may touch kernels/int8_common.cuh if shared).
Hard rules:
- Do NOT touch LayerNorm/residual handling, the public interface signatures,
  or anything outside kernels/.
- Do NOT retry the known negative results: cp.async double-buffering
  (INT8_ATTN_DB) and V register prefetch (INT8_ATTN_VPREFETCH).
- Keep registers <= 128 per thread at 256 threads/block (check with ptxas).
After editing, output exactly two lines:
CHANGE: <files + what you changed>
TARGET: <which metric you expect to improve and why>
EOF
    CHANGE_DESC=$(grep -E "^CHANGE:" "$LOG_DIR/claude_change_$i.log" | tail -1)
    TARGET_DESC=$(grep -E "^TARGET:" "$LOG_DIR/claude_change_$i.log" | tail -1)
    if [ -z "$(git status --porcelain kernels/)" ]; then
        log "claude made no kernel change — skipping iteration"
        ITER_RESULTS[$i]="SKIP (no change)"; continue
    fi

    # c. Compile check, up to 3 fix attempts.
    BUILD_OK=0
    for attempt in 1 2 3; do
        log "compile check (attempt $attempt) ..."
        if ERR=$(compile_check 2>&1); then BUILD_OK=1; echo "$ERR"; break; fi
        echo "$ERR" | tail -20
        run_claude "$LOG_DIR/claude_fixbuild_${i}_$attempt.log" <<EOF
The kernel change you just made fails to compile. Fix the compile error in
kernels/ without changing the optimization's intent. Error output:

$(echo "$ERR" | tail -40)
EOF
    done
    if [ "$BUILD_OK" -ne 1 ]; then
        log "compile still failing after 3 attempts — restoring"
        restore_tree; ITER_RESULTS[$i]="FAIL (compile)"; continue
    fi
    rm -rf ~/.cache/torch_extensions/py312_cu128/int8_ext 2>/dev/null

    # d. Five-gate accuracy validation, up to 2 repair attempts.
    VAL_OK=0
    for attempt in 0 1 2; do
        log "validate_int8.py (attempt $attempt) ..."
        if python3 validate_int8.py > "$LOG_DIR/validate_${i}_$attempt.log" 2>&1
        then VAL_OK=1; break; fi
        [ "$attempt" -eq 2 ] && break
        run_claude "$LOG_DIR/claude_fixval_${i}_$attempt.log" <<EOF
Your kernel change broke the 5-gate INT8 validation. Follow the repair order
in CLAUDE.md §5 (per-channel quantization -> QK^T back to fp16 -> adjust
scale computation -> worst stage back to fp16) or revert the precision-
affecting part of your change while keeping the perf part. Only touch
kernels/. Validation output (failures + repair suggestions):

$(grep -E "FAIL|XFAIL|Gate|RECOVERS|DATASET" "$LOG_DIR/validate_${i}_$attempt.log" | head -60)
EOF
        compile_check >/dev/null 2>&1 || { VAL_OK=0; break; }
        rm -rf ~/.cache/torch_extensions/py312_cu128/int8_ext 2>/dev/null
    done
    if [ "$VAL_OK" -ne 1 ]; then
        log "validation failing — restoring"
        restore_tree
        ITER_RESULTS[$i]="FAIL (accuracy gates)"; continue
    fi

    # e. Perf sweep vs baseline (>5% regression on either kernel -> revert).
    log "perf sweep ..."
    python3 sweep.py --kernel int8_attn > "$LOG_DIR/sweep_attn_$i.log" 2>&1
    python3 sweep.py --kernel int8_mlp  > "$LOG_DIR/sweep_mlp_$i.log"  2>&1
    ATTN_DELTA=$(compare_sweep /tmp/baseline_attn.csv results/int8_attn_sweep.csv); ATTN_OK=$?
    MLP_DELTA=$(compare_sweep /tmp/baseline_mlp.csv results/int8_mlp_sweep.csv);   MLP_OK=$?
    log "latency vs baseline: attn $ATTN_DELTA, mlp $MLP_DELTA"
    if [ "$ATTN_OK" -ne 0 ] || [ "$MLP_OK" -ne 0 ]; then
        log "latency regressed >5% — restoring"
        restore_tree
        ITER_RESULTS[$i]="FAIL (perf regression: attn $ATTN_DELTA mlp $MLP_DELTA)"
        continue
    fi

    # f. Record + commit. New baseline = this iteration's numbers.
    run_claude "$LOG_DIR/claude_record_$i.log" <<EOF
Append an iteration record to the bottom of CLAUDE.md following §7 exactly
(next iteration number after the existing ones, today's date). Facts:
- $CHANGE_DESC
- $TARGET_DESC
- Latency vs previous baseline: int8_attn $ATTN_DELTA, int8_mlp $MLP_DELTA
- All five validation gates passed (see below for gate-1 numbers)
$(grep -E "Gate1|SUMMARY|PASS" "$LOG_DIR/validate_${i}_0.log" | head -25)
Mark all five gate checkboxes as checked. Conclusion: pass. Only edit CLAUDE.md.
EOF
    cp results/int8_attn_sweep.csv /tmp/baseline_attn.csv
    cp results/int8_mlp_sweep.csv  /tmp/baseline_mlp.csv
    git add kernels/ results/ CLAUDE.md
    git commit -m "evolve iter $i: ${CHANGE_DESC#CHANGE: }

${TARGET_DESC}
Latency vs previous baseline: int8_attn ${ATTN_DELTA}, int8_mlp ${MLP_DELTA}
All 5 validation gates passed.

Co-Authored-By: Claude (evolve.sh) <noreply@anthropic.com>" >/dev/null
    log "iteration $i COMMITTED ($(git rev-parse --short HEAD))"
    PASS_COUNT=$((PASS_COUNT + 1))
    ITER_RESULTS[$i]="PASS (attn $ATTN_DELTA, mlp $MLP_DELTA)"
done

# ════════════════════════════════════════════════════════════════════════
# 3. Summary
# ════════════════════════════════════════════════════════════════════════
echo; log "════════ SUMMARY ════════"
for i in $(seq 1 "$MAX_ITER"); do
    echo "  iter $i: ${ITER_RESULTS[$i]:-not run}"
done
echo "  passed: $PASS_COUNT / $MAX_ITER"

python3 sweep.py --kernel int8_attn > /dev/null 2>&1
python3 sweep.py --kernel int8_mlp  > /dev/null 2>&1
TOTAL_ATTN=$(compare_sweep /tmp/baseline_attn.csv results/int8_attn_sweep.csv || true)
TOTAL_MLP=$(compare_sweep /tmp/baseline_mlp.csv results/int8_mlp_sweep.csv || true)
git checkout -- results/ 2>/dev/null
echo "  final latency vs last-accepted baseline: attn $TOTAL_ATTN, mlp $TOTAL_MLP"

log "asking claude for the next recommended direction ..."
run_claude "$LOG_DIR/claude_summary.log" <<EOF
Read the Iteration Log at the bottom of CLAUDE.md and the latest ncu output:
$(tail -30 "$LOG_DIR/ncu_$MAX_ITER.txt" 2>/dev/null)
In 5 lines or fewer, state the single most promising next optimization
direction and why, consistent with CLAUDE.md §4. Do not edit any files.
EOF
