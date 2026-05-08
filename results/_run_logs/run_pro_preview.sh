#!/usr/bin/env bash
# Per-task iteration count to match claude-opus-4-7 budgets.
set -u
cd /Users/victorgallego/metal-kernels

MODEL="gemini-3.1-pro-preview"
LOGDIR="results/_run_logs"
SUMMARY="$LOGDIR/run_all_${MODEL}.summary"

# task:iters pairs
PAIRS=(
    "saxpy:10"
    "heat2d:10"
    "nbody:10"
    "gradshaf:10"
    "hmc:10"
    "ising:10"
    "lbm:25"
)

: > "$SUMMARY"
echo "[start] $(date -Iseconds)" | tee -a "$SUMMARY"

for p in "${PAIRS[@]}"; do
    t="${p%%:*}"
    n="${p##*:}"
    log="$LOGDIR/${t}_${MODEL}.log"
    echo "[task] $t iters=$n -> $log  (start $(date -Iseconds))" | tee -a "$SUMMARY"
    start=$SECONDS
    uv run run_benchmark.py --task "$t" --model "$MODEL" --iterations "$n" \
        > "$log" 2>&1
    rc=$?
    elapsed=$((SECONDS - start))
    echo "[done] $t iters=$n rc=$rc elapsed=${elapsed}s" | tee -a "$SUMMARY"
done

echo "[end] $(date -Iseconds)" | tee -a "$SUMMARY"
