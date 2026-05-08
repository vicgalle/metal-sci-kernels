#!/usr/bin/env bash
# Sequentially run gemini-3-flash-preview over every registered task @ --iterations 10.
set -u
cd /Users/victorgallego/metal-kernels

MODEL="claude-opus-4-7"
ITERS=10
TASKS=(saxpy heat2d nbody)
LOGDIR="results/_run_logs"
SUMMARY="$LOGDIR/run_all_${MODEL}.summary"

: > "$SUMMARY"
echo "[start] $(date -Iseconds)" | tee -a "$SUMMARY"

for t in "${TASKS[@]}"; do
    log="$LOGDIR/${t}.log"
    echo "[task] $t -> $log  (start $(date -Iseconds))" | tee -a "$SUMMARY"
    start=$SECONDS
    uv run run_benchmark.py --task "$t" --model "$MODEL" --iterations "$ITERS" \
        > "$log" 2>&1
    rc=$?
    elapsed=$((SECONDS - start))
    echo "[done] $t rc=$rc elapsed=${elapsed}s" | tee -a "$SUMMARY"
done

echo "[end] $(date -Iseconds)" | tee -a "$SUMMARY"
