#!/bin/bash
# Watchdog launcher for step2.py with auto-resume on crash.
#
# Usage:
#   run_with_resume.sh <run_dir> [-- ARGS...]
#
# Example:
#   ./run_with_resume.sh ~/GOODRL/runs/20260515_tuned -- \
#       --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
#       --lr_a 0.0001 --lr_c 0.001 --warmup_critic 500 \
#       --grad_control 0 --algo_seed 3 --entloss_coef 0.01
#
# Behavior:
# - Creates run_dir/checkpoints/ and run_dir/step2.log
# - Runs step2.py, appending to step2.log
# - If process dies (any non-zero exit), waits 10s then restarts with --resume
# - Stops after MAX_RETRIES failures, or on clean exit (code 0)
# - Each attempt logs its boundary to step2.log

set -u

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
    echo "Usage: $0 <run_dir> -- [step2.py args...]" >&2
    exit 1
fi
shift

# Skip leading "--" if present
if [ "${1:-}" = "--" ]; then shift; fi

MAX_RETRIES="${MAX_RETRIES:-30}"
RETRY_SLEEP="${RETRY_SLEEP:-10}"
PYTHON="${PYTHON:-/home/xue/anaconda3/envs/drl_sched/bin/python}"
GOODRL_DIR="${GOODRL_DIR:-/home/xue/GOODRL}"
CKPT_INTERVAL="${CKPT_INTERVAL:-50}"

CKPT_DIR="$RUN_DIR/checkpoints"
LOG="$RUN_DIR/step2.log"
LATEST="$CKPT_DIR/state_latest.pth"

mkdir -p "$CKPT_DIR"
touch "$LOG"

echo "==================================================" >> "$LOG"
echo "Watchdog start: $(date)" >> "$LOG"
echo "Run dir: $RUN_DIR" >> "$LOG"
echo "Args: $@" >> "$LOG"
echo "==================================================" >> "$LOG"

for attempt in $(seq 1 $MAX_RETRIES); do
    RESUME_FLAG=""
    if [ -f "$LATEST" ]; then
        RESUME_FLAG="--resume $LATEST"
        echo "[$(date)] === Attempt $attempt: resuming from $LATEST ===" >> "$LOG"
    else
        echo "[$(date)] === Attempt $attempt: starting fresh ===" >> "$LOG"
    fi

    cd "$GOODRL_DIR" || { echo "cd failed" >> "$LOG"; exit 1; }

    "$PYTHON" -u step2.py "$@" \
        --checkpoint_dir "$CKPT_DIR" \
        $RESUME_FLAG >> "$LOG" 2>&1

    EXIT_CODE=$?
    echo "[$(date)] === Attempt $attempt ended, exit code: $EXIT_CODE ===" >> "$LOG"

    if [ "$EXIT_CODE" = "0" ]; then
        echo "[$(date)] Run completed successfully" >> "$LOG"
        echo "==================================================" >> "$LOG"
        exit 0
    fi

    # Detect if checkpoint was actually written this attempt
    if [ ! -f "$LATEST" ]; then
        echo "[$(date)] No checkpoint written yet (early crash). Likely code bug, not hardware. Aborting watchdog." >> "$LOG"
        exit 2
    fi

    echo "[$(date)] Will retry in ${RETRY_SLEEP}s..." >> "$LOG"
    sleep "$RETRY_SLEEP"
done

echo "[$(date)] Max retries ($MAX_RETRIES) reached. Giving up." >> "$LOG"
exit 3
