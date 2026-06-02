#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RL_DIR"

VENV_DIR=${VENV_DIR:-.venv}
SMOKE_OUTPUT=${SMOKE_OUTPUT:-./output_smoke_5step}
SMOKE_TRAINING_STEPS=${SMOKE_TRAINING_STEPS:-5}
SMOKE_MAX_TURNS=${SMOKE_MAX_TURNS:-5}
SMOKE_SAVE_FREQ=${SMOKE_SAVE_FREQ:-5}
LOG_DIR=${LOG_DIR:-logs}

[ -x "$VENV_DIR/bin/python" ] || {
    echo "ERROR: $VENV_DIR is missing. Run: bash scripts/install_vendor_env.sh" >&2
    exit 1
}
mkdir -p "$LOG_DIR" "$SMOKE_OUTPUT"

GPU_LOG="$LOG_DIR/gpu_monitor_smoke_5step.csv"
nvidia-smi \
    --query-gpu=timestamp,gpu_name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 5 > "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

echo "Running $SMOKE_TRAINING_STEPS-step smoke training in $SMOKE_OUTPUT"
echo "GPU monitor: $GPU_LOG"

SMOKE_OUTPUT="$SMOKE_OUTPUT" \
SMOKE_TRAINING_STEPS="$SMOKE_TRAINING_STEPS" \
SMOKE_MAX_TURNS="$SMOKE_MAX_TURNS" \
SMOKE_SAVE_FREQ="$SMOKE_SAVE_FREQ" \
    bash scripts/vendor_train.sh smoke

echo "Smoke training completed and checkpoint validation passed."
