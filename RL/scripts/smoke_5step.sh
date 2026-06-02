#!/bin/bash
set -euo pipefail

# DR-Venus 5步冒烟测试脚本（单卡版本）
# 集成：tmux后台运行 + wandb日志 + 显存监控

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RL_DIR"

VENV_DIR=${VENV_DIR:-.venv}
TMUX_SESSION="drvenus-smoke"
TENSORBOARD_SESSION="drvenus-smoke-tb"

# 加载 .env（已配置为单卡参数）
set -a
source .env
set +a

# 强制冒烟参数
export TOTAL_TRAINING_STEPS=5
export MAX_TURNS=5
export SAVE_FREQ=5

mkdir -p "$OUTPUT" "${LOG_DIR:-logs}"

# 激活环境
source "$VENV_DIR/bin/activate"

echo "=== DR-Venus 5步冒烟测试 (单卡) ==="
echo "OUTPUT: $OUTPUT"
echo "STEPS: 5, MAX_TURNS: 5"
echo "NUM_GPUS: ${NUM_GPUS:-1}, TP_SIZE: ${TP_SIZE:-1}, ULYSSES_SP_SIZE: ${ULYSSES_SP_SIZE:-1}"
echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN:-16384}"
echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE:-2}, ROLLOUT_N: ${ROLLOUT_N:-2}"
echo "GPU_UTIL: ${GPU_MEMORY_UTILIZATION:-0.75}"
echo ""

# 1. 启动显存监控（后台）
LOG_DIR=${LOG_DIR:-logs}
GPU_LOG="$LOG_DIR/gpu_monitor_smoke.csv"
echo "Starting GPU memory monitor -> $GPU_LOG"
nvidia-smi --query-gpu=timestamp,gpu_name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!
echo "GPU monitor PID: $GPU_MONITOR_PID"

# 2. 启动 TensorBoard
if ! tmux has-session -t "$TENSORBOARD_SESSION" 2>/dev/null; then
    TENSORBOARD_LOG_DIR=${TENSORBOARD_LOG_DIR:-tensorboard_log}
    mkdir -p "$TENSORBOARD_LOG_DIR"
    tmux new-session -d -s "$TENSORBOARD_SESSION" \
        "cd '$RL_DIR' && source $VENV_DIR/bin/activate && tensorboard --logdir $TENSORBOARD_LOG_DIR --host 0.0.0.0 --port 6006"
    echo "TensorBoard started in tmux: $TENSORBOARD_SESSION (port 6006)"
else
    echo "TensorBoard session already exists: $TENSORBOARD_SESSION"
fi

# 3. 启动训练（tmux 后台）
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
fi

# 写入训练启动脚本供 tmux 执行
cat > /tmp/drvenus_smoke_run.sh <<'RUNEOF'
#!/bin/bash
cd /root/DR-Venus-LocalSearch/RL
source .venv/bin/activate
set -a; source .env; set +a

# 强制冒烟参数
export TOTAL_TRAINING_STEPS=5
export MAX_TURNS=5
export SAVE_FREQ=5

echo "[$(date)] Smoke test STARTED (5 steps, single GPU)" | tee -a logs/smoke_timing.log

bash scripts/vendor_train.sh smoke

EXIT_CODE=$?
echo "[$(date)] Smoke test FINISHED with exit code: $EXIT_CODE" | tee -a logs/smoke_timing.log
RUNEOF
chmod +x /tmp/drvenus_smoke_run.sh

tmux new-session -d -s "$TMUX_SESSION" "bash /tmp/drvenus_smoke_run.sh 2>&1 | tee -a $OUTPUT/training.log"

echo ""
echo "=== 冒烟测试已启动 ==="
echo "tmux session: $TMUX_SESSION"
echo "  查看: tmux attach -t $TMUX_SESSION"
echo "  退出查看: Ctrl+B 然后按 D"
echo ""
echo "TensorBoard: $TENSORBOARD_SESSION (port 6006)"
echo "GPU 监控: $GPU_LOG (每5秒)"
echo "训练日志: $OUTPUT/training.log"
echo "显存监控PID: $GPU_MONITOR_PID (停止: kill $GPU_MONITOR_PID)"
