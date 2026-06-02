#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RL_DIR"

VENV_DIR=${VENV_DIR:-.venv}

load_config() {
    if [ -f .env ]; then
        local saved_exports
        saved_exports="$(mktemp)"
        export -p > "$saved_exports"
        set -a
        source .env
        set +a
        source "$saved_exports"
        rm -f "$saved_exports"
    fi

    LOG_DIR=${LOG_DIR:-logs}
    OUTPUT=${OUTPUT:-./output}
    LOCAL_SEARCH_SERVER_URL=${LOCAL_SEARCH_SERVER_URL:-http://localhost:8890}
    LOCAL_SEARCH_PID_FILE=${LOCAL_SEARCH_PID_FILE:-$LOG_DIR/local_search.pid}
    LOCAL_SEARCH_CONSOLE_LOG=${LOCAL_SEARCH_CONSOLE_LOG:-$LOG_DIR/local_search_console.log}
    RETRIEVAL_EVAL_SAMPLE_SIZE=${RETRIEVAL_EVAL_SAMPLE_SIZE:-1000}
    RETRIEVAL_EVAL_CONCURRENCY=${RETRIEVAL_EVAL_CONCURRENCY:-4}
    RETRIEVAL_EVAL_BATCH_SIZE=${RETRIEVAL_EVAL_BATCH_SIZE:-1}
    SMOKE_OUTPUT=${SMOKE_OUTPUT:-/root/autodl-tmp/output/dr-venus-smoke}
    SMOKE_TRAINING_STEPS=${SMOKE_TRAINING_STEPS:-1}
    SMOKE_MAX_TURNS=${SMOKE_MAX_TURNS:-5}
    SMOKE_GPU_MEMORY_UTILIZATION=${SMOKE_GPU_MEMORY_UTILIZATION:-0.75}
    SMOKE_SAVE_FREQ=${SMOKE_SAVE_FREQ:-1}
    TRAIN_TMUX_SESSION=${TRAIN_TMUX_SESSION:-drvenus-train}
    TENSORBOARD_TMUX_SESSION=${TENSORBOARD_TMUX_SESSION:-drvenus-tensorboard}
    TENSORBOARD_LOG_DIR=${TENSORBOARD_LOG_DIR:-tensorboard_log}
    TENSORBOARD_PORT=${TENSORBOARD_PORT:-6006}
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

check_checkpoint() {
    local output_dir="$1"
    if [ -x "$VENV_DIR/bin/python" ]; then
        "$VENV_DIR/bin/python" scripts/check_checkpoint.py --output_dir "$output_dir"
    else
        python3 scripts/check_checkpoint.py --output_dir "$output_dir"
    fi
}

ensure_environment() {
    [ -x "$VENV_DIR/bin/python" ] || fail \
        "$VENV_DIR is missing. Run: bash scripts/install_vendor_env.sh"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
}

search_health() {
    curl --noproxy '*' -fsS "$LOCAL_SEARCH_SERVER_URL/health"
}

search_is_healthy() {
    search_health >/dev/null 2>&1
}

start_search() {
    ensure_environment
    load_config
    mkdir -p "$LOG_DIR"

    if search_is_healthy; then
        echo "Local search is already healthy:"
        search_health
        echo
        return
    fi

    if [ -f "$LOCAL_SEARCH_PID_FILE" ]; then
        old_pid="$(cat "$LOCAL_SEARCH_PID_FILE" 2>/dev/null || true)"
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            fail "Local search PID $old_pid is running but health check failed. See $LOCAL_SEARCH_CONSOLE_LOG"
        fi
        rm -f "$LOCAL_SEARCH_PID_FILE"
    fi

    echo "Starting local search..."
    nohup bash scripts/start_local_search.sh > "$LOCAL_SEARCH_CONSOLE_LOG" 2>&1 < /dev/null &
    search_pid=$!
    echo "$search_pid" > "$LOCAL_SEARCH_PID_FILE"

    for _ in $(seq 1 90); do
        if search_is_healthy; then
            echo "Local search is ready:"
            search_health
            echo
            return
        fi
        if ! kill -0 "$search_pid" 2>/dev/null; then
            tail -n 120 "$LOCAL_SEARCH_CONSOLE_LOG" >&2 || true
            fail "Local search exited before becoming healthy."
        fi
        sleep 1
    done

    tail -n 120 "$LOCAL_SEARCH_CONSOLE_LOG" >&2 || true
    fail "Local search did not become healthy within 90 seconds."
}

stop_search() {
    load_config
    if [ ! -f "$LOCAL_SEARCH_PID_FILE" ]; then
        echo "No managed local-search PID file found."
        return
    fi
    search_pid="$(cat "$LOCAL_SEARCH_PID_FILE" 2>/dev/null || true)"
    if [[ "$search_pid" =~ ^[0-9]+$ ]] && kill -0 "$search_pid" 2>/dev/null; then
        echo "Stopping local search PID $search_pid..."
        kill "$search_pid"
        for _ in $(seq 1 30); do
            kill -0 "$search_pid" 2>/dev/null || break
            sleep 1
        done
    fi
    rm -f "$LOCAL_SEARCH_PID_FILE"
    echo "Managed local search stopped."
}

prepare_resources() {
    ensure_environment
    bash scripts/bootstrap_vendor.sh
    load_config
}

evaluate_retrieval() {
    start_search
    echo "Running local retrieval benchmark..."
    "$VENV_DIR/bin/python" scripts/evaluate_local_retrieval.py \
        --server_url "$LOCAL_SEARCH_SERVER_URL" \
        --sample_size "$RETRIEVAL_EVAL_SAMPLE_SIZE" \
        --batch_size "$RETRIEVAL_EVAL_BATCH_SIZE" \
        --concurrency "$RETRIEVAL_EVAL_CONCURRENCY"
}

run_preflight() {
    start_search
    echo "Running four-GPU preflight..."
    PRECHECK_ONLY=true bash train_igpo.sh
}

run_smoke() {
    start_search
    echo "Running smoke training in $SMOKE_OUTPUT ..."
    OUTPUT="$SMOKE_OUTPUT" \
    TOTAL_TRAINING_STEPS="$SMOKE_TRAINING_STEPS" \
    MAX_TURNS="$SMOKE_MAX_TURNS" \
    GPU_MEMORY_UTILIZATION="$SMOKE_GPU_MEMORY_UTILIZATION" \
    SAVE_FREQ="$SMOKE_SAVE_FREQ" \
        bash train_igpo.sh
    check_checkpoint "$SMOKE_OUTPUT"
}

run_training() {
    start_search
    if [ "${SKIP_PREFLIGHT:-false}" != "true" ]; then
        run_preflight
    fi
    echo "Starting formal training. RESUME_MODE defaults to auto."
    SKIP_PREFLIGHT=true bash train_igpo.sh
}

launch_training() {
    command -v tmux >/dev/null 2>&1 || fail "tmux is required. Install it with: sudo apt-get install -y tmux"
    if tmux has-session -t "$TRAIN_TMUX_SESSION" 2>/dev/null; then
        fail "tmux session $TRAIN_TMUX_SESSION already exists. Check it with: tmux attach -t $TRAIN_TMUX_SESSION"
    fi
    start_search
    run_preflight
    tmux new-session -d -s "$TRAIN_TMUX_SESSION" \
        "cd \"$RL_DIR\" && SKIP_PREFLIGHT=true bash scripts/vendor_train.sh train"
    echo "Formal training launched in tmux session: $TRAIN_TMUX_SESSION"
    echo "Attach: tmux attach -t $TRAIN_TMUX_SESSION"
    echo "Status: bash scripts/vendor_train.sh status"
}

stop_training() {
    command -v tmux >/dev/null 2>&1 || fail "tmux is not installed."
    if tmux has-session -t "$TRAIN_TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TRAIN_TMUX_SESSION"
        echo "Stopped tmux session: $TRAIN_TMUX_SESSION"
    else
        echo "Training tmux session does not exist: $TRAIN_TMUX_SESSION"
    fi
}

run_tensorboard() {
    ensure_environment
    exec tensorboard --logdir "$TENSORBOARD_LOG_DIR" --host 0.0.0.0 --port "$TENSORBOARD_PORT"
}

launch_tensorboard() {
    command -v tmux >/dev/null 2>&1 || fail "tmux is required. Install it with: sudo apt-get install -y tmux"
    if tmux has-session -t "$TENSORBOARD_TMUX_SESSION" 2>/dev/null; then
        echo "TensorBoard tmux session already exists: $TENSORBOARD_TMUX_SESSION"
        return
    fi
    tmux new-session -d -s "$TENSORBOARD_TMUX_SESSION" \
        "cd \"$RL_DIR\" && bash scripts/vendor_train.sh tensorboard"
    echo "TensorBoard launched on 0.0.0.0:$TENSORBOARD_PORT"
    echo "Use an SSH tunnel before opening it from your own browser."
}

show_status() {
    load_config
    echo "Repository: $RL_DIR"
    echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "Formal output: $OUTPUT"
    echo
    echo "Local search:"
    search_health 2>/dev/null || echo "unavailable: $LOCAL_SEARCH_SERVER_URL"
    echo
    if [ -f data/local_search_index/metadata.json ]; then
        echo "Index metadata:"
        cat data/local_search_index/metadata.json
        echo
    fi
    if command -v tmux >/dev/null 2>&1; then
        echo "tmux sessions:"
        tmux list-sessions 2>/dev/null || echo "none"
        echo
    fi
    latest_checkpoint=""
    latest_partial_checkpoint=""
    if [ -d "$OUTPUT" ]; then
        latest_partial_checkpoint="$(find "$OUTPUT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
        if [ -f "$OUTPUT/latest_checkpointed_iteration.txt" ]; then
            latest_step="$(cat "$OUTPUT/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
            if [[ "$latest_step" =~ ^[0-9]+$ ]]; then
                latest_checkpoint="$OUTPUT/global_step_$latest_step"
            fi
        fi
    fi
    echo "Latest resumable checkpoint: ${latest_checkpoint:-none}"
    if [ -n "$latest_checkpoint" ]; then
        check_checkpoint "$OUTPUT" || true
    fi
    if [ -n "$latest_partial_checkpoint" ] && [ "$latest_partial_checkpoint" != "$latest_checkpoint" ]; then
        echo "WARNING: checkpoint-like directory exists but is not marked resumable: $latest_partial_checkpoint"
    fi
    if [ -f "$OUTPUT/training.log" ]; then
        echo
        echo "Last 30 training log lines:"
        tail -n 30 "$OUTPUT/training.log"
    fi
}

show_logs() {
    load_config
    [ -f "$OUTPUT/training.log" ] || fail "$OUTPUT/training.log does not exist yet."
    exec tail -n 100 -f "$OUTPUT/training.log"
}

run_ready_check() {
    prepare_resources
    start_search
    evaluate_retrieval
    run_preflight
    run_smoke
    cat <<'EOF'

Vendor readiness workflow completed.

Review the smoke log, then launch formal training:
  bash scripts/vendor_train.sh launch

Start TensorBoard:
  bash scripts/vendor_train.sh tensorboard-start
EOF
}

usage() {
    cat <<'EOF'
Usage: bash scripts/vendor_train.sh <command>

Recommended workflow:
  ready              Download model/data, build the index, benchmark retrieval,
                     run preflight, and execute a one-step smoke training.
  launch             Start or resume formal training in a detached tmux session.
  status             Show retrieval health, checkpoints, tmux sessions, and logs.

Individual commands:
  prepare            Download model/data and build the local index.
  start-search       Start the managed local retrieval service.
  stop-search        Stop the managed local retrieval service.
  evaluate           Run the local retrieval benchmark.
  preflight          Validate packages, model, data, retrieval, and four GPUs.
  smoke              Run one short smoke training.
  check-checkpoint   Validate the latest resumable formal checkpoint.
  train              Start or auto-resume formal training in the foreground.
  stop-training      Stop the managed training tmux session.
  logs               Follow the formal training log.
  tensorboard        Run TensorBoard in the foreground.
  tensorboard-start  Start TensorBoard in a detached tmux session.
  diagnostics        Write a troubleshooting report under logs/.
  all                Run ready checks, then start formal training in foreground.
EOF
}

load_config

case "${1:-}" in
    ready) run_ready_check ;;
    prepare) prepare_resources ;;
    start-search) start_search ;;
    stop-search) stop_search ;;
    evaluate) evaluate_retrieval ;;
    preflight) run_preflight ;;
    smoke) run_smoke ;;
    check-checkpoint) check_checkpoint "$OUTPUT" ;;
    train|resume) run_training ;;
    launch) launch_training ;;
    stop-training) stop_training ;;
    status) show_status ;;
    logs) show_logs ;;
    tensorboard) run_tensorboard ;;
    tensorboard-start) launch_tensorboard ;;
    diagnostics)
        ensure_environment
        bash scripts/collect_diagnostics.sh
        ;;
    all)
        run_ready_check
        run_training
        ;;
    help|-h|--help|"") usage ;;
    *) usage; fail "Unknown command: $1" ;;
esac
