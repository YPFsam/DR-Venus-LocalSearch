#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RL_DIR"

VENV_DIR=${VENV_DIR:-.venv}
PROFILE_MODE=${PROFILE_MODE:-estimate}
PROFILE_INDEX_PASSAGES=${PROFILE_INDEX_PASSAGES:-100000}
PROFILE_FORCE_REBUILD_INDEX=${PROFILE_FORCE_REBUILD_INDEX:-false}
PROFILE_TOTAL_TRAINING_STEPS=${PROFILE_TOTAL_TRAINING_STEPS:-1}
PROFILE_CUDA_VISIBLE_DEVICES=${PROFILE_CUDA_VISIBLE_DEVICES:-0}
PROFILE_TIMESTAMP=${PROFILE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
PROFILE_OUTPUT=${PROFILE_OUTPUT:-./output_autodl_profile/$PROFILE_MODE-$PROFILE_TIMESTAMP}
PROFILE_EVAL_LOG_PATH=${PROFILE_EVAL_LOG_PATH:-./eval_log_autodl_profile/$PROFILE_MODE-$PROFILE_TIMESTAMP}
PROFILE_FORMAL_STEPS=${PROFILE_FORMAL_STEPS:-20}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

ensure_environment() {
    [ -x "$VENV_DIR/bin/python" ] || fail \
        "$VENV_DIR is missing. Run: bash scripts/install_vendor_env.sh"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
}

configure_mode() {
    case "$PROFILE_MODE" in
        sanity)
            PROFILE_ROLLOUT_N=${PROFILE_ROLLOUT_N:-1}
            PROFILE_MAX_TURNS=${PROFILE_MAX_TURNS:-5}
            PROFILE_MAX_MODEL_LEN=${PROFILE_MAX_MODEL_LEN:-16384}
            PROFILE_MAX_PROMPT_LEN=${PROFILE_MAX_PROMPT_LEN:-12288}
            PROFILE_MAX_RESPONSE_LEN=${PROFILE_MAX_RESPONSE_LEN:-2048}
            PROFILE_GPU_MEMORY_UTILIZATION=${PROFILE_GPU_MEMORY_UTILIZATION:-0.65}
            ;;
        estimate)
            PROFILE_ROLLOUT_N=${PROFILE_ROLLOUT_N:-2}
            PROFILE_MAX_TURNS=${PROFILE_MAX_TURNS:-20}
            PROFILE_MAX_MODEL_LEN=${PROFILE_MAX_MODEL_LEN:-32768}
            PROFILE_MAX_PROMPT_LEN=${PROFILE_MAX_PROMPT_LEN:-24576}
            PROFILE_MAX_RESPONSE_LEN=${PROFILE_MAX_RESPONSE_LEN:-4096}
            PROFILE_GPU_MEMORY_UTILIZATION=${PROFILE_GPU_MEMORY_UTILIZATION:-0.70}
            ;;
        stress)
            PROFILE_ROLLOUT_N=${PROFILE_ROLLOUT_N:-4}
            PROFILE_MAX_TURNS=${PROFILE_MAX_TURNS:-50}
            PROFILE_MAX_MODEL_LEN=${PROFILE_MAX_MODEL_LEN:-65536}
            PROFILE_MAX_PROMPT_LEN=${PROFILE_MAX_PROMPT_LEN:-57344}
            PROFILE_MAX_RESPONSE_LEN=${PROFILE_MAX_RESPONSE_LEN:-8192}
            PROFILE_GPU_MEMORY_UTILIZATION=${PROFILE_GPU_MEMORY_UTILIZATION:-0.75}
            ;;
        *)
            fail "Unsupported PROFILE_MODE=$PROFILE_MODE. Use sanity, estimate, or stress."
            ;;
    esac
    PROFILE_PPO_MINI_BATCH_SIZE=${PROFILE_PPO_MINI_BATCH_SIZE:-$PROFILE_ROLLOUT_N}
}

prepare_resources() {
    ensure_environment
    echo "Preparing AutoDL profile resources with $PROFILE_INDEX_PASSAGES passages..."
    INDEX_PASSAGES="$PROFILE_INDEX_PASSAGES" \
    FORCE_REBUILD_INDEX="$PROFILE_FORCE_REBUILD_INDEX" \
        bash scripts/bootstrap_vendor.sh
}

start_search() {
    ensure_environment
    bash scripts/vendor_train.sh start-search
}

verify_single_gpu() {
    ensure_environment
    export CUDA_VISIBLE_DEVICES="$PROFILE_CUDA_VISIBLE_DEVICES"
    "$VENV_DIR/bin/python" - "$PROFILE_MODE" <<'PY'
import sys
import torch

mode = sys.argv[1]
count = torch.cuda.device_count()
if count != 1:
    raise SystemExit(
        f"ERROR: single-GPU profile requires exactly one visible CUDA GPU, but PyTorch sees {count}. "
        "Set PROFILE_CUDA_VISIBLE_DEVICES to one GPU index."
    )
properties = torch.cuda.get_device_properties(0)
memory_gib = properties.total_memory / 1024**3
print(f"Visible profile GPU: {properties.name}; memory={memory_gib:.1f} GiB")
if memory_gib < 70 and mode != "sanity":
    print(
        "WARNING: this GPU has less than 70 GiB memory. Use PROFILE_MODE=sanity first. "
        "The estimate/stress modes may OOM and are less representative of a four-card 80 GB run."
    )
PY
}

write_profile_config() {
    mkdir -p "$PROFILE_OUTPUT" "$PROFILE_EVAL_LOG_PATH"
    cat > "$PROFILE_OUTPUT/profile_config.txt" <<EOF
PROFILE_MODE=$PROFILE_MODE
PROFILE_TOTAL_TRAINING_STEPS=$PROFILE_TOTAL_TRAINING_STEPS
PROFILE_ROLLOUT_N=$PROFILE_ROLLOUT_N
PROFILE_MAX_TURNS=$PROFILE_MAX_TURNS
PROFILE_MAX_MODEL_LEN=$PROFILE_MAX_MODEL_LEN
PROFILE_MAX_PROMPT_LEN=$PROFILE_MAX_PROMPT_LEN
PROFILE_MAX_RESPONSE_LEN=$PROFILE_MAX_RESPONSE_LEN
PROFILE_GPU_MEMORY_UTILIZATION=$PROFILE_GPU_MEMORY_UTILIZATION
PROFILE_INDEX_PASSAGES=$PROFILE_INDEX_PASSAGES
PROFILE_FORMAL_STEPS=$PROFILE_FORMAL_STEPS
EOF
}

run_profile() {
    ensure_environment
    configure_mode
    verify_single_gpu
    start_search
    if [ -e "$PROFILE_OUTPUT/training.log" ] || [ -e "$PROFILE_OUTPUT/global_step_1" ]; then
        fail "$PROFILE_OUTPUT already contains a profile run. Choose another PROFILE_OUTPUT."
    fi
    write_profile_config

    cat <<EOF
Starting single-GPU AutoDL profile.
  mode=$PROFILE_MODE
  output=$PROFILE_OUTPUT
  rollouts_per_step=$PROFILE_ROLLOUT_N
  max_turns=$PROFILE_MAX_TURNS
  max_model_len=$PROFILE_MAX_MODEL_LEN

This run is only for profiling. Do not resume formal four-GPU training from this checkpoint.
EOF

    CUDA_VISIBLE_DEVICES="$PROFILE_CUDA_VISIBLE_DEVICES" \
    NUM_GPUS=1 \
    TP_SIZE=1 \
    ULYSSES_SP_SIZE=1 \
    TRAIN_BATCH_SIZE=1 \
    PPO_MINI_BATCH_SIZE="$PROFILE_PPO_MINI_BATCH_SIZE" \
    ROLLOUT_N="$PROFILE_ROLLOUT_N" \
    ASYNC_NUM_WORKERS=1 \
    MAX_MODEL_LEN="$PROFILE_MAX_MODEL_LEN" \
    MAX_PROMPT_LEN="$PROFILE_MAX_PROMPT_LEN" \
    MAX_RESPONSE_LEN="$PROFILE_MAX_RESPONSE_LEN" \
    MAX_TURNS="$PROFILE_MAX_TURNS" \
    GPU_MEMORY_UTILIZATION="$PROFILE_GPU_MEMORY_UTILIZATION" \
    TOTAL_TRAINING_STEPS="$PROFILE_TOTAL_TRAINING_STEPS" \
    OUTPUT="$PROFILE_OUTPUT" \
    EVAL_LOG_PATH="$PROFILE_EVAL_LOG_PATH" \
    EXPERIMENT_NAME="dr-venus-autodl-single-a100-$PROFILE_MODE-$PROFILE_TIMESTAMP" \
    LOGGER_BACKENDS="['console','tensorboard']" \
    SAVE_FREQ=1 \
    RESUME_MODE=disable \
    MAX_ACTOR_CKPT_TO_KEEP=1 \
    MAX_CRITIC_CKPT_TO_KEEP=1 \
    MAX_LOCAL_CKPT_TO_KEEP=1 \
    TRACE_SAVE_INTERVAL=1 \
    TRACE_MAX_SAMPLES=2 \
        bash train_igpo.sh

    "$VENV_DIR/bin/python" scripts/check_checkpoint.py --output_dir "$PROFILE_OUTPUT"

    "$VENV_DIR/bin/python" scripts/summarize_autodl_profile.py \
        --metrics_dir "$PROFILE_EVAL_LOG_PATH" \
        --mode "$PROFILE_MODE" \
        --profile_rollouts "$PROFILE_ROLLOUT_N" \
        --profile_max_turns "$PROFILE_MAX_TURNS" \
        --formal_steps "$PROFILE_FORMAL_STEPS" \
        | tee "$PROFILE_OUTPUT/four_gpu_eta.txt"

    echo
    echo "AutoDL profile completed."
    echo "Summary: $PROFILE_OUTPUT/four_gpu_eta.txt"
    echo "Training log: $PROFILE_OUTPUT/training.log"
    echo "Metrics: $PROFILE_EVAL_LOG_PATH"
}

usage() {
    cat <<'EOF'
Usage: bash scripts/profile_single_gpu_autodl.sh <command>

Commands:
  prepare   Download the SFT checkpoint, prepare RL data, and build a 100K index.
  run       Execute one reduced single-GPU RL step and print a four-GPU ETA range.
  all       Run prepare, then run. This is the recommended first command on AutoDL.
  help      Show this message.

Profile modes:
  PROFILE_MODE=sanity    Fast pipeline validation; ETA confidence is very low.
  PROFILE_MODE=estimate  Default balance for one A100 80 GB; ETA confidence is low.
  PROFILE_MODE=stress    Closer workload; slower and may OOM; ETA confidence is better.

Examples:
  PROFILE_MODE=sanity bash scripts/profile_single_gpu_autodl.sh all
  PROFILE_MODE=estimate bash scripts/profile_single_gpu_autodl.sh run
  PROFILE_FORMAL_STEPS=20 PROFILE_MODE=stress bash scripts/profile_single_gpu_autodl.sh run
EOF
}

configure_mode

case "${1:-}" in
    prepare) prepare_resources ;;
    run) run_profile ;;
    all)
        prepare_resources
        run_profile
        ;;
    help|-h|--help|"") usage ;;
    *) usage; fail "Unknown command: $1" ;;
esac
