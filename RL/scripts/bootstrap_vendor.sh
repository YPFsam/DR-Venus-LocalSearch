#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RL_DIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

DEFAULT_MODEL_PATH="$RL_DIR/data/models/DR-Venus-4B-SFT"
if [ -z "${MODEL_PATH:-}" ] || [ "$MODEL_PATH" = "/absolute/path/to/DR-Venus-4B-SFT" ]; then
    MODEL_PATH="$DEFAULT_MODEL_PATH"
fi
INDEX_PASSAGES=${INDEX_PASSAGES:-500000}

if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: Hugging Face CLI not found. Run: pip install --upgrade huggingface_hub" >&2
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Downloading official SFT checkpoint to: $MODEL_PATH"
    hf download inclusionAI/DR-Venus-4B-SFT --local-dir "$MODEL_PATH"
else
    echo "Using existing SFT checkpoint: $MODEL_PATH"
fi

if [ ! -f .env ]; then
    cp .env.example .env
fi

if grep -q '^MODEL_PATH=/absolute/path/to/DR-Venus-4B-SFT$' .env; then
    python3 - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys

env_path = Path(".env")
model_path = sys.argv[1]
text = env_path.read_text(encoding="utf-8")
text = text.replace("MODEL_PATH=/absolute/path/to/DR-Venus-4B-SFT", f"MODEL_PATH={model_path}")
env_path.write_text(text, encoding="utf-8")
PY
    echo "Configured .env with MODEL_PATH=$MODEL_PATH"
else
    echo "Using existing .env. Verify MODEL_PATH before training."
fi

INDEX_PASSAGES="$INDEX_PASSAGES" bash scripts/prepare_local_rl.sh

cat <<'EOF'

Bootstrap completed.

Terminal 1:
  bash scripts/start_local_search.sh

Terminal 2:
  python3 scripts/evaluate_local_retrieval.py --sample_size 100
  PRECHECK_ONLY=true bash train_igpo.sh
  OUTPUT=./output_smoke TOTAL_TRAINING_STEPS=1 MAX_TURNS=5 GPU_MEMORY_UTILIZATION=0.75 bash train_igpo.sh
  bash train_igpo.sh

If troubleshooting is needed:
  bash scripts/collect_diagnostics.sh
EOF
