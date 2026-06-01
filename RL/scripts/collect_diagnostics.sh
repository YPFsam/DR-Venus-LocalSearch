#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RL_DIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

LOG_DIR=${LOG_DIR:-logs}
LOCAL_SEARCH_SERVER_URL=${LOCAL_SEARCH_SERVER_URL:-http://localhost:8890}
LOCAL_SEARCH_LOG_FILE=${LOCAL_SEARCH_LOG_FILE:-logs/local_search.log}
OUTPUT=${OUTPUT:-./output}
mkdir -p "$LOG_DIR"

REPORT="$LOG_DIR/diagnostics_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee "$REPORT") 2>&1

run_optional() {
    echo
    echo "### $*"
    "$@" || true
}

echo "DR-Venus diagnostics"
echo "Generated: $(date --iso-8601=seconds)"
echo "Working directory: $RL_DIR"

run_optional git rev-parse HEAD
run_optional git status --short
run_optional uname -a
run_optional python3 --version
run_optional nvidia-smi
run_optional free -h
run_optional df -h .
run_optional curl -fsS "$LOCAL_SEARCH_SERVER_URL/health"

if [ -f data/local_search_index/metadata.json ]; then
    run_optional cat data/local_search_index/metadata.json
fi

run_optional python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ["torch", "vllm", "ray", "transformers", "datasets", "rank-bm25", "flask", "wandb"]:
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
PY

if [ -f "$OUTPUT/training.log" ]; then
    run_optional tail -n 200 "$OUTPUT/training.log"
fi

if [ -f "$LOCAL_SEARCH_LOG_FILE" ]; then
    run_optional tail -n 200 "$LOCAL_SEARCH_LOG_FILE"
fi

echo
echo "Diagnostics written to: $REPORT"
