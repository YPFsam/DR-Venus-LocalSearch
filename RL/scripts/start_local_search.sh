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

INDEX_DIR=${INDEX_DIR:-data/local_search_index}
LOCAL_SEARCH_HOST=${LOCAL_SEARCH_HOST:-0.0.0.0}
LOCAL_SEARCH_PORT=${LOCAL_SEARCH_PORT:-8890}
LOCAL_SEARCH_LOG_FILE=${LOCAL_SEARCH_LOG_FILE:-logs/local_search.log}

exec python3 scripts/local_search_server.py \
    --index_dir "$INDEX_DIR" \
    --host "$LOCAL_SEARCH_HOST" \
    --port "$LOCAL_SEARCH_PORT" \
    --log_file "$LOCAL_SEARCH_LOG_FILE"
