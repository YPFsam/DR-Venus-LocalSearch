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
LOCAL_SEARCH_BACKEND=${LOCAL_SEARCH_BACKEND:-tantivy}
LOCAL_SEARCH_SERVER_URL=${LOCAL_SEARCH_SERVER_URL:-http://127.0.0.1:8890}
SAMPLE_SIZE=${SAMPLE_SIZE:-1000}
BATCH_SIZE=${BATCH_SIZE:-8}
CONCURRENCY=${CONCURRENCY:-1}
INDEX_LABEL=${INDEX_LABEL:-index}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

CONSOLE_LOG="$LOG_DIR/local_search_console_${INDEX_LABEL}.log"
EVAL_LOG="$LOG_DIR/retrieval_eval_${INDEX_LABEL}.log"

python3 scripts/local_search_server.py \
    --index_dir "$INDEX_DIR" \
    --host "$LOCAL_SEARCH_HOST" \
    --port "$LOCAL_SEARCH_PORT" \
    --backend "$LOCAL_SEARCH_BACKEND" \
    --log_file "$LOCAL_SEARCH_LOG_FILE" \
    > "$CONSOLE_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=false
for _ in $(seq 1 90); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: local search server exited before becoming healthy" >&2
        sed -n '1,240p' "$CONSOLE_LOG" >&2
        exit 1
    fi
    if curl --noproxy '*' -fsS "$LOCAL_SEARCH_SERVER_URL/health" > "$LOG_DIR/health_${INDEX_LABEL}.json"; then
        ready=true
        break
    fi
    sleep 1
done

if [ "$ready" != "true" ]; then
    echo "ERROR: local search server did not become healthy" >&2
    sed -n '1,240p' "$CONSOLE_LOG" >&2
    exit 1
fi

echo "Health:"
cat "$LOG_DIR/health_${INDEX_LABEL}.json"
echo
echo "Server RSS before evaluation:"
ps -o rss= -p "$SERVER_PID"
echo "Index disk usage:"
du -sh "$INDEX_DIR"
echo

python3 scripts/evaluate_local_retrieval.py \
    --server_url "$LOCAL_SEARCH_SERVER_URL" \
    --sample_size "$SAMPLE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --concurrency "$CONCURRENCY" \
    | tee "$EVAL_LOG"

echo
echo "Server RSS after evaluation:"
ps -o rss= -p "$SERVER_PID"
echo "Evaluation log: $EVAL_LOG"
