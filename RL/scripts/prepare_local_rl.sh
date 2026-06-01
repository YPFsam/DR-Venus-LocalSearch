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

INDEX_PASSAGES=${INDEX_PASSAGES:-500000}
INDEX_DIR=${INDEX_DIR:-data/local_search_index}
FORCE_REBUILD_INDEX=${FORCE_REBUILD_INDEX:-false}
LOCAL_SEARCH_BACKEND=${LOCAL_SEARCH_BACKEND:-tantivy}

case "$LOCAL_SEARCH_BACKEND" in
    tantivy) REQUIRED_INDEX_FILE="$INDEX_DIR/tantivy_index/meta.json" ;;
    sqlite_fts5) REQUIRED_INDEX_FILE="$INDEX_DIR/sqlite_fts.db" ;;
    rank_bm25) REQUIRED_INDEX_FILE="$INDEX_DIR/bm25_index.pkl" ;;
    *)
        echo "ERROR: unsupported LOCAL_SEARCH_BACKEND=$LOCAL_SEARCH_BACKEND" >&2
        exit 1
        ;;
esac

python3 scripts/prepare_redsearcher_data.py

if [ "$FORCE_REBUILD_INDEX" = "true" ] || \
   [ ! -f "$INDEX_DIR/passages.jsonl" ] || \
   [ ! -f "$REQUIRED_INDEX_FILE" ] || \
   [ ! -f "$INDEX_DIR/metadata.json" ]; then
    python3 scripts/build_bm25_index.py \
        --output_dir "$INDEX_DIR" \
        --topk_passages "$INDEX_PASSAGES" \
        --backend "$LOCAL_SEARCH_BACKEND"
else
    python3 - "$INDEX_DIR/metadata.json" "$LOCAL_SEARCH_BACKEND" "$INDEX_PASSAGES" <<'PY'
import json
from pathlib import Path
import sys

metadata_path = Path(sys.argv[1])
expected_backend = sys.argv[2]
expected_passages = int(sys.argv[3])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
actual_backend = metadata.get("backend")
actual_passages = int(metadata.get("num_passages", -1))
if actual_backend not in {expected_backend, "both"} or actual_passages != expected_passages:
    raise SystemExit(
        "ERROR: existing local index does not match the requested configuration: "
        f"metadata backend={actual_backend!r}, passages={actual_passages}; "
        f"requested backend={expected_backend!r}, passages={expected_passages}. "
        "Run with FORCE_REBUILD_INDEX=true to rebuild it intentionally."
    )
PY
    echo "Using existing $LOCAL_SEARCH_BACKEND index: $INDEX_DIR"
fi

echo "Local RL data and $LOCAL_SEARCH_BACKEND index are ready."
echo "Start retrieval with: bash scripts/start_local_search.sh"
