#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RL_DIR"

INDEX_PASSAGES=${INDEX_PASSAGES:-100000}
INDEX_DIR=${INDEX_DIR:-data/local_search_index}
FORCE_REBUILD_INDEX=${FORCE_REBUILD_INDEX:-false}

python3 scripts/prepare_redsearcher_data.py

if [ "$FORCE_REBUILD_INDEX" = "true" ] || \
   [ ! -f "$INDEX_DIR/passages.jsonl" ] || \
   [ ! -f "$INDEX_DIR/bm25_index.pkl" ] || \
   [ ! -f "$INDEX_DIR/metadata.json" ]; then
    python3 scripts/build_bm25_index.py \
        --output_dir "$INDEX_DIR" \
        --topk_passages "$INDEX_PASSAGES"
else
    echo "Using existing BM25 index: $INDEX_DIR"
fi

echo "Local RL data and BM25 index are ready."
echo "Start retrieval with: bash scripts/start_local_search.sh"
