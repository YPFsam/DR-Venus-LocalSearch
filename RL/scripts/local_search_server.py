#!/usr/bin/env python3
"""Local BM25 search HTTP server.

Replaces Serper + Jina + LLM summarization with a single local service.
Loads the BM25 index once at startup; all Ray workers share it via HTTP.

Architecture:
    [Ray Worker 1] ──HTTP──┐
    [Ray Worker 2] ──HTTP──┼──▶ [local_search_server:8890] ──▶ [BM25 index in RAM]
    [Ray Worker N] ──HTTP──┘

Start before training:
    python scripts/local_search_server.py --port 8890 --index_dir data/local_search_index

Usage in train_igpo.sh:
    LOCAL_SEARCH_SERVER_URL=http://localhost:8890
"""

import argparse
import json
import os
import pickle
import string
import sys
import time
from typing import Dict, List, Optional

# ── Global state (loaded once at startup) ──────────────────────────────────
_passages: List[Dict] = []
_passage_map: Dict[str, Dict] = {}
_bm25 = None


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization with punctuation removal."""
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    return text.split()


def load_index(index_dir: str):
    """Load passages and BM25 index from disk."""
    global _passages, _passage_map, _bm25

    passages_path = os.path.join(index_dir, "passages.jsonl")
    bm25_path = os.path.join(index_dir, "bm25_index.pkl")

    if not os.path.exists(passages_path):
        print(f"ERROR: {passages_path} not found. Run build_bm25_index.py first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(bm25_path):
        print(f"ERROR: {bm25_path} not found. Run build_bm25_index.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading passages from {passages_path}...")
    t0 = time.time()
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            _passages.append(p)
            _passage_map[p["id"]] = p
    print(f"  {_passages} passages loaded in {time.time() - t0:.1f}s")

    print(f"Loading BM25 index from {bm25_path}...")
    t0 = time.time()
    with open(bm25_path, "rb") as f:
        _bm25 = pickle.load(f)
    print(f"  BM25 index loaded in {time.time() - t0:.1f}s")


def search(query: str, topk: int = 10) -> List[Dict]:
    """Query BM25 and return top-k passages."""
    import numpy as np

    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)

    if len(scores) == 0:
        return []

    k = min(topk, len(scores))
    if k < len(scores):
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    else:
        top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        idx = int(idx)
        score = float(scores[idx])
        if score <= 0:
            continue
        p = _passages[idx]
        results.append({
            "id": p["id"],
            "title": p["title"],
            "text": p["text"],
            "score": score,
        })
    return results


def get_document(doc_id: str) -> Optional[Dict]:
    """Get a passage by its ID."""
    return _passage_map.get(doc_id)


# ── Flask app ──────────────────────────────────────────────────────────────
def create_app():
    """Create Flask application with search and visit endpoints."""
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "num_passages": len(_passages)})

    @app.route("/search", methods=["POST"])
    def search_endpoint():
        """
        Search endpoint compatible with tool_search.py.

        Request:  {"queries": ["q1", "q2"], "topk": 10}
        Response: {"results": [{"query": "q1", "passages": [{"id": ..., "title": ..., "text": ..., "score": ...}]}]}
        """
        data = request.get_json(force=True)
        queries = data.get("queries", [])
        topk = data.get("topk", 10)

        all_results = []
        for q in queries:
            passages = search(q, topk=topk)
            all_results.append({"query": q, "passages": passages})

        return jsonify({"results": all_results})

    @app.route("/document/<doc_id>", methods=["GET"])
    def document_endpoint(doc_id):
        """
        Document endpoint compatible with tool_visit.py.

        Response: {"id": ..., "title": ..., "text": ...} or {"error": "not found"}
        """
        doc = get_document(doc_id)
        if doc is None:
            return jsonify({"error": f"Document {doc_id} not found"}), 404
        return jsonify(doc)

    return app


def main():
    parser = argparse.ArgumentParser(description="Local BM25 search HTTP server")
    parser.add_argument("--port", type=int, default=8890, help="Server port (default: 8890)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--index_dir", default="data/local_search_index",
                        help="Directory containing passages.jsonl and bm25_index.pkl")
    parser.add_argument("--topk", type=int, default=10, help="Default top-k for search (default: 10)")
    args = parser.parse_args()

    load_index(args.index_dir)
    app = create_app()
    print(f"\nLocal search server starting on http://{args.host}:{args.port}")
    print(f"  Passages: {len(_passages)}")
    print(f"  Endpoints: /health, /search, /document/<id>")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
