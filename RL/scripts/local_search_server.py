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
import logging
import os
import pickle
import string
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

# ── Global state (loaded once at startup) ──────────────────────────────────
_passages: List[Dict] = []
_passage_map: Dict[str, Dict] = {}
_bm25 = None
logger = logging.getLogger("local_search")


def configure_logging(log_file: str):
    """Log search timing to the console and a bounded local file."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


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
        logger.error("%s not found. Run build_bm25_index.py first.", passages_path)
        sys.exit(1)
    if not os.path.exists(bm25_path):
        logger.error("%s not found. Run build_bm25_index.py first.", bm25_path)
        sys.exit(1)

    logger.info("Loading passages from %s", passages_path)
    t0 = time.time()
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            _passages.append(p)
            _passage_map[p["id"]] = p
    logger.info("%d passages loaded in %.1fs", len(_passages), time.time() - t0)

    logger.info("Loading BM25 index from %s", bm25_path)
    t0 = time.time()
    with open(bm25_path, "rb") as f:
        _bm25 = pickle.load(f)
    logger.info("BM25 index loaded in %.1fs", time.time() - t0)


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
        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            return jsonify({"error": "queries must be a list of strings"}), 400
        if not isinstance(topk, int) or topk <= 0 or topk > 100:
            return jsonify({"error": "topk must be an integer between 1 and 100"}), 400

        started = time.perf_counter()
        all_results = []
        query_latencies_ms = []
        try:
            for q in queries:
                query_started = time.perf_counter()
                passages = search(q, topk=topk)
                query_latencies_ms.append((time.perf_counter() - query_started) * 1000)
                all_results.append({"query": q, "passages": passages})
        except Exception:
            logger.exception("Search request failed")
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        mean_query_ms = sum(query_latencies_ms) / len(query_latencies_ms) if query_latencies_ms else 0.0
        max_query_ms = max(query_latencies_ms, default=0.0)
        logger.info(
            "search batch queries=%d topk=%d elapsed_ms=%.1f mean_query_ms=%.1f "
            "max_query_ms=%.1f result_counts=%s",
            len(queries),
            topk,
            elapsed_ms,
            mean_query_ms,
            max_query_ms,
            [len(group["passages"]) for group in all_results],
        )
        if os.environ.get("LOCAL_SEARCH_LOG_QUERIES", "false").lower() == "true":
            logger.info("search query_text=%s", json.dumps([query[:300] for query in queries], ensure_ascii=False))

        return jsonify({"results": all_results})

    @app.route("/document/<path:doc_id>", methods=["GET"])
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
    parser.add_argument("--log_file", default="logs/local_search.log",
                        help="Rotating log file for query timing (default: logs/local_search.log)")
    args = parser.parse_args()

    configure_logging(args.log_file)
    load_index(args.index_dir)
    app = create_app()
    logger.info("Local search server starting on http://%s:%d", args.host, args.port)
    logger.info("Passages: %d; endpoints: /health, /search, /document/<id>", len(_passages))
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
