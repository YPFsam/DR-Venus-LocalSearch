#!/usr/bin/env python3
"""Local search HTTP server.

Replaces Serper + Jina + LLM summarization with a single local service. Tantivy
is the default inverted-index backend. SQLite FTS5 and the original rank_bm25
full-scan backend remain available for comparison.

Architecture:
    [Ray Worker 1] ──HTTP──┐
    [Ray Worker 2] ──HTTP──┼──▶ [local_search_server:8890] ──▶ [selected index]
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
import sqlite3
import string
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

# ── Global state (loaded once at startup) ──────────────────────────────────
_passages: List[Dict] = []
_passage_map: Dict[str, Dict] = {}
_bm25 = None
_backend = "tantivy"
_num_passages = 0
_sqlite_path = ""
_sqlite_local = threading.local()
_tantivy_index = None
_tantivy_searcher = None
logger = logging.getLogger("local_search")

_STOP_WORDS = {
    "a", "about", "after", "all", "also", "an", "and", "are", "as", "at", "be", "before", "between",
    "by", "can", "did", "do", "does", "during", "each", "find", "for", "from", "had", "has", "have", "how",
    "i", "if", "in", "into", "is", "it", "its", "may", "more", "most", "not", "of", "on", "or", "other",
    "please", "should", "than", "that", "the", "their", "then", "there", "these", "they", "this", "to", "use",
    "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
}
_MAX_FTS_QUERY_TERMS = 8


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


def _validate_index_metadata(index_dir: str, backend: str, num_passages: int) -> None:
    """Reject incomplete or mixed-version index directories before serving."""
    metadata_path = os.path.join(index_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise SystemExit(f"ERROR: {metadata_path} not found. Rebuild the local search index first.")
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    metadata_backend = metadata.get("backend")
    if metadata_backend not in {backend, "both"}:
        raise SystemExit(
            f"ERROR: index metadata backend is {metadata_backend!r}, requested {backend!r}. "
            "Rebuild the local search index or select the matching backend."
        )
    metadata_passages = int(metadata.get("num_passages", -1))
    if metadata_passages != num_passages:
        raise SystemExit(
            f"ERROR: index metadata declares {metadata_passages} passages but {backend} loaded "
            f"{num_passages}. Rebuild the local search index."
        )


def load_index(index_dir: str, backend: str):
    """Load the selected search backend."""
    global _backend, _num_passages, _passages, _passage_map, _bm25, _sqlite_path
    global _tantivy_index, _tantivy_searcher

    _backend = backend
    if backend == "tantivy":
        try:
            import tantivy
        except ImportError as exc:
            raise SystemExit("ERROR: 'tantivy' not installed. Run: pip install tantivy") from exc
        tantivy_path = os.path.abspath(os.path.join(index_dir, "tantivy_index"))
        if not os.path.isdir(tantivy_path):
            logger.error("%s not found. Rebuild the local search index first.", tantivy_path)
            sys.exit(1)
        _tantivy_index = tantivy.Index.open(tantivy_path)
        _tantivy_searcher = _tantivy_index.searcher()
        _num_passages = int(_tantivy_searcher.num_docs)
        _validate_index_metadata(index_dir, backend, _num_passages)
        logger.info("Tantivy index ready: %s (%d passages)", tantivy_path, _num_passages)
        return

    if backend == "sqlite_fts5":
        _sqlite_path = os.path.abspath(os.path.join(index_dir, "sqlite_fts.db"))
        if not os.path.exists(_sqlite_path):
            logger.error("%s not found. Rebuild the local search index first.", _sqlite_path)
            sys.exit(1)
        connection = _get_sqlite_connection()
        _num_passages = int(connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0])
        _validate_index_metadata(index_dir, backend, _num_passages)
        logger.info("SQLite FTS5 index ready: %s (%d passages)", _sqlite_path, _num_passages)
        return

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
    _num_passages = len(_passages)
    _validate_index_metadata(index_dir, backend, _num_passages)


def _get_sqlite_connection() -> sqlite3.Connection:
    connection = getattr(_sqlite_local, "connection", None)
    if connection is None:
        connection = sqlite3.connect(f"file:{_sqlite_path}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        _sqlite_local.connection = connection
    return connection


def search(query: str, topk: int = 10) -> List[Dict]:
    """Query the selected backend and return top-k passages."""
    if _backend == "tantivy":
        return _search_tantivy(query, topk)
    if _backend == "sqlite_fts5":
        return _search_sqlite_fts(query, topk)
    return _search_rank_bm25(query, topk)


def _search_rank_bm25(query: str, topk: int) -> List[Dict]:
    """Query rank_bm25 by scoring the full corpus."""
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


def _search_tantivy(query: str, topk: int) -> List[Dict]:
    """Query Tantivy's inverted index and return stored passages."""
    tokens = _tokenize(query)
    if not tokens:
        return []
    parsed_query, _ = _tantivy_index.parse_query_lenient(
        " ".join(tokens),
        default_field_names=["title", "text"],
        field_boosts={"title": 2.0, "text": 1.0},
    )
    hits = _tantivy_searcher.search(parsed_query, limit=topk).hits
    results = []
    for score, address in hits:
        document = _tantivy_searcher.doc(address).to_dict()
        results.append(
            {
                "id": document["id"][0],
                "title": document["title"][0],
                "text": document["text"][0],
                "score": float(score),
            }
        )
    return results


def _search_sqlite_fts(query: str, topk: int) -> List[Dict]:
    """Query SQLite FTS5 using selective low-frequency terms and BM25 ranking."""
    tokens = [
        token for token in dict.fromkeys(_tokenize(query))
        if len(token) > 2 and token not in _STOP_WORDS
    ]
    if not tokens:
        tokens = list(dict.fromkeys(_tokenize(query)))
    if not tokens:
        return []
    connection = _get_sqlite_connection()
    placeholders = ", ".join("?" for _ in tokens)
    document_frequencies = dict(connection.execute(
        f"SELECT term, doc FROM passages_vocab WHERE term IN ({placeholders})",
        tokens,
    ).fetchall())
    selected_tokens = sorted(
        (token for token in tokens if token in document_frequencies),
        key=lambda token: document_frequencies[token],
    )[:_MAX_FTS_QUERY_TERMS]
    if not selected_tokens:
        return []
    match_expression = " OR ".join(f'"{token}"' for token in selected_tokens)
    rows = connection.execute(
        """
        SELECT passages.id, passages.title, passages.text, -bm25(passages_fts, 2.0, 1.0) AS score
        FROM passages_fts
        JOIN passages ON passages.rowid = passages_fts.rowid
        WHERE passages_fts MATCH ?
        ORDER BY bm25(passages_fts, 2.0, 1.0)
        LIMIT ?
        """,
        (match_expression, topk),
    ).fetchall()
    return [
        {"id": row[0], "title": row[1], "text": row[2], "score": float(row[3])}
        for row in rows
    ]


def get_document(doc_id: str) -> Optional[Dict]:
    """Get a passage by its ID."""
    if _backend == "tantivy":
        parsed_query, _ = _tantivy_index.parse_query_lenient(f'id:"{doc_id}"')
        hits = _tantivy_searcher.search(parsed_query, limit=1).hits
        if not hits:
            return None
        document = _tantivy_searcher.doc(hits[0][1]).to_dict()
        return {"id": document["id"][0], "title": document["title"][0], "text": document["text"][0]}
    if _backend == "sqlite_fts5":
        row = _get_sqlite_connection().execute(
            "SELECT id, title, text FROM passages WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "title": row[1], "text": row[2]}
    return _passage_map.get(doc_id)


# ── Flask app ──────────────────────────────────────────────────────────────
def create_app():
    """Create Flask application with search and visit endpoints."""
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "backend": _backend, "num_passages": _num_passages})

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
    parser = argparse.ArgumentParser(description="Local search HTTP server")
    parser.add_argument("--port", type=int, default=8890, help="Server port (default: 8890)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--index_dir", default="data/local_search_index",
                        help="Directory containing the local search index")
    parser.add_argument("--backend", choices=["tantivy", "sqlite_fts5", "rank_bm25"], default="tantivy",
                        help="Search backend (default: tantivy)")
    parser.add_argument("--log_file", default="logs/local_search.log",
                        help="Rotating log file for query timing (default: logs/local_search.log)")
    args = parser.parse_args()

    configure_logging(args.log_file)
    load_index(args.index_dir, args.backend)
    app = create_app()
    logger.info("Local search server starting on http://%s:%d", args.host, args.port)
    logger.info("Backend: %s; passages: %d; endpoints: /health, /search, /document/<id>",
                _backend, _num_passages)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
