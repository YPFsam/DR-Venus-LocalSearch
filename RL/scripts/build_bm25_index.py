#!/usr/bin/env python3
"""Build a local search index from Wikipedia passages for offline retrieval.

This script downloads Wikipedia passages (from HuggingFace datasets or a local file),
splits them into manageable chunks, and builds a Tantivy inverted index by
default. SQLite FTS5 and the original `rank_bm25` full-scan backend remain
available for comparison.

Usage:
    # Option 1: Download Wikipedia from HuggingFace (recommended)
    python build_bm25_index.py --output_dir data/local_search_index --topk_passages 100000

    # Option 2: Use a local JSONL file
    python build_bm25_index.py --output_dir data/local_search_index --input_file data/wiki_passages.jsonl

Output:
    data/local_search_index/
    ├── passages.jsonl       # Passage corpus (id, title, text)
    ├── tantivy_index/       # Tantivy inverted index (default)
    ├── sqlite_fts.db        # Optional SQLite FTS5 inverted index
    ├── bm25_index.pkl       # Optional rank_bm25 full-scan index
    └── metadata.json        # Index metadata (num_passages, avg_length, etc.)
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List


def load_passages_from_huggingface(max_passages: int = 2_000_000) -> List[Dict]:
    """Load Wikipedia passages from HuggingFace datasets.

    Uses the official wikimedia/wikipedia dataset (20231101.en dump).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading Wikipedia dataset from HuggingFace (max {max_passages} passages)...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    passages = []
    article_count = 0
    for i, article in enumerate(ds):
        article_count = i + 1
        text = article.get("text", "").strip()
        title = article.get("title", "").strip()
        if not text or len(text) < 50:
            continue

        # Split long articles into passages (~200 words each)
        words = text.split()
        chunk_size = 200
        for j in range(0, len(words), chunk_size):
            if len(passages) >= max_passages:
                break
            chunk = " ".join(words[j : j + chunk_size])
            if len(chunk.strip()) < 30:
                continue
            passages.append({
                "id": f"wiki_{len(passages)}",
                "title": title,
                "text": chunk,
            })

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} articles, {len(passages)} passages...")

        if len(passages) >= max_passages:
            break

    print(f"Loaded {len(passages)} passages from {article_count} articles.")
    return passages


def load_passages_from_file(filepath: str) -> List[Dict]:
    """Load passages from a JSONL file.

    Each line should be a JSON object with 'title' and 'text' fields.
    """
    passages = []
    print(f"Loading passages from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                title = obj.get("title", "")
                text = obj.get("text", "")
                if not text:
                    continue
                passages.append({
                    "id": obj.get("id", f"doc_{line_num}"),
                    "title": title,
                    "text": text,
                })
            except json.JSONDecodeError:
                continue
            if line_num % 100000 == 0:
                print(f"  Loaded {line_num} lines, {len(passages)} valid passages...")

    print(f"Loaded {len(passages)} passages from {filepath}.")
    return passages


def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization: lowercase + split on whitespace + remove punctuation."""
    import string
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    return text.split()


def build_bm25_index(passages: List[Dict]):
    """Build BM25 index using rank_bm25 library."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("ERROR: 'rank_bm25' not installed. Run: pip install rank_bm25")
        sys.exit(1)

    print(f"Tokenizing {len(passages)} passages...")
    t0 = time.time()
    tokenized_corpus = []
    for i, p in enumerate(passages):
        tokens = tokenize_simple(p["title"] + " " + p["text"])
        tokenized_corpus.append(tokens)
        if (i + 1) % 100000 == 0:
            print(f"  Tokenized {i + 1}/{len(passages)} passages...")

    print(f"Tokenization took {time.time() - t0:.1f}s")
    print(f"Building BM25 index...")
    t0 = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built in {time.time() - t0:.1f}s")
    return bm25


def remove_stale_backend_artifacts(output_dir: Path, backend: str) -> None:
    """Remove indexes that no longer match the freshly written passage corpus."""
    selected = {
        "tantivy": {"tantivy"},
        "sqlite_fts5": {"sqlite_fts5"},
        "rank_bm25": {"rank_bm25"},
        "both": {"sqlite_fts5", "rank_bm25"},
    }[backend]
    artifacts = {
        "tantivy": output_dir / "tantivy_index",
        "sqlite_fts5": output_dir / "sqlite_fts.db",
        "rank_bm25": output_dir / "bm25_index.pkl",
    }
    for artifact_backend, path in artifacts.items():
        if artifact_backend in selected or not path.exists():
            continue
        print(f"Removing stale {artifact_backend} index: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def reset_staging_directory(staging_dir: Path, output_dir: Path) -> None:
    """Create an empty sibling staging directory without touching the live index."""
    if staging_dir.resolve().parent != output_dir.resolve().parent:
        raise RuntimeError(f"Refusing to remove unexpected staging directory: {staging_dir}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)


def promote_staging_directory(staging_dir: Path, output_dir: Path) -> None:
    """Replace the live index only after every staged artifact is complete."""
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
    if backup_dir.resolve().parent != output_dir.resolve().parent:
        raise RuntimeError(f"Refusing to replace unexpected index directory: {output_dir}")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except Exception:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def main():
    parser = argparse.ArgumentParser(description="Build index for local retrieval")
    parser.add_argument("--output_dir", default="data/local_search_index",
                        help="Output directory for index files")
    parser.add_argument("--input_file", default=None,
                        help="Local JSONL file with passages (if not using HuggingFace)")
    parser.add_argument("--topk_passages", type=int, default=100_000,
                        help="Max number of passages to index (default: 100K)")
    parser.add_argument("--backend", choices=["tantivy", "sqlite_fts5", "rank_bm25", "both"], default="tantivy",
                        help="Index backend to build (default: tantivy)")
    args = parser.parse_args()

    # Load passages
    if args.input_file:
        passages = load_passages_from_file(args.input_file)
    else:
        passages = load_passages_from_huggingface(args.topk_passages)

    if not passages:
        print("ERROR: No passages loaded.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    reset_staging_directory(staging_dir, output_dir)

    passages_path = staging_dir / "passages.jsonl"
    print(f"Saving passages to {passages_path}...")
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    remove_stale_backend_artifacts(staging_dir, args.backend)

    if args.backend in {"rank_bm25", "both"}:
        bm25 = build_bm25_index(passages)
        bm25_path = staging_dir / "bm25_index.pkl"
        print(f"Saving BM25 index to {bm25_path}...")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

    if args.backend in {"sqlite_fts5", "both"}:
        from build_sqlite_fts_index import build_sqlite_fts_index

        build_sqlite_fts_index(
            passages_file=Path(passages_path),
            output_file=staging_dir / "sqlite_fts.db",
        )

    if args.backend == "tantivy":
        from build_tantivy_index import build_tantivy_index

        build_tantivy_index(
            passages_file=Path(passages_path),
            output_dir=staging_dir / "tantivy_index",
        )

    avg_len = sum(len(p["text"].split()) for p in passages) / len(passages)
    metadata = {
        "num_passages": len(passages),
        "avg_passage_length": avg_len,
        "source": args.input_file or "wikimedia/wikipedia-20231101.en",
        "backend": args.backend,
    }
    metadata_path = staging_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    promote_staging_directory(staging_dir, output_dir)

    print(f"\nIndex built successfully!")
    print(f"  Passages: {len(passages)}")
    print(f"  Avg passage length: {avg_len:.1f} words")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
