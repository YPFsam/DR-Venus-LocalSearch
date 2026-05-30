#!/usr/bin/env python3
"""Build a local BM25 index from Wikipedia passages for offline retrieval.

This script downloads Wikipedia passages (from HuggingFace datasets or a local file),
splits them into manageable chunks, and builds a BM25 index using the `rank_bm25` library.

Usage:
    # Option 1: Download Wikipedia from HuggingFace (recommended)
    python build_bm25_index.py --output_dir data/local_search_index --topk_passages 2000000

    # Option 2: Use a local JSONL file
    python build_bm25_index.py --output_dir data/local_search_index --input_file data/wiki_passages.jsonl

    # Option 3: Use Elasticsearch instead of rank_bm25
    python build_bm25_index.py --output_dir data/local_search_index --use_es --es_host localhost:9200

Output:
    data/local_search_index/
    ├── passages.jsonl       # Passage corpus (id, title, text)
    ├── bm25_index.pkl       # Pre-built BM25 index (rank_bm25 format)
    └── metadata.json        # Index metadata (num_passages, avg_length, etc.)
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import List, Dict, Optional


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
        trust_remote_code=True,
    )

    passages = []
    for i, article in enumerate(ds):
        if i >= max_passages:
            break
        text = article.get("text", "").strip()
        title = article.get("title", "").strip()
        if not text or len(text) < 50:
            continue

        # Split long articles into passages (~200 words each)
        words = text.split()
        chunk_size = 200
        for j in range(0, len(words), chunk_size):
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

    print(f"Loaded {len(passages)} passages from {min(i + 1, max_passages)} articles.")
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


def build_elasticsearch_index(passages: List[Dict], es_host: str = "localhost:9200",
                               index_name: str = "wikipedia"):
    """Build BM25 index in Elasticsearch.

    Requires Elasticsearch to be running. Start with:
        docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.10.0
    """
    try:
        from elasticsearch import Elasticsearch, helpers
    except ImportError:
        print("ERROR: 'elasticsearch' not installed. Run: pip install elasticsearch")
        sys.exit(1)

    es = Elasticsearch(f"http://{es_host}")
    if not es.ping():
        print(f"ERROR: Cannot connect to Elasticsearch at {es_host}")
        sys.exit(1)

    # Delete existing index if any
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # Create index with BM25 analyzer
    es.indices.create(
        index=index_name,
        body={
            "settings": {
                "number_of_shards": 1,
                "index": {"similarity": {"default": {"type": "BM25"}}},
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                }
            },
        },
    )

    # Bulk index
    actions = [
        {
            "_index": index_name,
            "_id": p["id"],
            "_source": {"title": p["title"], "text": p["text"]},
        }
        for p in passages
    ]
    print(f"Indexing {len(actions)} passages into Elasticsearch...")
    helpers.bulk(es, actions, chunk_size=5000)
    es.indices.refresh(index=index_name)
    print(f"Done. Indexed {len(actions)} passages.")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for local retrieval")
    parser.add_argument("--output_dir", default="data/local_search_index",
                        help="Output directory for index files")
    parser.add_argument("--input_file", default=None,
                        help="Local JSONL file with passages (if not using HuggingFace)")
    parser.add_argument("--topk_passages", type=int, default=2_000_000,
                        help="Max number of passages to index (default: 2M)")
    parser.add_argument("--use_es", action="store_true",
                        help="Use Elasticsearch instead of rank_bm25")
    parser.add_argument("--es_host", default="localhost:9200",
                        help="Elasticsearch host (default: localhost:9200)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load passages
    if args.input_file:
        passages = load_passages_from_file(args.input_file)
    else:
        passages = load_passages_from_huggingface(args.topk_passages)

    if not passages:
        print("ERROR: No passages loaded.")
        sys.exit(1)

    if args.use_es:
        build_elasticsearch_index(passages, args.es_host)
    else:
        # Save passages
        passages_path = os.path.join(args.output_dir, "passages.jsonl")
        print(f"Saving passages to {passages_path}...")
        with open(passages_path, "w", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # Build and save BM25 index
        bm25 = build_bm25_index(passages)
        bm25_path = os.path.join(args.output_dir, "bm25_index.pkl")
        print(f"Saving BM25 index to {bm25_path}...")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

        # Save metadata
        avg_len = sum(len(p["text"].split()) for p in passages) / len(passages)
        metadata = {
            "num_passages": len(passages),
            "avg_passage_length": avg_len,
            "source": args.input_file or "wikimedia/wikipedia-20231101.en",
        }
        metadata_path = os.path.join(args.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nIndex built successfully!")
        print(f"  Passages: {len(passages)}")
        print(f"  Avg passage length: {avg_len:.1f} words")
        print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
