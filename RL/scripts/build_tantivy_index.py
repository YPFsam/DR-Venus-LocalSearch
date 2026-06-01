#!/usr/bin/env python3
"""Build a Tantivy inverted index from a passage JSONL file."""

import argparse
import json
import shutil
import time
from pathlib import Path


def build_tantivy_index(passages_file: Path, output_dir: Path) -> int:
    """Build a stored Tantivy index for search and local document visits."""
    try:
        import tantivy
    except ImportError as exc:
        raise SystemExit("ERROR: 'tantivy' not installed. Run: pip install tantivy") from exc

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")
    schema_builder.add_text_field("title", stored=True)
    schema_builder.add_text_field("text", stored=True)
    schema = schema_builder.build()

    print(f"Building Tantivy index: {output_dir}")
    started = time.time()
    index = tantivy.Index(schema, path=str(output_dir))
    writer = index.writer(heap_size=256_000_000)
    count = 0
    with passages_file.open("r", encoding="utf-8") as passages:
        for line in passages:
            passage = json.loads(line)
            writer.add_document(
                tantivy.Document.from_dict(
                    {
                        "id": str(passage["id"]),
                        "title": str(passage.get("title", "")),
                        "text": str(passage["text"]),
                    },
                    schema,
                )
            )
            count += 1
            if count % 100_000 == 0:
                print(f"  Indexed {count} passages...")

    writer.commit()
    writer.wait_merging_threads()
    print(f"Tantivy index built in {time.time() - started:.1f}s ({count} passages)")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages_file", default="data/local_search_index/passages.jsonl")
    parser.add_argument("--output_dir", default="data/local_search_index/tantivy_index")
    args = parser.parse_args()
    build_tantivy_index(Path(args.passages_file), Path(args.output_dir))


if __name__ == "__main__":
    main()
