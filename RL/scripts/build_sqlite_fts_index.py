#!/usr/bin/env python3
"""Build a SQLite FTS5 inverted index from a passage JSONL file."""

import argparse
import json
import sqlite3
import time
from pathlib import Path


def build_sqlite_fts_index(passages_file: Path, output_file: Path, batch_size: int = 10_000) -> int:
    """Build a contentless FTS5 index plus a document store."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.unlink(missing_ok=True)

    print(f"Building SQLite FTS5 index: {output_file}")
    started = time.time()
    connection = sqlite3.connect(output_file)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute(
            """
            CREATE TABLE passages (
                rowid INTEGER PRIMARY KEY,
                id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE passages_fts USING fts5(
                title,
                text,
                content='',
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        connection.execute("CREATE VIRTUAL TABLE passages_vocab USING fts5vocab(passages_fts, 'row')")

        pending = []
        count = 0
        with passages_file.open("r", encoding="utf-8") as passages:
            for line in passages:
                passage = json.loads(line)
                count += 1
                pending.append((count, str(passage["id"]), str(passage.get("title", "")), str(passage["text"])))
                if len(pending) >= batch_size:
                    _insert_batch(connection, pending)
                    pending.clear()
                    if count % 100_000 == 0:
                        print(f"  Indexed {count} passages...")
            if pending:
                _insert_batch(connection, pending)

        connection.execute("INSERT INTO passages_fts(passages_fts) VALUES ('optimize')")
        connection.commit()
    finally:
        connection.close()

    print(f"SQLite FTS5 index built in {time.time() - started:.1f}s ({count} passages)")
    return count


def _insert_batch(connection: sqlite3.Connection, rows: list[tuple[int, str, str, str]]) -> None:
    connection.executemany("INSERT INTO passages(rowid, id, title, text) VALUES (?, ?, ?, ?)", rows)
    connection.executemany(
        "INSERT INTO passages_fts(rowid, title, text) VALUES (?, ?, ?)",
        ((rowid, title, text) for rowid, _, title, text in rows),
    )
    connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages_file", default="data/local_search_index/passages.jsonl")
    parser.add_argument("--output_file", default="data/local_search_index/sqlite_fts.db")
    parser.add_argument("--batch_size", type=int, default=10_000)
    args = parser.parse_args()
    build_sqlite_fts_index(Path(args.passages_file), Path(args.output_file), args.batch_size)


if __name__ == "__main__":
    main()
