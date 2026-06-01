#!/usr/bin/env python3
"""Download and convert the official REDSearcher 1K RL dataset to veRL parquet."""

import argparse
import os
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K/"
    "resolve/main/data/train-00000-of-00001.parquet"
)


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Using existing download: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(destination)
    print(f"Downloaded: {destination}")


def convert_dataset(source_path: Path, output_path: Path, expected_rows: int) -> None:
    source_table = pq.read_table(source_path)
    required_columns = {"problem", "answer", "difficulty"}
    missing_columns = required_columns - set(source_table.column_names)
    if missing_columns:
        raise ValueError(f"Missing source columns: {sorted(missing_columns)}")
    if source_table.num_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} REDSearcher rows, found {source_table.num_rows}. "
            "Check whether the upstream dataset changed."
        )

    converted_rows = []
    for index, row in enumerate(source_table.to_pylist()):
        problem = str(row["problem"]).strip()
        answer = str(row["answer"]).strip()
        difficulty = str(row["difficulty"]).strip()
        if not problem or not answer:
            raise ValueError(f"Row {index} has an empty problem or answer")
        converted_rows.append(
            {
                "data_source": "redsearcher",
                "prompt": [{"role": "user", "content": problem}],
                "ability": "search & question answering",
                "reward_model": {"ground_truth": answer, "style": "unknown"},
                "extra_info": {
                    "index": f"redsearcher_{index:04d}",
                    "split": "train",
                    "difficulty": difficulty,
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted_rows), output_path)
    print(f"Wrote {len(converted_rows)} rows: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--download_path", default="data/downloads/redsearcher_rl_1k_source.parquet")
    parser.add_argument("--output", default="data/redsearcher_rl_1k.parquet")
    parser.add_argument("--expected_rows", type=int, default=1000)
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    download_path = Path(os.path.expanduser(args.download_path))
    output_path = Path(os.path.expanduser(args.output))
    download_file(args.source_url, download_path, args.force_download)
    convert_dataset(download_path, output_path, args.expected_rows)


if __name__ == "__main__":
    main()

