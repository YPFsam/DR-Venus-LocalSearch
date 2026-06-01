#!/usr/bin/env python3
"""Measure local BM25 latency and lexical answer coverage on RL prompts.

The answer-hit metric is a conservative diagnostic, not an RL quality score:
it checks whether the normalized ground-truth answer appears in any full
top-k passage returned by the local server.
"""

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests


def normalize_text(value: Any) -> str:
    return re.sub(r"\W+", "", str(value).casefold(), flags=re.UNICODE)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def read_rows(train_file: Path, sample_size: int) -> list[dict]:
    rows = pq.read_table(train_file).to_pylist()
    if sample_size > 0:
        rows = rows[:sample_size]
    return rows


def prompt_text(row: dict) -> str:
    for message in row.get("prompt") or []:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def ground_truth(row: dict) -> str:
    return str((row.get("reward_model") or {}).get("ground_truth", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="data/redsearcher_rl_1k.parquet")
    parser.add_argument("--server_url", default="http://localhost:8890")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    rows = read_rows(Path(args.train_file), args.sample_size)
    if not rows:
        raise SystemExit("ERROR: no training rows found")

    session = requests.Session()
    session.trust_env = False
    server_url = args.server_url.rstrip("/")
    health = session.get(f"{server_url}/health", timeout=10)
    health.raise_for_status()

    latencies_ms = []
    hit_count = 0
    evaluated_count = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        queries = [prompt_text(row) for row in batch]
        t0 = time.perf_counter()
        response = session.post(
            f"{server_url}/search",
            json={"queries": queries, "topk": args.topk},
            timeout=120,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.raise_for_status()
        result_groups = response.json().get("results", [])
        if len(result_groups) != len(batch):
            raise RuntimeError("local search server returned an unexpected result count")

        latencies_ms.extend([elapsed_ms / len(batch)] * len(batch))
        for row, result_group in zip(batch, result_groups):
            expected = normalize_text(ground_truth(row))
            if not expected:
                continue
            evaluated_count += 1
            passages = result_group.get("passages", [])
            if any(expected in normalize_text(passage.get("text", "")) for passage in passages):
                hit_count += 1

    summary = {
        "server": health.json(),
        "sample_size": len(rows),
        "topk": args.topk,
        "lexical_answer_hit_at_k": hit_count / evaluated_count if evaluated_count else 0.0,
        "evaluated_answers": evaluated_count,
        "latency_ms_per_query_mean": statistics.mean(latencies_ms),
        "latency_ms_per_query_p50": percentile(latencies_ms, 0.50),
        "latency_ms_per_query_p95": percentile(latencies_ms, 0.95),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
