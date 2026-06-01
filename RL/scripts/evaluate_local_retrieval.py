#!/usr/bin/env python3
"""Measure local retrieval latency and lexical answer coverage on RL prompts.

The answer-hit metric is a conservative diagnostic, not an RL quality score:
it checks whether the normalized ground-truth answer appears in any full
top-k passage returned by the local server.
"""

import argparse
import concurrent.futures
import json
import math
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests


_thread_local = threading.local()


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


def session_for_thread() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _thread_local.session = session
    return session


def evaluate_batch(server_url: str, topk: int, batch: list[dict]) -> tuple[list[float], int, int]:
    queries = [prompt_text(row) for row in batch]
    t0 = time.perf_counter()
    response = session_for_thread().post(
        f"{server_url}/search",
        json={"queries": queries, "topk": topk},
        timeout=120,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.raise_for_status()
    result_groups = response.json().get("results", [])
    if len(result_groups) != len(batch):
        raise RuntimeError("local search server returned an unexpected result count")

    hit_count = 0
    evaluated_count = 0
    for row, result_group in zip(batch, result_groups):
        expected = normalize_text(ground_truth(row))
        if not expected:
            continue
        evaluated_count += 1
        passages = result_group.get("passages", [])
        if any(expected in normalize_text(passage.get("text", "")) for passage in passages):
            hit_count += 1
    return [elapsed_ms / len(batch)] * len(batch), hit_count, evaluated_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="data/redsearcher_rl_1k.parquet")
    parser.add_argument("--server_url", default="http://localhost:8890")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent HTTP batches (default: 1)")
    parser.add_argument("--progress_every", type=int, default=10,
                        help="Print progress every N request batches; 0 disables progress output")
    args = parser.parse_args()

    rows = read_rows(Path(args.train_file), args.sample_size)
    if not rows:
        raise SystemExit("ERROR: no training rows found")
    if args.batch_size <= 0:
        raise SystemExit("ERROR: --batch_size must be positive")
    if args.concurrency <= 0:
        raise SystemExit("ERROR: --concurrency must be positive")

    server_url = args.server_url.rstrip("/")
    health = session_for_thread().get(f"{server_url}/health", timeout=10)
    health.raise_for_status()

    batches = [rows[start : start + args.batch_size] for start in range(0, len(rows), args.batch_size)]
    latencies_ms = []
    hit_count = 0
    evaluated_count = 0
    processed_rows = 0
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = executor.map(
            lambda batch: evaluate_batch(server_url, args.topk, batch),
            batches,
        )
        for completed_batches, (batch_latencies, batch_hits, batch_evaluated) in enumerate(results, 1):
            latencies_ms.extend(batch_latencies)
            hit_count += batch_hits
            evaluated_count += batch_evaluated
            processed_rows += len(batch_latencies)
            if args.progress_every > 0 and (
                completed_batches % args.progress_every == 0 or processed_rows == len(rows)
            ):
                partial_hit_rate = hit_count / evaluated_count if evaluated_count else 0.0
                mean_latency_ms = statistics.mean(latencies_ms)
                print(
                    f"[retrieval-eval] processed={processed_rows}/{len(rows)} "
                    f"lexical_answer_hit_at_k={partial_hit_rate:.4f} "
                    f"mean_latency_ms={mean_latency_ms:.1f} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )

    elapsed_s = time.perf_counter() - started
    summary = {
        "server": health.json(),
        "sample_size": len(rows),
        "topk": args.topk,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "lexical_answer_hit_at_k": hit_count / evaluated_count if evaluated_count else 0.0,
        "evaluated_answers": evaluated_count,
        "latency_ms_per_query_mean": statistics.mean(latencies_ms),
        "latency_ms_per_query_p50": percentile(latencies_ms, 0.50),
        "latency_ms_per_query_p95": percentile(latencies_ms, 0.95),
        "throughput_queries_per_second": len(rows) / elapsed_s,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
