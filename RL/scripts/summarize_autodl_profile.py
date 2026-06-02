#!/usr/bin/env python3
"""Summarize a single-GPU AutoDL profile and print a conservative four-GPU ETA."""

import argparse
import json
import math
from pathlib import Path


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def latest_metric(metrics_dir: Path) -> tuple[Path, dict]:
    metric_files = sorted(
        metrics_dir.glob("metric_step_*.json"),
        key=lambda path: int(path.stem.removeprefix("metric_step_")),
    )
    if not metric_files:
        raise SystemExit(f"ERROR: no metric_step_*.json files found under {metrics_dir}")
    path = metric_files[-1]
    return path, json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--mode", choices=["sanity", "estimate", "stress"], required=True)
    parser.add_argument("--profile_rollouts", type=int, required=True)
    parser.add_argument("--profile_max_turns", type=int, required=True)
    parser.add_argument("--formal_rollouts", type=int, default=64)
    parser.add_argument("--formal_max_turns", type=int, default=50)
    parser.add_argument("--formal_steps", type=int, default=20)
    args = parser.parse_args()

    metric_path, metrics = latest_metric(Path(args.metrics_dir))
    profile_step_seconds = float(metrics["perf/time_per_step"])
    profile_tokens = int(metrics.get("perf/total_num_tokens", 0))
    profile_throughput = float(metrics.get("perf/throughput", 0.0))

    workload_ratio = args.formal_rollouts / args.profile_rollouts
    ideal_four_gpu_seconds = profile_step_seconds * workload_ratio / 4
    turn_ratio = args.formal_max_turns / args.profile_max_turns

    if args.mode == "sanity":
        planning_low = ideal_four_gpu_seconds
        planning_high = ideal_four_gpu_seconds * max(4.0, turn_ratio * 2.0)
        confidence = "very low"
    elif args.mode == "estimate":
        planning_low = ideal_four_gpu_seconds * 0.9
        planning_high = ideal_four_gpu_seconds * max(2.0, turn_ratio * 1.5)
        confidence = "low"
    else:
        planning_low = ideal_four_gpu_seconds * 0.8
        planning_high = ideal_four_gpu_seconds * max(1.6, turn_ratio * 1.2)
        confidence = "medium-low"

    low_total = planning_low * args.formal_steps
    high_total = planning_high * args.formal_steps
    stage_metrics = {
        key.removeprefix("timing_s/"): value
        for key, value in sorted(metrics.items())
        if key.startswith("timing_s/")
    }

    print("Single-GPU AutoDL profile summary")
    print(f"  metric_file: {metric_path}")
    print(f"  mode: {args.mode}")
    print(f"  profile_step_time: {format_duration(profile_step_seconds)}")
    print(f"  profile_rollouts_per_step: {args.profile_rollouts}")
    print(f"  profile_max_turns: {args.profile_max_turns}")
    print(f"  profile_total_tokens: {profile_tokens}")
    print(f"  profile_throughput_tokens_per_second_per_gpu: {profile_throughput:.2f}")
    print()
    print("Four-GPU planning estimate")
    print(f"  formal_rollouts_per_step: {args.formal_rollouts}")
    print(f"  formal_max_turns: {args.formal_max_turns}")
    print(f"  ideal_linear_lower_bound_per_step: {format_duration(ideal_four_gpu_seconds)}")
    print(
        f"  conservative_range_per_step: "
        f"{format_duration(planning_low)} .. {format_duration(planning_high)}"
    )
    print(
        f"  conservative_range_for_{args.formal_steps}_steps: "
        f"{format_duration(low_total)} .. {format_duration(high_total)}"
    )
    print(f"  confidence: {confidence}")
    print()
    print("Important: this is a capacity-planning estimate, not a benchmark-equivalent projection.")
    print("A single GPU cannot reproduce TP=2, SP=4, NCCL, 64-rollout concurrency, or 131K context behavior.")
    print("Use the first 3-5 formal four-GPU steps to replace this estimate with perf/time_per_step.")
    if stage_metrics:
        print()
        print("Measured timing stages")
        for key, value in stage_metrics.items():
            if isinstance(value, int | float) and math.isfinite(float(value)):
                print(f"  {key}: {format_duration(float(value))}")


if __name__ == "__main__":
    main()
