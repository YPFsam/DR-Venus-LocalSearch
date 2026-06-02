#!/usr/bin/env python3
"""Validate the latest veRL checkpoint before treating it as resumable."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SHARD_RE = re.compile(r"^(model|optim|extra_state)_world_size_(\d+)_rank_(\d+)\.pt$")


def fail(message: str) -> None:
    print(f"[checkpoint-check] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_step(output_dir: Path, requested_step: int | None) -> int:
    if requested_step is not None:
        return requested_step
    tracker = output_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        fail(f"missing tracker file: {tracker}")
    try:
        return int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        fail(f"invalid tracker file: {tracker}: {exc}")


def validate_shards(actor_dir: Path) -> int:
    groups: dict[str, dict[int, set[int]]] = {}
    for path in actor_dir.iterdir():
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        kind, world_size_text, rank_text = match.groups()
        if path.stat().st_size <= 0:
            fail(f"empty checkpoint shard: {path}")
        groups.setdefault(kind, {}).setdefault(int(world_size_text), set()).add(int(rank_text))

    world_sizes = set()
    for kind in ("model", "optim", "extra_state"):
        versions = groups.get(kind, {})
        if len(versions) != 1:
            fail(f"expected exactly one {kind} shard set in {actor_dir}, found {sorted(versions)}")
        world_size, ranks = next(iter(versions.items()))
        expected_ranks = set(range(world_size))
        if ranks != expected_ranks:
            fail(f"incomplete {kind} shards: expected ranks {sorted(expected_ranks)}, found {sorted(ranks)}")
        world_sizes.add(world_size)

    if len(world_sizes) != 1:
        fail(f"checkpoint shard world sizes do not match: {sorted(world_sizes)}")
    return world_sizes.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--step", type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    step = read_step(output_dir, args.step)
    step_dir = output_dir / f"global_step_{step}"
    actor_dir = step_dir / "actor"

    if not actor_dir.is_dir():
        fail(f"actor checkpoint directory does not exist: {actor_dir}")
    if not (step_dir / "data.pt").is_file():
        fail(f"dataloader state is missing: {step_dir / 'data.pt'}")
    if not (actor_dir / "huggingface" / "config.json").is_file():
        fail(f"Hugging Face config is missing: {actor_dir / 'huggingface' / 'config.json'}")

    world_size = validate_shards(actor_dir)
    print(
        f"[checkpoint-check] PASS: step={step} world_size={world_size} "
        f"path={step_dir}"
    )


if __name__ == "__main__":
    main()
