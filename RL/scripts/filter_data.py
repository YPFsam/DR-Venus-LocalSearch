#!/usr/bin/env python3
"""Filter and split training data for reduced-scale experiments.

This script:
1. Reads the original train.parquet (80K samples)
2. Optionally filters by data_source
3. Samples a subset (e.g., 40K) with balanced data_source distribution
4. Writes the filtered subset to a new parquet file

Usage:
    # Filter to 40K samples (balanced across data sources)
    python filter_data.py --input data/train.parquet --output data/train_40k.parquet --num_samples 40000

    # Filter to 40K with specific data sources
    python filter_data.py --input data/train.parquet --output data/train_40k.parquet \
        --num_samples 40000 --data_sources nq,2wiki,tq

    # Just shuffle and keep all
    python filter_data.py --input data/train.parquet --output data/train_shuffled.parquet
"""

import argparse
import json
import os
import sys

import pandas as pd


def _estimate_token_count(obj) -> int:
    """Rough token count estimate: ~4 chars per token for English text."""
    if isinstance(obj, str):
        return len(obj) // 4
    elif isinstance(obj, list):
        # List of message dicts — serialize to JSON and estimate
        total = 0
        for item in obj:
            if isinstance(item, dict):
                for v in item.values():
                    total += len(str(v))
            else:
                total += len(str(item))
        return total // 4
    else:
        return len(str(obj)) // 4


def main():
    parser = argparse.ArgumentParser(description="Filter training data")
    parser.add_argument("--input", required=True, help="Input parquet file path")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to keep (default: keep all)")
    parser.add_argument("--data_sources", type=str, default=None,
                        help="Comma-separated data sources to include (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Filter out samples with estimated token count > this value")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    if "data_source" in df.columns:
        print(f"  Data sources distribution:")
        for src, count in df["data_source"].value_counts().items():
            print(f"    {src}: {count}")

    # Filter by data_source
    if args.data_sources:
        sources = [s.strip() for s in args.data_sources.split(",")]
        df = df[df["data_source"].isin(sources)]
        print(f"\n  After data_source filter ({sources}): {len(df)} samples")

    # Filter by max token estimate
    if args.max_tokens and "prompt" in df.columns:
        df["_est_tokens"] = df["prompt"].apply(_estimate_token_count)
        df = df[df["_est_tokens"] <= args.max_tokens]
        df = df.drop(columns=["_est_tokens"])
        print(f"  After max_tokens filter (<= {args.max_tokens}): {len(df)} samples")

    # Sample
    if args.num_samples and args.num_samples < len(df):
        if "data_source" in df.columns:
            # Stratified sampling to maintain balance
            grouped = df.groupby("data_source", group_keys=False)
            n_per_group = {}
            total_sources = len(grouped)
            base = args.num_samples // total_sources
            remainder = args.num_samples % total_sources

            for i, (name, _) in enumerate(grouped):
                n_per_group[name] = base + (1 if i < remainder else 0)

            samples = []
            for name, group in grouped:
                n = min(n_per_group[name], len(group))
                samples.append(group.sample(n=n, random_state=args.seed))

            df = pd.concat(samples).sample(frac=1, random_state=args.seed)
        else:
            df = df.sample(n=args.num_samples, random_state=args.seed)
        print(f"\n  Sampled {len(df)} examples")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\n  Saved to {args.output} ({len(df)} samples)")

    if "data_source" in df.columns:
        print(f"  Final data source distribution:")
        for src, count in df["data_source"].value_counts().items():
            print(f"    {src}: {count}")


if __name__ == "__main__":
    main()
