#!/usr/bin/env python3
"""Validate the local four-GPU RL environment before allocating training workers."""

import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path


REQUIRED_MODULES = [
    "datasets",
    "flask",
    "huggingface_hub",
    "numpy",
    "pandas",
    "pyarrow",
    "qwen_agent",
    "rank_bm25",
    "ray",
    "requests",
    "tantivy",
    "torch",
    "transformers",
    "vllm",
]


def fail(message: str) -> None:
    print(f"[preflight] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def validate_modules() -> None:
    missing = [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        fail(f"Missing Python modules: {', '.join(missing)}. Install the documented dependencies first.")


def validate_tool_protocol() -> None:
    from scrl.llm_agent.dr_agent_loop import _Flag, _parse_single_response
    from tool_server.tool_prompt import SUMMARY_PROMPT, SYSTEM_PROMPT

    if "<tool_call>" not in SYSTEM_PROMPT or "<tool_callDemand>" in SYSTEM_PROMPT:
        fail("SYSTEM_PROMPT must instruct the same <tool_call> protocol parsed by the agent loop.")
    if "<think>" not in SUMMARY_PROMPT or "<thinkDemand>" in SUMMARY_PROMPT:
        fail("SUMMARY_PROMPT must use the official <think> protocol.")

    flag, _, payload = _parse_single_response(
        '<think>look up evidence</think><tool_call>{"name":"search","arguments":{"query":["test"]}}</tool_call>'
    )
    if flag != _Flag.CALL or payload.get("name") != "search":
        fail("Async agent loop cannot parse the documented <tool_call> protocol.")
    print("[preflight] Tool protocol: <tool_call> parser and prompt agree")


def validate_model_path(model_path: str, allow_remote_model: bool) -> None:
    if not model_path or model_path.startswith("/path/to/"):
        fail("MODEL_PATH is not configured. Point it to the downloaded DR-Venus-4B-SFT directory.")

    path = Path(model_path).expanduser()
    if path.is_dir():
        if not (path / "config.json").is_file():
            fail(f"MODEL_PATH exists but does not contain config.json: {path}")
        print(f"[preflight] Model checkpoint: {path}")
        return

    if allow_remote_model and "/" in model_path:
        print(
            "[preflight] WARNING: MODEL_PATH is not local. Transformers may download it at runtime: "
            f"{model_path}"
        )
        return

    fail(
        f"MODEL_PATH does not exist locally: {model_path}. Download inclusionAI/DR-Venus-4B-SFT "
        "or explicitly set ALLOW_REMOTE_MODEL_PATH=true."
    )


def validate_parquet(path_value: str, expected_rows: int, label: str) -> None:
    import pyarrow.parquet as pq

    path = Path(path_value)
    if not path.is_file():
        fail(f"{label} parquet does not exist: {path}")

    table = pq.read_table(path)
    required_columns = {"data_source", "prompt", "reward_model", "extra_info"}
    missing_columns = required_columns - set(table.column_names)
    if missing_columns:
        fail(f"{label} parquet is missing columns: {sorted(missing_columns)}")
    if expected_rows > 0 and table.num_rows != expected_rows:
        fail(f"{label} parquet expected {expected_rows} rows, found {table.num_rows}: {path}")

    first_row = table.slice(0, 1).to_pylist()
    if not first_row:
        fail(f"{label} parquet is empty: {path}")
    reward_model = first_row[0].get("reward_model") or {}
    if not reward_model.get("ground_truth"):
        fail(f"{label} parquet has no reward_model.ground_truth in its first row: {path}")
    print(f"[preflight] {label} parquet: {path} ({table.num_rows} rows)")


def visible_gpu_count() -> int:
    import torch

    return torch.cuda.device_count()


def validate_gpu_config(args: argparse.Namespace) -> None:
    visible_gpus = visible_gpu_count()
    if visible_gpus < args.num_gpus:
        fail(f"Training requests {args.num_gpus} GPUs, but PyTorch sees only {visible_gpus}.")
    if visible_gpus != args.num_gpus:
        print(
            f"[preflight] WARNING: PyTorch sees {visible_gpus} GPUs; training will allocate {args.num_gpus}. "
            "Set CUDA_VISIBLE_DEVICES if isolation is required."
        )
    if args.num_gpus % args.tp_size != 0:
        fail(f"NUM_GPUS={args.num_gpus} must be divisible by TP_SIZE={args.tp_size}.")
    if args.ulysses_sp_size > args.num_gpus or args.num_gpus % args.ulysses_sp_size != 0:
        fail(f"ULYSSES_SP_SIZE={args.ulysses_sp_size} must divide NUM_GPUS={args.num_gpus}.")
    effective_rollout_batch = args.train_batch_size * args.rollout_n
    if effective_rollout_batch % args.ppo_mini_batch_size != 0:
        fail(
            f"TRAIN_BATCH_SIZE * rollout_n ({effective_rollout_batch}) must be divisible by "
            f"PPO_MINI_BATCH_SIZE ({args.ppo_mini_batch_size})."
        )
    print(f"[preflight] GPUs: {args.num_gpus}; TP={args.tp_size}; SP={args.ulysses_sp_size}")


def validate_local_search(server_url: str) -> None:
    health_url = server_url.rstrip("/") + "/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(health_url, timeout=5) as response:
            payload = json.load(response)
    except Exception as exc:
        fail(f"Local search health check failed: {health_url}: {exc}")

    if payload.get("status") != "ok" or int(payload.get("num_passages", 0)) <= 0:
        fail(f"Local search health response is invalid: {payload}")
    print(f"[preflight] Local search: {health_url} ({payload['num_passages']} passages)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--expected_train_rows", type=int, default=1000)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--ulysses_sp_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=64)
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--use_local_search", default="true")
    parser.add_argument("--local_search_server_url", default="http://localhost:8890")
    parser.add_argument("--allow_remote_model", default="false")
    args = parser.parse_args()

    validate_modules()
    validate_tool_protocol()
    validate_model_path(args.model_path, parse_bool(args.allow_remote_model))
    validate_parquet(args.train_file, args.expected_train_rows, "Training")
    validate_parquet(args.val_file, 0, "Validation")
    validate_gpu_config(args)
    if parse_bool(args.use_local_search):
        validate_local_search(args.local_search_server_url)
    print("[preflight] All checks passed.")


if __name__ == "__main__":
    main()
