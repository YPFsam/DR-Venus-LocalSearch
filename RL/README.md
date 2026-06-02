# DR-Venus RL: Four-GPU Local Retrieval Edition

This directory contains a resource-constrained DR-Venus RL setup for one
machine with four 80 GB A100 or A800 GPUs. It starts from the official
[`inclusionAI/DR-Venus-4B-SFT`](https://huggingface.co/inclusionAI/DR-Venus-4B-SFT)
checkpoint and uses a local Wikipedia retrieval service instead of Serper and
Jina.

For a GPU-provider handoff, use the complete Chinese runbook:
[`VENDOR_TRAINING_GUIDE.zh-CN.md`](VENDOR_TRAINING_GUIDE.zh-CN.md). It documents
the one-command readiness workflow and detached formal training launcher.

Before renting four GPUs, an AutoDL single-A100 profiling workflow is documented
in [`AUTODL_SINGLE_GPU_PROFILE.zh-CN.md`](AUTODL_SINGLE_GPU_PROFILE.zh-CN.md).
It validates one reduced RL step and prints a conservative four-GPU ETA range.

The official RL checkpoint card states that RL uses
[`Zchu/REDSearcher_RL_1K`](https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K):
1,000 curated query-answer pairs. The tracked `data/train.parquet` file contains
80,000 generic QA examples and is retained for reference only. It is not the
default training file in this local setup.

## 1. Local Defaults

| Setting | Published repository script default | Four-GPU local default |
|---|---:|---:|
| Starting checkpoint | SFT checkpoint | `inclusionAI/DR-Venus-4B-SFT` |
| RL training data | repository `data/train.parquet` | converted REDSearcher RL 1K |
| Retrieval | Serper + Jina | local Wikipedia Tantivy HTTP service |
| Outcome reward | LLM judge | deterministic F1 |
| GPUs | 16 x A100 in the original run | 4 x 80 GB A100/A800 |
| Context window | 261K | 131K |
| Rollout tensor parallel size | 4 | 2 |
| Ulysses sequence parallel size | 8 | 4 |
| Training batch size | 16 | 8 |
| PPO mini-batch size | 128 | 64 |
| Agent loop workers | 8 | 4 |
| Maximum turns | 200 | 50 |

This is an offline retrieval adaptation, not an exact reproduction of the
official online-search run. Its metrics should be reported separately.

## 2. Machine Requirements

- Linux or WSL Ubuntu with four visible 80 GB A100/A800 GPUs.
- Python 3.12 for the verified one-click installer.
- A CUDA driver compatible with the PyTorch and vLLM builds you install.
- Enough host RAM and disk for the chosen Wikipedia passage count.
- Network access while downloading the model, the 1K RL data, and Wikipedia.
  After preparation, local-search training does not need external APIs.

The default corpus is 500,000 Wikipedia passages. Tantivy is the default
backend: it uses an inverted index, so a query does not score every passage in
Python. Use 100,000 passages only for a faster smoke preparation, and increase
beyond 500,000 only after measuring RAM usage, retrieval latency, and answer
coverage.

The local corpus is not an internet-scale corpus and does not guarantee that
REDSearcher questions are covered. Measure retrieval quality before paying for
a formal RL run.

## 3. Quick Vendor Bootstrap

After the provider has installed the NVIDIA driver, it can use
the bundled environment installer and readiness workflow:

```bash
cd DR-Venus/RL
bash scripts/install_vendor_env.sh
source .venv/bin/activate
bash scripts/vendor_train.sh ready
```

`vendor_train.sh ready` downloads the official SFT checkpoint when missing,
creates `.env` when missing, downloads and converts the official REDSearcher RL
1K data, builds the Tantivy index when missing, benchmarks retrieval, runs the
four-GPU preflight, executes a one-step smoke training, and validates the smoke
checkpoint shards. It does not rebuild a matching existing index. The default
bootstrap index contains 500,000 passages.

After the smoke run succeeds, launch detached formal training:

```bash
bash scripts/vendor_train.sh launch
```

The remaining sections document each step separately for troubleshooting and
custom deployments.

## 4. Install Dependencies

Create an isolated environment:

```bash
cd DR-Venus/RL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

The bundled installer pins the stack verified by the single-GPU A800 smoke run
and installs a prebuilt `flash-attn` wheel. This avoids an expensive source
build on the provider machine:

```bash
bash scripts/install_vendor_env.sh
```

For a custom stack, set `VLLM_SPEC` and `FLASH_ATTN_WHEEL_URL` together. The
wheel must match the Python, CUDA and PyTorch ABI. The runtime preflight checks
the packages needed by this project.

## 5. Download the Official SFT Checkpoint

Install the Hugging Face CLI and download the official SFT checkpoint into a
local directory. Using a local path avoids simultaneous worker downloads during
RL startup.

```bash
pip install --upgrade huggingface_hub
hf download inclusionAI/DR-Venus-4B-SFT \
  --local-dir /data/models/DR-Venus-4B-SFT
```

The Hugging Face CLI documentation for `--local-dir` is available
[here](https://huggingface.co/docs/huggingface_hub/main/guides/cli).
For an offline training machine, run the download elsewhere and copy the
resulting model directory to the same local path.

## 6. Configure the Run

Copy the template and edit at least `MODEL_PATH`:

```bash
cp .env.example .env
sed -n '1,160p' .env
```

The important defaults are:

```bash
MODEL_PATH=/data/models/DR-Venus-4B-SFT
NUM_GPUS=4
TRAIN_FILE=data/redsearcher_rl_1k.parquet
USE_LOCAL_SEARCH=true
TRAIN_REWARD_TYPE=f1
GPU_MEMORY_UTILIZATION=0.80
SAVE_FREQ=5
RESUME_MODE=auto
MAX_ACTOR_CKPT_TO_KEEP=2
MAX_CRITIC_CKPT_TO_KEEP=2
MAX_LOCAL_CKPT_TO_KEEP=2
```

Local retrieval plus F1 reward does not require Serper, Jina, a page
summarizer, or an LLM judge API.

## 7. Prepare RL Data and Local Retrieval

The convenience script downloads the official REDSearcher 1K source parquet,
converts it to the veRL schema, streams Wikipedia from Hugging Face, and builds
a local Tantivy index:

```bash
bash scripts/prepare_local_rl.sh
```

The generated files are:

```text
data/redsearcher_rl_1k.parquet
data/local_search_index/passages.jsonl
data/local_search_index/tantivy_index/
data/local_search_index/metadata.json
```

The default index size is 500,000 passages. To build a smaller smoke index:

```bash
INDEX_PASSAGES=100000 bash scripts/prepare_local_rl.sh
```

Existing index files are reused only when their backend and passage count match
the requested configuration. To rebuild them intentionally:

```bash
FORCE_REBUILD_INDEX=true INDEX_PASSAGES=500000 bash scripts/prepare_local_rl.sh
```

Use 100,000 passages for a fast smoke preparation and 500,000 passages for the
first paid formal run. A 1,000,000-passage index is reasonable only after
measuring retrieval quality and latency; fitting on disk does not prove that
the corpus covers the training questions. Index builds use a staging directory
and replace the live index only after every artifact is complete.

Reference measurements on a 23 GiB WSL host showed why the Tantivy default
matters. With 100,000 passages, the original `rank_bm25` fallback used about
1.45 GiB server RSS and 2.42 seconds per query. Tantivy used about 100 MiB RSS
and 19 ms per query. With 500,000 passages, `rank_bm25` used about 6.4 GiB RSS
and 11.25 seconds per query, while Tantivy used about 240 MiB RSS and 72 ms per
query in a serial 100-query test. In a 1,000-query test with four concurrent
single-query requests, 500,000-passage Tantivy averaged 53 ms per query and
74 queries/s. A 1,000,000-passage Tantivy index averaged 92 ms and 43 queries/s,
but its conservative lexical `answer hit@10` did not improve. Hardware differs,
so benchmark the target machine before a formal run.

SQLite FTS5 and `rank_bm25` remain available as diagnostic fallback backends:

```bash
LOCAL_SEARCH_BACKEND=sqlite_fts5 FORCE_REBUILD_INDEX=true \
  bash scripts/prepare_local_rl.sh
```

If the training machine cannot access Hugging Face, copy a local JSONL corpus
to the machine and build the index from it. Each line must contain `title` and
`text`; `id` is optional.

```bash
python3 scripts/prepare_redsearcher_data.py
python3 scripts/build_bm25_index.py \
  --input_file /data/corpora/wiki_passages.jsonl \
  --output_dir data/local_search_index
```

The REDSearcher conversion can also be performed on a connected machine and the
generated `data/redsearcher_rl_1k.parquet` copied to the training machine.

## 8. Start Retrieval and Validate the Machine

Start the retrieval service in its own terminal:

```bash
cd DR-Venus/RL
source .venv/bin/activate
bash scripts/start_local_search.sh
```

The server binds to `0.0.0.0:8890`, so a WSL service is reachable from the
Windows host through `localhost`. Check it from another terminal:

```bash
NO_PROXY=localhost,127.0.0.1 curl -fsS http://localhost:8890/health
```

Run the complete preflight without allocating RL workers:

```bash
python3 scripts/evaluate_local_retrieval.py --sample_size 100
PRECHECK_ONLY=true bash train_igpo.sh
```

The preflight verifies:

- required Python modules;
- local SFT checkpoint and `config.json`;
- converted 1K training parquet and validation parquet schema;
- visible GPU count and TP/SP divisibility;
- GRPO batch alignment;
- local retrieval `/health`.

`evaluate_local_retrieval.py` separately reports query latency and a
conservative lexical `answer hit@k` diagnostic. Use `--sample_size 1000` before
a formal run. It is a retrieval sanity check, not an RL evaluation metric.

To run the retrieval service and the diagnostic in one temporary session:

```bash
INDEX_LABEL=500k SAMPLE_SIZE=1000 CONCURRENCY=4 \
  bash scripts/run_local_retrieval_eval.sh
```

## 9. Smoke Run and Formal Training

Run one short training step first:

```bash
OUTPUT=./output_smoke TOTAL_TRAINING_STEPS=1 MAX_TURNS=5 GPU_MEMORY_UTILIZATION=0.75 \
  bash train_igpo.sh
```

After the smoke run succeeds, start the configured one-epoch run:

```bash
bash train_igpo.sh
```

Training logs and checkpoints are written to `OUTPUT`, which defaults to
`./output`. Validation traces are written to `EVAL_LOG_PATH`, which defaults to
`./eval_log`.

The training command also writes `OUTPUT/training.log`. Local retrieval writes
a rotating `logs/local_search.log` with batch latency and result counts. Full
query text is omitted by default; set `LOCAL_SEARCH_LOG_QUERIES=true` only when
diagnosing retrieval behavior. Rollout traces are stored under
`OUTPUT/rollout_traces/`, limited to eight samples per step by default.

## 10. Resume and Monitor Training

The default `RESUME_MODE=auto` resumes from the checkpoint named by
`OUTPUT/latest_checkpointed_iteration.txt` when the same `OUTPUT` directory is
preserved. A leftover `global_step_N/` directory without that tracker is not a
valid resume point. Each checkpoint contains actor shards, dataloader state, and IGPO
warmup state. The default `SAVE_FREQ=5` writes one checkpoint every five
training steps. `MAX_LOCAL_CKPT_TO_KEEP=2` retains the newest two complete
checkpoints across process restarts to control disk usage. Actor and critic
checkpoint managers use the same retention limit during a running process.

To resume automatically:

```bash
bash train_igpo.sh
```

To validate the newest formal checkpoint before resuming:

```bash
bash scripts/vendor_train.sh check-checkpoint
```

To select an explicit checkpoint:

```bash
RESUME_MODE=resume_path \
RESUME_FROM_PATH=./output/global_step_5 \
  bash train_igpo.sh
```

Do not resume from a smoke run into a formal run unless that is intentional.
Use a separate `OUTPUT` directory for smoke tests.

The default logger is local TensorBoard:

```bash
tensorboard --logdir tensorboard_log --host 0.0.0.0 --port 6006
```

When the GPU machine is remote, use an SSH tunnel and open
`http://localhost:6006` locally:

```bash
ssh -L 6006:localhost:6006 user@gpu-host
```

For web-based real-time monitoring, enable W&B in `.env`:

```bash
LOGGER_BACKENDS="['console','tensorboard','wandb']"
WANDB_API_KEY=replace-with-a-dedicated-api-key
WANDB_ENTITY=your-team-or-user
WANDB_RUN_ID=dr-venus-4b-local-search-4gpu
WANDB_RESUME=allow
```

The provider does not need your W&B password. Give it a dedicated API key or
place the key in `.env` yourself, and ensure `.env` is not committed or copied
back with checkpoints. Keeping the same `WANDB_RUN_ID` allows a resumed
training process to continue writing to the same W&B run.

## 11. Optional Online Mode

The original online tool path is still available:

```bash
USE_LOCAL_SEARCH=false bash train_igpo.sh
```

Online mode additionally needs a project `.env` with `SERPER_KEY_ID`,
`JINA_API_KEYS`, `API_KEY`, `API_BASE`, and `SUMMARY_MODEL_NAME`. If
`TRAIN_REWARD_TYPE=llm`, also configure `JUDGE_MODEL_NAME` and optionally
`JUDGE_API_BASE` and `JUDGE_API_KEY`.

## 12. Troubleshooting

**Preflight reports a missing model directory**

Set `MODEL_PATH` in `.env` to the downloaded `DR-Venus-4B-SFT` directory. A
remote Hugging Face repo ID is intentionally rejected by default. To permit
runtime downloads explicitly, set `ALLOW_REMOTE_MODEL_PATH=true`.

**Local search health check fails**

Run `bash scripts/start_local_search.sh` in another terminal. If index files are
missing, run `bash scripts/prepare_local_rl.sh`.

**CUDA out of memory**

Keep `GPU_MEMORY_UTILIZATION=0.75` for the first run. If needed, reduce
`MAX_MODEL_LEN`, `MAX_TURNS`, or `TRAIN_BATCH_SIZE`. Preserve the TP/SP
divisibility checks enforced by preflight.

**Local retrieval is too slow**

Check `/health` and confirm that `backend` is `tantivy`, then run
`scripts/run_local_retrieval_eval.sh` with `CONCURRENCY=4`. The `rank_bm25`
fallback scans every passage in Python and is not suitable for a paid
long-horizon run.

**Collect a troubleshooting bundle**

Run the diagnostics script and send the generated text file together with the
relevant TensorBoard or W&B run link. It reports package versions, GPU status,
disk and RAM usage, cgroup OOM counters, Ray worker errors, local-search health,
and recent logs without printing `.env`:

```bash
bash scripts/collect_diagnostics.sh
```
