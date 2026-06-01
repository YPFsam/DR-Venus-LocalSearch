# DR-Venus RL: Four-GPU Local Retrieval Edition

This directory contains a resource-constrained DR-Venus RL setup for one
machine with four 80 GB A100 or A800 GPUs. It starts from the official
[`inclusionAI/DR-Venus-4B-SFT`](https://huggingface.co/inclusionAI/DR-Venus-4B-SFT)
checkpoint and uses a local Wikipedia BM25 service instead of Serper and Jina.

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
| Retrieval | Serper + Jina | local Wikipedia BM25 HTTP service |
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
- Python 3.10 or newer.
- A CUDA driver compatible with the PyTorch and vLLM builds you install.
- Enough host RAM and disk for the chosen Wikipedia passage count.
- Network access while downloading the model, the 1K RL data, and Wikipedia.
  After preparation, local-search training does not need external APIs.

The default BM25 corpus is 100,000 Wikipedia passages so a new machine can run
the full workflow without loading a multi-million-passage Python index. Increase
the corpus only after measuring RAM usage and retrieval latency. `rank_bm25`
scores the corpus in Python for every query, so a much larger production corpus
should use a scalable retrieval backend.

The 100,000-passage default is a deployment baseline, not an internet-scale
corpus and not a guarantee that REDSearcher questions are covered. Measure
retrieval quality before paying for a formal RL run.

## 3. Quick Vendor Bootstrap

After the provider has installed the NVIDIA driver, CUDA-compatible Python
environment, PyTorch, vLLM, and the packages in `requirements.txt`, it can run:

```bash
cd DR-Venus/RL
source .venv/bin/activate
bash scripts/bootstrap_vendor.sh
```

This command downloads the official SFT checkpoint when missing, creates `.env`
when missing, downloads and converts the official REDSearcher RL 1K data, and
builds the BM25 index when missing. It does not rebuild an existing index.

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

Install a CUDA-compatible PyTorch and vLLM stack first, then install the project
dependencies. Follow the current
[vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
when choosing wheels for the target machine.

```bash
pip install vllm
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install -e .
```

If `flash-attn` installation fails, verify that the CUDA toolkit, compiler, and
installed PyTorch ABI match before retrying. The runtime preflight checks the
Python packages needed by this project.

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

Local BM25 retrieval plus F1 reward does not require Serper, Jina, a page
summarizer, or an LLM judge API.

## 7. Prepare RL Data and Local Retrieval

The convenience script downloads the official REDSearcher 1K source parquet,
converts it to the veRL schema, streams Wikipedia from Hugging Face, and builds
a local BM25 index:

```bash
bash scripts/prepare_local_rl.sh
```

The generated files are:

```text
data/redsearcher_rl_1k.parquet
data/local_search_index/passages.jsonl
data/local_search_index/bm25_index.pkl
data/local_search_index/metadata.json
```

The default index size is 100,000 passages. To request another size:

```bash
INDEX_PASSAGES=500000 bash scripts/prepare_local_rl.sh
```

Existing index files are reused. To rebuild them intentionally:

```bash
FORCE_REBUILD_INDEX=true INDEX_PASSAGES=500000 bash scripts/prepare_local_rl.sh
```

Use 100,000 passages for the first smoke run. For a paid formal run, compare
100,000 and 500,000 passages with `evaluate_local_retrieval.py`. A 1,000,000
passage index can be tried only after measuring host RAM and latency: the
bundled `rank_bm25` backend scans the full corpus for each query, so 10x more
passages also makes each search substantially more expensive. Do not select
1,000,000 passages solely because the index fits on disk.

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

Start the BM25 service in its own terminal:

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

The default `RESUME_MODE=auto` resumes from the newest
`OUTPUT/global_step_N/` checkpoint when the same `OUTPUT` directory is
preserved. Each checkpoint contains actor shards, dataloader state, and IGPO
warmup state. The default `SAVE_FREQ=5` writes one checkpoint every five
training steps. `MAX_LOCAL_CKPT_TO_KEEP=2` retains the newest two complete
checkpoints across process restarts to control disk usage. Actor and critic
checkpoint managers use the same retention limit during a running process.

To resume automatically:

```bash
bash train_igpo.sh
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

Start with fewer passages and measure query latency. The bundled server is a
simple single-process `rank_bm25` baseline; it is intentionally easy to deploy,
but it is not a high-throughput search engine.

**Collect a troubleshooting bundle**

Run the diagnostics script and send the generated text file together with the
relevant TensorBoard or W&B run link. It reports package versions, GPU status,
disk and RAM usage, local-search health, and recent logs without printing
`.env`:

```bash
bash scripts/collect_diagnostics.sh
```
