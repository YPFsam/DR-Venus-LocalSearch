# DR-Venus 4 卡本地检索 RL：GPU 供应商执行说明书

本文档用于在一台 Linux GPU 服务器上，从官方
[`inclusionAI/DR-Venus-4B-SFT`](https://huggingface.co/inclusionAI/DR-Venus-4B-SFT)
checkpoint 开始训练本仓库的 RL 阶段。默认方案使用 4 张 80 GB A100/A800、
REDSearcher RL 1K 数据和本地 Tantivy 检索，不依赖 Serper、Jina 或外部 LLM
judge。

如果需要在租用 4 卡机器前先用 AutoDL 单张 A100 做预算验证，请参考
[`AUTODL_SINGLE_GPU_PROFILE.zh-CN.md`](AUTODL_SINGLE_GPU_PROFILE.zh-CN.md)。

## 1. 供应商需要准备的机器

建议使用原生 Ubuntu Linux，不建议把正式付费训练放在 WSL 中。

最低要求：

| 项目 | 要求 |
|---|---|
| GPU | 4 x 80 GB NVIDIA A100 或 A800 |
| 系统 | Ubuntu 22.04 或兼容 Linux |
| Python | 3.12（默认一键脚本使用已验证的 3.12 wheel） |
| 主机内存 | 至少 128 GiB，建议 256 GiB |
| 可用磁盘 | 至少 300 GiB，建议预留 500 GiB |
| 网络 | 准备阶段能够访问 GitHub 和 Hugging Face |
| 基础软件 | `git`、`curl`、`tmux`、编译工具、NVIDIA 驱动、CUDA toolkit |

默认脚本安装已验证的 `flash-attn` 预编译 wheel，避免在供应商机器上耗时编译。开始前请确认以下命令均可执行：

```bash
nvidia-smi
python3 --version
```

安装常用系统工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl tmux build-essential python3 python3-venv python3-pip ninja-build
```

CUDA toolkit 的版本应与服务器驱动及脚本安装的 PyTorch wheel 兼容。不要在未知环境中
随意替换已验证可用的 CUDA toolkit。

## 2. 拉取代码并安装 Python 环境

拉取本地检索版本：

```bash
git clone https://github.com/YPFsam/DR-Venus-LocalSearch.git
cd DR-Venus-LocalSearch/RL
```

运行环境安装脚本：

```bash
bash scripts/install_vendor_env.sh
source .venv/bin/activate
```

脚本会：

1. 检查 Linux、Python 和 `nvidia-smi`。
2. 安装 `uv`。
3. 创建项目本地 `.venv`。
4. 安装已验证的 `vllm==0.8.5` GPU stack。
5. 下载并安装与 Python 3.12、PyTorch 2.6 匹配的 `flash-attn==2.7.4.post1`
   预编译 wheel。
6. 安装项目依赖和本地 `igpo` 包。
7. 检查 PyTorch 是否能够访问 CUDA。

如供应商已有验证过的 vLLM 环境，可跳过该脚本，直接激活已有环境并执行：

```bash
pip install -r requirements.txt
pip install -e .
```

如果需要有意识地更换 vLLM 版本或 `flash-attn` wheel，可显式传入：

```bash
VLLM_SPEC='vllm==<verified-version>' \
FLASH_ATTN_WHEEL_URL='<matching-wheel-url>' \
  bash scripts/install_vendor_env.sh
```

## 3. 一键准备资源并做 Smoke Test

运行：

```bash
bash scripts/vendor_train.sh ready
```

该命令会自动完成：

1. 下载官方 `inclusionAI/DR-Venus-4B-SFT` 到 `RL/data/models/DR-Venus-4B-SFT/`。
2. 生成本地 `.env`，自动写入 SFT checkpoint 路径。
3. 下载并转换官方 `Zchu/REDSearcher_RL_1K` 数据，得到
   `data/redsearcher_rl_1k.parquet`。
4. 流式下载英文 Wikipedia，构建默认 500,000 passages 的 Tantivy 倒排索引。
5. 后台启动本地检索服务，监听 `0.0.0.0:8890`。
6. 对完整 1,000 条 RL 问题执行本地检索质量和性能测试。
7. 检查 Python 依赖、SFT checkpoint、训练数据、4 张 GPU、TP/SP 参数和检索服务。
8. 使用独立目录 `output_smoke/` 执行 1 个训练 step、最多 5 轮工具调用的 smoke test。
9. 校验 smoke checkpoint 的 tracker、dataloader 状态和每个 FSDP shard；保存失败时 `ready`
   会直接失败，不允许进入正式训练。

首次准备需要下载 checkpoint 和 Wikipedia，请预留时间。索引采用 staging 目录构建：
只有全部文件构建成功后才会替换正式索引；中途中断不会留下可被训练误用的半成品。

如 Hugging Face 限流，可在执行前配置只读 token：

```bash
export HF_TOKEN='<huggingface-read-token>'
bash scripts/vendor_train.sh ready
```

## 4. 启动正式训练

`ready` 成功后，启动或续训正式任务：

```bash
bash scripts/vendor_train.sh launch
```

训练会放入名为 `drvenus-train` 的后台 `tmux` session。SSH 断开不会停止训练。

进入训练终端：

```bash
tmux attach -t drvenus-train
```

离开但不停止训练：按 `Ctrl+B`，再按 `D`。

查看当前状态：

```bash
bash scripts/vendor_train.sh status
```

持续查看训练日志：

```bash
bash scripts/vendor_train.sh logs
```

## 5. 断点续训

默认配置：

```bash
RESUME_MODE=auto
SAVE_FREQ=5
MAX_LOCAL_CKPT_TO_KEEP=2
MAX_ACTOR_CKPT_TO_KEEP=2
MAX_CRITIC_CKPT_TO_KEEP=2
```

训练每 5 个 step 保存 checkpoint，只保留最近两个完整 checkpoint，避免磁盘无限增长。
只要保留同一个 `RL/output/` 目录，重新执行：

```bash
bash scripts/vendor_train.sh launch
```

脚本只会从 `latest_checkpointed_iteration.txt` 标记的完整 `output/global_step_N/` 自动续训。
不要将残留目录或 `output_smoke/` 用于正式训练。可手动检查：

```bash
bash scripts/vendor_train.sh check-checkpoint
```

## 6. 实时查看训练曲线

### 6.1 TensorBoard：默认启用，不需要账号

在 GPU 服务器执行：

```bash
bash scripts/vendor_train.sh tensorboard-start
```

TensorBoard 后台监听 `0.0.0.0:6006`。建议不要直接把该端口暴露到公网，而是在自己的电脑上建立 SSH 隧道：

```bash
ssh -N -L 6006:127.0.0.1:6006 <user>@<gpu-server>
```

然后在自己的浏览器打开：

```text
http://localhost:6006
```

### 6.2 W&B：可选的网页远程监控

默认启用 `console` 和 `tensorboard`，不在仓库内存放任何 W&B 凭据。如需使用 W&B，
在 GPU 机器的私有 `RL/.env` 中设置 `WANDB_API_KEY`，并将
`LOGGER_BACKENDS` 改为 `['console','tensorboard','wandb']`。

不要把 `.env` 提交到 Git。

## 7. 日志、索引和 checkpoint 位置

| 路径 | 内容 |
|---|---|
| `output/training.log` | 正式训练主日志 |
| `output/global_step_N/` | 可恢复 checkpoint |
| `output/rollout_traces/` | 部分 rollout 调试轨迹 |
| `output_smoke/` | 独立 smoke test 输出 |
| `logs/local_search.log` | 本地检索耗时和返回条数 |
| `logs/local_search_console.log` | 本地检索服务启动日志 |
| `tensorboard_log/` | TensorBoard 曲线 |
| `data/local_search_index/` | Wikipedia passages 和 Tantivy 索引 |

默认 500K passages 是正式训练的起跑配置。不要因为磁盘足够就直接扩大到 1M：
本地实测中 1M 的检索延迟更高，但保守 `answer hit@10` 没有改善。

## 8. 常用维护命令

```bash
# 检查状态
bash scripts/vendor_train.sh status

# 重新运行本地检索测试
bash scripts/vendor_train.sh evaluate

# 重新执行 GPU 和配置预检
bash scripts/vendor_train.sh preflight

# 校验最新的正式 checkpoint 是否可用于续训
bash scripts/vendor_train.sh check-checkpoint

# 停止脚本管理的检索服务
bash scripts/vendor_train.sh stop-search

# 紧急停止后台训练 tmux session
bash scripts/vendor_train.sh stop-training

# 收集排障信息
bash scripts/vendor_train.sh diagnostics
```

故障发生时，请将以下内容回传：

1. `bash scripts/vendor_train.sh diagnostics` 生成的 `logs/diagnostics_*.txt`。
2. `output/training.log`。
3. `logs/local_search_console.log`。
4. 执行失败的完整命令。

诊断脚本不会输出 `.env`，因此不会主动泄露 API Key。

## 9. 需要重建本地索引时

默认复用已经完成的 500K Tantivy 索引。如需明确重建：

```bash
FORCE_REBUILD_INDEX=true INDEX_PASSAGES=500000 \
  bash scripts/vendor_train.sh prepare
```

只做更快的 100K smoke 索引：

```bash
FORCE_REBUILD_INDEX=true INDEX_PASSAGES=100000 \
  bash scripts/vendor_train.sh prepare
```

正式训练前应恢复为 500K，并重新运行：

```bash
bash scripts/vendor_train.sh ready
```

## 10. 最短执行清单

供应商正常情况下只需要依次执行：

```bash
git clone https://github.com/YPFsam/DR-Venus-LocalSearch.git
cd DR-Venus-LocalSearch/RL
bash scripts/install_vendor_env.sh
source .venv/bin/activate
bash scripts/vendor_train.sh ready
bash scripts/vendor_train.sh launch
bash scripts/vendor_train.sh tensorboard-start
```

之后使用 `bash scripts/vendor_train.sh status` 查看状态。任务异常退出后，再次执行
`bash scripts/vendor_train.sh launch` 即可自动续训。
