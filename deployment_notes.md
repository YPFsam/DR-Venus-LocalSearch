# DR-Venus-LocalSearch 部署踩坑指南

> 基于 2026-06-02 在 AutoDL（A800-80GB，单卡）上的实际部署经验整理。
> 适用于在其他 GPU 云服务商（如恒源云、矩池云、揽睿星舟等）上复现。
> **单卡训练计算路径已跑通，但 checkpoint 保存失败；端到端单卡冒烟和 4 卡仍待验证。**

---

## 0. 已验证环境版本

以下版本组合已在 AutoDL A800-80GB 上完成过 1 step 的训练计算。由于 checkpoint 保存失败，
它是后续排障的基准环境，不代表端到端 smoke 已通过：

| 组件 | 版本 | 安装方式 |
|------|------|----------|
| Python | 3.12.11 | miniconda |
| PyTorch | 2.6.0+cu124 | `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` |
| CUDA (系统 nvcc) | 12.4.131 | 系统自带 |
| vLLM | 0.8.5 | `pip install vllm==0.8.5` |
| flash-attn | 2.7.4.post1 | **预编译 wheel**，见踩坑 3.2 |
| transformers | 4.57.6 | `pip install 'transformers>=4.56,<5'` |
| trl | 0.15.2 | `pip install 'trl<0.16'` |
| ray | 2.55.1 | vllm 依赖自动安装 |
| hydra-core | 1.3.2 | requirements.txt |

---

## 1. 硬件要求

| 配置 | GPU | 系统内存 | 系统盘 | 数据盘 | 适用场景 |
|------|-----|----------|--------|--------|----------|
| 最低 | 1×80GB | 64GB+ | 30GB | 50GB | 冒烟测试（MAX_MODEL_LEN=16384） |
| 推荐 | 4×80GB | 256GB+ | 30GB | 100GB+ | 正式训练（131K 上下文） |

**说明**：
- 单卡 80GB 可以跑完 4B 模型的一次 IGPO 计算（MAX_MODEL_LEN 需降到 16384）
- 正式训练的 131K 上下文需要 4 卡（序列并行 ULYSSES_SP_SIZE=4）
- 每个 checkpoint 约 18GB（actor 全量参数），数据盘需预留足够空间

---

## 2. 磁盘规划（重要）

### 踩坑：系统盘爆满

项目安装后磁盘占用：
- `.venv`（PyTorch + vLLM + 依赖）：**~11GB**
- 模型权重 `DR-Venus-4B-SFT`：**~8.3GB**
- Wikipedia 搜索索引（500K passages）：**~1.2GB**
- 训练数据：**~1MB**（很小）
- uv/pip 缓存（安装过程）：**~9.5GB + 4GB**
- 训练 checkpoint（每个 step）：**~18GB**（4B 模型全量参数）

**总计需要 35-40GB**（含 1 个 checkpoint），AutoDL 默认系统盘只有 30GB，必须用数据盘。

### 踩坑：checkpoint 默认写到系统盘

`vendor_train.sh smoke` 的 `SMOKE_OUTPUT` 默认值是 `./output_smoke`（项目目录，在系统盘）。
即使 `.env` 设置了 `OUTPUT` 到数据盘，smoke 模式会用 `SMOKE_OUTPUT` 覆盖。

**解决**：修改 `vendor_train.sh` 中 `SMOKE_OUTPUT` 默认路径，或在 `.env` 中设置 `SMOKE_OUTPUT`。

### 解决方案

所有大文件放数据盘（`/root/autodl-tmp`）：

```bash
# venv 放数据盘，软链接到项目目录
DATA_DISK=/root/autodl-tmp
mkdir -p $DATA_DISK/venvs $DATA_DISK/models $DATA_DISK/output $DATA_DISK/eval_log
cd /root/DR-Venus-LocalSearch/RL
ln -sf $DATA_DISK/venvs/dr-venus .venv

# uv/pip 缓存放数据盘（安装完可删除节省空间）
export UV_CACHE_DIR=$DATA_DISK/uv_cache
export PIP_CACHE_DIR=$DATA_DISK/pip_cache

# 模型放数据盘
# .env 里设置 MODEL_PATH=$DATA_DISK/models/DR-Venus-4B-SFT

# 输出/checkpoint 放数据盘
# .env 里设置 OUTPUT=$DATA_DISK/output/dr-venus-smoke
```

### 其他服务商注意

- **恒源云**：数据盘通常是 `/root/autodl-fs/`，不是 `autodl-tmp`
- **矩池云**：数据盘通常是 `/root/maoxian/` 或 `/data/`
- **揽睿星舟**：通常是 `/root/workspace/` 或挂载路径不同
- **通用做法**：`df -h` 看哪个分区空间大，就用哪个

---

## 3. CUDA 版本与 flash-attn（核心踩坑）

### 踩坑 3.1：系统 CUDA 和 PyTorch CUDA 版本不匹配

```
系统 nvcc --version → CUDA 12.4
PyTorch 编译版本 → CUDA 13.0 (torch 2.11.0+cu130)
```

**影响**：`flash-attn` 需要用系统 `nvcc` 编译，版本不匹配 → 编译失败。

**原因**：`install_vendor_env.sh` 用 `uv pip install vllm --torch-backend=auto`，uv 会自动选最新的 vLLM，而最新 vLLM 可能绑定了比你系统更新的 CUDA 版本。

### 踩坑 3.2：flash-attn 从源码编译太慢（教训）

flash-attn 2.6.3 有 **5248 个 .cu 文件**，在 A800 上编译需要 **25-30 小时**。
flash-attn 2.5.9 有 780 个文件，仍需数小时。

**重要教训**：**不要从源码编译 flash-attn！先用预编译 wheel 确定版本，再装匹配的 torch/CUDA。**

### 踩坑 3.3：pip 拒绝重命名的 wheel 文件

```bash
# 下载后改名为 /tmp/flash_attn.whl
pip install /tmp/flash_attn.whl  # ERROR: not a valid wheel filename
```

**解决**：wheel 文件名必须保留原始格式，pip 用文件名解析元数据。

### 正确的安装流程（推荐）

**核心思路**：先确定 flash-attn 预编译 wheel，再安装匹配的 torch 和 vLLM。

```bash
# 1. 确定你的环境：Python 版本、CUDA 版本
python3 --version  # e.g. 3.12
nvcc --version     # e.g. CUDA 12.4

# 2. 去 GitHub 找对应的预编译 wheel
# https://github.com/Dao-AILab/flash-attention/releases
# 文件名格式：flash_attn-{ver}+cu{cuda}torch{torch_ver}cxx11abi{TRUE/FALSE}-cp{py}-cp{py}-linux_x86_64.whl
# 例如 CUDA 12 + torch 2.6 + Python 3.12：
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# 3. 下载（保留原始文件名！）
cd /tmp && curl -fLO -C - "$WHEEL_URL"
# 或
# wget "$WHEEL_URL" -O "/tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# 4. 安装匹配的 PyTorch（根据 wheel 里的 torch 版本）
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 5. 安装匹配的 vLLM
pip install vllm==0.8.5

# 6. 安装 flash-attn（--no-deps 防止 pip 自动升级 torch！）
pip install /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl --no-deps

# 7. 验证
python -c "import torch; print(torch.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "from flash_attn.bert_padding import index_first_axis; print('OK')"
```

**注意**：`--no-deps` 很关键！不加的话 pip 会自动升级 torch 到 flash-attn 要求的版本，导致 CUDA 版本不匹配。

---

## 4. Python 包版本问题

### 踩坑 4.1：trl 版本不兼容

```
ImportError: cannot import name 'AutoModelForCausalLMWithValueHead' from 'trl'
```

**原因**：新版 trl（0.16+）移除了 `AutoModelForCausalLMWithValueHead`。

**解决**：
```bash
pip install 'trl<0.16'  # 0.15.2 有这个类
```

### 踩坑 4.2：transformers 5.x 移除 API

```
ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
```

**原因**：transformers 5.x 有 breaking changes。

**解决**：
```bash
pip install 'transformers>=4.56,<5'
```

### 踩坑 4.3：pyext 不兼容 Python 3.12

```
AttributeError: module 'inspect' has no attribute 'getargspec'
```

**原因**：`inspect.getargspec` 在 Python 3.12 中被移除。

**解决**：跳过 `pyext`，它只用于 prime_code reward，F1 reward 不需要。

### 踩坑 4.4：soundfile 缺失

```
ModuleNotFoundError: No module named 'soundfile'
```

**原因**：qwen-agent 的依赖。

**解决**：
```bash
pip install soundfile
```

---

## 5. 训练配置与运行踩坑

### 踩坑 5.1：torch.compile 循环导入

```
ImportError: cannot import name 'scaled_mm_configs' from 'torch._inductor.kernel.mm_common'
```

**原因**：PyTorch 2.6+ 的 `torch._inductor.kernel` 模块有循环导入 bug。

**解决**：在 `train_igpo.sh` 开头添加：
```bash
export TORCHDYNAMO_DISABLE=1
```
并在 Hydra 参数中添加：
```
actor_rollout_ref.actor.use_torch_compile=false
actor_rollout_ref.ref.use_torch_compile=false
```

### 踩坑 5.2：vLLM V1 引擎未设置

```
ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False.
```

**原因**：veRL fork 的 `vllm_async_server.py` 直接 import 了 `vllm.v1.engine.async_llm.AsyncLLM`，但 vLLM 0.8.x 需要显式启用 V1。

**解决**：在 `train_igpo.sh` 开头添加：
```bash
export VLLM_USE_V1=1
```

### 踩坑 5.3：async rollout _postprocess tensor 大小不匹配（代码 bug）

```
RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 598 but got size 564 for tensor number 2 in the list.
```

**原因**：`verl/experimental/agent_loop/agent_loop.py` 的 `_postprocess` 方法中，`response_ids` 做了 padding 到统一长度，但 `prompt_ids`、`input_ids`、`attention_mask`、`position_ids` 没有做对应的 prompt 侧 padding。在 async rollout 模式下，不同样本的多轮推理产生不同长度的序列，导致 `torch.cat` 失败。

**修复**：在 `_postprocess` 中添加 prompt 侧 left-padding（见 git diff）。

### 踩坑 5.4：wandb API key 格式错误

```
AuthenticationError: WANDB_API_KEY invalid: API key must have 40+ characters, has 36.
```

**解决**：提供正确的 40+ 字符 wandb key，或在 `.env` 中改用 `LOGGER_BACKENDS="['console','tensorboard']"` 跳过 wandb。

### 踩坑 5.5：保存 checkpoint 时 Ray Worker 崩溃（根因待确认）

```
ray.exceptions.ActorUnavailableError: The actor is unavailable: RpcError: Socket closed
```

**已知事实**：异常发生在 `actor_rollout_wg.save_checkpoint()` 内部，不能当作普通退出清理。
当前日志不足以区分主机内存 OOM、SIGSEGV 或 vLLM V1 后台进程断连。

**影响**：本次 step 的计算已完成，但 checkpoint 未确认完整保存。必须使用
`bash scripts/vendor_train.sh check-checkpoint` 验证 tracker、dataloader 状态和 FSDP shards。
正式训练前需要重新执行 smoke，并回传 `logs/diagnostics_*.txt` 排查根因。

---

## 6. 单卡 vs 4 卡参数配置

### 单卡冒烟测试参数

| 参数 | 单卡值 | 4卡值 | 说明 |
|------|--------|-------|------|
| NUM_GPUS | 1 | 4 | GPU 数量 |
| TP_SIZE | 1 | 2 | vLLM 张量并行 |
| ULYSSES_SP_SIZE | 1 | 4 | 序列并行度 |
| MAX_MODEL_LEN | 16384 | 131000 | vLLM 上下文窗口 |
| MAX_PROMPT_LEN | 8192 | 120000 | prompt 最大长度 |
| MAX_RESPONSE_LEN | 4096 | 8192 | 每轮生成上限 |
| ASYNC_PROMPT_PAD | 512 | 1024 | async prompt padding |
| TRAIN_BATCH_SIZE | 2 | 8 | 训练 batch |
| PPO_MINI_BATCH_SIZE | 4 | 64 | PPO mini batch |
| ROLLOUT_N | 2 | 8 | 每样本 rollout 数 |
| ASYNC_NUM_WORKERS | 1 | 4 | 并发 agent worker |
| GPU_MEMORY_UTIL | 0.75 | 0.80 | vLLM 显存占比 |

### 单卡冒烟测试性能数据

| 阶段 | 耗时 |
|------|------|
| Rollout（4样本×~5轮） | ~750s |
| IGPO KV-Cache IG 计算 | ~4.6s |
| F1 Reward 计算 | ~0.04s |
| PPO 更新（actor+ref） | ~24.5s |
| Checkpoint 保存 | ~5.8s 后 actor 异常，未完成 |
| **总计 1 step** | **~872s（14.5分钟）** |
| **GPU 显存峰值** | **~56GB / 80GB** |

---

## 7. 网络和代理

### AutoDL 学术加速

```bash
source /etc/network_turbo  # AutoDL 学术资源加速
```

### HuggingFace 镜像（国内环境）

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 手动下载数据集（代理不兼容时）

```bash
curl -L -o data/downloads/redsearcher_rl_1k_source.parquet \
    "https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K/resolve/main/data/train-00000-of-00001.parquet"
```

---

## 8. 完整安装流程（其他服务商直接用）

```bash
# === 变量配置（根据服务商修改 DATA_DISK）===
DATA_DISK=/root/autodl-tmp  # AutoDL
# DATA_DISK=/root/autodl-fs  # 恒源云
# DATA_DISK=/data            # 矩池云

# 1. Clone 项目
cd /root && git clone https://github.com/YPFsam/DR-Venus-LocalSearch.git

# 2. 创建目录和软链接
mkdir -p $DATA_DISK/venvs $DATA_DISK/models $DATA_DISK/output $DATA_DISK/eval_log
cd /root/DR-Venus-LocalSearch/RL
ln -sf $DATA_DISK/venvs/dr-venus .venv

# 3. 创建 venv
python3 -m venv $DATA_DISK/venvs/dr-venus
source .venv/bin/activate

# 4. 安装 PyTorch（匹配系统 CUDA 12.4）
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 5. 安装 vLLM
pip install vllm==0.8.5

# 6. 安装 flash-attn 预编译 wheel（--no-deps 防止 torch 被升级！）
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
WHEEL_PATH="/tmp/${WHEEL_URL##*/}"
curl -fL -C - -o "$WHEEL_PATH" "$WHEEL_URL"
pip install "$WHEEL_PATH" --no-deps
rm "$WHEEL_PATH"

# 7. 安装项目依赖
pip install 'trl<0.16' 'transformers>=4.56,<5' soundfile
pip install -r requirements.txt  # 当前 requirements 已将 pyext 标记为可选

# 8. 验证安装
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"
python -c "import vllm; print(f'vllm={vllm.__version__}')"
python -c "import flash_attn; print(f'flash_attn={flash_attn.__version__}')"
python -c "from trl import AutoModelForCausalLMWithValueHead; print('trl OK')"

# 9. 配置 .env
cp .env.example .env
# 编辑 MODEL_PATH=$DATA_DISK/models/DR-Venus-4B-SFT
# 编辑 OUTPUT=$DATA_DISK/output/dr-venus-smoke
# 编辑 EVAL_LOG_PATH=$DATA_DISK/eval_log

# 10. 准备数据和模型
export HF_ENDPOINT=https://hf-mirror.com  # 国内需要
bash scripts/bootstrap_vendor.sh  # 下载模型和构建搜索索引

# 11. 冒烟测试（单卡）
# 确保 .env 中 NUM_GPUS=1, TP_SIZE=1, ULYSSES_SP_SIZE=1
bash scripts/smoke_5step.sh
# 或直接：
# bash scripts/vendor_train.sh smoke

# 12. 正式训练（4卡）
# 确保 .env 中 NUM_GPUS=4, TP_SIZE=2, ULYSSES_SP_SIZE=4
bash scripts/vendor_train.sh launch
```

---

## 9. 版本兼容矩阵

| 组件 | 已验证版本 | 兼容范围 | 备注 |
|------|-----------|----------|------|
| Python | 3.12.11 | 3.10-3.12 | 3.12 可用但 pyext 不兼容 |
| PyTorch | 2.6.0+cu124 | 2.6+ | 必须与系统 CUDA 匹配 |
| CUDA (系统) | 12.4.131 | 12.x | nvcc 版本必须与 PyTorch cu??? 匹配 |
| vLLM | 0.8.5 | 0.8.x | 需要 VLLM_USE_V1=1 |
| flash-attn | 2.7.4.post1 | 2.5+ | **必须用预编译 wheel** |
| transformers | 4.57.6 | 4.56-4.x | **不能用 5.x** |
| trl | 0.15.2 | <0.16 | 新版移除了 ValueHead |
| ray | 2.55.1 | 2.40+ | vllm 依赖 |
| hydra-core | 1.3.2 | 1.3.x | requirements.txt |

---

## 10. Checklist（新环境快速检查）

```bash
# GPU
nvidia-smi  # 确认 GPU 可用

# CUDA 版本
nvcc --version  # 记下版本，需要和 PyTorch 匹配

# 磁盘
df -h  # 确认数据盘有 50GB+ 可用空间

# Python
python3 --version  # 3.10+

# 网络（国内环境）
source /etc/network_turbo  # AutoDL 学术加速
export HF_ENDPOINT=https://hf-mirror.com  # HF 镜像

# 验证安装
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
python -c "import vllm; print(vllm.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "from trl import AutoModelForCausalLMWithValueHead; print('trl OK')"
python -c "from flash_attn.bert_padding import index_first_axis; print('flash_attn OK')"
```
