# AutoDL 单卡 A100 验证与 4 卡训练时间估算

本文档用于在 AutoDL 的单张 A100 80 GB 实例上验证 DR-Venus RL 本地检索版本，并获得
4 卡正式训练的粗略时间区间。

单卡 profile **不是**正式训练，也不能直接精确预测 4 卡速度。它的价值是：

1. 在租用 4 卡机器前验证 CUDA、vLLM、`flash-attn`、模型下载、数据准备和本地检索。
2. 验证完整 RL step 是否能执行，包括 rollout、IGPO、actor 更新和 checkpoint 保存。
3. 获得单卡分项耗时，为 4 卡预算提供数量级估算。

## 1. 为什么单卡估算存在误差

正式 4 卡配置与单卡 profile 并不等价：

| 项目 | 正式 4 卡训练 | 单卡 `estimate` profile |
|---|---:|---:|
| GPU | 4 x A100/A800 80 GB | 1 x A100 80 GB |
| TP | 2 | 1 |
| Ulysses SP | 4 | 1 |
| 每 step 查询数 | 8 | 1 |
| 每个查询 rollout 数 | 8 | 2 |
| 每 step rollout 总数 | 64 | 2 |
| 最大工具轮数 | 50 | 20 |
| 最大上下文 | 131K | 32K |
| 异步 worker | 4 | 1 |

单卡无法复现多卡的 NCCL 通信、TP/SP 并行、64 条 rollout 的并发调度和 131K 长上下文。
长上下文 prefill 也不是严格线性增长。因此：

- `sanity` 档位只能验证是否跑通，ETA 误差可能超过 2 倍。
- 默认 `estimate` 档位适合预算预估，通常应接受约 `-50% ~ +100%` 的误差。
- `stress` 档位更接近正式负载，但仍可能有约 `-40% ~ +80%` 的误差，而且更容易 OOM。

最终可靠 ETA 必须在 4 卡机器跑完前 3 到 5 个正式 step 后，根据
TensorBoard 的 `perf/time_per_step` 重新计算。

## 2. AutoDL 实例要求

建议选择：

| 项目 | 要求 |
|---|---|
| GPU | 1 x A100 80 GB |
| 系统盘和数据盘 | 合计至少 150 GiB 可用空间 |
| 主机内存 | 建议至少 64 GiB |
| 网络 | 能访问 GitHub 和 Hugging Face |

AutoDL 上的 A100 可能是 40 GB 或 80 GB。40 GB 实例可以运行 `sanity` 档位验证流程，
但默认 `estimate` 和 `stress` 更容易 OOM，且对正式 4 x 80 GB 训练的参考价值较低。

拉取代码：

```bash
git clone https://github.com/YPFsam/DR-Venus-LocalSearch.git
cd DR-Venus-LocalSearch/RL
```

安装环境：

```bash
bash scripts/install_vendor_env.sh
source .venv/bin/activate
```

## 3. 推荐执行顺序

先跑较快的 `sanity` 档位：

```bash
PROFILE_MODE=sanity bash scripts/profile_single_gpu_autodl.sh all
```

它会自动下载官方 SFT checkpoint、准备 REDSearcher RL 1K、构建 100K Tantivy 索引、
启动本地检索，并执行一个缩小后的完整 RL step。

`sanity` 跑通后，再执行默认的时间估算档位：

```bash
PROFILE_MODE=estimate bash scripts/profile_single_gpu_autodl.sh run
```

如果默认 `estimate` 能稳定运行，且你愿意为更接近正式负载的估算多花一些单卡时间，
可尝试：

```bash
PROFILE_MODE=stress bash scripts/profile_single_gpu_autodl.sh run
```

`stress` 使用 64K 上下文和 50 轮上限。如果显存不足，保留 `estimate` 结果即可，不要为
单卡 profile 反复调参消耗预算。

## 4. 脚本会生成什么

每次执行会创建独立目录：

```text
output_autodl_profile/<mode>-<timestamp>/
eval_log_autodl_profile/<mode>-<timestamp>/
```

重点文件：

| 文件 | 内容 |
|---|---|
| `output_autodl_profile/.../training.log` | 单卡 RL step 完整日志 |
| `output_autodl_profile/.../profile_config.txt` | 本次 profile 参数 |
| `output_autodl_profile/.../four_gpu_eta.txt` | 自动生成的 4 卡耗时区间 |
| `eval_log_autodl_profile/.../metric_step_1.json` | TensorBoard 同源的原始指标 |

profile 只有在 `scripts/check_checkpoint.py` 确认 tracker、dataloader 状态和 FSDP shards
完整后才会输出 ETA。训练 step 完成但 checkpoint 保存失败，不算通过。

输出中的 `ideal_linear_lower_bound_per_step` 只是理想线性下界，不应拿来直接报价。
预算应采用 `conservative_range_per_step` 和 `conservative_range_for_20_steps`。

## 5. 常用命令

仅准备模型、数据和 100K 本地索引：

```bash
bash scripts/profile_single_gpu_autodl.sh prepare
```

只执行 profile：

```bash
PROFILE_MODE=estimate bash scripts/profile_single_gpu_autodl.sh run
```

指定使用第 0 张 GPU：

```bash
PROFILE_CUDA_VISIBLE_DEVICES=0 PROFILE_MODE=estimate \
  bash scripts/profile_single_gpu_autodl.sh run
```

如 AutoDL 实例已经存在其他规格的本地索引，需要明确重建 100K profile 索引：

```bash
PROFILE_FORCE_REBUILD_INDEX=true PROFILE_MODE=sanity \
  bash scripts/profile_single_gpu_autodl.sh all
```

## 6. 重要限制

单卡 profile 的 checkpoint 仅用于验证，不要复制到正式 4 卡机器续训：

```text
output_autodl_profile/
```

正式训练仍应从官方 `inclusionAI/DR-Venus-4B-SFT` 起步，并使用正式目录：

```text
output/
```

如果单卡 profile 发生 OOM，优先使用 `sanity` 或默认 `estimate` 档位。不要因此直接断定
4 卡正式配置不可运行：正式训练使用 TP=2 和 SP=4，内存分布与单卡不同。
