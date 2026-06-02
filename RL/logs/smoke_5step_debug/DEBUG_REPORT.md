# Smoke Test 5-Step Debug Report

> 时间：2026-06-02 18:14 ~ 18:46（卡死于 18:24）
> 环境：AutoDL 单卡 A800-80GB，torch 2.6.0+cu124，vLLM 0.8.5，flash-attn 2.7.4.post1
> 配置：NUM_GPUS=1, TP_SIZE=1, ULYSSES_SP_SIZE=1, MAX_MODEL_LEN=16384, TRAIN_BATCH_SIZE=2, ROLLOUT_N=2

## 现象

5 步冒烟测试中，step 1-3 正常完成，step 4 开始后 **卡死（deadlock）**，20+ 分钟无任何输出。

```
Step 1: 111.8s ✅ (gen=65s, num_turns/mean=5.75)
Step 2: 113.2s ✅ (gen=95s, num_turns/mean=7.25, ig_active_samples=1 ← IG 开始激活)
Step 3: 259.5s ✅ (num_turns/mean=7.25, response_length/clip_ratio=0.25)
Step 4: 卡死 ❌ (无输出，GPU util=0%，进程活着但无响应)
```

## 死锁时的系统状态

- GPU: 33082 MiB / 81920 MiB 占用，**利用率 0%**
- TaskRunner (PID 4376): 155 个线程活着，但 `ray.get()` 挂起
- AsyncvLLMServer (PID 5434): core worker 日志还在更新（心跳），但无实际请求处理
- AgentLoopWorker (PID 6406): worker_stdout_logs 为空
- vLLM server 日志无新输出

## 关键日志文件

```
logs/smoke_5step_debug/
├── training.log              # 训练主日志（含 step 1-3 的完整 metrics）
├── gpu_monitor.csv           # GPU 显存监控（每 5s）
├── rollout_traces/           # step 1-3 的 rollout trace（JSON）
│   ├── step_1/0000-0003.json
│   ├── step_2/0000-0003.json
│   └── step_3/0000-0003.json  ← step 4 卡死前最后的 trace
├── worker-*_4376.out         # TaskRunner (主控进程)
├── worker-*_4879.out         # WorkerDict (FSDP actor/ref model)
├── worker-*_5434.out         # AsyncvLLMServer
├── worker-*_5434.err         # AsyncvLLMServer stderr
├── worker-*_6406.out         # AgentLoopWorker (rollout worker)
├── worker-*_7000.out         # 另一个 AgentLoopWorker
└── worker-*_7000.err
```

## 相关代码路径

1. **Async rollout 入口**：`verl/experimental/agent_loop/agent_loop.py`
   - `AgentLoopManager.generate_sequences()` → 调用 AgentLoopWorker
   - `AgentLoopWorker.generate_sequences()` → `_run_agent_loop()` → `_postprocess()`

2. **vLLM 通信**：`verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   - 通过 ZMQ IPC 与 vLLM V1 EngineCore 通信
   - `init_engine()` → `AsyncLLM.from_vllm_config()`

3. **训练主循环**：`verl/trainer/ppo/ray_trainer.py`
   - `fit()` → 循环调用 `generate_sequences()` → `compute_reward()` → `update_actor()`

## 根因分析

### 最可能的原因：ZMQ 管道通信死锁

vLLM V1 使用后台线程模式（`VLLM_USE_V1=1`），veRL 通过 ZMQ IPC 管道与 vLLM 通信。

日志中的关键警告：
```
WARNING: Detected VLLM_USE_V1=1 with Engine in background thread.
Usage should be considered experimental.
```

死锁发生时的调用链推测：
```
TaskRunner.fit() 
  → async_rollout_manager.generate_sequences() 
    → ray.get(worker.generate_sequences()) 
      → AgentLoopWorker._run_agent_loop()
        → vLLM generate (via ZMQ) ← 卡在这里
```

Step 3 的 `response_length/clip_ratio=0.25` 说明部分样本已经逼近 `MAX_RESPONSE_LEN=4096` 的截断边界。
随着训练进行，模型行为变化导致生成了更长或更复杂的序列，可能触发了 vLLM V1 后台线程的某个边界条件 bug。

### 其他可能性（日志不足以排除）

1. **主机内存 OOM**：`RAY_memory_monitor_refresh_ms=0` 关闭了 Ray 内存监控，
   单卡 FSDP NO_SHARD 模式下 CPU 内存可能不足。但 `free -h` 显示有 1TB RAM，不太可能。
2. **NCCL hang**：单卡不应该有 NCCL 通信，但初始化了 NCCL（日志有 `NCCL version 2.21.5+cuda12.4`）。
3. **Ray actor 调度问题**：AgentLoopWorker 的 `ray.get()` 永久阻塞。

### 不太可能的原因

- GPU OOM：显存只用了 33GB/80GB
- 代码逻辑 bug：step 1-3 都正常，同样的代码路径
- 数据问题：训练数据是固定的 parquet 文件

## 建议的排查方向

1. **加 ZMQ 超时**：在 `vllm_async_server.py` 的 ZMQ 通信中加超时，避免永久阻塞
2. **加 debug 日志**：在 `AgentLoopWorker._run_agent_loop()` 的每个 turn 开始/结束时打印日志
3. **降低 MAX_RESPONSE_LEN**：从 4096 降到 2048，避免截断边界问题
4. **尝试 sync 模式**：设 `USE_ASYNC_ROLLOUT=false` 看是否稳定
5. **尝试 VLLM_USE_V1=0**：用 vLLM V0 引擎（但 veRL fork 的代码直接 import 了 V1 的 AsyncLLM）
6. **加 SAVE_FREQ=1**：每步保存 checkpoint，卡死后可续训

## Step 3 Rollout Trace 分析（卡死前最后一步）

Sample 0（8 轮）：模型搜索了 "iridoid glycoside lepidopteran" 等关键词，
但本地 Wikipedia 检索结果不相关（返回了 "Structural testing"、"Butterfly ray" 等），
模型不断重试不同 query，最终达到 max_turns 限制被强制终止。

这说明：
- tool call 识别正常（✅ 工具协议修复生效）
- 本地搜索在执行（✅ tool_calls 有 search 结果）
- 搜索质量不高（Wikipedia 500K passages 对专业生物学术问题覆盖不足）
- 模型在 step 3 后行为变化可能触发了 vLLM 的边界条件
