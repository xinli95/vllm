# vLLM 101 — System Overview

> 这是一个系列教程的起点。每个 component 会有独立的 deep dive 文档。

## 目录

| # | 文档 | 主题 |
|---|------|------|
| 00 | `00_overview.md` (本文) | 系统架构全景 |
| 01 | `01_kv_cache.md` | PagedAttention & KV Cache Manager |
| 02 | `02_scheduler.md` | Scheduler & 请求调度 |
| 03 | `03_gpu_model_runner.md` | GPU Model Runner & CUDAGraph |
| 04 | `04_attention_backend.md` | Attention Backend 抽象层 |
| 05 | `05_tensor_parallelism.md` | Tensor / Pipeline Parallelism |
| 06 | `06_structured_output.md` | Structured Output & Logits Processors |
| 07 | `07_spec_decode.md` | Speculative Decoding (EAGLE / Medusa) |
| 08 | `08_engine_core_ipc.md` | Engine Core & ZMQ 进程间通信 |

---

## 一、vLLM 是什么？

vLLM 是一个面向生产环境的高性能 LLM 推理引擎。核心论文创新是
**PagedAttention**——把 KV Cache 像操作系统管理内存页一样管理，消除显存碎片，
大幅提升 GPU 利用率和吞吐量。

代码目前运行在 **V1 架构**（`vllm/v1/`），采用多进程 + ZMQ 通信设计，
相比早期单进程架构更易扩展、延迟更低。

---

## 二、整体进程架构（V1）

```
┌──────────────────────────────────────────────────────────┐
│                    Client (HTTP)                         │
└─────────────────────┬────────────────────────────────────┘
                      │ REST / SSE
┌─────────────────────▼────────────────────────────────────┐
│            API Server Process(es)   [默认 1 个]           │
│  • FastAPI + OpenAI-compatible endpoints                  │
│  • Tokenization / 多模态数据加载                           │
│  • Detokenization + streaming 回包                        │
│  Code: vllm/entrypoints/openai/api_server.py             │
└─────────────────────┬────────────────────────────────────┘
                      │ ZMQ (many-to-many)
┌─────────────────────▼────────────────────────────────────┐
│         Engine Core Process(es)   [1 个 per DP rank]      │
│  • Scheduler — 选 batch、管队列                            │
│  • KV Cache Manager — PagedAttention 块分配               │
│  • StructuredOutputManager — grammar bitmask 生成         │
│  Code: vllm/v1/engine/core.py                            │
└──────────┬──────────────────────────────┬────────────────┘
           │ IPC / NCCL                   │
     ┌─────▼──────┐                 ┌─────▼──────┐
     │ GPU Worker │  ...TP ranks... │ GPU Worker │
     │ rank=0     │                 │ rank=N-1   │
     │ ModelRunner│                 │ ModelRunner│
     │ forward()  │                 │ forward()  │
     └────────────┘                 └────────────┘
```

### 进程数量一览

| 进程类型 | 数量 | 说明 |
|----------|------|------|
| API Server | `A`（默认等于 DP size）| 处理 HTTP 请求 & 输入处理 |
| Engine Core | `DP`（默认 1）| 调度器 & KV Cache 管理 |
| GPU Worker | `DP × PP × TP`（= GPU 总数）| 每 GPU 一个，执行 forward |
| DP Coordinator | 1（仅 DP>1 时）| 跨 DP rank 负载均衡 |

**典型单机 4-GPU 部署**（`vllm serve -tp=4`）：
1 API Server + 1 Engine Core + 4 GPU Workers = **6 个进程**

---

## 三、核心组件地图

```
vllm/
├── entrypoints/            # 入口：LLM class（离线）、OpenAI API server（在线）
│   ├── llm.py
│   └── openai/api_server.py
│
├── v1/                     # V1 架构主目录
│   ├── engine/
│   │   ├── core.py         # Engine Core 主循环（调度 → 执行 → 回包）
│   │   ├── async_llm.py    # AsyncLLMEngine（在线服务用）
│   │   └── llm_engine.py   # LLMEngine（离线 LLM class 用）
│   │
│   ├── core/
│   │   ├── sched/
│   │   │   └── scheduler.py    # Scheduler：batch 组装、请求队列管理
│   │   ├── kv_cache_manager.py # PagedAttention 块分配 & prefix cache
│   │   └── block_pool.py       # 物理 KV Cache 块池
│   │
│   ├── worker/
│   │   ├── gpu_worker.py       # 单 GPU 控制进程
│   │   └── gpu_model_runner.py # input 准备、forward 执行、CUDAGraph
│   │
│   ├── attention/
│   │   ├── backend.py          # Attention Backend 抽象基类
│   │   └── backends/           # FlashAttention / FlashInfer / XFormers 实现
│   │
│   ├── structured_output/      # Structured Output 引擎侧管理
│   │   ├── __init__.py         # StructuredOutputManager
│   │   ├── backend_xgrammar.py # xgrammar 后端（默认）
│   │   ├── backend_guidance.py # guidance 后端
│   │   └── backend_outlines.py # outlines 后端
│   │
│   ├── spec_decode/            # 推测解码
│   │   ├── eagle.py            # EAGLE draft model
│   │   ├── medusa.py           # Medusa heads
│   │   └── ngram_proposer.py   # N-gram 推测
│   │
│   └── sample/
│       └── logits_processor/   # Logits Processors（min-p, logit bias 等）
│
├── model_executor/
│   ├── models/                 # LLaMA / Qwen / Mistral 等 nn.Module 实现
│   └── layers/                 # Linear, Attention, MoE, RotaryEmbedding 等
│
├── distributed/
│   ├── parallel_state.py       # TP/PP/DP 进程组管理
│   ├── communication_op.py     # all-reduce, all-gather 等
│   └── kv_transfer/            # 跨节点 KV Cache 传输（P/D 分离）
│
└── config/                     # VllmConfig：全局配置对象
```

---

## 四、一次请求的完整生命周期

```
1. 请求到达 API Server
   └─ validate → tokenize → 构造 Request 对象

2. 发给 Engine Core（ZMQ PUSH）
   └─ 进入 Scheduler.waiting_queue

3. Scheduler.schedule() [每个 step 调用]
   ├─ KVCacheManager.allocate_slots()
   │   ├─ 查 prefix cache（hash 匹配已有 KV 块）
   │   └─ 分配新的 physical KV block（按需分页）
   ├─ 构造 SchedulerOutput（哪些 req prefill / 哪些 decode）
   └─ StructuredOutputManager.grammar_bitmask()
       └─ 异步编译 grammar → 生成 token bitmask

4. SchedulerOutput → GPU Worker（IPC）
   └─ ModelRunner.execute_model()
       ├─ _update_states()：更新 persistent batch（加/删 req）
       ├─ prepare_inputs()：
       │   ├─ input_ids, positions tensor
       │   └─ AttentionMetadata（block_table, seq_lens 等）
       ├─ model.forward()
       │   ├─ Embedding → Transformer layers（TP all-reduce 在此）
       │   └─ 输出 logits [batch, vocab_size]
       └─ Sampler.forward()
           ├─ apply logits processors（temperature, top-k/p, logit bias）
           ├─ apply structured output bitmask（若有）
           └─ sample token（greedy / multinomial）

5. ModelRunnerOutput → Engine Core（IPC）
   └─ 判断 EOS / max_tokens → 更新 Request 状态

6. Engine Core → API Server（ZMQ PUSH）
   └─ Detokenizer.decode() → stream token 给客户端
```

---

## 五、Structured Output 在哪里？

Structured output（结构化输出）横跨多个组件，不是单一模块：

```
API Server
  └─ 验证请求的 response_format / guided_json 参数
      └─ 选择 backend（xgrammar / guidance / outlines / lm-format-enforcer）

Engine Core（Scheduler）
  └─ StructuredOutputManager
      ├─ grammar_init()：异步编译 grammar（用 ThreadPoolExecutor，避免阻塞调度循环）
      └─ grammar_bitmask()：每 step 生成 [batch, vocab_size/32] 的 int32 bitmask

GPU Worker（ModelRunner）
  └─ Sampler
      └─ apply_bitmask()：把 bitmask 作用到 logits 上（屏蔽不合法 token）
          └─ grammar.accept_token()：推进 FSM 状态
```

**核心思路**：用有限状态机（FSM）或 grammar 约束每步可生成的 token 集合，
通过 bitmask 把不合法 token 的 logit 置为 -inf，从而强制模型输出符合 schema 的内容。
详见 `06_structured_output.md`。

---

## 六、关键配置对象：VllmConfig

所有组件共享一个 `VllmConfig` 实例，它封装了所有子配置：

```python
@dataclass
class VllmConfig:
    model_config: ModelConfig        # 模型路径、dtype、max_seq_len
    cache_config: CacheConfig        # block_size、num_gpu_blocks、swap_space
    parallel_config: ParallelConfig  # tp_size、pp_size、dp_size
    scheduler_config: SchedulerConfig # max_num_seqs、max_num_batched_tokens
    device_config: DeviceConfig      # cuda / cpu / tpu
    lora_config: LoRAConfig | None
    speculative_config: SpeculativeConfig | None
    structured_outputs_config: StructuredOutputsConfig | None
    # ...
```

> 设计原则：把配置做成一个大对象传递，避免每个构造函数都需要显式传参。
> 新增功能只需在 `VllmConfig` 里加字段，无需改动中间层构造函数。

---

## 七、Deep Dive 路线图

推荐的学习顺序（由浅入深）：

```
01_kv_cache        → 理解 PagedAttention 核心创新
02_scheduler       → 理解 batch 组装、chunked prefill、prefix cache 交互
03_gpu_model_runner → 理解 GPU 侧执行流程、CUDAGraph、persistent batch
04_attention_backend → FlashAttention / FlashInfer 的接入方式
05_tensor_parallelism → weights sharding、all-reduce 通信
06_structured_output → grammar FSM、bitmask 机制、多后端设计
07_spec_decode     → EAGLE draft、token verification、接受率
08_engine_core_ipc → ZMQ 通信协议、背压机制、DP 协调
```

每个文档都会包含：核心数据结构、关键代码路径、设计取舍、以及可以实验的代码片段。
