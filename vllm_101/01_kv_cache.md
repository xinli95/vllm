# 01 — PagedAttention & KV Cache Manager

> **前置阅读**：[00_overview.md](00_overview.md)
> **关键代码**：
> - [vllm/v1/core/kv_cache_utils.py](../vllm/v1/core/kv_cache_utils.py) — 数据结构：`KVCacheBlock`、`FreeKVCacheBlockQueue`、block hashing
> - [vllm/v1/core/block_pool.py](../vllm/v1/core/block_pool.py) — `BlockPool`：块分配、释放、prefix cache 哈希表
> - [vllm/v1/core/kv_cache_manager.py](../vllm/v1/core/kv_cache_manager.py) — `KVCacheManager`：Scheduler 面向的高层 API
> - [vllm/v1/kv_cache_interface.py](../vllm/v1/kv_cache_interface.py) — `KVCacheSpec`、`KVCacheConfig`：描述 KV Cache 的格式

---

## 一、为什么需要 PagedAttention？

### 传统方法的问题

LLM 推理中，每个 token 都需要存储自己的 Key / Value 向量（KV Cache），
供后续 token 做 attention 时复用。

传统做法是为每条请求 **预分配一块连续显存**，大小 = `max_seq_len × num_layers × 2 × num_heads × head_dim × dtype_size`。
这带来两个严重问题：

```
请求 A (实际 512 tokens, 分配 2048 token 槽位)
┌──────────────────────────────────────────────────────────┐
│ [已用 512 token KV] [              内部碎片 1536          ]│
└──────────────────────────────────────────────────────────┘

请求 B (实际 100 tokens, 分配 2048 token 槽位)
┌──────────────────────────────────────────────────────────┐
│ [已用 100 token KV] [              内部碎片 1948          ]│
└──────────────────────────────────────────────────────────┘
```

1. **内部碎片**：每条请求都必须预留到 `max_seq_len`，即便实际只用了 5%。
   论文测量，这种浪费高达 **60–80%** 显存。
2. **无法共享前缀**：两条以相同 system prompt 开头的请求，
   各自持有一份 KV Cache 副本，不能复用。

### PagedAttention 的解法

借鉴操作系统虚拟内存的思想：把 KV Cache 切成固定大小的**物理块（page）**，
按需分配，不同请求的块可以不连续，前缀相同的请求可以**共享物理块**。

```
Block 0   Block 1   Block 2   Block 3   Block 4
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ sys   │ │ sys   │ │ req_A │ │ req_B │ │ req_A │
│ prompt│ │ prompt│ │ cont. │ │ cont. │ │ cont2 │
│ (共享)│ │ (共享)│ │       │ │       │ │       │
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘
req_A 的逻辑序列:  [Block 0] → [Block 1] → [Block 2] → [Block 4]
req_B 的逻辑序列:  [Block 0] → [Block 1] → [Block 3]
```

- 物理内存利用率接近 100%（只有最后一个 block 可能不满）
- 相同前缀的 KV 只计算一次，后续请求直接引用

---

## 二、核心数据结构

### 2.1 KVCacheBlock — 物理块的 CPU 侧元数据

```python
# vllm/v1/core/kv_cache_utils.py:109
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                          # 物理块编号（0 ~ num_gpu_blocks-1）
    ref_cnt: int = 0                       # 引用计数
    _block_hash: BlockHashWithGroupId | None = None  # 前缀 cache 用的哈希
    prev_free_block: KVCacheBlock | None = None      # 空闲队列双向链表
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False                  # 是否是占位 null block
```

**三种状态**：

| `ref_cnt` | `block_hash` | 状态 |
|-----------|--------------|------|
| > 0 | None | 活跃块，正被某请求使用，KV 尚未写满 |
| > 0 | 已设置 | 活跃块，KV 已写满，同时被 prefix cache 引用 |
| = 0 | 已设置 | 空闲但已缓存：可被 LRU 驱逐，也可被命中复用 |
| = 0 | None | 纯空闲块，等待分配 |

`block_id` 同时也是 GPU tensor 中的索引——GPU Worker 用 `block_id` 直接查 block table，
CPU 侧的 `KVCacheBlock` 只是对应的元数据镜像。

### 2.2 FreeKVCacheBlockQueue — O(1) LRU 空闲队列

标准 Python `deque` 支持首尾操作但不支持 O(1) 中间删除，
因此 vLLM 自己实现了一个**带 fake head/tail 的双向链表**：

```python
# vllm/v1/core/kv_cache_utils.py:158
class FreeKVCacheBlockQueue:
    fake_free_list_head → [Block_X] → [Block_Y] → ... → fake_free_list_tail
```

- `popleft()` / `popleft_n(n)` — 从队头取 N 个块（LRU 驱逐从这里发生）
- `append(block)` / `append_n(blocks)` — 回收到队尾（最近使用）
- `remove(block)` — O(1) 从中间摘出（prefix cache 命中时用）

**LRU 顺序维护**：当一条请求完成，其块按**逆序**归还到队尾。
逆序确保靠近序列末尾（token 越多）的块在队列前面，更早被驱逐。
这是因为序列的前缀更有可能被其他请求共享，应该尽量留在 cache 中。

### 2.3 BlockPool — 物理块池

```python
# vllm/v1/core/block_pool.py:130
class BlockPool:
    blocks: list[KVCacheBlock]              # 所有块（按 block_id 索引）
    free_block_queue: FreeKVCacheBlockQueue # LRU 空闲队列
    cached_block_hash_to_block: BlockHashToBlockMap  # hash → block 的 prefix cache 表
    null_block: KVCacheBlock                # block_id=0 的占位块
```

`BlockHashToBlockMap` 是 prefix cache 的核心查找表：
```
block_hash_A → KVCacheBlock(id=3)
block_hash_B → {id=7: KVCacheBlock, id=12: KVCacheBlock}  # hash 冲突时用 dict
```

> 注意：vLLM 允许同一个 hash 对应多个物理块（相同内容的块被多个请求各自分配了一份）。
> 这样避免了 block_id 变化带来的 block table 不稳定性。

---

## 三、Block Hashing — Prefix Cache 的键

prefix cache 能否命中，核心在于：两个 block 的 token 内容是否完全相同。
vLLM 用 rolling hash 来做到 O(1) 查找。

### 哈希的构成

每个 block 的 hash = `hash(前一个block的hash, 本block的token_ids, extra_keys)`

```python
# extra_keys 包含：
# - multimodal 内容的 hash（图片/音频不同，KV 就不同）
# - LoRA adapter 名字（不同 LoRA 的 KV 不能共享）
# - cache_salt（用户指定的隔离标识）
```

类型定义：
```python
BlockHash = NewType("BlockHash", bytes)         # 单个 block 的 hash bytes
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)  # hash + 4字节 group_id
```

group_id 的存在是因为混合模型（如 Mamba + Attention）有多个 KV Cache group，
同样的 token 序列在不同 group 里的 KV 格式不同，需要分开存储。

### hash 何时计算？

在 `Request` 对象创建时，`block_hasher` 回调会预先计算所有 block 的 hash，
并存储在 `request.block_hashes: list[BlockHash]` 中。
这样 `get_computed_blocks()` 查询时可以直接读取，无需重新计算。

---

## 四、KVCacheSpec — 描述每层 KV Cache 的格式

不同的模型层（标准 attention、sliding window、MLA、Mamba…）对 KV Cache
的内存布局要求完全不同。`KVCacheSpec` 体系提供统一的描述接口：

```
KVCacheSpec (抽象基类)
├── AttentionSpec
│   ├── FullAttentionSpec         # 标准全注意力（含 MLA、sliding window 的全量变体）
│   │   ├── MLAAttentionSpec      # DeepSeek MLA：compressed KV
│   │   └── SinkFullAttentionSpec # Sink attention
│   ├── SlidingWindowSpec         # 滑动窗口注意力（独立 KV cache group）
│   ├── ChunkedLocalAttentionSpec # 分块本地注意力
│   ├── CrossAttentionSpec        # Encoder-Decoder 交叉注意力（如 Whisper）
│   └── EncoderOnlyAttentionSpec
└── MambaSpec                     # Mamba SSM 状态缓存
```

以 `AttentionSpec` 为例，其 `page_size_bytes` 计算：
```python
# vllm/v1/kv_cache_interface.py:138
real_page_size = 2 * block_size * num_kv_heads * head_size * dtype_size
# 因子 2 = K 和 V 各一份
# 若 kv_quant_mode.is_per_token_head，还需加 scale tensor 的空间
```

这些 Spec 决定了启动时 GPU Worker 如何分配 KV Cache tensor，
以及 `num_gpu_blocks` 如何根据可用显存计算出来。

---

## 五、KVCacheManager — Scheduler 的接口

`KVCacheManager` 是 Scheduler 与底层 Block Pool 之间的门面（Facade），
隐藏了 coordinator / multi-group 的复杂性，对外暴露三个核心方法：

```python
class KVCacheManager:
    def get_computed_blocks(request) → (KVCacheBlocks, num_computed_tokens)
    def allocate_slots(request, num_new_tokens, ...) → KVCacheBlocks | None
    def free(request) → None
```

`KVCacheBlocks` 是 Scheduler 侧的块引用：

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[group_idx][block_idx] → KVCacheBlock
    # 外层 tuple: kv_cache_groups（混合模型有多个）
    # 内层 list:  该 group 内按顺序排列的物理块
```

Scheduler 拿到 `KVCacheBlocks` 后，调用 `get_block_ids()` 转成 int 列表，
打包进 `SchedulerOutput` 发给 GPU Worker。GPU Worker 用 block_id 数组
构建 block table，告诉 attention kernel 去哪里读写 KV。

---

## 六、allocate_slots() — 一次调度的完整分配流程

这是 KV Cache 系统最核心的操作，每次 `scheduler.schedule()` 时对每条请求调用一次。

```
tokens 布局（完整序列视角）：

────────────────────────────────────────────────────────────────────
│ < comp > │ < new_comp > │ < ext_comp > │ < new >  │ < lookahead >│
────────────────────────────────────────────────────────────────────
               ↑ prefix cache 新命中    ↑ 本步要计算的 token
```

- `comp`：`request.num_computed_tokens`（已有 KV 的 token）
- `new_comp`：本步新命中 prefix cache 的 token 数（不需重算）
- `ext_comp`：外部 KV 传输已提供的 token 数（P/D 分离场景）
- `new`：本步需要 GPU 计算 KV 的 token 数
- `lookahead`：spec decode 的 draft token 预留槽位

### 分配步骤（简化版）

```python
def allocate_slots(request, num_new_tokens, num_new_computed_tokens,
                   new_computed_blocks, num_lookahead_tokens, ...):

    # Step 1: 释放 sliding window 窗口外的块（节省空间）
    coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    # Step 2: 计算还需要分配多少新块
    num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(...)

    # Step 3: 检查空闲块是否够用（不够 → 返回 None，Scheduler 跳过此请求）
    if num_blocks_to_allocate > block_pool.get_num_free_blocks():
        return None

    # Step 4: 把 prefix cache 命中的块挂到请求上（增加 ref_cnt）
    coordinator.allocate_new_computed_blocks(...)

    # Step 5: 分配真正的新物理块
    new_blocks = coordinator.allocate_new_blocks(...)

    # Step 6: 把已写满的块加入 prefix cache（计算 hash、存入哈希表）
    # 只 cache "确认的" token，不 cache draft token（避免无效缓存）
    num_tokens_to_cache = min(total_computed_tokens + num_new_tokens,
                              request.num_tokens)  # request.num_tokens = 不含 draft
    coordinator.cache_blocks(request, num_tokens_to_cache)

    return KVCacheBlocks(new_blocks)
```

### 分配失败 → Scheduler 的处理

`allocate_slots` 返回 `None` 意味着：当前空闲块不足以容纳这条请求的本步新 token。
Scheduler 会将此请求暂时跳过（保留在 running queue），等待其他请求完成释放块。
如果请求长期无法调度，Scheduler 会触发 **preemption**（见 `02_scheduler.md`）。

---

## 七、Prefix Cache 命中的完整路径

以下场景：两条请求共享相同的 system prompt（前 2 个 block）：

```
Step 1: 请求 A 完成 prefill，前 2 个 block 被 cache
        Block 0 (hash_0): ref_cnt=1, in cache
        Block 1 (hash_1): ref_cnt=1, in cache
        Block 2 (req_A private): ref_cnt=1, NOT in cache（未写满或是最后一块）

Step 2: 请求 A 解码中，释放（完成后）
        Block 0: ref_cnt=0, 保留 hash, 进入 free_queue 尾部（可被驱逐但优先保留）
        Block 1: ref_cnt=0, 同上

Step 3: 请求 B 到来，scheduler 调用 get_computed_blocks(request_B)
        → find_longest_cache_hit(request_B.block_hashes)
        → 命中 Block 0（hash_0 匹配）和 Block 1（hash_1 匹配）

Step 4: allocate_slots 调用 allocate_new_computed_blocks([Block 0, Block 1])
        → block_pool.touch([Block 0, Block 1])
            → 从 free_queue 中 remove（O(1)）
            → ref_cnt: 0 → 1
        → 请求 B 的 block list 前两项 = [Block 0, Block 1]

Step 5: allocate_new_blocks 为请求 B 的新 token 分配新物理块
        → Block 3（全新分配）
```

这样请求 B 的第一个 prefill step 只需要计算 block 2 之后的 token，
Block 0 和 Block 1 的 KV 直接重用，无需重算。

---

## 八、LRU 驱逐机制

当 `block_pool.get_new_blocks(n)` 被调用时：

```python
def get_new_blocks(self, num_blocks):
    ret = self.free_block_queue.popleft_n(num_blocks)  # 从队头取（LRU）
    for block in ret:
        self._maybe_evict_cached_block(block)  # 如果有 hash，从 cache 表删除
        block.ref_cnt = 0 → 1
```

`_maybe_evict_cached_block` 做的事：
```python
def _maybe_evict_cached_block(self, block):
    if block.block_hash is None:
        return  # 无 hash = 普通空闲块，无需驱逐
    # 从哈希表移除
    self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)
    block.reset_hash()  # block_hash = None
    # （如果启用 kv_cache_events，发出 BlockRemoved 事件）
```

**驱逐顺序**：`free_block_queue` 是 LRU 队列，队头是最久未使用的块。
对于有 hash 的 cached block，"最近使用"指最近一次被某请求命中并释放的时间。

---

## 九、KVCacheSpec 与显存规划

启动时，vLLM 执行如下流程确定 `num_gpu_blocks`：

```
1. GPU Worker 做一次 "profile run"（用最大 batch size 和 seq len 跑一次 forward）
   → 测量 non-KV 显存使用量（weights, activations, workspace）

2. 剩余显存 = gpu_memory_utilization * total_vram - non_kv_memory

3. 对每个 KVCacheSpec 计算 page_size_bytes

4. num_gpu_blocks = 剩余显存 / max(page_size_bytes across all groups)

5. 将 num_gpu_blocks 广播给 Engine Core，BlockPool 据此初始化
```

---

## 十、实现细节：null_block

```python
# block_pool.py:176
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True
```

`null_block`（block_id=0）是一个特殊的占位块，用于：
- sliding window attention 中，超出窗口范围的 "slot" 用 null_block 填充
- Mamba 的 align mode 中类似用途

它的 `ref_cnt` 不被维护（永远不会被释放回 free queue 或驱逐）。

---

## 十一、小结：关键设计取舍

| 设计决策 | 原因 |
|----------|------|
| CPU 侧 `KVCacheBlock` 只存元数据，GPU tensor 按 block_id 索引 | 避免 CPU-GPU 数据同步开销 |
| FreeKVCacheBlockQueue 自己实现双向链表（不用 deque） | 支持 O(1) 中间删除（prefix cache 命中时 touch 操作） |
| prefix cache 不 de-duplicate 相同内容的块 | 保持 block_id 稳定，block table append-only |
| request 创建时就预计算所有 block hash | 调度时 O(1) 查 cache，不阻塞调度循环 |
| 只 cache `request.num_tokens` 以内的 token | draft token（spec decode）不被缓存，避免 reject 后产生脏缓存 |
| 按逆序归还块到 free queue | 尾块（最新）优先被驱逐，前缀块尽量留在 cache |

---

**下一篇**：[02_scheduler.md](02_scheduler.md) — 了解 Scheduler 如何利用 KVCacheManager
编排 batch，处理 prefill/decode 混合，以及 preemption 机制。
