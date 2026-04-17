# 02 — Scheduler & 请求调度

> **前置阅读**：[01_kv_cache.md](01_kv_cache.md)
> **关键代码**：
> - `vllm/v1/core/sched/scheduler.py`
> - `vllm/v1/core/sched/request_queue.py`
> - `vllm/v1/core/sched/output.py`
> - `vllm/v1/request.py`

---

## TODO

- [ ] Scheduler 在整体架构中的位置（Engine Core 的核心循环）
- [ ] RequestQueue：waiting / running / swapped 三态
- [ ] schedule() 的执行流程
- [ ] Prefill vs Decode batch 的区别
- [ ] Chunked Prefill：为什么拆分长 prefill
- [ ] SchedulerOutput 数据结构（NewRequestData / CachedRequestData）
- [ ] 与 KVCacheManager 的协作：allocate → schedule → free
- [ ] 调度策略：FCFS vs 优先级队列
- [ ] Preemption（抢占）：swap out 到 CPU / recompute
- [ ] max_num_seqs / max_num_batched_tokens 参数的意义
