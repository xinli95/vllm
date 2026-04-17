# 08 — Engine Core & 进程间通信

> **前置阅读**：[00_overview.md](00_overview.md)
> **关键代码**：
> - `vllm/v1/engine/core.py`
> - `vllm/v1/engine/core_client.py`
> - `vllm/v1/engine/async_llm.py`
> - `vllm/v1/engine/coordinator.py`
> - `vllm/v1/executor/multiproc_executor.py`
> - `vllm/v1/utils.py`

---

## TODO

- [ ] Engine Core 主循环：schedule → execute → output 的 busy loop
- [ ] ZMQ 通信拓扑：API Server ↔ Engine Core（PUSH/PULL + PUB/SUB）
- [ ] EngineCoreRequest / EngineCoreOutput 的序列化（msgpack / pickle）
- [ ] 背压机制：如何避免 Engine Core 被请求淹没
- [ ] AsyncLLMEngine vs LLMEngine：在线 vs 离线的不同事件循环
- [ ] CoreClient：同进程（InprocClient）vs 跨进程（MPClient）模式
- [ ] Executor：EngineCore 如何向 GPU Worker 分发任务
- [ ] GPU Worker 结果回收：output_queue 与 IPC shared memory
- [ ] DP Coordinator：多 DP rank 的负载均衡与 MoE 同步
- [ ] Tensor IPC：GPU tensor 跨进程传输（zero-copy）
- [ ] 错误处理与进程恢复
