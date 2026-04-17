# 05 — Tensor Parallelism & Pipeline Parallelism

> **前置阅读**：[03_gpu_model_runner.md](03_gpu_model_runner.md)
> **关键代码**：
> - `vllm/distributed/parallel_state.py`
> - `vllm/distributed/communication_op.py`
> - `vllm/model_executor/layers/linear.py`
> - `vllm/model_executor/layers/vocab_parallel_embedding.py`
> - `vllm/v1/executor/multiproc_executor.py`

---

## TODO

- [ ] 为什么需要模型并行（单卡装不下）
- [ ] Tensor Parallelism（TP）：Column / Row parallel linear
- [ ] QKV sharding 与 attention head 分配
- [ ] MLP / FFN 的 TP 切分
- [ ] all-reduce 的时机与通信量分析
- [ ] Pipeline Parallelism（PP）：micro-batch 流水
- [ ] Data Parallelism（DP）：多 Engine Core 负载均衡
- [ ] 进程组初始化：ProcessGroup、NCCL communicator
- [ ] Executor：MultiProcExecutor 如何分发 SchedulerOutput
- [ ] Expert Parallelism（MoE EP）：路由 token 到不同 GPU
