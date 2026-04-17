# 03 — GPU Model Runner & CUDAGraph

> **前置阅读**：[02_scheduler.md](02_scheduler.md)
> **关键代码**：
> - `vllm/v1/worker/gpu_model_runner.py`
> - `vllm/v1/worker/gpu_worker.py`
> - `vllm/v1/worker/gpu_input_batch.py`
> - `vllm/v1/worker/block_table.py`

---

## TODO

- [ ] Worker vs ModelRunner 的职责分工
- [ ] Persistent Batch（InputBatch）：避免每 step 重建 tensor
- [ ] _update_states()：处理 batch 变化（add / remove / reorder）
- [ ] prepare_inputs()：构造 input_ids、positions、attention_metadata
- [ ] BlockTable：逻辑块 → 物理块的 GPU 侧映射
- [ ] AttentionMetadata：各 backend 需要哪些信息
- [ ] CUDAGraph capture & replay 机制
- [ ] Padding 与 continuous batching 的 tradeoff
- [ ] execute_model() 完整调用链
- [ ] Sampler：从 logits 到 next token
