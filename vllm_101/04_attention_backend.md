# 04 — Attention Backend 抽象层

> **前置阅读**：[03_gpu_model_runner.md](03_gpu_model_runner.md)
> **关键代码**：
> - `vllm/v1/attention/backend.py`
> - `vllm/v1/attention/selector.py`
> - `vllm/v1/attention/backends/`
> - `vllm/model_executor/layers/attention/`

---

## TODO

- [ ] 为什么需要 Attention Backend 抽象（多硬件、多实现）
- [ ] AttentionBackend 接口：build_metadata、forward
- [ ] Prefill 与 Decode 的不同 attention 计算路径
- [ ] FlashAttention 2/3：paged KV 的实现方式
- [ ] FlashInfer：page table 管理、workspace buffer
- [ ] XFormers / vanilla fallback
- [ ] MLA（Multi-head Latent Attention，DeepSeek）
- [ ] Sliding Window Attention
- [ ] Backend 选择逻辑（selector.py）
- [ ] AttentionMetadata 各字段含义（slot_mapping、block_tables 等）
