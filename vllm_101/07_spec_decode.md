# 07 — Speculative Decoding

> **前置阅读**：[03_gpu_model_runner.md](03_gpu_model_runner.md)
> **关键代码**：
> - `vllm/v1/spec_decode/eagle.py`
> - `vllm/v1/spec_decode/medusa.py`
> - `vllm/v1/spec_decode/ngram_proposer.py`
> - `vllm/v1/spec_decode/draft_model.py`
> - `vllm/v1/spec_decode/metadata.py`

---

## TODO

- [ ] 推测解码的核心思想：draft + verify，节省自回归步数
- [ ] 接受率（acceptance rate）与实际加速比的关系
- [ ] Draft Model 类型：
  - [ ] EAGLE / EAGLE-2：feature-based draft，共享 target model 隐层
  - [ ] Medusa：多个并行 draft heads
  - [ ] N-gram Proposer：基于已有输出做 n-gram 匹配
  - [ ] Suffix Decoding
- [ ] Verification 步骤：target model 一次 forward 验证多个 draft token
- [ ] Token Tree / Beam 结构：一次 draft 多条路径
- [ ] KV Cache 与 spec decode 的交互：draft token 的 KV 如何管理
- [ ] Scheduler 对 spec decode 的支持：scheduled_spec_decode_tokens
- [ ] Structured Output + Spec Decode 的 bitmask 扩展
- [ ] 性能 tradeoff：draft model 大小 vs 接受率
