# 06 — Structured Output & Logits Processors

> **前置阅读**：[03_gpu_model_runner.md](03_gpu_model_runner.md)
> **关键代码**：
> - `vllm/v1/structured_output/__init__.py`（StructuredOutputManager）
> - `vllm/v1/structured_output/backend_xgrammar.py`
> - `vllm/v1/structured_output/backend_guidance.py`
> - `vllm/v1/sample/logits_processor/`
> - `vllm/model_executor/layers/logits_processor.py`

---

## TODO

### Structured Output
- [ ] 用户侧接口：`response_format`、`guided_json`、`guided_regex`
- [ ] StructuredOutputManager 的生命周期（Engine Core 持有）
- [ ] Grammar 编译：异步 ThreadPoolExecutor，避免阻塞调度循环
- [ ] Bitmask 生成：`grammar_bitmask()` 的 int32 tensor 格式
- [ ] 多后端统一抽象：StructuredOutputBackend / StructuredOutputGrammar 接口
- [ ] xgrammar：基于 token trie 的高效 FSM
- [ ] guidance / outlines / lm-format-enforcer 后端对比
- [ ] Speculative Decoding + Structured Output 的联合约束
- [ ] Thinking mode（reasoning）下的 structured output 处理

### Logits Processors
- [ ] Logits Processor 的职责：在 sampling 前修改 logits 分布
- [ ] argmax-invariant vs non-argmax-invariant 的区分
- [ ] BatchUpdate 数据结构：add / remove / move 操作
- [ ] 内置处理器：min-p、logit bias、min-tokens
- [ ] 硬编码在 Sampler 中的处理：temperature、top-k、top-p、repetition penalty
- [ ] 自定义 Logits Processor 的编写方式
