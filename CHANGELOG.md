# Changelog

本项目参考 Keep a Changelog 风格记录重要变化。教材是一次性成体系建设，日志用于追踪勘误、实现进度和兼容性，不承诺实时追逐模型榜单。

## [Unreleased]

### Added

- 面向开发者与算法工程师的新项目规格。
- Markdown/MkDocs、Notebook、可复用源码和生产项目四层结构。
- 模型谱系、环境矩阵、求职路线和实现验收矩阵。
- NumPy attention 正确性基线、PyTorch/JAX 微型 GPT 和可逆字节 tokenizer。
- 三本经过执行验证的 Attention、MiniGPT 与 RAG Notebook。
- 原生 BM25、租户 ACL、RRF、Recall@k 与 MRR。
- 带审批、幂等、预算和冲突检测的 Agent 工具执行内核。
- PyTorch LoRA Linear、合并导出与冻结/等价性测试。
- 文档与代码双 CI、环境诊断和 Notebook 执行器。
- OpenAI-compatible SSE 压测、TTFT/TPOT/吞吐定义与 vLLM 单卡路线。
- JSONL 评测 runner、文本指标、配对 bootstrap 与发布门禁。
- Transformers 离线训练/生成 smoke test 与真实 checkpoint 检查。
- LangChain/LlamaIndex canonical RAG adapter 与框架等价测试。
- 面试题、系统设计题、简历项目和工程项目索引。
- Dense retrieval、sentence-transformers embedding 与 cross-encoder adapter。
- GPT/DeepSeek/Qwen、Claude、Gemini 三类云 API 契约与离线测试。
- PEFT tiny GPT LoRA smoke test 与 TRL 单卡 SFT 入口。
- SQLite Agent ledger、原子 claim、pending reconciliation 防重放。
- GPT、Llama、Qwen、DeepSeek、Claude、Gemini 独立模型章节。
- Markdown 结构切分、稳定 chunk id、增量 upsert/delete 与授权引用审计。
- Agent pending 调用的外部成功确认、放弃/补偿和不可重放审计流程。
- QLoRA 分项显存估算、OOM 降级顺序和 NF4 单卡训练入口。
- JSON Schema/引用指标、overall/切片汇总与 Markdown 评测报告。
- RAG 摄取、检索、引用、生产，Agent 架构/runtime/评测，SFT/QLoRA、推理/vLLM 与评测方法共十二篇重点进阶专章。
- 分级相关性的 nDCG@k 指标与全包 strict mypy 验证。
- 内容准确性政策、官方核验台账、模型/API 版本边界与自动化准确性检查。
- 可执行 KV Cache 容量公式、严格的 SSE token 计数和 text-only 云响应校验。

### Changed

- 将学习目标从通用科普调整为研究生教材、工程落地和求职准备。
- 将 RAG、Agent、微调、推理部署和评测设为重点实践方向。
- 明确区分 Gemini Interactions API 与 `generateContent`，并按 TRL 实际 generation/assistant mask 规则修订单卡 SFT 指引。

## [0.1.0] - 2026-08-05

### Added

- LLM 基础、训练、推理、应用、质量、安全与前沿的首版中文教材。
- MkDocs Material 站点、搜索、MathJax、Mermaid 和严格构建检查。
