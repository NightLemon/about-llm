# 工程项目索引

项目代码位于仓库根目录 projects/。教材解释原理，项目负责把原理变成可运行、可测量、可恢复的系统。

| 项目目录 | 当前能力 | 验证 |
|---|---|---|
| rag-foundations | 结构切分/增量更新、BM25/dense/RRF、ACL、引用审计、检索指标 | CPU 单测 |
| rag-framework-adapters | LangChain/LlamaIndex 无损转换 | 框架单测与 demo |
| safe-agent | 工具审批、幂等、预算、冲突、SQLite reconciliation/补偿审计 | CPU 单测 |
| single-gpu-finetuning | LoRA 从零、PEFT/TRL、QLoRA 估算与训练入口 | CPU 单测；真实 QLoRA 待目标 GPU |
| transformers-basics | 离线 tiny GPT 训练/生成、checkpoint 检查 | CPU smoke test |
| cloud-api-contracts | GPT/DeepSeek/Qwen、Claude、Gemini 请求响应契约 | 离线协议测试 |
| inference-serving | SSE 压测、TTFT/TPOT/吞吐、vLLM 路线 | CPU 协议单测；GPU 待目标环境 |
| evaluation-gate | JSONL runner、文本/检索/结构/引用指标、切片报告、bootstrap、门禁 | CPU 单测 |

## 推荐顺序

1. 先运行 rag-foundations，理解召回与 ACL。
2. 用 rag-framework-adapters 比较抽象成本，不改变检索结果。
3. 用 evaluation-gate 固定 case 与基线。
4. 用 single-gpu-finetuning 比较 Prompt、RAG 与 LoRA。
5. 用 inference-serving 测真实服务质量和性能。
6. 最后让 safe-agent 调用经过验证的检索与工具。

## 项目完成标准

每个生产项目最终应有：

- 明确的输入输出、数据与权限契约；
- 非 LLM 或最简单可行基线；
- 质量、安全、延迟和成本指标；
- 单元、集成、对抗和故障恢复测试；
- 可复现配置、版本和运行命令；
- 可观测 trace、发布门禁与回滚；
- 已知限制和不适用范围。

只有 README 和架构图不算项目完成；只有能运行但没有评测也不算。
