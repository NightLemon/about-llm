# Notebooks

Notebook 用于观察现象，不承载唯一实现。核心逻辑位于 src/about_llm/ 并由 tests/ 覆盖。

执行约定：

- 从仓库根目录启动 Jupyter。
- 先安装 README 中对应依赖组。
- 每本必须能 Restart & Run All。
- 默认不访问网络、不下载模型、不使用付费 API。
- 输出只展示小型结果，不提交大型二进制或模型权重。

当前顺序：

1. 01_attention_three_ways.ipynb：NumPy、PyTorch、JAX 的缩放点积注意力与因果 mask。
2. 02_minigpt_forward.ipynb：字节 tokenizer、PyTorch/JAX 微型 GPT、loss 与因果性。
3. 03_rag_retrieval_and_evaluation.ipynb：BM25、dense、RRF、ACL 与检索指标。

## 覆盖边界与下一步

三本 Notebook 覆盖“观察张量/模型前向/检索现象”的入门主线，不代表仓库全部主题都有 Notebook。训练恢复、PEFT 导出、云 API、Agent runtime、推理调度和评测门禁需要文件、进程、数据库或故障注入，优先使用 `projects/` 中的可运行入口与 `tests/`，不要为了 Notebook 形式复制唯一实现。

| 想学习的主题 | 首选入口 |
|---|---|
| Attention、MiniGPT、RAG 检索 | 本目录三本 Notebook |
| Tokenizer、MoE、generation contract | `projects/transformers-basics/` |
| LoRA/QLoRA、偏好优化、训练恢复 | `projects/single-gpu-finetuning/` |
| Agent 权限、审批、恢复、outbox | `projects/safe-agent/` |
| batching、量化、KV cache、服务指标 | `projects/inference-serving/` |
| 评测统计、证据验证、发布门禁 | `projects/evaluation-gate/` |

完整的实验顺序、交付物和证据边界见 `docs/practice/labs.md`；项目成熟度见 `docs/practice/project-index.md`。
