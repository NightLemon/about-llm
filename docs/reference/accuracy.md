# 内容准确性与核验台账

本页定义教材中的事实如何分类、验证和维护。目标不是用“官方”二字替代证据，而是让读者知道每条结论在什么边界内成立、仓库实际验证了什么、哪些仍需目标环境实测。

## 四类陈述与证据

| 类别 | 典型内容 | 首选证据 | 必须写清的边界 |
|---|---|---|---|
| 数学与算法 | attention 公式、复杂度、KV Cache 容量 | 推导、可执行小例子、单元测试 | 变量定义、单位、假设和省略项 |
| 代码与 API | 请求字段、chat template、CLI 参数 | 固定版本官方文档、schema、契约测试 | provider、API/runtime 版本、`checked_at` |
| 产品事实 | 模型能力、配额、价格、区域和数据政策 | 对应产品的官方页面 | 检查日期；不得把当前状态写成永久事实 |
| 工程估算 | 显存、吞吐、延迟和成本 | 明确公式加目标环境基准 | 硬件、dtype、batch、长度、并发、软件版本和误差来源 |

论文能说明论文中的方法，不能自动证明某个当前闭源产品采用相同内部实现。官方文档能证明文档在检查时给出的接口承诺，也不等于本仓库已经在真实账号、网络、GPU 或付费端点上运行成功。

## 核验边界

- 离线单元测试证明给定输入下的本地实现行为，不证明云端 API 当前可用。
- CPU smoke test 不证明 CUDA kernel、vLLM、bitsandbytes 或目标 GPU 的兼容性与峰值显存。
- 一台机器上的 benchmark 只对记录的 workload 和版本有效，不是普适性能排名。
- JSON mode 只保证产生有效 JSON；OpenAI Structured Outputs 在受支持范围内约束 JSON Schema，但两者都不保证内容真实、业务规则正确或工具调用已获授权。
- schema、引用格式和安全分类器通过，只是系统质量证据的一部分，不能替代语义评测与权限控制。

## 写作与实现规则

1. 未公开的闭源模型参数量、层数、训练数据、路由和后训练配方保持未知，不从旧论文或同品牌模型外推。
2. 开放权重模型的架构结论绑定 checkpoint revision，以实际 `config`、tokenizer、chat template、generation config、model card 和许可为准。
3. 时间敏感 API 或产品事实记录 `checked_at`；稳定教材不维护“永久 latest”模型、价格或上下文窗口表。
4. 数字示例写明单位、公式和排除项。容量估算与实测峰值分开，decimal GB 与 binary GiB 不混用。
5. 性能结论绑定硬件、dtype/量化、输入输出长度分布、batch/并发、runtime/kernel 版本和采样参数。
6. OpenAI-compatible 只表示部分请求形状兼容；provider 的扩展字段、错误、usage、流式事件和工具语义分别验证。
7. 工具调用是模型提出的候选动作，不是授权。参数校验、身份、权限、幂等和审计由外部 runtime 承担。

## 官方核验台账

下列页面于 **2026-08-05** 人工核对。链接是事实来源，不是运行证明；采用时仍应固定实际依赖/API 版本。

| 主题 | 已核对结论 | 官方来源 |
|---|---|---|
| OpenAI Structured Outputs | 支持范围内遵循 JSON Schema；JSON mode 仅保证有效 JSON | [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) |
| Anthropic Messages | `system`、content blocks、input/output usage 与 stop reason 属于 Messages 契约 | [Messages API](https://docs.anthropic.com/en/api/messages) |
| Gemini 文本生成 | 新项目推荐 GA Interactions API；本仓库 `generateContent` adapter 是另一套需独立建模的契约 | [Gemini text generation](https://ai.google.dev/gemini-api/docs/text-generation) |
| TRL `SFTTrainer` | `assistant_only_loss` 依赖 assistant/generation mask；部分已知模板可自动修补，仍需检查实际 mask/labels | [SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer) |
| vLLM CLI | `vllm serve` 提供 OpenAI-compatible server；参数和支持矩阵应按安装版本与 stable 文档核对 | [vLLM stable CLI](https://docs.vllm.ai/en/stable/cli/) |
| Transformers chat template | 模板序列化 role/control tokens；训练与生成格式、generation prompt 和 assistant mask 要按 tokenizer 模板验证 | [Chat templates](https://huggingface.co/docs/transformers/en/chat_templating) |

## 已有可执行证据

- NumPy attention 与 PyTorch attention 对照，覆盖 causal mask 和数值稳定性。
- KV Cache 理想化 dense K/V 公式由 `estimate_kv_cache_bytes` 实现；32 层、8 KV heads、head dim 128、8192 tokens、BF16 的结果精确为 1 GiB，测试同时验证 batch 缩放与非法维度。
- OpenAI-compatible 流式基准必须取得服务端 `completion_tokens`；SSE event/chunk 数不作为 token 数。
- 云 API adapter 的离线契约测试覆盖三类字段映射和密钥脱敏；text-only parser 遇到无文本响应会显式失败。
- LoRA 冻结、初始化、merge 等价性以及 RAG/Agent/评测核心不变量均有单元测试。

## 尚未由本仓库证明

- 当前机器没有完成真实 CUDA/vLLM/QLoRA 的目标硬件验证；文档中的命令是运行路线，不是成功声明。
- 离线 adapter 测试没有访问 OpenAI、Anthropic、Gemini、DeepSeek 或 Qwen 的真实付费端点，也不证明账号配额、区域或最新产品行为。
- 任何吞吐、TTFT、TPOT、显存与费用结果都需要在目标环境保存 workload manifest 后才能成为可比较证据。
- 闭源模型未公开的内部架构与训练配方不在可核验范围内。

## 维护流程

1. 修改公式、API 或数字前先确定陈述类别与证据等级。
2. API/产品事实优先核对固定版本官方文档，记录 provider、版本和 `checked_at`。
3. 将能执行的不变量写成最小测试；不能执行的事实明确标注“文档核对”或“待目标环境验证”。
4. 运行 `python scripts/check_content_accuracy.py`，再运行完整测试、严格类型检查和 MkDocs strict build。
5. 目标环境基准保存原始结果、配置、硬件与软件版本；只在这些条件内解读结论。
6. 发现勘误时更新正文、测试、台账与 `CHANGELOG.md`，避免只改其中一处。
