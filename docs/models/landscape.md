# GPT、Llama、Qwen、DeepSeek、Claude 与 Gemini

## 如何比较模型

不要把品牌名当架构。统一从六个维度比较：权重是否开放、训练和后训练公开程度、上下文与模态、工具/API 能力、部署位置、许可证与数据治理。闭源模型的参数量、训练语料和内部路由若未官方披露，就写“未公开”，不从传闻补齐。

## 模型家族

### GPT

GPT 奠定 decoder-only 自回归预训练与规模化路线。公开研究代际展示了预训练、few-shot/in-context learning、指令/偏好后训练和工具调用的演进；当前产品模型细节以 OpenAI 官方模型/API 文档为准。学习重点不是背型号，而是理解 completion、chat、tool、structured output 等接口怎样改变应用契约。

### Llama

Meta 发布权重的 decoder-only 家族，是本地推理、微调和系统研究的重要基线。代际中常见 RoPE、RMSNorm、SwiGLU、GQA、扩展词表/上下文与多语言改进。具体层数、头数、词表、许可按所选 checkpoint 的 config/model card，不把一家族概括套到全部版本。

### Qwen

阿里云通义千问家族覆盖 dense/MoE、代码、数学、视觉与音频等方向，多语言与中文是重要实践场景。工程中重点检查 chat template、tool calling 协议、长上下文设置、量化与 Transformers 版本兼容。云端 API 与开放权重模型是两条部署路线。

### DeepSeek

DeepSeek 家族公开研究涉及 MoE、Multi-head Latent Attention、代码/数学和推理后训练等。学习时拆开三个问题：稀疏架构如何降低激活计算、KV 表示如何影响推理成本、可验证奖励/test-time compute 如何改变推理行为。API 模型与开放权重版本可能不同，不能互换假设。

### Claude

Anthropic 的闭源模型家族，常用于长上下文、工具调用和企业 API。可以学习 Constitutional AI、RLHF/RLAIF 等公开研究与产品接口，但不要声称当前模型一定采用某篇论文的全部方法。工程比较关注消息角色、tool use、streaming、缓存、限额与数据政策。

### Gemini

Google 的闭源/产品模型家族，强调原生多模态、长上下文以及与 Google 平台集成。Gemini API、Vertex AI 和面向用户的产品在身份、区域、治理与功能上不同。工程选型需验证结构化输出、函数调用、媒体输入、缓存和配额。

## 公开权重与云 API

| 问题 | 公开权重 | 云 API |
|---|---|---|
| 数据驻留 | 可完全本地 | 依供应商和区域政策 |
| 运维 | 自己负责 GPU、服务和升级 | 供应商托管 |
| 定制 | 可量化、微调、修改 serving | 以 API 能力为限 |
| 规模弹性 | 需容量规划 | 通常更易弹性，但有限额 |
| 可复现 | 可固定权重哈希与 runtime | 模型可能更新，需版本或快照 |
| 成本 | 固定资源加运维 | 按 token、请求或缓存计费 |

## 实验选型

- 原理与单测：微型自建模型。
- 单卡 Transformers/LoRA：选许可证允许、尺寸适合的 Llama、Qwen、DeepSeek 系 checkpoint。
- 中文 RAG：至少比较一个多语言 Embedding/重排器与 BM25。
- 云 Agent：用统一 adapter 跑 GPT、Claude、Gemini、DeepSeek、Qwen API 的同一协议测试，但保留各家独有能力。
- 性能：只比较相同硬件、量化、输入输出长度和质量约束下的数据。

## 型号信息的时间边界

型号、价格、上下文长度和 API 字段变化快。本教材正文讲稳定比较方法；具体项目用配置记录 provider、model、revision 和 checked_at，需要时从官方文档核对，而不是把易过时表格写成永久事实。
