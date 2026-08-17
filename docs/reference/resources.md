# 资料与论文

优先阅读一手资料。这里按概念价值而非“最新榜单”组织；访问日期和版本对快速变化的实现文档尤其重要。

需要跟进近期研究时，使用独立的[近期论文解读](../papers/index.md)和固定日期快照。那里记录“为什么现在读”、原文证据、限制与不可外推项；本页继续维护长期有效的基础书目。

## 基础

- Jurafsky & Martin, *Speech and Language Processing*（在线草稿）：NLP 全景与术语基础。
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016：深度学习数学与优化基础。
- Bishop & Bishop, *Deep Learning: Foundations and Concepts*, 2024：概率视角的现代教材。
- Hugging Face, *NLP Course*：tokenizer、Transformer 与生态实践。

## Transformer 与语言模型

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017：Transformer 原始架构。
- Devlin et al., [BERT](https://arxiv.org/abs/1810.04805), 2018：双向 masked LM。
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020：规模与上下文学习。
- Su et al., [RoFormer](https://arxiv.org/abs/2104.09864), 2021：RoPE。
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), 2022：IO-aware 精确注意力。

## 规模、数据与训练

- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020。
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022。
- Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499), 2021。
- Penedo et al., [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116), 2023：网页数据流水线案例。
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054), 2019：大模型状态分片。

## 后训练

- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022：InstructGPT/RLHF。
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021。
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314), 2023。
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023。
- Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073), 2022。

## RAG 与 Agent

- Lewis et al., [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401), 2020。
- Karpukhin et al., [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906), 2020。
- Nogueira & Cho, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085), 2019。
- Yao et al., [ReAct](https://arxiv.org/abs/2210.03629), 2022。
- Schick et al., [Toolformer](https://arxiv.org/abs/2302.04761), 2023。

## 持续更新：API 与互操作标准

本节只列需要按版本和访问日期持续复核的官方入口，不把当前产品状态写成永久事实。

- Model Context Protocol，[Introduction](https://modelcontextprotocol.io/docs/getting-started/intro)、[2025-11-25 transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)、[lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle) 与 [tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)：tools/resources/prompts、stdio/HTTP、capability、结构化结果与安全边界。
- A2A Protocol，[Specification](https://a2a-protocol.org/latest/specification/)、[v1.0.0 JSON Schema](https://a2a-protocol.org/v1.0.0/spec/a2a.json) 与官方 [Python SDK](https://github.com/a2aproject/a2a-python)：Agent Card、message/task/artifact、JSON-RPC/HTTP+JSON/gRPC binding、长任务与跨 Agent 协作；核对版本与 SDK compatibility，不把单一 loopback control 写成完整 conformance。
- Google Gemini，[Interactions API](https://ai.google.dev/gemini-api/docs/interactions) 与 [GenerateContent API](https://ai.google.dev/api/generate-content)：分别核对状态、steps、工具、流式事件与迁移边界。
- Anthropic，[Messages API](https://platform.claude.com/docs/en/api/messages)：核对顶层 `system`、content blocks、usage 与 stop reason。
- OpenAI，[API documentation](https://developers.openai.com/api/docs/)：按所用 endpoint、model snapshot 与功能版本核对请求、工具、stream 和 usage。

协议/产品文档变化快。项目接入时把 URL、访问日期、API/协议版本和 SDK 版本写入 manifest，并用目标账号与网络环境另做 smoke test。

## 推理与服务

- Kwon et al., [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023。
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2022。
- Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323), 2022。
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978), 2023。

## 评测、解释与安全

- Liang et al., [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110), 2022。
- Srivastava et al., [BIG-bench](https://arxiv.org/abs/2206.04615), 2022。
- Zheng et al., [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685), 2023。
- Bommasani et al., [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258), 2021。
- Panfilov et al., [Stealing Reasoning Traces from Proprietary LLM APIs](https://arxiv.org/abs/2608.09867), 2026：opaque reasoning block 的跨上下文重放、轨迹发布风险与 context-bound envelope；论文注明披露后原攻击截至 2026 年 8 月已不可复现。
- OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/)：应用威胁清单；使用时核对当前版本。
- NIST, [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：风险治理框架。
- Mitchell et al., [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993), 2018。
- Gebru et al., [Datasheets for Datasets](https://arxiv.org/abs/1803.09010), 2018。

## 多模态与长序列

- Radford et al., [CLIP](https://arxiv.org/abs/2103.00020), 2021。
- Alayrac et al., [Flamingo](https://arxiv.org/abs/2204.14198), 2022。
- Liu et al., [LLaVA](https://arxiv.org/abs/2304.08485), 2023。
- Beltagy et al., [Longformer](https://arxiv.org/abs/2004.05150), 2020。
- Gu & Dao, [Mamba](https://arxiv.org/abs/2312.00752), 2023。

## 阅读原则

论文链接只代表值得学习，不代表仓库为其全部结论背书。阅读时核对版本、数据、规模、对照、推理预算与复现；对当下模型/法规/产品信息再查官方最新来源。
