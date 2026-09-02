# 导航链接的规范名

同一个页面在不同 `.doc-nav` 行里出现时，应使用同一个名字。名字不一致会让读者
把同一页当成几个不同页面——这是导航体验里最容易被忽略、也最容易修的问题。

本页只约束 **`{ .doc-nav }` 导航行**里的链接文字。正文里的行内链接可以按语境自由措辞。

## 两条规则

1. **规范名优先**：下表里出现的页面，导航行必须使用「规范名」列的写法。
2. **方向性标签例外**：`返回…`、`运行…`、`进入…` 这类带动作方向的标签保留原样，
   因为它们表达的是"从哪来、去做什么"，而不是页面本身的名字。
   例：`[返回项目索引](../practice/project-index.md)` 不需要改成 `项目索引`。

带锚点的链接（如 `labs.md#lab-1b`）指向的是页面里的某一节，不受本表约束。

## 规范名表

| 页面 | 规范名 |
|---|---|
| `applications/agent-task-lifecycle.md` | 一次 Agent 退款任务 |
| `applications/agents.md` | Agent 总览 |
| `applications/rag-request-lifecycle.md` | 一次 RAG 请求 |
| `applications/rag-retrieval.md` | 召回与重排 |
| `applications/rag.md` | RAG 总览 |
| `career/interview-questions.md` | 面试题与回答方法 |
| `career/resume-projects.md` | 简历项目 |
| `career/system-design.md` | 系统设计 |
| `core/generation.md` | 生成与解码 |
| `core/transformer.md` | Transformer |
| `evidence/gemini-controls.md` | Gemini 接入证据 |
| `evidence/project-controls.md` | 项目实验台账 |
| `evidence/rag-answer-controls.md` | RAG 请求证据 |
| `frontier/reasoning-long-context-moe.md` | 前沿总览 |
| `guide/environment.md` | 环境配置 |
| `guide/knowledge-map.md` | 知识地图 |
| `guide/learning-paths.md` | 学习路径 |
| `guide/repo-map.md` | 仓库地图 |
| `models/cloud-api-contracts.md` | 云 API 契约 |
| `models/deepseek.md` | DeepSeek |
| `models/landscape.md` | 模型全景 |
| `models/qwen.md` | Qwen |
| `practice/labs.md` | 实验目录 |
| `practice/labs/lab-0c-cloud-budget.md` | 实验 0C |
| `practice/labs/lab-5-rag-request.md` | 实验 5 |
| `practice/labs/lab-6-agent-lifecycle.md` | 实验 6 |
| `practice/labs/lab-7b-nano-vllm-qwen3.md` | Qwen3 + nano-vLLM |
| `practice/project-index.md` | 项目索引 |
| `practice/projects/cloud-api-contracts.md` | Cloud API 项目 |
| `practice/projects/evaluation-gate.md` | Evaluation Gate |
| `practice/projects/inference-serving.md` | Inference Serving 项目 |
| `practice/projects/single-gpu-finetuning.md` | 单卡微调项目 |
| `practice/projects/transformers-basics.md` | Transformers 项目 |
| `quality/evaluation-methodology.md` | 评测方法 |
| `quality/evaluation.md` | 评测总览 |
| `quality/reasoning-artifact-security.md` | Reasoning 工件安全 |
| `reference/accuracy.md` | 内容准确性台账 |
| `systems/inference-request-lifecycle.md` | 一次请求如何穿过推理引擎 |
| `systems/serving.md` | 服务与可观测性 |
| `training/alignment.md` | 对齐 |
| `training/peft-qlora-engineering.md` | PEFT/QLoRA |
| `training/sft-data-pipeline.md` | SFT 数据管线 |

表中只列出**曾经出现过多个名字**的页面。只被一种写法引用的页面不需要登记；
如果将来给它加了第二种写法，再把它补进来。

## 新增页面时

给新页面写第一个导航链接时，先想清楚它在导航里叫什么，然后**在所有导航行里都用这个名字**。
只有当同一页开始出现第二种写法时，才需要把它登记到上表。
