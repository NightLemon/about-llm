# Qwen 家族

## 学习价值

Qwen 覆盖中文/多语言、代码、数学、视觉、音频、dense 与 MoE 等多条路线，适合做中文 RAG、工具调用、单卡微调和多模态实验。家族跨度大，不能用一个架构描述所有 Qwen checkpoint。

## 检查 checkpoint

加载前后确认：

- model_type 与 architectures；
- dense 还是 MoE、激活参数与总参数；
- num_attention_heads 与 num_key_value_heads；
- hidden/intermediate size；
- RoPE 与最大位置配置；
- tokenizer 词表、特殊 token 和 chat template；
- tool calling 模板与 generation config；
- model card 推荐 Transformers 版本和许可。

中文“字符数”与 token 数不同；用目标 tokenizer 实测 Prompt、文档和输出预算。

## 中文 RAG

至少比较：

- BM25 字符/分词基线；
- 多语言/中文 embedding；
- cross-encoder reranker；
- chunk 的中文标点/标题结构；
- 精确实体、型号和英文缩写；
- 中英混合 query；
- 引用、无答案与冲突证据。

模型生成能力不能补偿检索 ACL；云端 Qwen API 与本地开放权重也要分别评测。

## 工具与结构化输出

使用 checkpoint 官方 chat template 序列化 tool schema/结果。模型只提出调用，外部 runtime 校验参数、身份、权限和副作用。更换 Qwen 版本时把模板和 parser 一起回归。

## 单卡实践

从小尺寸 Instruct checkpoint 建立 Transformers 基线；记录 revision、dtype、量化、上下文和峰值显存。LoRA target modules 由实际 module names 决定，不能从 Llama 教程机械复制。

## 面试追问

1. 中文 tokenizer 效率如何影响成本和有效上下文？
2. dense/MoE 的参数量怎样公平比较？
3. tool template 变化为什么会破坏调用？
4. 中文 RAG 为什么仍需要 BM25？
5. 如何比较云 API 与本地 Qwen 的总成本？
