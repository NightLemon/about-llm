# Qwen 家族

## 学习目标与证据边界

读完本章应能为中文/中英混合任务选择并检查 Qwen checkpoint，解释 dense/MoE、Base/Instruct、文本/代码/多模态版本的边界，并能验证 tokenizer、chat template 与 tool calling 是否和部署 runtime 一致。

**先修知识**：decoder-only Transformer、MoE、tokenization、RAG、LoRA/QLoRA、Transformers chat template。

Qwen 家族覆盖多语言、代码、数学、视觉、音频、dense 与 MoE 等多条路线，不能用一个架构描述所有 Qwen checkpoint。本文讲稳定检查方法；具体层数、head、专家数、上下文、thinking/tool 模式和许可都以固定 revision 的 config、model card 与官方仓库为准。

## 先识别你拿到的是什么

“Qwen 模型”至少还缺五个限定：

1. **代际与 checkpoint**：不同公开代际配置和模板不同；
2. **Base 或 Instruct**：续写基座和对话后训练模型不是同一接口；
3. **dense 或 MoE**：总参数、激活参数、加载内存与每 token 计算的含义不同；
4. **文本/代码/视觉/音频**：processor、输入模态和生成头可能不同；
5. **本地权重或云 API**：provider 端模型、路由、配额和协议不能由本地 model card 推断。

检查 checkpoint 时记录：

```text
model id + commit hash
model_type / architectures
hidden_size / intermediate_size / num_hidden_layers
num_attention_heads / num_key_value_heads
dense or MoE: total params / active params / experts / top-k
rope and max position configuration
tokenizer files / special tokens / chat template
generation config / tool template / license
recommended Transformers/runtime versions
```

若 config 使用模型专有字段，不要按字段名猜公式；先查同 revision 的官方实现和技术报告。

可用 `inspect_checkpoint.py <model-id> --revision <commit-hash>` 保存 normalized config/generation snapshots、resolved metadata、模板直接生成的 token IDs，并比较 tokenizer/model/generation 三方 special-token IDs。snapshot 可能含库默认值/metadata，不是原始 JSON byte hash；generation config 加载失败也不能仅凭 `OSError` 区分文件缺失、认证或网络问题。完整 commit hash 才是预期的不可变输入；脚本参数必填不等于 branch/tag 不会移动。脚本不加载权重或 processor，也不证明有效 runtime defaults、许可、质量与支持矩阵。

## 中文与多语言 tokenizer

中文“字符数”、UTF-8 字节数与 token 数不是固定比例。数字、空格、标点、繁简体、罕见字、中英混排、代码和 JSON 会显著改变切分。成本和上下文预算必须用目标 tokenizer 实测。

至少建立以下 tokenizer 回归集：

- 中文新闻、口语、古文与繁体；
- 产品型号、日期、金额、电话号码和长数字；
- 中英混合术语、URL、Markdown、LaTeX；
- Python/SQL/JSON，尤其缩进、引号和转义；
- emoji、组合字符、罕见 Unicode 与恶意控制字符；
- system/user/assistant/tool 多轮模板。

报告 `tokens/汉字`、`tokens/byte`、p50/p95 序列长度和截断率。不能只用一段中文示例断言“某 tokenizer 更适合中文”；序列更短也不自动等于任务质量更高。

## 架构检查：dense、GQA 与 MoE

公开 Qwen 文本 checkpoint 常属于 decoder-only causal Transformer，但具体 norm、MLP、RoPE、bias、weight tying 和 attention 结构随代际/版本变化。像 Llama 一样，先从 config 计算 KV Cache；`num_key_value_heads` 不等于 `num_attention_heads` 时通常意味着 GQA/MQA 形式。

MoE 版本需要同时报告：

- 总参数与每 token 激活参数；
- routed/shared experts（若该 checkpoint 存在）；
- 每 token 选择专家数与路由规则；
- router/负载均衡损失；
- 单卡是否需要容纳全部专家权重；
- expert parallel 的 all-to-all 与负载分布。

“激活参数像小模型”只描述部分计算，不代表加载显存、通信或实际 tokens/s 与同规模 dense 模型相同。

仓库的 `moe_routing.py` 是通用 CPU 教学 fixture：固定 top-k/capacity/tie-break，区分 assignment drop 与整 token drop，并执行线性 expert combine。它不读取任何 Qwen config/weight，也没有实现特定 Qwen MoE 的 routed/shared expert、auxiliary loss 或 expert-parallel kernel；学习具体 checkpoint 时必须重新从固定 revision 的 config 与官方实现建立契约。

同目录 `configs/moe-gqa.example.json` 也只是 `authored_moe_gqa` 公式 fixture：它证明本仓库检查器能同时报告 MoE markers，并仍按显式标准 GQA 字段计算理想 K/V payload；它不对应任何 Qwen 代际，不能证明专家总数、激活参数或 routing 语义。若出现已知 MLA marker，检查器会 fail closed；“没有命中当前 marker 列表”也不等于已经证明该架构不是其他 latent/proprietary attention。

## Base、Instruct 与思考模式

Base checkpoint 适合续写、继续预训练和自定义后训练；Instruct checkpoint 依赖官方 chat template。对话模板通常编码 role、turn boundary、generation prompt、tool schema/result 和停止 token。

部分公开 Qwen checkpoint 或模板支持可配置的思考/非思考行为。是否存在、怎样开启、reasoning 文本是否暴露以及对应 token 预算，必须查看具体 model card 和 tokenizer template。不要跨版本复制参数，也不要通过脆弱的字符串删除“思考标签”；parser 应基于该版本明确协议，并保留原始输出用于审计。

思考模式比较要固定：最大输出、实际 token、采样、候选数、验证器、wall time 和任务成功率。更长轨迹不保证答案更正确。

## Tool calling 与结构化输出

工具 schema 往往不是简单附在普通用户文本后，而是由 chat template 序列化成训练见过的控制格式。正确流程：

1. 用 checkpoint tokenizer 渲染 tools、messages 和 tool result；
2. 检查 token ids、generation prompt 与 stop tokens；
3. 对模型输出做版本化 parser；
4. schema/范围/资源归属校验；
5. 外部 Agent runtime 执行 ACL、审批、幂等和审计。

模型只提出调用。即使官方示例能自动执行工具，生产系统也不能把该便捷循环当作授权层。

结构化输出评测包含语法合法率、schema 合法率、字段语义、枚举/单位、未知字段、恶意字符串和越权资源 id。不要只统计 JSON 可解析率。

## 中文 RAG 实践

Qwen 的中文生成能力不能补偿检索错误或 ACL 泄漏。至少比较：

| 组件 | 必测基线 |
|---|---|
| sparse | 字符 n-gram、中文分词 BM25 或兼顾英文 token 的 BM25 |
| dense | 多语言/中文 embedding，并固定 query/document 前缀 |
| reranker | cross-encoder 与无重排基线 |
| chunking | 中文标题/段落/表格结构，不只固定字符数 |
| query | 精确实体、型号、英文缩写、中英混合、错别字 |
| generation | 有答案、无答案、冲突、过期证据与引用 |
| security | tenant/ACL 过滤、间接提示注入和敏感字段 |

更换 Qwen generation checkpoint 时保持检索结果固定，先测生成器；更换 embedding/reranker 时保存候选列表，先测召回与排序。否则无法归因提升来自哪一层。

## 单卡 LoRA/QLoRA

LoRA target modules 由实际 module names 决定，不能从 Llama 教程机械复制。MoE checkpoint 还要决定是否训练 shared/routed expert、router 或普通 attention/MLP 投影；不同选择的训练参数、通信和过拟合风险不同。

单卡实验顺序：

1. 固定 revision 与许可，检查 tokenizer/template；
2. 运行 Base/Instruct Prompt 基线和 RAG 基线；
3. micro-batch 1、短序列、少量样本过拟合，检查 labels；
4. LoRA rank/target 消融；
5. 保存/reload/merge 数值回归；
6. 领域、中文长尾、通用、安全和格式切片评测；
7. 记录峰值显存、训练 token、时间和 adapter 大小。

云端 Qwen API 与本地开放权重分别评测：它们可能使用不同模型、模板、路由、量化与内容策略。

## 多模态版本

视觉/音频版本通常需要 `AutoProcessor` 或专用 processor，不是把图片路径塞进文本 tokenizer。输入要记录媒体 MIME、尺寸/时长、采样、压缩、tile/patch 设置和 processor revision。

中文多模态评测至少覆盖 OCR 小字、表格/图表数值、空间关系、文档布局、视频时间定位、音频转写和文本线索遮蔽。图像中的文字是低信任输入，不能提升为 system 指令或工具授权。

## 可运行实验

选择一个小尺寸 Qwen Instruct checkpoint 并固定 commit：

1. 运行 `inspect_checkpoint.py` 导出 config/template；
2. 对中文、英文、数字、代码各 100 条统计 token 长度；
3. 渲染普通对话、system、tools、tool result，保存 token fixture；
4. 用 Transformers 跑 greedy 基线和结构化输出小集；
5. 比较 BM25、dense、hybrid、reranker 的中文 RAG 指标；
6. 在显存允许时做短序列 LoRA，并和 Prompt/RAG 基线配对评测。

实验报告必须能回答“哪个 checkpoint、哪个模板、哪个 tokenizer、哪组数据、什么硬件与预算”。

## 常见错误

- 用“Qwen”一个词代替代际、尺寸、Base/Instruct、模态和 revision；
- 用字符数估 token 预算；
- 把云 API 行为当作本地 checkpoint 行为；
- 从其他架构复制 LoRA target modules；
- 只看 MoE 激活参数而忽略总权重与通信；
- 手写工具模板或用字符串切 reasoning/tool 输出；
- 中文总体分数上升，却不检查数字、实体、中英混合和权限切片。

## 面试追问

1. 中文 tokenizer 效率怎样影响成本、batch 和有效上下文？
2. dense/MoE 的总参数、激活参数和实际显存怎样公平比较？
3. tool template 变化为什么会破坏调用，即使 messages JSON 没变？
4. 中文 RAG 为什么仍需要 BM25 和精确实体召回？
5. 云 API 与本地 Qwen 如何做质量—延迟—成本对比？
6. 思考/非思考模式怎样在同预算下评测？
7. 多模态输入为何需要独立 processor 与安全边界？

## 一手资料

- Qwen Team，[Qwen3 official repository](https://github.com/QwenLM/Qwen3)，公开模型卡、部署和模板入口；具体版本以所选 revision 为准。
- Qwen Team，[Qwen documentation](https://qwen.readthedocs.io/)，官方使用与部署文档。
- Hugging Face，[Chat templates](https://huggingface.co/docs/transformers/en/chat_templating)，模板渲染与 generation mask。
- 目标 checkpoint 的 config、tokenizer、model card 与 license；它们高于跨代概述。
