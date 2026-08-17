# 微调与参数高效训练

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要在 Prompt、RAG、全参数训练和 PEFT 之间做选择的工程师。
- **先修**：Transformer、tokenization、训练/验证切分和基础优化。
- **首次阅读**：判断是否需要训练 → 数据与 labels → LoRA/QLoRA → 评测 → 发布。
- **完成信号**：能说明为何选择某种干预，并设计带基线、held-out 与回归切片的实验。
- **卡住时**：先读 [SFT 数据流水线](sft-data-pipeline.md)，不要从训练命令开始。

</div>

微调改变模型在给定输入下的行为分布。它适合学习稳定任务、输出格式、领域表达和工具协议，但不是知识库、权限系统、事实验证器或服务优化器。

## 先判断问题在哪一层

“效果不好”还不是训练理由。先把失败分类，再选择最小干预：

| 主要失败 | 首选基线 | 微调可能有用的部分 | 微调不能替代 |
|---|---|---|---|
| 缺少易变或私有事实 | RAG、数据库、工具 | 学习怎样引用和使用证据 | 新鲜数据、ACL、引用核验 |
| JSON 或固定格式不稳定 | schema、constrained decoding、few-shot | 降低格式错误率 | 解析、业务校验、执行授权 |
| 工具选择不稳定 | typed tool contract、状态机 | 学习 proposal pattern | 参数校验、幂等、人工确认 |
| 稳定领域任务较弱 | zero/few-shot 基线 | 学习任务映射或领域表达 | held-out 评测、数据许可 |
| 偏好或拒答边界不稳定 | rubric、SFT 基线 | 改变被测分布上的行为概率 | 外部 policy、事实 verifier |
| 延迟或显存过高 | 小模型、量化、路由、batching | 蒸馏后恢复部分质量 | runtime profiling 与容量验收 |

若错误来自检索漏召回、权限过滤、Prompt 拼接或服务超时，继续增加训练样本是在错误层修问题。

## SFT 学到了什么

监督微调使用“输入 → 目标输出”示例最小化 next-token loss。对话训练常只监督 assistant token：prompt 仍参与 attention，却用 loss mask 排除，不要求模型复述用户消息。

设第 \(i\) 个 token 的标签为 \(y_i\)，监督位置集合为 \(S\)，则常见目标为：

\[
\mathcal{L}_{\text{SFT}}
=-\frac{1}{|S|}\sum_{i\in S}\log p_\theta(y_i\mid x_{<i}).
\]

关键不是“样本里有 assistant 字段”，而是 template、tokenization、shift、mask 和 truncation 之后，正确 token 真正进入了 \(S\)。训练前应打印少量样本的 token、角色边界和最终 labels。

## 数据决定了训练上限

### 划分与泄漏

按来源、任务模板、用户或时间划分，而不是先随机打散近重复样本。检查 exact duplicate 只是起点；改写、翻译、共享模板和同一文档切片也会让验证集过于乐观。

保留独立 test 集，不用它挑 rank、checkpoint、Prompt 或停止点。若迭代中反复查看 test，它就已经变成开发集，需要新的最终评测。

### 对话模板与 labels

训练与部署应使用同一角色语义、special token 和 chat template。至少检查：

- system、user、assistant、tool 的顺序是否合法；
- assistant-only mask 是否覆盖预期回复；
- EOS 是被监督、仅用作停止，还是两者兼有；
- prompt 太长时从哪一侧截断；
- tool calls 与 tool results 是否保持结构而非意外字符串化；
- padding token 是否被 loss mask 排除。

详细的 schema、模板、mask 和数据审计流程见 [SFT 数据流水线](sft-data-pipeline.md)。

### Packing 与有效 token

Sequence packing 可以减少 padding，但样本边界、position、attention mask 与 loss mask 必须一起设计。仅插入 EOS 不一定阻止后一条样本读取前一条样本。

比较不同 batch 时，用有效监督 token 作为 loss 分母。若两个 micro-batch 分别有 \(n_1\) 和 \(n_2\) 个监督 token，直接平均两个 local mean 会给短批过高权重；应累加 loss sum，再除以整个 update window 的 \(n_1+n_2\)。

## 选择训练方法

| 方法 | 更新内容 | 主要优势 | 主要代价 |
|---|---|---|---|
| 全参数微调 | 全部权重 | 表达容量最大 | 显存、存储和遗忘风险最高 |
| LoRA | 低秩增量 | 训练与多任务存储较小 | target modules、rank 和服务切换需验证 |
| QLoRA | 量化冻结基座 + LoRA | 进一步降低基座常驻显存 | 量化误差和 runtime 兼容更复杂 |
| Prefix/Prompt tuning | 连续前缀参数 | 参数极少 | 能力上限和部署支持依任务而变 |
| Adapter/IA³ | 小型模块或通道缩放 | 模块化 | 插入位置与服务支持不统一 |

先用最简单方法建立基线。参数更少不自动意味着训练更快，文件更小也不等于峰值显存更低。

## LoRA

对冻结权重 \(W\in\mathbb{R}^{d_{out}\times d_{in}}\)，LoRA 学习低秩增量：

\[
W' = W + \Delta W
   = W + \frac{\alpha}{r}BA,
\]

其中 \(A\in\mathbb{R}^{r\times d_{in}}\)、\(B\in\mathbb{R}^{d_{out}\times r}\)，且 \(r\ll\min(d_{in},d_{out})\)。可训练参数约为 \(r(d_{in}+d_{out})\)。

需要一起报告：

- target modules，而不只写“用了 LoRA”；
- rank \(r\)、缩放 \(\alpha\)、dropout 与初始化；
- 可训练参数数和占比；
- adapter 是否动态加载或 merge；
- base、tokenizer、template 与 adapter 的完整身份。

Rank 越高表达容量越大，但也更耗显存并更容易拟合噪声。应在固定预算和同一 held-out 集上比较，而不是按 train loss 选择。

## QLoRA

QLoRA 将冻结基座权重量化，forward 时按 runtime 规则反量化，并以较高精度训练 LoRA 参数。它不意味着“训练全程都是 4-bit”：adapter、梯度、优化器状态、激活和部分算子仍使用更高精度。

实验中明确区分：权重存储 dtype、计算 dtype、量化 group/zero point、double quantization、optimizer、activation checkpointing 和实际峰值显存。某个 adapter 文件只有几 MB，不能用来推断训练显存或服务延迟。

具体的单卡预算、PEFT 导出与验证见 [LoRA、QLoRA 与单卡工程](peft-qlora-engineering.md)。

## 训练预算与稳定性

一次可比较的训练至少固定：

- 模型、tokenizer、template 与数据版本；
- 最大长度、有效 token batch、accumulation 和 packing；
- optimizer、学习率、warmup、weight decay 和 clipping；
- 精度、量化、gradient checkpointing 与硬件；
- seed、最大 token/step/时间预算和 early stopping 规则。

同时记录 train/validation loss、任务指标、gradient/overflow、吞吐和峰值显存。Loss 突降先检查 label leakage、重复 shift、padding 与 mask；loss 不降再检查监督 token 是否为空、学习率、冻结范围和截断。

单次训练无法说明随机性。高风险比较至少重复多个 seed，报告原始结果与离散程度；预算不足时，应明确只有一次探索性运行。

## 怎样证明训练有用

“训练跑通”至少拆成五个问题：

| 层 | 关键问题 | 合适证据 |
|---|---|---|
| 数据 | 实际监督了哪些 token？ | split、template、token IDs、mask、截断样例 |
| 机制 | 声明的参数真的更新了吗？ | trainable/frozen 清单、finite gradient、optimizer step |
| 目标 | 未用于该步更新的数据是否改善？ | validation 曲线、预定义停止指标 |
| 行为 | 部署式生成是否更好？ | held-out 任务、格式、通用能力和安全切片 |
| 发布 | 目标 runtime 能否加载和回滚？ | 独立重载、兼容性、容量、canary 与 rollback |

使用同一 case、预处理和 decoding 比较 base + Prompt、RAG（若适用）、LoRA 与其他候选。报告 effect size 和失败样例，而不只报“提升百分比”。同 batch loss 下降只能证明优化目标在当前 batch 上变化，不能代替 held-out 行为。

## Checkpoint 与精确恢复

能加载模型参数，不等于能继续同一条训练轨迹。若训练实际使用了某项状态，checkpoint 就必须保存或明确回退：

| 状态 | 遗漏后的典型后果 |
|---|---|
| model / adapter | 参数回退或错配 base |
| optimizer | moments 与 step 丢失，下一次更新不同 |
| scheduler / global step | 学习率轨迹漂移 |
| GradScaler | overflow 判断与 skip/update 不同 |
| Python/NumPy/Torch/CUDA RNG | dropout、增强或采样序列变化 |
| sampler、permutation、cursor | 漏样本、重样本或顺序变化 |
| accumulation position 与 `.grad` | 半个 update window 被丢弃或重复 |
| distributed/sharded state | rank 间参数、optimizer 或数据进度不一致 |

常见策略有两种：

1. **只在 optimizer commit boundary 保存**：保证 gradients 已清空；崩溃后从最近提交点重放，接受 at-least-once 数据读取。
2. **保存窗口中间态**：同时保存 pending sample identity、分母、gradients、相关 RNG 和 cursor，恢复更快但契约更复杂。

DataLoader 的 sampler-emitted、main-loop-consumed 和 optimizer-committed 不是同一个 cursor；prefetch 会放大差异。先定义哪条进度线是恢复边界，再做真正退出进程后的 uninterrupted/split 对照。只比较最终 loss“差不多”不足以证明 exact resume。

## 导出、发布与回滚

训练 checkpoint 面向继续训练；服务 bundle 面向不可变加载。后者通常还需要 base、tokenizer、chat template、adapter 或 merged weights、generation config、runtime contract 和安全/质量结果。

Merge、量化或格式转换会产生新的部署对象，应重新验证：

- 独立进程能否加载；
- logits 或任务行为是否在预定容差内；
- tokenizer/template 是否匹配；
- 目标硬件上的延迟、吞吐和显存；
- 安全与通用能力是否回归；
- 上一版本能否完整回滚。

文件 hash 可以发现意外漂移，但无密钥 hash 不认证来源。发布判断仍需访问控制、来源管理和可审阅的评测记录。

## 一个最小实验

1. 选择一个有自动评分和人工可审阅样例的小任务。
2. 固定 100–500 条 train、独立 validation/test 和三类失败切片。
3. 建立 base zero/few-shot 基线。
4. 打印 3 条最终 token/label，确认监督边界。
5. 训练两个 rank 或学习率配置，预算相同。
6. 用固定 decoding 比较 held-out 质量、格式和通用回归。
7. 独立重载最佳 adapter，并保存最差样例。

项目代码与运行入口见 [Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md)。项目中的控制脚本用于检查某个机制，不应把固定录制数字当作本实验的学习目标。

## 何时不要微调

- 需要的是最新事实：优先 RAG 或工具。
- 少量示例已经稳定解决：先保留简单 Prompt。
- 没有可靠评测集：先建立基线和失败分类。
- 数据来源、许可或隐私不清楚：先解决数据治理。
- 只想“减少幻觉”：还需要证据、验证、拒答和权限边界。

## 自测

1. 为什么把 prompt labels 设为 `-100` 不会阻止 response 读取 prompt？
2. 两个 micro-batch 的监督 token 数不同，为什么不能直接平均 local mean loss？
3. LoRA adapter 很小，为什么不能据此推断训练峰值显存？
4. 能重新加载权重，为什么仍不能声称 exact resume？
5. 一次训练 loss 下降后，还需要哪几类 held-out 与发布证据？
