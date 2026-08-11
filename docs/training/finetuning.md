# 微调与参数高效训练

本章解释方法谱系与选择。实际训练前继续阅读[SFT 数据、模板与训练闭环](sft-data-pipeline.md)和[LoRA、QLoRA 与单卡工程](peft-qlora-engineering.md)。

## 微调解决什么

预训练教模型延续语料分布；监督微调（SFT）用“输入 → 理想输出”示例教会任务格式、对话行为、领域风格和工具协议。它主要改变行为分布，不保证注入的每条事实都可靠、可更新。

## 数据格式

对话样本应明确 system、user、assistant、tool 角色，使用与部署一致的 chat template。常见策略只在 assistant 回复上计算损失，避免让模型学习生成用户提示；也可对全部 token 训练，但含义不同。

质量通常比数量重要。数据应覆盖正常请求、边界条件、澄清、拒答、工具失败和格式修复。近重复模板会虚增样本量。划分测试集时按来源/任务/用户隔离。

本仓库的 SFT reference core 用严格 JSONL、exact/group 跨 split gate、有序 train/combined binding、显式 normalization/view/n-gram/threshold 的 lexical Jaccard candidate gate，以及默认拒绝的 source/license/purpose/expiry registry 与有限敏感候选扫描，把这部分变成可执行基线。独立审计进程把结果收敛为不含 held-out 原文的严格 readiness artifact，trainer 只重读 train 并核对 ordered identity；这减少了测试集暴露面，但未签名 hash 不是来源认证。lexical candidate 不等于语义重复，registry allow 不是法律意见，有限扫描未命中也不证明无 PII/secret；它仍不覆盖 consent、完整许可审查、embedding/翻译级污染或真实域 detector calibration。两个训练入口另有目标 tokenizer-reported assistant-mask 与截断 preflight，但它也不等于独立 mask 语义或最终 labels 验证。运行方法和证据边界见[SFT 数据、模板与训练闭环](sft-data-pipeline.md)。

偏好训练采用独立 artifact：combined 文件保留 A/B presentation、tie/invalid 和 held-out split，DPO trainer 文件必须逐记录等于其中有序的 binary train subset。审计进程另绑定字符 n-gram candidate policy、source registry 和有限敏感候选扫描；这些是待复核 gate，不是语义无污染、法律许可或无敏感信息证明。严格 readiness 不含 held-out 原文；目标 tokenizer 加载后，preflight 要求 prompt token IDs 同时是 prompt+chosen/rejected 的精确前缀，并拒绝空/同 token completion 与会触发 `max_length` 的 pair。这个 tokenization gate 防止模板切片和静默截断改变 DPO loss 边界，但不验证标注者、position bias 或对齐效果；具体入口在 `projects/single-gpu-finetuning/`，项目状态见[工程项目索引](../practice/project-index.md)。

## 全参数微调

更新全部权重，容量最大，适合数据足、预算高或需要显著领域迁移的场景。代价是显存大、每任务保存完整模型，并有灾难性遗忘风险。优化器状态常比权重本身占更多显存。

## LoRA

冻结原权重 \(W\)，学习低秩增量：

\[
W'=W+\Delta W=W+\frac{\alpha}{r}BA
\]

其中 \(A\in\mathbb{R}^{r\times d_{in}}\)，\(B\in\mathbb{R}^{d_{out}\times r}\)，且 \(r\ll d\)。可作用于 Q/K/V/O 投影、MLP 或更多线性层。

关键参数：秩 \(r\)、缩放 \(\alpha\)、dropout、target modules、学习率。更高秩不必然更好；应看任务复杂度和数据量。adapter 可在推理时合并进权重，也可动态加载，但多 adapter 服务有调度与显存代价。

## QLoRA

把冻结的基座权重量化（常见 4-bit）以节省显存，计算时反量化，并以较高精度训练 LoRA 参数。它不是把所有训练都变成 4-bit：adapter、梯度、优化器和部分计算仍使用更高精度。量化误差、计算 dtype、双重量化和分页优化器实现都会影响结果。

## 其他 PEFT

- Adapter：在层间插入小型可训练模块。
- Prefix/Prompt tuning：学习连续虚拟 token 或 K/V 前缀。
- BitFit：只训练 bias。
- IA³ 等：学习通道缩放。

选择考虑质量、训练/服务复杂度、能否合并、任务数和切换频率。

## 超参数与训练

微调通常比预训练用更小学习率和更少步骤。监控训练/验证 loss、任务指标、格式合法率、通用能力和安全回归。按有效 token 数加权 loss，避免大量短样本或 padding 扭曲结果。sequence packing 可提效，但要正确隔离样本边界。

## Checkpoint 与精确恢复

“能重新加载权重”不等于“能从中断处继续同一条训练轨迹”。可恢复训练 checkpoint 至少要把模型参数、optimizer state、scheduler/global step、混合精度 scaler、所有实际使用的 RNG、sampler/shuffle 状态、数据 cursor 与数据身份放在同一个一致性边界；gradient accumulation 还要保存窗口内梯度与 accumulation position，DataLoader worker/prefetch、分布式 sampler、FSDP/ZeRO shard 也各有额外状态。若某项未使用，可以明确省略；若使用了却没保存，便不能声称 exact resume。

仓库的 `minigpt_training_checkpoint.py` 提供一个范围刻意收紧的 CPU control：pickle-free strict artifact 保存 FP32 MiniGPT 全部参数、单 param-group AdamW 的 per-parameter step/一阶/二阶矩、线性 LR 进度、Byte-BPE/config/tied-weight contract、数据 shape+content fingerprint、shuffle permutation/cursor/epoch、独立 data-generator RNG 与 dropout 使用的 Torch CPU RNG。它只允许在梯度已清空的 optimizer-step boundary 保存。7×5 token 数据、batch 2、dropout 0.2 的固定实验中，6 次更新在第 3 次后保存/恢复；恢复段的 batch、epoch、LR、loss 以及最终模型/optimizer/stream/RNG 与不中断运行逐位一致。

这个结论只覆盖当前 CPU、PyTorch、FP32、MiniGPT architecture revision 和训练契约。当前 AdamW step 是 FP32 tensor，所以 reference 把总 update 限制在 $2^{24}$ 以内，确保整数仍可精确表示。artifact 不嵌入数据 payload，只用 fingerprint 拒绝数据漂移；也没有保存 Python、NumPy 或 CUDA RNG，因为该 control 没有使用它们。它不支持 AMP scaler、gradient accumulation、DataLoader worker/prefetch、distributed/sharded state、目标 Llama/Qwen checkpoint 或 CUDA。无密钥 SHA-256 不认证来源，exclusive-create + file `fsync` 不证明断电原子发布；固定 loss 也不单调下降，因此实验不证明收敛或模型质量。

## 何时不要微调

- 只是需要最新事实：优先 RAG 或工具。
- 需求可由少量示例稳定表达：先试 Prompt。
- 没有可靠评测集：先建立基线与失败分类。
- 数据少且包含秘密：评估隐私、记忆和访问控制。
- 只想“减少幻觉”：单纯 SFT 通常不够，需要证据、验证与拒答机制。

## 实验设计

至少比较：基座 + Prompt、RAG、LoRA、全参微调（若预算允许）。保持评测集和推理配置一致，记录训练/推理成本。检查基座能力回归与未见任务泛化，不要只看训练同分布数据。

## 自测

1. 为什么教模型最新产品价格通常不应首选微调？
2. LoRA 的秩影响参数量和表达能力的方式是什么？
3. QLoRA 中哪些部分通常不是 4-bit？
