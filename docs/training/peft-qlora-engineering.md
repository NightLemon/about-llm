# LoRA、QLoRA 与单卡工程

参数高效微调（PEFT）减少可训练参数，但不自动减少所有显存，也不保证与全参数微调等价。理解每一块内存和 adapter 的线性代数，才能在消费级 GPU 上做可信实验。

## LoRA 直觉

冻结基座权重 \(W_0\in\mathbb R^{d_{out}\times d_{in}}\)，只训练低秩增量：

\[
y=xW_0^\top + \frac{\alpha}{r}xA^\top B^\top,
\quad A\in\mathbb R^{r\times d_{in}},
\quad B\in\mathbb R^{d_{out}\times r}
\]

参数从 \(d_{out}d_{in}\) 变为 \(r(d_{in}+d_{out})\)。常把 `B` 零初始化，使训练开始时增量为 0，模型函数与基座一致；`A` 随机初始化打破对称。

`alpha/r` 控制增量尺度。rank 增大容量与显存，未必线性提高质量。LoRA dropout 只作用 adapter 路径。推理可保留 adapter 动态切换，也可 merge：

\[
W'=W_0+\frac{\alpha}{r}BA
\]

merge 前后应做 logits/生成数值等价测试。

## Target modules

只调 `q_proj/v_proj` 参数少；加入 `k/o`、MLP `gate/up/down` 容量更大。不同架构命名不同，不能照抄 target list。打印匹配模块和 trainable parameter count，若匹配 0 个要失败。

attention-only 适合轻量风格/指令，all-linear 往往对领域适配更强但显存和过拟合风险增加。用同数据与预算做消融。

Embedding/lm_head 是否训练取决于新增 token 和任务。若扩词表却冻结对应行，新 token 无法良好学习。tie weights 时合并/保存要特别检查。

## QLoRA 做了什么

QLoRA 将冻结基座以 4-bit 存储并在计算时反量化到 BF16/FP16，LoRA adapter 保持高精度训练。经典配置包括 NF4、double quantization、paged optimizer。4-bit 主要压缩 base weights，不是“训练全部都 4-bit”。

NF4 针对近似正态权重分布设计量化 level；double quant 再量化 scale 以节省少量内存。compute dtype 决定矩阵计算稳定性：支持时优先 BF16，FP16 需要关注 overflow/scaler。

量化基座通常不直接更新。`prepare_model_for_kbit_training` 会处理 layer norm、输入梯度和 checkpointing 等细节，版本变化需验证。

## 显存账本

训练峰值包括：

\[
M=M_{base}+M_{adapter}+M_{grad}+M_{optimizer}+M_{activation}+M_{workspace}+M_{runtime}
\]

7B 的 4-bit 原始权重约 3.5 GB，但量化 metadata、dequant workspace、CUDA context、allocator、adapter/gradient/optimizer 和激活会显著增加。序列长度增大时 activation 与 attention 中间量增长；朴素 attention 的 score 对长度可呈平方增长。

仓库 `estimate_qlora_memory` 给出分项一阶估算。它用于排除明显不可行配置，不替代目标 GPU 上的 `torch.cuda.max_memory_allocated/reserved` 实测。reserved 与 allocated 的差可提示碎片/缓存，但不等于泄漏。

## 单卡降级顺序

1. micro-batch 降到 1，用 gradient accumulation 保持有效 batch；
2. gradient checkpointing，以额外前向计算换 activation；
3. 使用兼容的 SDPA/FlashAttention 和 length bucketing；
4. 基于长度分布降低 max sequence，而非任意截断；
5. 减少 target modules/rank；
6. 换更小基座。

每一步都会改变速度、容量或训练语义，必须新建 run 配置。`empty_cache()` 不能释放仍被 tensor 引用的内存，也不会让真实峰值消失。

## Gradient accumulation

有效 batch 相同不代表完全等价：dropout mask、optimizer 更新频率、scheduler step、gradient clipping 和 batch-dependent operation 可能不同。loss 应按 accumulation 正确缩放；最后不足一个 accumulation window 的 batch 要处理。

记录 tokens/update 比 samples/update 更合理，因为样本长度差异大。使用 packing 后尤其如此。

## Gradient checkpointing

前向不保存所有中间激活，反向时重算。它降低 activation 内存但增加计算；与 KV cache 通常不兼容，训练时关闭 `use_cache`。检查模型是否真的启用，测 step time 与峰值，而不是只看配置字段。

## Optimizer

AdamW 对每个可训练参数通常有一阶、二阶状态，可能还有 FP32 master weight。LoRA 参数少，所以可接受。8-bit/paged optimizer 进一步降低或迁移状态，但引入 kernel/分页行为；性能要在目标硬件测。

weight decay 通常不作用 bias/LayerNorm；对 LoRA A/B 的最佳设置需要实验。学习率常比全参数微调高，但依模型、数据和 target modules 变化。

## 分布式与单卡的边界

单卡 QLoRA 是否可行不能只按“多少 B”判断：总参数、量化 metadata、层数/隐藏维度、序列长度、micro-batch、target modules、attention backend 和可用显存都会改变峰值。先用分项估算筛选，再在目标卡做最小 dry-run；模型本身无法容纳时，FSDP/ZeRO/offload 是另一套权衡，CPU/NVMe offload 可能极慢，不能只看“能跑”。

多 GPU 时不要在量化模型上盲目使用普通 data parallel。确认 bitsandbytes、Accelerate、FSDP/DeepSpeed 的兼容矩阵和保存方式。仓库不把未在目标组合运行的配置写成已验证。

## Adapter 保存与合并

保存内容包括 adapter state、PEFT config、base model id/revision、tokenizer revision/template、target modules、训练配置和数据 manifest hash。只写“基于 Llama”无法重现。

动态 adapter 优点是一个基座服务多个任务，缺点是路由、batching 和版本管理复杂。merge 后部署简单，但失去动态切换，量化基座 merge 可能需要先反量化并占用更多 RAM。

合并测试：固定输入，比较 unmerged 与 merged logits；保存/重载再比较；重新跑任务、安全和性能评测。量化部署也重新评测。

仓库 `smoke_peft.py` 已把这条链路落实为随机 tiny GPT-2 CPU control：保存 exact base 与 adapter safetensors，从独立 base 重载 adapter，`safe_merge` 后保存 full weights，再从磁盘重载 merged model。它同时保存 WordLevel tokenizer、special tokens 与 chat template；8-step fixture 的 adapter/merged reload logit error 都是 0，merge error 约 $8.94\times10^{-8}$，chat-template token IDs 在重载前后均为 `[5,7,2,9,2]`，frozen base 参数未变。

仓库 strict verifier 覆盖目录内全部 13 个文件，而非只列三个 weight：canonical manifest 绑定 identity、base/adapter/merged/tokenizer contract、path-sorted file size/SHA-256 和 descriptor-set digest；加载前要求三个 safetensors 可解析、base/merged 完整 config 与 tensor key/dtype/shape signature 一致、每个 target 有 LoRA A/B tensor，并拒绝额外、缺失、symlink、路径穿越、duplicate/non-canonical JSON、资源超限和语义漂移。结构一致仍不证明权重数值正确。PEFT 自身不会自动强制仓库 manifest，base path 或 identity string 也不认证内容；必须把 `verify_peft_export_directory` 放在任何 published-artifact load 之前。Unkeyed digest 可被协同重算，exclusive-create + file `fsync` 不证明目录原子发布；verify 后再由框架按路径打开文件还存在并发替换 TOCTOU，需由不可变发布目录、ACL/lease 或内容寻址句柄补足。Fixture 仍没有训练恢复状态、量化 merge、目标 checkpoint 或 CUDA，所以它证明当前标准 artifact plumbing、完整文件集校验和数值等价，不证明来源、license、目标质量或跨版本/runtime 兼容。

## 训练恢复不是 adapter 保存

adapter 目录通常只面向推理或后续加载，不自动包含 optimizer、scheduler、scaler、RNG、sampler/data cursor 和未完成 accumulation window。要声称训练可恢复，先定义一致性边界，再做两条同起点实验：一条不中断运行到第 `N` 步，另一条在第 `K` 步写盘、终止进程、重载后运行到 `N`；逐步比较 sample identity、LR、loss，并在终点比较 adapter/base 可训练参数、optimizer state 与所有消费过的 RNG。只比较最终 loss 接近不够。

仓库 `minigpt_resume_toy.py` 已在 CPU FP32 MiniGPT + 单组 AdamW 上给出这种 bit-exact control，并验证 loader 构造随机模型不会污染调用进程的 Torch RNG。它没有 LoRA/QLoRA、AMP、accumulation、worker、CUDA 或 shard，因此不能替代目标 PEFT 训练的恢复演练。目标单卡路径至少还应覆盖 adapter 与 `modules_to_save`、bitsandbytes optimizer state、GradScaler（若使用 FP16）、CUDA RNG、sampler/data cursor、scheduler step 和 accumulation 边界；量化基座的 identity 也必须绑定，不能只保存 adapter。

## 多 Adapter

为租户/任务各训 adapter 可隔离更新，但 adapter 数过多会造成运维碎片。避免用 adapter 存频繁变化事实。组合多个 LoRA（加权、串联、融合）不保证线性组合行为，必须测相互干扰。

高敏租户 adapter 仍可能记忆数据；权限隔离不只是隐藏文件名，加载和输出都要控制。

## 常见失败

- `trainable params = 0`：target name 不匹配或全部被冻结。
- loss 不降：mask 全为 -100、template 错、LR/optimizer 问题。
- loss 很低生成差：数据泄漏、teacher forcing 与解码差异、格式不一致。
- NaN：FP16 overflow、异常样本、LR 过大、量化/kernel 不稳定。
- OOM 在评测/保存：generation KV cache、merge 或 optimizer checkpoint 峰值未计入。
- adapter 加载差：base/tokenizer revision 错、漏 modules_to_save。
- merge 后偏差大：dtype/量化顺序、scale 或 tied weights 问题。

## 仓库实验路线

1. `LoRALinear`：验证冻结、零初始化和 merge 代数。
2. `about-llm-sft-data audit`：严格 schema、exact/group 泄漏门禁与 manifest identity。
3. `smoke_peft.py`：随机 tiny GPT 实际训练 PEFT，无网络下载。
4. `smoke_trl_sft.py`：离线贯通真实 template mask、collator assistant-only labels 与 tiny-batch overfit。
5. `about-llm-sft-data prepare-training`：在可读 held-out 的审计进程执行 exact/binding/lexical 与 source-policy/有限敏感候选 gate，生成不含 held-out 原文的 readiness。
6. `train_trl_sft.py --data-preflight-only`：train-only 进程严格重载 readiness 并绑定当前 train，再运行固定 revision、assistant-only loss 的 LoRA SFT。
7. `about-llm-preference-data prepare-training`：把 binary train-only pair 按顺序绑定到完整 preference split artifact，执行 prompt/candidate lexical 与 source/sensitive gate，不把 tie/invalid 强制成 winner。
8. `train_trl_dpo.py --data-preflight-only`：无下载地验证 preference readiness；正式运行在模型权重加载前用目标 tokenizer 阻断 prefix mismatch、空/同 token completion 和截断。
9. `train_qlora.py --estimate-only`：CPU 上做容量计划。
10. `minigpt_resume_toy.py`：CPU FP32、单 AdamW group 的严格 checkpoint 与 bit-exact split-run control。
11. 目标 CUDA：NF4 SFT/偏好 QLoRA dry-run，记录版本、峰值与 token/s，并演练真实 LoRA/QLoRA restart。
12. 完整 run：与 base/prompt/RAG 基线做统一 test。

两个训练入口只把 `messages` 交给 dataset adapter，治理字段留在 audit manifest；这避免随意 metadata schema 干扰 Arrow/trainer 输入。审计侧的 `sft-data-audit.json`、`sft-split-audit.json`、`sft-data-binding.json` 与 `sft-governance-audit.json` 分别绑定训练集、combined split gate、有序精确关系和当时 policy/candidate 决策；trainer 侧只需要当前 train 和最小 `sft-training-readiness.json`，不需要 validation/test 文件权限。readiness 的无密钥 hash 可检出意外漂移但不可认证签发者，governance pass 也不等于法律许可或敏感信息不存在。`sft-template-mask-audit.json` 证明实际目标 tokenizer 返回对齐、非空、二值且未截断的 assistant mask；它仍依赖模板自己的 generation 标注，不独立证明语义边界或最终 collator labels 正确。真实运行先 8 条样本/10 step，再 100 step，再完整数据。每阶段抽样可视化 token/mask/label，并验证梯度、checkpoint 和生成。

DPO 路线使用另一套 `preference-training-readiness.json`，把 binary train 身份、exact/group split、prompt/candidate lexical policy 与 source/sensitive governance 绑定到 combined artifact。它复用同一 source policy，但使用 preference 自己的 dataset/detector/audit 版本；pass 仍不是许可、consent、无敏感信息或语义无重复证明。`preference-tokenization-audit.json` 复现 TRL 0.29 的 prompt 与两侧完整对话 tokenization，并把其 prefix warning 升级为阻断；因为长度超过 `max_length` 时 TRL 会按 `keep_start/keep_end` 截断 tensor，入口选择训练前拒绝而不是静默改变学习信号。LoRA 时 `ref_model=None + peft_config` 让 TRL 通过禁用当前 adapter 回到冻结基座 reference，避免再复制一套基座；QLoRA 仍必须在目标 CUDA/bitsandbytes 组合实测。

## 实验比较

LoRA/QLoRA/全参比较需控制基座、数据、template、有效 token batch、更新步数和评测。QLoRA 可能因量化噪声不同，不应默认等价。报告 trainable/total params、峰值 allocated/reserved、tokens/s、总时长、adapter size、任务/通用/安全指标。

## 面试追问

**LoRA 为什么有效？** 经验上任务适配所需权重更新常位于较低内在维度，低秩增量提供受限但足够的方向；这是有用假设，不保证所有任务/层都低秩。

**QLoRA 为何仍 OOM？** 4-bit 只压缩冻结基座存储；激活、adapter、梯度、optimizer、dequant workspace、KV/generation 和 runtime 仍占高精度内存。

**rank 越大越好吗？** 容量增加也提高显存、训练成本和过拟合；收益依 target module/数据。应做 rank-target 联合消融并看验证/通用回归。

**微调能替代 RAG 吗？** 行为/格式适合微调，动态可追溯知识适合 RAG。两者可组合：SFT 教模型如何使用证据，RAG 提供最新事实。
