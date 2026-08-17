# 微调与参数高效训练

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要选择 Prompt、RAG、全参或 PEFT 路线的工程师。
- **先修**：Transformer、tokenization、训练/验证切分和基础优化。
- **首次阅读**：按失败类型选方案 → 五层证据 → 数据格式 → LoRA/QLoRA → 发布验收。
- **完成信号**：能说明为何选择某种路线，并给出质量、显存和回归门槛。
- **卡住时**：先读[SFT 数据闭环](sft-data-pipeline.md)，不要直接从训练命令开始。

</div>

本章解释方法谱系与选择。实际训练前继续阅读[SFT 数据、模板与训练闭环](sft-data-pipeline.md)和[LoRA、QLoRA 与单卡工程](peft-qlora-engineering.md)。

## 先按失败类型选择干预

“模型效果不好”不是可执行诊断。先把失败定位到知识、行为、协议、系统或治理层，再决定是否训练：

| 主要失败 | 首选基线 | 微调可以承担什么 | 微调不能替代什么 |
|---|---|---|---|
| 易变、私有且需要引用的事实缺失 | RAG、数据库或工具 | 教模型遵循证据、引用和拒答格式 | 新鲜事实源、ACL、引用核验 |
| JSON、语气或稳定任务格式不一致 | 明确 Prompt、schema、constrained decoding | 用 SFT 降低格式和行为错误率 | 解析器、业务校验、执行授权 |
| 工具选择或参数映射不稳定 | 有限状态机、typed tool contract、失败分类 | 学习稳定的 proposal pattern | 参数验证、幂等、权限和副作用确认 |
| 稳定领域任务映射较弱 | few-shot 与任务基线；必要时比较 DAPT | SFT/LoRA 学习任务行为，DAPT 补领域分布适配 | held-out 证据、来源许可、动态知识更新 |
| 偏好、拒答或风格边界不稳定 | rubric、SFT 与偏好数据审计 | 改变被测分布上的行为概率 | 外部安全 policy、事实 verifier、人工升级 |
| 延迟、显存或单次成本过高 | 小模型、路由、量化、batching | 蒸馏或适配小模型以恢复任务质量 | runtime profiling、容量与 SLO 验收 |

一个系统可以同时使用 RAG 与微调：RAG 提供当前证据，SFT 教模型怎样使用证据。若错误来自检索召回、权限过滤或服务超时，继续增加 SFT 样本通常是在错误层修问题。

## “微调成功”是分层证据

训练链路至少有五层互不替代的命题。应逐层保存证据，而不是用最后一个绿色数字覆盖前面的未知项：

| 层级 | 要回答的问题 | 最小证据 | 仍然不能声称 |
|---|---|---|---|
| 数据与接口 | 实际监督了哪些 token？ | 数据 identity、split、template/token IDs、mask、截断与最终 labels | 许可完备、无语义泄漏、标签正确 |
| 机制执行 | 声明的参数真的更新了吗？ | trainable/frozen 参数账本、finite gradient、optimizer step、基座不变、adapter 独立重载 | objective 改善、收敛或质量提升 |
| 训练目标 | 优化目标在未用于该步更新的数据上改善吗？ | train/validation 曲线、预定义 early-stop 指标、多 seed 或不确定区间 | 真实生成行为或业务价值改善 |
| 行为与回归 | 部署式生成是否在目标切片更好？ | 固定 decoding 的 held-out 任务、格式、通用能力、安全和失败 taxonomy | 分布外泛化、线上因果效果、生产可靠性 |
| 发布与运行 | 目标 runtime 能否安全加载、运营和回滚？ | immutable identity、独立加载/merge/量化回归、容量、canary 与 rollback | 模型永不失败、来源自动可信或组织级合规 |

机制证据只回答训练路径是否按声明执行。同 batch loss 下降不是 held-out 行为证据；反过来，少量样例生成看起来更好，也不能证明 mask、数据隔离和基座 identity 正确。训练 checkpoint 和服务 artifact 是两种不同契约：前者面向继续训练，后者面向不可变加载、推理兼容、发布和回滚。

### 先写可证伪的实验合同

训练前固定：目标失败 taxonomy、主指标与最小实际改善、不可退化切片、Prompt/RAG/base 对照、模型与 tokenizer/template revision、generation config、数据截止时间、seed、token/时间/费用预算和停止规则。训练中不能因为某个 slice 不好看就事后删除它，也不能在 test 上反复挑 rank、checkpoint 或 Prompt。

公平比较至少保持同一评测 case、输入预处理和部署式 decoding；同时报告训练 token、更新数、峰值显存、墙钟、artifact 大小和推理成本。LoRA、QLoRA 与全参训练若使用不同有效 token batch、数据顺序或量化 runtime，应把这些列为共同变化，不能把全部差异归因于 PEFT 方法。

### 发布是一次新的验证

训练目录中可恢复的 optimizer、scheduler、RNG、sampler 和数据 cursor，不等于可服务 artifact 已包含 exact base、tokenizer、chat template、adapter、generation config 与 runtime contract；只保存 adapter 也不等于可精确恢复训练。量化或 merge 会产生新的部署 artifact，需要新的 identity，并重跑 task/safety/latency/内存验证。回滚单元应是完整 bundle，而不是只把 adapter 文件名改回旧值。

## 微调解决什么

预训练教模型延续语料分布；监督微调（SFT）用“输入 → 理想输出”示例教会任务格式、对话行为、领域风格和工具协议。它主要改变行为分布，不保证注入的每条事实都可靠、可更新。

## 数据格式

对话样本应明确 system、user、assistant、tool 角色，使用与部署一致的 chat template。常见策略只在 assistant 回复上计算损失，避免让模型学习生成用户提示；也可对全部 token 训练，但含义不同。

质量通常比数量重要。数据应覆盖正常请求、边界条件、澄清、拒答、工具失败和格式修复。近重复模板会虚增样本量。划分测试集时按来源/任务/用户隔离。

本仓库的 SFT reference core 用严格 JSONL、exact/group 跨 split gate、有序 train/combined binding、显式 normalization/view/n-gram/threshold 的 lexical Jaccard candidate gate，以及默认拒绝的 source/license/purpose/expiry registry 与有限敏感候选扫描，把这部分变成可执行基线。独立审计进程把结果收敛为不含 held-out 原文的严格 readiness artifact，trainer 只重读 train 并核对 ordered identity；这减少了测试集暴露面，但未签名 hash 不是来源认证。lexical candidate 不等于语义重复，registry allow 不是法律意见，有限扫描未命中也不证明无 PII/secret；它仍不覆盖 consent、完整许可审查、embedding/翻译级污染或真实域 detector calibration。两个训练入口另有目标 tokenizer-reported assistant-mask 与截断 preflight，但它也不等于独立 mask 语义或最终 labels 验证。运行方法和证据边界见[SFT 数据、模板与训练闭环](sft-data-pipeline.md)。

偏好训练采用独立 artifact：combined 文件保留 A/B presentation、tie/invalid 和 held-out split，DPO trainer 文件必须逐记录等于其中有序的 binary train subset。审计进程另绑定字符 n-gram candidate policy、source registry 和有限敏感候选扫描；这些是待复核 gate，不是语义无污染、法律许可或无敏感信息证明。严格 readiness 不含 held-out 原文；目标 tokenizer 加载后，preflight 要求 prompt token IDs 同时是 prompt+chosen/rejected 的精确前缀，并拒绝空/同 token completion 与会触发 `max_length` 的 pair。这个 tokenization gate 防止模板切片和静默截断改变 DPO loss 边界，但不验证标注者、position bias 或对齐效果；具体入口在 `projects/single-gpu-finetuning/`，项目状态见[工程项目索引](../practice/project-index.md)。

SFT 另有一条固定目标权重 tool-aware final-label control：原生 Qwen 模板在多轮、并行 tool calls、tool preamble 三条 fixture 上的 assistant mask 全零；审核模板保持 47 / 301 / 200 个 token IDs 相同，在 Arrow 前生成 8 / 51 / 31 个 assistant tokens，并让真实 TRL collator 得到 `[3, 301]`、90 个监督 labels 与 813 个 `-100`。CPU FP32 no-grad loss `1.251716` 证明 labels 被目标模型消费，不证明 backward、optimizer、任意 provider schema、tool 结果真实性、收敛、泛化或数据合法性；详见[SFT 数据闭环](sft-data-pipeline.md#target-qwen-sft-final-label-control)。

## 全参数微调

更新全部权重，容量最大，适合数据足、预算高或需要显著领域迁移的场景。代价是显存大、每任务保存完整模型，并有灾难性遗忘风险。优化器状态常比权重本身占更多显存。

## LoRA

冻结原权重 \(W\)，学习低秩增量：

\[
W'=W+\Delta W=W+\frac{\alpha}{r}BA
\]

其中 \(A\in\mathbb{R}^{r\times d_{in}}\)，\(B\in\mathbb{R}^{d_{out}\times r}\)，且 \(r\ll d\)。可作用于 Q/K/V/O 投影、MLP 或更多线性层。

关键参数：秩 \(r\)、缩放 \(\alpha\)、dropout、target modules、学习率。更高秩不必然更好；应看任务复杂度和数据量。adapter 可在推理时合并进权重，也可动态加载，但多 adapter 服务有调度与显存代价。

仓库的目标 checkpoint control 已在固定 Qwen2.5-0.5B-Instruct revision 上真实执行一次 CPU FP32 assistant-only LoRA backward/AdamW step：24 层 `q_proj/v_proj`、`r=4` 共 270,336 个可训练参数，48 个 B tensors 均由零变为非零；494,032,768 个冻结基座参数的前后 byte fingerprint 相同。保存的标准 PEFT adapter 在重新核对并新加载的同 revision 基座上得到 bit-exact logits。这里的 loss 从约 0.003864 升到 0.584557，所以证据是“目标权重训练/导出链路已执行”，不是“训练有效、收敛或质量改善”。完整报告、负例与边界见[LoRA、QLoRA 与单卡工程](peft-qlora-engineering.md#target-qwen-lora-control)。

同一 checkpoint 还以两条 authored preference pair 执行一次真实 TRL/PEFT DPO step：初始 loss≈`log(2)`，同 batch step 后为 `0.333352`，两条 relative margin 均为正，96 个 LoRA gradient tensors finite；冻结参数、non-adapter state、model/generation config 指纹前后相同。两次 adapter-disabled reference forward 的 replay max-abs drift 为 `0.547077`，因此报告明确区分“reference 身份冻结”和“数值 bitwise replay”。这只补齐目标权重 DPO 机制证据，不是人类偏好、held-out 质量、收敛或安全证据；详见[偏好对齐](alignment.md#target-qwen-dpo-control)。

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

独立的 CPU AMP control 补的是“为什么 scaler 必须保存”，不是扩展上述 MiniGPT 文件格式。真实 FP16 autocast/GradScaler 在三个 overflow window 中观察 scale `8→4→2→1` 且 AdamW state 不变；进程内恢复 scale=1 后下一条边界梯度执行 step=2，漏恢复而使用 fresh scale=8 则 overflow 并跳步。它还以 `24→unscale 3→clip 约 0.5` 对照错误的 `clip 24→unscale 约 0.0625`。这条 control 仍然只是单参数、进程内 replay，不能与 MiniGPT control 事后拼成统一证据。

`checkpoint_resume_control.py` 是第三条真正统一的 CPU control。一个含显式 inverted-dropout mask 的 6 参数线性模型先执行 1 次有限 AdamW update，再连续触发 3 次 overflow；scale `8→4→2→1` 时 optimizer step 保持 1，`StepLR(step_size=2,gamma=0.5)` 的 `last_epoch/step_count` 也保持 `1/2`。第 4 个 attempt 后，phase-1 进程把 model、optimizer、scheduler、GradScaler、Torch CPU/Python RNG、stateful shuffle generator/permutation/cursor/epoch、进度与 dataset hash 写入约 21 KiB checkpoint，然后真正退出。不同 PID 的 resume 进程用 `torch.load(..., weights_only=True)` 重开文件；后 4 个 attempt 的 batch、随机因子、mask、loss、gradient norm、scale、optimizer/scheduler step 与 LR 均和独立 uninterrupted 进程逐项相同，最终 model/optimizer/scheduler/scaler/RNG/data state 的 tensor-byte fingerprint 也完全一致。

负对照分别证明字段不是装饰：overflow 后错误推进 scheduler 会在 optimizer 仍为 step=1 时把 scheduler 推到 `last_epoch=4`；漏 scheduler state 会错过下一次 LR 衰减；漏 scale=1 的 scaler 会让边界 attempt 在 fresh scale=8 下 overflow；漏 Torch/Python RNG 会保持 batch 不变但改变随机因子与 mask；漏 data-stream state 会保持 RNG trace 不变却重放错误 batch。这里“是否执行 optimizer step”由 authored AdamW fixture 的所有 per-parameter `step` 一致递增来观测，不是可直接照搬到任意 optimizer 的通用 API。

统一 resume control 只覆盖当前 PyTorch 2.13.0+cpu、CPU FP16 autocast、单机本地临时文件、custom stateful shuffle 与 zero-grad optimizer boundary。`torch.save` 容器仍基于 pickle；`weights_only=True`、16 MiB pre-load cap、closed top-level schema 与 dataset hash 缩小风险，但不认证来源、不保密，也不把任意不可信 artifact 变安全。same-directory temp + file `fsync` + `os.replace` 已执行，却没有故障注入或目录 `fsync`，因此不证明断电/crash 原子性。实验没有 NumPy/CUDA RNG、真实 `DataLoader` worker/prefetch、gradient accumulation 中间态、DDP/FSDP/ZeRO、目标 LLM/PEFT Trainer、CUDA、收敛、性能或质量证据。

另一个 `ddp_amp_overflow_consensus_control.py` 只补“多个 rank 是否对同一 update 作相同决定”。默认 DDP reduction 前的 rank-local Inf 在当前双进程 CPU/Gloo fixture 中传播到两边，因此两边 AdamW/StepLR 都 skip。若先完成 finite reduction，再在 rank 0 的 `unscale_` 前人为损坏 gradient，两个 rank-local scaler 会分别 skip/update，造成 parameter、optimizer、scheduler、LR 与 scaler 分叉；optimizer 前对 local non-finite flag 做 `all_reduce(MAX)` 可让两边共同跳过。这个 post-reduction 故障是 authored counterfactual，不是 DDP 常态；示例的 `update(new_scale=...)` 只定义共同 scale，growth tracker 与 native overflow transition 不同。它没有 checkpoint、随机层、多参数/bucket、CUDA/NCCL、目标 Trainer 或训练质量证据，也不能与跨 PID resume control 事后拼成“分布式 exact resume 已完成”。

`dataloader_prefetch_resume_control.py` 再把 data cursor 的含义拆开。真实 `DataLoader(num_workers=2,prefetch_factor=2,multiprocessing_context="spawn")` 在训练进程只接收固定 permutation 前 3 条 `[8,3,1]` 时，tracking sampler 已把 cursor 推到 7；已发进 worker queue、但尚未交给训练循环的是 `[7,0,9,4]`。phase-1 写下两种 cursor 后退出：不同 PID 从 **application-consumed cursor=3** 重建 loader 可恢复完整 sample-ID 顺序；若把 sampler emitted cursor=7 当 checkpoint position，则组合序列变成 `[8,3,1,2,6,5]`，静默漏四条。

恢复 sample order 仍不等于恢复 tensor。相同 loader seed 可让独立 phase-1 重放相同 prefix，但 fresh workers 从各自 RNG 序列开头重新出发，resume tail 与 uninterrupted tail 的 worker-local `torch.rand` 最大差约 0.6544。fixture 另用 `(namespace,sample_id)` 派生局部 generator，tail 逐位相同；生产 key 还应按语义加入 dataset/transform revision、epoch/visit 等，否则多 epoch 会永远得到同一增强。这只证明 map-style、batch 1、`in_order=True`、非 persistent CPU workers 的单 epoch sample-delivery 边界；它没有保存 queue payload/worker state，不证明任意随机 transform、IterableDataset、persistent workers、optimizer commit 与 sample consumption 原子、distributed sampler、训练质量或性能，也不能与 model checkpoint control 拼成完整训练 exact resume。

`optimizer_commit_resume_control.py` 随后把第三条 cursor 变成真实随机训练事件。六个独立顶层 PID 的每段都启动两个 spawn workers；CPU Float64 线性模型在 main process 用 seed `20260815` 生成 inverted-Bernoulli mask，再执行 MSE、`SGD(momentum=0.9)`、`StepLR(step_size=2,gamma=0.5)` 与 accumulation steps=2。phase-1 在第三个 microbatch 已 stochastic forward/backward 时，sampler emitted=7、main loop consumed=3，而 optimizer/scheduler 只提交前两条，所以 optimizer-committed cursor=2。当前 8,985-byte base checkpoint 不含 `.grad`，但保存 commit-boundary model、SGD momentum、StepLR 与 Torch CPU RNG。从 2 恢复同时还原 commit-boundary RNG 并重放 sample `1`，最终 ledger、model/optimizer/scheduler/RNG 与 uninterrupted bit-exact、参数最大差 0。

第一个隔离负例加载 sidecar 的正确 crash RNG，却故意漏掉 pending gradients并从 consumed=3 起步，因此 ledger 漏 `1`，但未来 mask tail 与终态 RNG 仍和 baseline 相同。末尾 partial window 被正确重缩放，所以两边 optimizer/StepLR 都是 5 次、终态 LR 都是 `0.0125`，参数最大差仍为 `0.005767858566116724`；差异由漏半窗口而非 RNG shift 或 step 数引起。

同一 control 也真实执行保存半窗口 gradients 的第二种正确协议。当前 7,905-byte sidecar 绑定 base SHA-256、三种 cursor、pending window `[1]`、accumulation position=1、steps/loss divisor=2、两个逐参数 finite Float64 gradient tensors 与 crash-observed Torch RNG；最后发布的 827-byte 严格 canonical JSON manifest 以 closed schema 绑定数据 identity、两个文件的 name/schema/size/hash、sidecar→base digest 和 publication sequence。不同 PID 必须先验证 `publication_state=complete` 与 artifact identities，再让 `torch.load(weights_only=True)` 消费经过同一 hash 复核的 bytes；从 consumed=3 继续的首个完成窗口是 `[1,7]`，最终 ledger、model/optimizer/scheduler/RNG 同样和 uninterrupted bit-exact、参数最大差 0。

第二个隔离负例恢复完整 gradients 与 ledger，却故意沿用 base 的 commit-boundary RNG、不加载 sidecar 的 crash RNG。它仍有完整 sample ledger、5 次 optimizer/StepLR step 和 LR `0.0125`，但 mask offset 错误，终态 RNG 不同，参数最大差为 `0.017878893573032573`。因此 ledger、global step、LR 与 gradient sidecar 完整仍不足以单独证明 exact resume。

因此，若不保存 accumulation position 与未提交 gradients，恢复点必须回退到最近 commit boundary，并允许 at-least-once delivery/replay；若选择 sidecar 协议，则窗口 identity、分母、gradients 和相关 RNG 都是 checkpoint state。父进程把 base-only、payloads-without-manifest、manifest-without-sidecar 与 post-manifest tamper 四种快照送进同一 completeness gate，均在反序列化前拒绝；完整 bundle 通过，而 base-only 仍可独立用于 commit replay。

manifest-last 能识别当前进程故障留下的 incomplete/mismatched bundle，不把三次单文件 temp+file-`fsync`+`os.replace` 变成原子 publication：没有 parent-directory `fsync`、断电/文件系统 fault injection、原子目录切换、远程存储语义或 sample/optimizer 事务。无密钥 hashes 也不认证发布者或阻止整套 bundle 协同替换。业务副作用或不可重复数据源仍需幂等/去重；该 stochastic CPU control 已覆盖 main-process Torch RNG 与 StepLR，但不覆盖 worker/Python/NumPy/CUDA RNG、原生 Dropout/任意随机模型、GradScaler、queue state、多 epoch、distributed/CUDA 或目标 Trainer。

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
