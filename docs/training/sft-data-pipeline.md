# SFT 数据、模板与训练闭环

监督微调（Supervised Fine-tuning, SFT）看起来只是 next-token loss，真正困难的是让数据、chat template、loss mask、推理格式和评测保持同一契约。模型经常不是“没学会”，而是训练了错误 token 或部署时换了模板。

## 先判断是否需要微调

把问题分成三类：

- 知识缺失且经常变化：优先 RAG/工具；微调不适合频繁事实更新。
- 行为、格式、风格或领域推理模式：SFT/偏好优化可能合适。
- 基座本身能力不足：少量微调通常无法创造未具备的复杂能力，先换模型或分解任务。

建立 base zero-shot、few-shot、RAG 和 SFT 四个可比较系统。若 prompt/RAG 已达标，微调的部署和回归成本可能不值得。

## 数据契约

每条样本至少有：

~~~json
{
  "id": "case-17",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "expert-authored",
  "license": "internal-approved",
  "task": "extract",
  "language": "zh",
  "risk": "normal",
  "group_id": "customer-42",
  "split": "train",
  "metadata": {"annotation_batch": "2026-08-a"}
}
~~~

仓库的 `about-llm.sft-jsonl.v1` 契约要求除可选 `metadata` 外的上述字段全部存在，并拒绝未知字段、重复 JSON key、`NaN`/`Infinity`、空字符串、无配对 Unicode surrogate 和不合法对话顺序。system 只能可选地出现在开头，第一个非 system 消息必须是 user，最后一条必须是 assistant，确保至少有明确监督目标。

`id/source/license` 用于治理，`task/language/risk` 用于切片，`group_id` 用于防泄漏划分，`split` 只能是 `train/validation/test`。不要只保存渲染后的长字符串；原始结构让模板可升级和检查。审计进程要求全为 train 的单独文件和包含三种 split 的 combined artifact，并验证前者按顺序逐记录等于后者的 train 子集；trainer 只接收 train 文件和审计进程签发的 readiness artifact。这样可以让训练身份绑定跨 split 审计，同时不把 validation/test 原文访问权交给训练进程。

## 数据来源与许可

专家撰写质量高但贵；历史日志真实但含隐私、旧流程和错误答案；teacher model 合成覆盖广但会复制其偏差和措辞。记录来源比例并分层评测。合成数据必须经过规则、去重、难度和人工抽样，不能因为生成了百万条就认为有信息量。

确认数据许可、用户同意、保留期限和模型供应商政策。PII/secret 扫描只是辅助；高敏字段优先在数据进入训练区前删除或替换。删除请求需要能从 source id 追到训练 shard/checkpoint 和后续策略。

## 清洗与去重

基础检查：role 顺序、空内容、非法 Unicode、过长样本、模板残片、HTML/代码完整性、明显 secret。exact hash 去完全重复；MinHash/LSH 或 embedding 发现近重复。

去重的关键是跨 split：同一用户、工单 thread、文档模板或同一合成 seed 的变体不能散到 train/test。先按 group/source 分组再划分，而不是逐行随机。

不要过度清洗掉困难和失败样本。保留可学习的边界案例、拒答、澄清和工具错误；只删除无法确定目标或违反政策的数据。

### 可执行 exact gate

仓库提供不依赖模型下载的严格审计：

~~~powershell
python -m about_llm.finetuning_cli audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --require-splits train,validation,test --output outputs/sft-split-audit.json
~~~

门禁检查 required split、重复 id、规范序列化后完全相同的 `messages`、跨 split `group_id` 和跨 split exact content。输出同时给有序数据 fingerprint、保留重复次数但忽略行顺序的 fingerprint，以及绑定契约版本、split policy 和 gate 规则的 manifest fingerprint。字符统计单位是 Unicode code point，**不是 tokenizer token**。

这个 exact gate 故意不把范围说大：它不判断 lexical/semantic near duplicate，也不判断许可是否合法，不扫描 PII/secret，也没有运行目标 tokenizer、chat template 或 assistant mask。空白、标点或等义改写不同就不会被 exact hash 捕获。hash 只证明显式 canonical 字段在该序列化规则下的身份，不证明数据正确、安全或可合法使用。

### 可执行 source policy 与敏感候选 gate

`license` 字符串本身不是授权证据。仓库提供严格 source registry：每条规则精确绑定 `source + license`、`allow/deny`、允许的 `training/evaluation` purpose、evidence reference 与可选 expiry；未知组合默认拒绝，policy 的 `reviewed_at` 不能晚于显式审计时间。`risk` 也必须属于 policy 声明的标签集合，避免未知标签被当作普通样本。

~~~powershell
python -m about_llm.finetuning_cli governance-audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --policy projects/single-gpu-finetuning/governance-policy.example.json --evaluated-at 2026-08-06T12:00:00Z --output outputs/sft-governance-audit.json
~~~

`--evaluated-at` 必须使用固定 UTC 秒级时间，不能偷偷读取 wall clock；这样 expiry 决策可重放。示例 policy 只说明仓库自写 fixture 在该内部规则下被允许，不是外部律师意见。真实 policy 的 `evidence_ref` 应指向可访问的合同/许可快照、负责人和审批记录，而不是再次写一个 license 名称。

同一 gate 运行有限、高精度优先的候选扫描：email、私钥 header、若干 key/token/JWT 形态和通过 Luhn 的 card-like number。报告只保留 record、message、Unicode code-point span、detector 和绑定整条 record identity 的 candidate fingerprint，不保存命中原文。候选默认阻断；人工接受必须以 reviewer/rationale/evidence 绑定精确 candidate fingerprint，数据一改或 span 移动就需重审，policy 中未使用的陈旧 exception 也会阻断。

这不是“PII/secret scanner 完成”：它不识别人名、地址、电话、医疗/生物特征、上下文敏感推断或未知 token 格式，也没有真实域 precision/recall 校准。无候选不证明无敏感信息；接受 exception 只记录一次人工处置，不证明内容安全或合法。即使报告不含命中原文，record identity、位置和 source metadata 仍可能敏感，必须限制访问。

### 可执行 lexical candidate gate

小中型审计 artifact 可继续运行透明的全对比较：

~~~powershell
python -m about_llm.finetuning_cli near-audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --profile nfc_whitespace --ngram-size 5 --threshold 0.9 --output outputs/sft-near-duplicate-audit.json
~~~

它在 full conversation、全部 user content、全部 assistant content 三个 view 上分别构造 Unicode code point 字符 n-gram **集合**，计算精确 Jaccard：

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

报告给每个候选的分子、分母、两侧 shingle 数、record/split、比较总数、策略和 manifest identity。默认只比较跨 split pair；`--include-within-split` 才加入同 split。短于 n-gram size 的文本整体作为一个 shingle，因此大 n 对短文本会退化为接近 exact match。集合 Jaccard 也会丢失重复频次。

profile 必须显式选择。`nfc_whitespace` 只做 NFC 与空白折叠；`nfkc_casefold_whitespace` 还会合并兼容字符、宽度和大小写，召回更多候选但可能破坏代码、标识符、表格或格式任务的含义。`prepare-training` 要求显式 profile，把 near manifest 与 exact train/combined binding 绑定；存在未处理 candidate 时 readiness 为失败，trainer fail closed。

这里的“candidate”不是人工确认的 duplicate。阈值必须按语言、长度和任务用人工样本校准；无 finding 也不能证明没有改写、翻译、答案片段或 embedding-level 污染。Readiness 的 fail-closed gate 仍使用全对精确比较，复杂度为 \(O(N^2)\)；不能只把近似 LSH 的“未命中”当作无污染证明。

仓库另有 deterministic MinHash/LSH candidate core 和 `minhash_lsh_toy.py`：稳定 shingle hash、seeded affine MinHash、band collision、exact candidate recheck、item/report fingerprint，以及可选 exhaustive recall audit。固定 5-item fixture 将 10 个全对缩成 3 个候选，精确阈值 0.8 下只有 1 个正例、2 个 false positive，本快照 recall=1、precision=1/3；另一个 1-hash 反例会漏掉 Jaccard=2/3 的 pair。候选召回不是保证，exhaustive recall audit 本身仍是 \(O(N^2)\)，也只覆盖该快照。大数据应在目标语言/来源/长度切片抽样做 exact ground truth，报告 recall/precision 与漏检，并继续对候选复算精确分子/分母；当前 core 尚未替代 readiness gate，也不检测 semantic/translation duplicate。

## Chat template 是模型接口

template 定义 BOS/EOS、role token、turn 分隔和 generation prompt。同一 `messages` 在不同模型上会生成不同 token。训练、评测、推理必须调用 tokenizer 自带或版本化的同一模板，禁止手写一个近似格式。

检查：

1. `apply_chat_template(..., tokenize=False)` 人工阅读；
2. token ids 中 BOS/EOS 是否重复；
3. assistant 开始/结束边界；
4. tool call/response role 是否支持；
5. 推理时是否正确添加 generation prompt。

锁定 tokenizer revision。只锁 model 权重不锁 tokenizer/template 仍不可复现。

## Loss mask

语言模型 loss：

\[
\mathcal L=-\frac{1}{\sum_t m_t}\sum_t m_t\log p_\theta(x_t\mid x_{<t})
\]

`m_t` 决定哪些 token 被监督。assistant-only SFT 通常令 system/user/tool 输入为 0、assistant 输出为 1。若所有 token 都算 loss，模型会学习复述用户和 system；若边界错一位，会监督 role token 或漏掉首 token。

不能靠字符串查找 `<assistant>` 生成 mask：tokenizer 会合并空格或特殊 token。使用 template 提供的 assistant mask；模板不支持时明确实现 token-level span，并用小样本可视化每个 token、id、role、label。

多轮对话要决定监督所有 assistant turn 还是只监督最后一轮。前者数据更多，后者避免早期答案上下文与训练目标混淆；选择进入实验配置。

## Packing 与 padding

padding 会浪费计算；packing 把多条短样本拼入同一序列。正确 packing 必须：样本间有 EOS/边界、attention 不跨样本（或因果顺序不会造成不当泄漏）、labels 对齐、position strategy 与模型支持一致。

按长度 bucket 减少 padding。统计 p50/p95/p99 token 长度和截断率。截断不能悄悄删掉 assistant 答案；可从长文裁剪输入、过滤或使用长上下文配置，但策略应按任务设计。

## 训练配置

关键参数：有效 batch = micro batch × gradient accumulation × data parallel replicas。学习率与 warmup 按 trainable parameters、batch 和数据量调，不机械复制论文。

保存：模型/adapter、optimizer、scheduler、scaler、global step、epoch、sampler/RNG state、数据 manifest、代码 commit 和完整配置。仓库采用两阶段权限边界：`prepare-training` 在独立审计进程中读取 train + combined，执行 train-only、combined split、顺序敏感子集绑定、lexical candidate 与 governance policy/candidate gate，写出 exact/split/binding/near/governance/readiness artifact；两个训练入口只读取 train + readiness，严格重算 train ordered fingerprint/manifest 并拒绝未知字段、重复 key、错误版本、失败 gate、篡改或陈旧 artifact。`--data-preflight-only` 可在模型下载前单独验证这条边界。

~~~powershell
python -m about_llm.finetuning_cli prepare-training --train-jsonl projects/single-gpu-finetuning/train.example.jsonl --audit-jsonl projects/single-gpu-finetuning/audit.example.jsonl --profile nfc_whitespace --ngram-size 5 --threshold 0.9 --governance-policy projects/single-gpu-finetuning/governance-policy.example.json --governance-evaluated-at 2026-08-06T12:00:00Z --output-dir outputs/sft-prepare
python projects/single-gpu-finetuning/train_trl_sft.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/train.example.jsonl --readiness-json outputs/sft-prepare/sft-training-readiness.json --output-dir outputs/sft-run --data-preflight-only
~~~

readiness 不含 held-out 原文，但不是签名凭证。其无密钥 SHA-256 只能在文件未被攻击者整体替换的前提下发现意外漂移；能替换 artifact 的主体也能重算 hash。生产环境仍需最小权限、受控发布通道，必要时使用签名/证明和独立审计日志。readiness 也不是 tokenizer/mask 验证报告。目标 tokenizer 加载后还会执行结构化 assistant-mask preflight、拒绝静默右截断并写 `sft-template-mask-audit.json`；后者仍不独立证明 mask 语义或最终 collator labels 正确。只有 adapter 文件不能精确恢复中断训练。

梯度裁剪前记录 norm；监控 loss、学习率、token/s、step time、显存和有效 token 数。train loss 下降只证明拟合训练 token，不证明任务质量。

## 验证与早停

验证集与最终 test 分开。训练中可看 validation loss 和任务 proxy；超参选择完成后只少量运行 test。开放生成必须用部署 generation 配置跑任务指标，teacher-forced loss 与实际解码质量不完全一致。

每个 checkpoint 比较：格式合法率、任务质量、通用能力、安全拒答、过度拒答、长度和重复。早停基于预先定义的组合，而不是挑最好看的一个指标。

## 数据配比与 curriculum

领域数据过采样可提升目标任务，也可能导致 catastrophic forgetting。混入通用 instruction、拒答和格式样本，并报告比例。按难度 curriculum 有时提高稳定性，但也可能让模型在简单分布过拟合；要做随机顺序基线。

对稀有任务可 loss weighting 或 sampling weighting，但验证分布应反映真实业务，不跟着训练采样比例改变。

## 过拟合小批次

正式训练前用 8–32 条样本做 overfit test：loss 应明显下降，生成能复现目标格式。这能发现 mask、template、optimizer、冻结和数据读取错误。过拟合失败时不要直接加 GPU。仓库的 `python projects/single-gpu-finetuning/smoke_trl_sft.py` 已在随机 tiny GPT-2 上离线贯通 strict records、真实 Transformers template mask、TRL collator labels 和 optimizer step；这是控制流证据，不是目标模型实验。

然后做 100–1000 步 smoke run，检查 checkpoint 恢复、评测和 adapter 加载，再启动完整实验。

## 泄漏检测

除了 exact/near duplicate，检查 reference answer 独特片段、同一 source 的不同切片、时间穿越和 teacher 生成时是否访问 test。RAG+SFT 比较中，test 答案不能出现在训练 instruction 或 few-shot。

代码任务按 repository/problem family 划分；客服按用户/thread；文档问答按 source document。随机行划分常产生虚高指标。

## 训练后检查

- adapter 加载前后参数和 trainable count；
- merge 前后 logits/生成在容差内一致；
- base 与 adapter revision 兼容；
- 部署 template 与训练一致；
- 量化后重新评测，不假设质量不变；
- 未见 prompt 模板、长输入、空输入和注入测试；
- 模型卡记录数据、限制、硬件、指标和已知失败。

## 实验记录表

| 类别 | 必填 |
|---|---|
| Identity | run id、git commit、seed、owner |
| Model | id、revision、tokenizer/template、dtype |
| Data | manifest hash、来源比例、split、长度、去重 |
| Objective | mask、packing、loss、label policy |
| Optimization | LR、batch、steps、scheduler、clip |
| PEFT | rank/alpha/dropout、target modules、quantization |
| Evidence | checkpoint、日志、峰值显存、评测报告 |

## 面试追问

**SFT loss 降了但效果变差，为什么？** 可能数据目标与评测不一致、模板/mask 错、过拟合、训练分布窄、生成配置变化或通用能力退化；需要 token 可视化和基线/切片评测定位。

**为什么 assistant-only loss？** 用户/system 是条件而非希望模型生成的目标；监督它们会浪费容量并鼓励复述。但某些 continued pretraining/全序列任务目标不同，不能把规则绝对化。

**如何证明没有数据泄漏？** 无法只凭一句声明；提供 group-level split、去重配置、source manifest、时间截断、test 隔离权限和抽样审计结果。
