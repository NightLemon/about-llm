# SFT 数据、模板与训练闭环

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：SFT 数据、模板和训练验收工程师。
- **先修**：[训练数据工程](data.md)、tokenization 和 chat template。
- **首次阅读**：是否需要微调 → 数据契约 → chat template → loss mask → 验证。
- **完成信号**：能证明模板、标签 mask、切分和训练输入完全一致。
- **卡住时**：回到[Tokenization](../core/tokenization.md)的模板与 special token。

</div>

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

仓库的 `about-llm.sft-jsonl.v2` 契约要求除可选 `metadata/tools` 外的上述字段全部存在，并拒绝未知字段、重复 JSON key、`NaN`/`Infinity`、空治理字符串、无配对 Unicode surrogate 和不合法对话顺序。system 只能可选地出现在开头，第一个非 system 消息必须是 user，最后一条必须是 assistant，确保至少有明确监督目标。

v2 的 tool 子集采用显式 function schema。assistant 可用空 `content` 加一个或多个 `tool_calls`，但每个 call ID 在整段对话中必须唯一、name 必须有对应 definition；紧随其后的 tool response 必须同时携带匹配的 `tool_call_id/name`，并在用户或 assistant 继续前逐个清空全部 pending calls：

~~~json
{
  "messages": [
    {"role": "user", "content": "杭州天气?"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call-1", "type": "function", "function": {
        "name": "weather", "arguments": {"city": "Hangzhou"}
      }}
    ]},
    {"role": "tool", "content": "sunny", "tool_call_id": "call-1", "name": "weather"},
    {"role": "assistant", "content": "晴天。"}
  ],
  "tools": [{"type": "function", "function": {
    "name": "weather", "description": "Get weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
  }}]
}
~~~

该 contract 是本仓库固定子集，不声称等于所有 provider 的 wire schema。arguments/parameters 必须是 strict JSON object，programmatic 构造也会做不可变深快照，避免调用方后续 mutation 让 fingerprint 漂移。

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

门禁检查 required split、重复 id、规范序列化后完全相同的 `messages + tools`、跨 split `group_id` 和跨 split exact content。输出同时给有序数据 fingerprint、保留重复次数但忽略行顺序的 fingerprint，以及绑定契约版本、split policy 和 gate 规则的 manifest fingerprint；另报告 tool definition/call/response 数。字符统计单位是 Unicode code point，**不是 tokenizer token**。

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

截至 2026-08-13，TRL 0.29.1 的 `assistant_only_loss=True` 是 conversational preprocessing 开关：它要求 chat template 提供 generation mask，并会拒绝已经只含 `input_ids/assistant_masks` 的预分词 Dataset。若为避免 Arrow 改写 nested tool arguments 而在 Dataset 前生成 masks，应显式设 `assistant_only_loss=False`、保留 `assistant_masks` 列，并实际调用同一个 configured collator 检查 final labels；该 collator 会独立按 masks 投影 `-100`。对 prompt-completion 数据，`completion_only_loss=None` 默认只监督 completion，设为 `False` 才监督完整序列。无论走哪条路径，都不能只相信配置名。

多轮对话要决定监督所有 assistant turn 还是只监督最后一轮。前者数据更多，后者避免早期答案上下文与训练目标混淆；选择进入实验配置。

### 从模板 mask 到 TRL 最终 labels 的固定证据 { #target-qwen-sft-final-label-control }

仓库为固定 `Qwen/Qwen2.5-0.5B-Instruct` revision 保留一条独立 control。它先证明 checkpoint 原生模板不含 `{% generation %}`，并在多轮、并行 tool calls、tool preamble 三条 authored fixture 上实际返回全零 assistant mask；再加载审核模板，要求 47 / 301 / 200 个 input IDs 与原生完全相同，并让 8 / 51 / 31 个 mask tokens 精确等于 control 独立保存的 assistant serialization，包括所有 assistant turn、Qwen tool-call markup 和 `<|im_end|>\n`。

raw `Dataset.from_list` 会把并行调用中异构的 `{"city":...}` / `{"expression":...}` arguments 合并成一个 Arrow struct，并向各自缺少的 key 注入 `null`，使 rendered prompt 漂移。因此入口先在 Python 中渲染 token/mask，再把纯整数列交给 Dataset。真实 TRL 0.29.1 configured collator 必须得到 `[3, 301]`：548 attention token、355 padding token、90 个监督 label 与 813 个 `-100`，监督位置等于 input IDs，其余有效 token 与 padding 全部忽略。固定 Qwen CPU FP32 no-grad loss `1.251716` 只说明该 batch 能进入目标权重；它不执行 backward 或 optimizer，也不是训练、收敛和质量证据。

~~~powershell
python projects/single-gpu-finetuning/run_qwen_target_sft_label_control.py --verify projects/single-gpu-finetuning/qwen2.5-0.5b-sft-label.recorded-report.json
~~~

这条证据只覆盖固定 Qwen schema 下的三条多轮/tool 记录，不能外推到任意 provider schema、multimodal、任意新消息或 tool 执行/结果真实性；“审核模板与原生 input IDs 相同”也只对该 fixture 成立。authored fixture/readiness/hash 不证明数据合法性、语义质量或来源认证，离线 verifier 不重放 tokenizer/model，verify→loader reopen TOCTOU 仍在。运行入口与机器边界见[站内项目页](../practice/projects/single-gpu-finetuning.md#run)。

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

### Assistant mask 会改变 accumulation 分母

SFT 中每条序列的监督 token 数不只是由长度决定，还受多轮 assistant span、tool call serialization、padding、truncation 与 `-100` mask 影响。若目标是 assistant token mean，不能对每个 micro-batch 的 masked mean loss 等权平均；应累积 masked loss sum 与 `labels != -100` 的 count，并以整个 optimizer-update window 的全局有效 count 缩放。Sequence mean 或 task-balanced weighting 可以是合理目标，但必须显式命名、实现和评测，不能把它误报为 token mean。

运行 `python projects/single-gpu-finetuning/gradient_accumulation_toy.py` 可看到 `[1,3]` 有效 token 反例：sum/count 路径与 full batch 一致，等权 micro-batch mean 改变 loss 和梯度，ignored positions 的梯度为零。迁移到真实 Trainer 前，还要核对 framework 是否已经除以 accumulation steps、DDP 是 sum 还是 mean、何时 all-reduce count，以及 AMP unscale、clip、`no_sync`、optimizer/scheduler 的顺序。该 toy 没有执行目标 tokenizer/model、optimizer、CUDA 或分布式 runtime。

`python projects/single-gpu-finetuning/ddp_token_mean_control.py` 是与上述 toy 分开的 reducer 集成证据：两个 CPU/Gloo rank 真实 `all_reduce` 得到全局 count 4，并执行默认 DDP backward。对 `[1,3]` rank counts，local loss sum 乘 `D/N=1/2` 得到 full-batch `(0.575,-0.575)`；只乘 `1/N=1/4` 得到 `(0.2875,-0.2875)`，rank-local mean 得到 `(0.35,-0.35)`。它没有执行 accumulation window、`no_sync`、AMP、optimizer、目标 Trainer/tokenizer/model、GPU 或多节点，不能替代真实 SFT 集成测试。

`ddp_accumulation_no_sync_control.py` 进一步执行两个 micro-batch/rank 的 accumulation、正确 `no_sync` scope、同步后的 global-norm clipping 与 plain SGD step。固定 counts `[[1,2],[3,1]]` 时，`D/N=2/7` 得到 pre-clip `(+19/35,-19/35)`，built-in DDP 的 pre/post-clip gradient 和参数更新均与 full batch 一致。计数 hook 证明 `no_sync` 必须包住 forward 和 backward：正确 scope 只有一次 reference all-reduce hook，只包 backward 有两次；本 fixture 后者数值仍相同，不能把“通信多了”误写成“梯度一定错”。它仍没有执行真实 TRL collator/model、AMP、随机层、多 bucket、AdamW、GPU、多节点或质量评测。

`python projects/single-gpu-finetuning/amp_grad_scaler_control.py` 是另一条单进程数值控制：scaled accumulation gradient 24 必须先 unscale 为 3，再 clip 到约 0.5；反向顺序会把 optimizer gradient 错缩为约 0.0625。含 `inf` 的任一 micro-batch 会使整个 AdamW update 被跳过，不能推进 scheduler 或把该窗口记成完成更新。报告还用 scale=1/8 的边界梯度证明漏恢复 GradScaler 会改变下一步 execute/skip 决策。它没有接入当前 SFT collator、TRL、DDP、磁盘 checkpoint 或 CUDA，因此训练入口仍要做目标路径集成测试。

### DataLoader prefetch 与恢复 cursor

多 worker loader 会提前向 worker queue 发 index。仓库的 `dataloader_prefetch_resume_control.py` 在真实 CPU spawn workers 上固定 `num_workers=2,prefetch_factor=2,batch_size=1,in_order=True`：主循环只收到 `[8,3,1]` 时 sampler cursor 已从 0 走到 7，`[7,0,9,4]` 已发出但未交付。把 emitted cursor 直接保存为“已训练位置”后，resume 只见 `[2,6,5]`；保存应用实际 consumed cursor=3，重建后才得到完整顺序。

这仍只恢复 sample identity。fresh workers 的 local Torch RNG 从头开始，tail 不同；按 `(namespace,sample_id)` 构造局部 generator 的 authored transform 则逐位重放。真实增强 key 通常还要绑定数据/变换版本、epoch、重复访问序号，并审计 collision；若变换依赖邻样本、时间、外部服务或全局状态，sample-keyed RNG 也不够。control 没有执行 collator、model、optimizer、IterableDataset、persistent worker、pin memory、distributed sampler 或 queue-state serialization；consumed 也不是 optimizer-committed。生产 resume 应逐 batch 记录 source/sample IDs 并做 kill/reload 对照，不凭 sampler 的当前整数猜进度。

`optimizer_commit_resume_control.py` 进一步执行 main-process inverted-Bernoulli mask、真实 Float64 backward、SGD momentum、StepLR 与两步 accumulation：第三条 `[8,3,1]` 已被 main loop 消费并 stochastic backward 时，只有 `[8,3]` 已提交，故 emitted/consumed/committed=`7/3/2`。base checkpoint 不含 `.grad`，但保存 commit-boundary model/optimizer/scheduler/Torch RNG；从 2 恢复 RNG 并重放 sample `1` 与 uninterrupted bit-exact。第一个负例从 3 起步并使用正确 crash RNG，却漏 gradients/sample `1`；即使 partial-window 缩放保持同为 5 次 optimizer/StepLR step、LR 同为 `0.0125`，参数最大差仍约 `0.0057678586`，而未来 RNG 轨迹相同。

第二条正确路径从 consumed=3 加载绑定 base digest 的 sidecar；它恢复 pending `[1]`、position/divisor、两个逐参数 gradients 与 crash-observed Torch RNG，首个完成窗口为 `[1,7]`，终态也 bit-exact。sidecar 协议现在要求最后发布 strict canonical manifest，先核对 complete state、数据 identity、base/sidecar name/schema/size/hash 与发布顺序，再按同一 identity 加载 bytes。另一个隔离负例恢复 gradients/ledger 却使用 commit-boundary RNG：step/LR 相同，参数最大差约 `0.0178788936`，终态 RNG 不同。base-only、两 payload 无 manifest、manifest 缺 sidecar、sidecar post-manifest tamper 四种快照均在 `torch.load` 前拒绝；base-only 仍能走 commit-boundary replay。

真实 SFT 还要把有效-token分母、assistant mask、GradScaler、Python/NumPy/CUDA/worker RNG 与 adapter/optimizer state 纳入同一协议；tiny CPU control 只覆盖 main-process Torch RNG 与 StepLR。manifest-last 只检测当前 incomplete/mismatched bundle，base+sidecar+manifest 仍非 sample/optimizer 原子发布，也无 directory `fsync`、断电/filesystem 故障、来源认证或不可变目录证据，因此不证明目标 Trainer 恢复。

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
