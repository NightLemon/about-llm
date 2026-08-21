# SFT 数据、模板与训练闭环

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备构造 SFT 数据、训练 adapter 或验收标签语义的工程师。
- **先修**：[训练数据工程](data.md)、tokenization 和 causal LM loss。
- **首次阅读**：跟一条售后对话走过 record、template、assistant mask、collator 和评测。
- **完成信号**：能逐 token 指出哪些位置参与 loss，并证明部署模板与训练模板一致。
- **卡住时**：先只处理一条 user → assistant 对话，不接 Trainer。

</div>

我们想让模型学会这样的售后行为：先查订单，再根据工具结果回答，而不是凭参数记忆猜测。

```text
user:      订单 order-1001 能退吗？
assistant: call lookup_order(order-1001)
tool:      delivered / paid 300 CNY / return_window_open
assistant: 可以申请退款，最高 300 元。
```

这条样本看起来已经可以塞进 Trainer。真正进入 loss 之前，它还要经历来源审查、group split、chat template、
tokenization、assistant mask、padding/collation 和 packing。任一环节漂移，模型都可能在学习另一个任务。

监督微调（Supervised Fine-tuning，SFT）的难点因此不是写出 next-token loss，而是让
原始 record、训练 labels、推理格式和最终评测共享同一契约。

## 先确认问题适合用微调解决

把需求分成三类：

| 需求 | 首选起点 | 原因 |
|---|---|---|
| 经常变化的订单、政策和知识 | RAG / tool | 参数更新慢，source of truth 在外部系统 |
| 格式、风格、工具选择和稳定行为 | Prompt baseline 后评估 SFT | 可以从一致示例中学习行为分布 |
| 基座没有的复杂能力 | 换模型、分解任务或更多训练 | 少量 SFT 很难凭空创造底层能力 |

本例的订单状态仍由 tool 提供，SFT 只教模型何时调用、怎样使用结果和怎样组织答复。

至少比较 base zero-shot、few-shot、tool/RAG baseline 与 SFT。若简单 Prompt 已达到质量和成本目标，
引入 adapter、训练数据治理与回归面可能没有收益。

## 一条样本怎样走到 optimizer

| 阶段 | 产生什么 | 本例要检查什么 |
|---|---|---|
| Structured record | messages、tools、source、group、split | Tool call 与 response 能否成对 |
| Governance | source policy、sensitive candidates | 数据是否允许用于 training |
| Split / dedup | exact、lexical candidates、group assignment | 同一客户或 thread 是否跨 split |
| Chat template | rendered text / token IDs | Role 与 tool markup 是否匹配 checkpoint |
| Assistant mask | 每个 token 的 0/1 标记 | 只监督两个 assistant turns |
| Collator | `input_ids`、`attention_mask`、`labels` | 非监督位置与 padding 是否为 `-100` |
| Trainer | loss sum、有效 label count、gradients | Denominator 是否跨 accumulation 正确 |
| Evaluation | 部署 generation 与 held-out cases | 学到的是任务行为，不只是训练文本 |

表中前六步可以在模型下载前或一次 no-grad forward 中验证。先把它们跑通，
比训练几小时后再猜“为什么 loss 很奇怪”便宜得多。

## Structured record 保留模板升级空间

本例可以写成：

```json
{
  "id": "support-017",
  "messages": [
    {"role": "user", "content": "订单 order-1001 能退吗？"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call-1",
        "type": "function",
        "function": {
          "name": "lookup_order",
          "arguments": {"order_id": "order-1001"}
        }
      }]
    },
    {
      "role": "tool",
      "content": "delivered / paid 300 CNY / return_window_open",
      "tool_call_id": "call-1",
      "name": "lookup_order"
    },
    {"role": "assistant", "content": "可以申请退款，最高 300 元。"}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "lookup_order",
      "description": "Read one authorized order.",
      "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": false
      }
    }
  }],
  "source": "expert-authored",
  "license": "internal-approved",
  "task": "tool-grounded-support",
  "language": "zh",
  "risk": "normal",
  "group_id": "customer-42",
  "split": "train",
  "metadata": {"annotation_batch": "2026-08-a"}
}
```

仓库的 `about-llm.sft-jsonl.v2` 要求治理字段、合法 role 顺序和最终 assistant supervision target。
它拒绝 duplicate JSON keys、`NaN/Infinity`、未知字段、空治理字段和无配对 Unicode surrogate。

Tool subset 还要求：

- Tool name 在 definitions 中存在；
- 每个 call ID 在对话中唯一；
- 随后的 tool response 匹配 ID 与 name；
- 所有 pending calls 清空后才能进入下一条 user 或 assistant message。

这是仓库固定的训练 schema，不代表所有 provider wire formats。保留 structured messages，
可以在 tokenizer 或 template 升级时重新渲染；只保存拼好的长字符串会失去 role、tool 和治理边界。

## 来源、敏感信息与 split 在训练区外先审计

专家样本通常质量较高但昂贵；历史日志贴近真实分布，却含旧流程、错误答案和隐私；
teacher model 可以扩展覆盖，也会复制自身偏差与措辞。记录来源比例，并在各来源切片上评测。

Source registry 精确绑定 `source + license`、允许的 purpose、evidence reference、review time 和 expiry。
未知组合默认拒绝。License 字符串本身只是 record 字段，真实 evidence 应指向可访问合同、许可快照和负责人。

有限 scanner 可以寻找 email、private-key header、部分 key/token/JWT 形态和通过 Luhn 的 card-like number。
命中项进入人工处置，exception 绑定 record identity、detector、span、reviewer、rationale 与 evidence。
无命中只说明这些 detectors 没有发现候选，不代表数据中不存在姓名、地址、医疗信息或未知 secret。

删除请求要能沿 source ID 追到 train shard、run、checkpoint 和 adapter。高敏信息最好在进入训练区之前隔离，
避免它继续复制到日志、cache 和中间 artifact。

### Group split 先于逐行随机

`group_id=customer-42` 表示同一客户或 thread 的相关样本必须落在同一 split。
代码任务通常按 repository/problem family，文档问答按 source document，客服按 user/thread 分组。

Exact hash 可以删除完全相同 records；字符 n-gram Jaccard 用于产生 lexical near-duplicate candidates：

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

Normalization profile、n-gram size 和 threshold 都会改变候选。`nfc_whitespace` 保留大小写与兼容字符，
`nfkc_casefold_whitespace` 召回更多表面变体，也可能破坏代码、标识符和格式任务的含义。

Candidate 是待复核 pair，不是已确认 duplicate。翻译、语义改写和答案片段还需要其他检测或人工审计。
大数据可以用 MinHash/LSH 缩小候选集，但对候选仍重算 exact Jaccard，并在目标切片评估 recall。

运行离线审计：

~~~powershell
python -m about_llm.finetuning_cli audit `
  --jsonl projects/single-gpu-finetuning/audit.example.jsonl `
  --require-splits train,validation,test `
  --output outputs/sft-split-audit.json

python -m about_llm.finetuning_cli governance-audit `
  --jsonl projects/single-gpu-finetuning/audit.example.jsonl `
  --policy projects/single-gpu-finetuning/governance-policy.example.json `
  --evaluated-at 2026-08-06T12:00:00Z `
  --output outputs/sft-governance-audit.json

python -m about_llm.finetuning_cli near-audit `
  --jsonl projects/single-gpu-finetuning/audit.example.jsonl `
  --profile nfc_whitespace `
  --ngram-size 5 `
  --threshold 0.9 `
  --output outputs/sft-near-duplicate-audit.json
~~~

`evaluated-at` 显式固定 expiry 判断时间；审计结果同时绑定 record、split 和 policy revisions。
Exact / lexical gates 不能代替许可判断、语义去重或完整 PII detection。

## Chat template 决定模型看见什么

Template 插入 BOS/EOS、role tokens、turn separators、tool markup 和 generation prompt。
同一份 structured messages 在不同 checkpoint 上可以得到完全不同的 token IDs。

训练、评测和部署应使用同一 tokenizer/template revision。检查顺序是：

1. 先用 `apply_chat_template(..., tokenize=False)` 人工阅读；
2. 检查 BOS/EOS 是否重复；
3. 找到 user、assistant、tool 的开始和结束边界；
4. 确认多 tool calls 和 response 顺序受支持；
5. 推理时确认 generation prompt 与训练结尾一致；
6. 保存 rendered text、token IDs 和 revision。

只锁定 model weights、没有锁 tokenizer 和 template，仍然无法重放输入。

## Assistant mask 决定模型学什么

Causal LM 的 masked token loss 可以写成：

\[
\mathcal L=
-\frac{1}{\sum_t m_t}
\sum_t m_t\log p_\theta(x_t\mid x_{<t}).
\]

\(m_t\) 为 1 的位置参与监督。Assistant-only SFT 通常将 system、user 和 tool input 设为 0，
assistant 的 tool-call serialization 与最终回答设为 1。

在本例中，模型应该学习两段 assistant output：工具 proposal 和最终自然语言答复。
若所有 tokens 都参与 loss，模型还会学习复述用户、工具结果与 system 文本；边界错一位则可能监督 role marker，
或漏掉 assistant 的第一个 token。

不能用字符串搜索 `<assistant>` 构造 mask。Tokenizer 可能合并空格，role marker 也可能是 special token。
优先使用 template 提供的 generation/assistant mask；否则实现明确的 token-level spans，
并为小样本打印 token、ID、role、mask 与最终 label。

多轮对话还要决定监督所有 assistant turns，还是只监督最后一轮。两者都可以是合法训练目标，
但要写入配置并分别评测，不能让 collator 默认值替产品做决定。

### TRL 配置名不是最终 labels 的证据

截至仓库核对日 2026-08-13，TRL 0.29.1 的 `assistant_only_loss=True` 要求 conversational preprocessing
能从 template 得到 generation mask。若 Dataset 已经预先生成 `input_ids/assistant_masks`，可以显式关闭该预处理开关，
保留 mask 列，再让同一个 configured collator 把非监督位置投影成 `-100`。

最终要检查的是 collator 输出：

```text
labels[t] == input_ids[t]   when assistant_mask[t] == 1
labels[t] == -100           for other valid tokens and padding
```

### 在固定 Qwen 版本上检查 final labels { #target-qwen-sft-final-label-control }

仓库为指定的 `Qwen/Qwen2.5-0.5B-Instruct` revision 保存了一份运行记录。
它用三条多轮/tool 固定样例比较原生模板与审核模板的 input IDs，检查 assistant masks，
再让真实 TRL collator 生成 final labels，并执行一次目标模型 CPU FP32 no-grad loss。

~~~powershell
python projects/single-gpu-finetuning/run_qwen_target_sft_label_control.py `
  --verify projects/single-gpu-finetuning/qwen2.5-0.5b-sft-label.recorded-report.json
~~~

这次运行检查 records、template、collator 和 checkpoint revision 是否正确连接起来。
它没有执行 backward、optimizer 或模型质量评测。具体 token counts 与 loss 见
[单 GPU 微调项目](../practice/projects/single-gpu-finetuning.md#run)。

## Padding、packing 与 truncation 不能改变标签语义

按长度 bucket 可以减少 padding。Packing 把多条短样本放进同一 sequence，还要明确：

- 样本间 BOS/EOS 或 boundary；
- attention 是否跨样本；
- labels 与 assistant masks 怎样拼接；
- position IDs 是否重置；
- 截断发生在哪个 segment；
- loss denominator 使用哪些 valid objective tokens。

静默右截断可能保留完整 user prompt，却删掉 assistant answer，于是样本占显存但没有监督 token。
统计 p50/p95/p99 长度、截断率和每条样本的监督 token 数；过长样本可以裁剪输入、过滤或进入长上下文路线。

高 packing efficiency 只说明 padding 少。Attention boundary、role mask 或 labels 错误时，100% 利用率仍可能训练错误目标。

## Accumulation 的分母是全窗口有效 labels

每条 SFT sequence 的监督 token 数受 assistant turns、tool serialization、padding 和 truncation 影响。
若目标是 assistant-token mean，就要跨整个 optimizer update 累积 masked loss sum 与 `labels != -100` count。

等权平均每个 micro-batch 的 masked mean，会让监督 token 较少的 batch 权重过高。
Sequence mean 或 task-balanced weighting 可以是另一种明确 objective，但不能把它误报为 token mean。

运行最小反例：

~~~powershell
python projects/single-gpu-finetuning/gradient_accumulation_toy.py
python projects/single-gpu-finetuning/ddp_token_mean_control.py
python projects/single-gpu-finetuning/ddp_accumulation_no_sync_control.py
python projects/single-gpu-finetuning/amp_grad_scaler_control.py
~~~

第一条用 `[1,3]` 有效 labels 说明 sum/count 与 full batch 一致，而等权 micro-batch mean 改变 gradient。
后续小实验分别覆盖 DDP 默认 reducer 的 \(D/N\) scaling、`no_sync` + clipping + SGD，
以及 AMP 的 unscale-before-clip 和 overflow skip。

这些都是小型 CPU 实验，没有接入完整目标 SFT Trainer、CUDA 或真实数据质量评测。
迁移时还要核对 framework 是否已经除 accumulation steps、何时 all-reduce count，
以及 optimizer、scheduler 和 GradScaler 在 skip 时是否共同保持不变。

## Readiness 把 held-out 数据挡在 Trainer 外

仓库采用两阶段边界：

```text
audit process:
  train + combined splits + policy
  -> exact / near / governance / binding reports
  -> readiness artifact

trainer process:
  train file + readiness
  -> rehash train identity
  -> tokenizer / template / mask preflight
  -> model loading and training
```

Trainer 不需要读取 validation/test 原文。Audit 进程验证独立 train file 按顺序等于 combined artifact 中的 train subset，
并把 split、near-duplicate 和 governance decisions 绑定到 readiness。

~~~powershell
python -m about_llm.finetuning_cli prepare-training `
  --train-jsonl projects/single-gpu-finetuning/train.example.jsonl `
  --audit-jsonl projects/single-gpu-finetuning/audit.example.jsonl `
  --profile nfc_whitespace `
  --ngram-size 5 `
  --threshold 0.9 `
  --governance-policy projects/single-gpu-finetuning/governance-policy.example.json `
  --governance-evaluated-at 2026-08-06T12:00:00Z `
  --output-dir outputs/sft-prepare

python projects/single-gpu-finetuning/train_trl_sft.py `
  --model-id <model> `
  --revision <commit> `
  --train-jsonl projects/single-gpu-finetuning/train.example.jsonl `
  --readiness-json outputs/sft-prepare/sft-training-readiness.json `
  --output-dir outputs/sft-run `
  --data-preflight-only
~~~

无密钥 readiness hash 可以发现当前文件与记录 identity 漂移，但不能认证签发者；
能替换全套 artifact 的主体也能重新计算 hash。生产还需要最小权限、受控发布和独立审计日志。

Readiness 也不等于 tokenizer/mask audit。模型 revision 加载后，训练入口仍要拒绝静默截断，
保存 template/mask report，并检查 configured collator 的 final labels。

## 训练先从 8–32 条样本过拟合开始

小批次 overfit 是管道验收，不是泛化实验。Loss 应明显下降，生成能复现目标格式；
若失败，优先检查 template、mask、冻结参数、optimizer、数据读取和 learning rate，而不是增加 GPU。

仓库的离线 smoke test 会检查输入记录，再贯通 Transformers template mask、TRL collator labels 和 optimizer step：

~~~powershell
python projects/single-gpu-finetuning/smoke_trl_sft.py
~~~

接着运行 100–1000 steps smoke，检查 checkpoint、resume、adapter 加载和 held-out evaluation，
最后才启动完整实验。

训练配置至少保存：model/adapter、optimizer、scheduler、scaler、global step、sampler/RNG、
data manifest、tokenizer/template、mask policy、packing、代码 revision 和完整参数。

Train loss 下降说明模型更能预测训练 labels，不会自然证明部署任务更好。

## Resume 的位置必须与 optimizer commit 对齐

多 worker DataLoader 会提前把 sample IDs 发到 worker queues。此时 sampler cursor 可能已经走到 7，
main loop 只消费到 3，而 optimizer 只提交到 2。把 emitted cursor 当成“已训练位置”，resume 会跳过队列中的样本。

Gradient accumulation 崩溃时还可能存在 partial-window gradients。两种恢复策略是：

1. 回到 optimizer-committed boundary，恢复该时刻 RNG，并重放未提交 samples；
2. 保存 pending sample IDs、position/divisor、gradients 与 crash-time RNG sidecar 后继续。

仓库的 CPU 实验对两条路径都做 bit-exact 比较，并用“漏 gradients”和“恢复错误 RNG”的负例展示参数漂移。
Manifest-last 只能检测当前 bundle 是否完整，仍没有让 sample、optimizer 和所有 checkpoint files 成为跨故障原子事务。

真实 SFT 还要纳入 assistant mask、有效-token denominator、adapter、optimizer、scheduler、GradScaler、
Python/NumPy/CUDA/worker RNG，以及分布式 rank 共识。详细故障矩阵见
[单 GPU 微调项目](../practice/projects/single-gpu-finetuning.md)。

## Validation 与 test 回答不同问题

Validation 用于 early stopping 和超参数选择，最终 test 限制查看次数。
Teacher-forced loss 与真实 generation quality 不完全一致，开放生成要使用部署时的 decoding、template 和 tool schema 运行。

每个 checkpoint 至少比较：

- 目标任务与格式合法率；
- 通用能力与语言切片；
- 安全拒答和 over-refusal；
- 长度、重复与 tool behavior；
- 延迟、显存和每成功任务成本。

领域数据过采样可能提高目标任务，同时造成 catastrophic forgetting。记录通用 instruction、拒答、格式和领域样本比例，
并在真实 validation 分布上评测，不让 validation 跟随训练采样权重变化。

Curriculum、loss weighting 和 sampling weighting 都需要随机顺序或统一采样 baseline。
早停依据事前定义的指标与 hard gates，不能从多个 checkpoint 中挑一个最好看的单项分数。

训练后还要检查 adapter 与 base revision 兼容、merge 前后 logits/生成容差、部署 template、
量化后的重新评测，以及未见 Prompt、长输入、空输入和 injection cases。

## 一份实验记录应该能重建决策

| 类别 | 最少保存什么 |
|---|---|
| Identity | Run ID、git revision、seed、owner |
| Model | Model / adapter revision、tokenizer/template、dtype |
| Data | Manifest、来源比例、split、长度、dedup / governance |
| Objective | Assistant mask、packing、loss denominator、label policy |
| Optimization | LR、batch、steps、scheduler、clip、scaler |
| PEFT | Rank / alpha / dropout、target modules、quantization |
| Evidence | Checkpoints、logs、显存、逐 case evaluation 与失败样本 |

模型卡据此记录训练数据范围、已知限制、硬件、评测和不支持的场景。

## 自测与面试追问

1. 售后样本中哪些 tokens 应参与 assistant-only loss，tool response 为什么通常不参与？
2. Chat template 输出看起来正确，为什么仍要检查 collator 的 final labels？
3. 同一客户的不同工单为什么可能需要按 group 切分？
4. 两个 micro-batches 的有效 labels 分别为 1 和 9，等权 mean 会怎样改变 objective？
5. Sampler emitted=7、consumed=3、optimizer committed=2 时，从哪个位置恢复，各需保存什么？
6. SFT loss 下降而 held-out 任务变差时，你会沿哪几层定位？
7. 怎样用权限和 artifact 边界证明 Trainer 没有读取 held-out 原文？
