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

这条样本看起来已经可以交给 Trainer，但它离真正参与 loss 还有一段路。先核对来源和使用权限，再按客户分组切分数据。
之后还要经过对话模板、分词、assistant mask、补齐和组 batch，最后才可能与其他短样本装入同一序列。
任一环节漂移，模型都可能在学习另一个任务。

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

至少比较基座模型的零样本、少样本、tool/RAG baseline 与 SFT。若简单 Prompt 已经达到质量和成本目标，
新增 Adapter、训练数据治理和回归风险可能得不偿失。

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

仓库的 `about-llm.sft-jsonl.v2` 要求治理字段完整、role 顺序合法，并且对话最终包含可监督的 assistant 输出。
解析器会拒绝重复 JSON 字段、`NaN/Infinity`、未知字段、空治理字段和无配对的 Unicode surrogate。

Tool subset 还要求：

- Tool name 在 definitions 中存在；
- 每个 call ID 在对话中唯一；
- 随后的 tool response 匹配 ID 与 name；
- 所有 pending calls 清空后才能进入下一条 user 或 assistant message。

这是仓库使用的训练 schema，不代表所有 provider 的线上请求格式。保留结构化 messages 后，tokenizer 或模板升级时
仍能重新渲染；如果只保存拼好的长字符串，role、tool 和治理边界都会丢失。

## 来源、敏感信息与 split 在训练区外先审计

专家样本通常质量较高但昂贵；历史日志贴近真实分布，却含旧流程、错误答案和隐私；
teacher model 可以扩展覆盖，也会复制自身偏差与措辞。记录来源比例，并在各来源切片上评测。

来源登记表把 `source + license` 与允许用途、证明材料、复核时间和过期时间绑定。没有登记的组合默认不能用于训练。
记录中的 license 字符串只是一项声明；真正的依据应指向可访问的合同或许可快照，并注明负责人。

仓库提供的有限扫描器会寻找 email、私钥头、部分 key/token/JWT 形态，以及通过 Luhn 校验的银行卡样式数字。

命中后进入人工处置。如果需要例外放行，应记录样本身份、检测器、位置、复核人、理由和依据。

扫描范围仅限上述模式，姓名、地址、医疗信息和未知 secret 仍需其他检查。

删除请求还要能从 source ID 追到训练 shard、run、checkpoint 和 Adapter。高敏信息最好在进入训练区前就被隔离，
避免继续复制到日志、缓存和中间工件。

### Group split 先于逐行随机

`group_id=customer-42` 表示同一客户或会话的相关样本必须落在同一个 split。代码任务通常按仓库或题目家族分组，
文档问答按来源文档分组，客服数据按用户或会话分组。

精确 hash 可以找到完全相同的记录；字符 n-gram Jaccard 用于产生词面近重复候选：

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

Normalization profile、n-gram size 和 threshold 都会改变候选。`nfc_whitespace` 保留大小写与兼容字符，
`nfkc_casefold_whitespace` 召回更多表面变体，也可能破坏代码、标识符和格式任务的含义。

候选 pair 只是待复核对象，不等于已确认重复。翻译、语义改写和答案片段污染仍需要其他检测或人工审计。
数据量很大时，可以用 MinHash/LSH 缩小候选集；进入候选集后仍要重算精确 Jaccard，并在目标切片上评估召回率。

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

`evaluated-at` 固定“是否过期”的判断时刻。审计结果还会绑定记录、数据划分和策略版本。
精确与词面检查不能代替许可判断、语义去重或完整的个人信息检测。

## Chat template 决定模型看见什么

对话模板会插入 BOS/EOS、角色 token、轮次分隔符、工具标记和生成提示。同一份结构化消息换到另一个 checkpoint 后，
可能得到完全不同的 token IDs。

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

\(m_t=1\) 的位置参与监督。Assistant-only SFT 通常不监督 system、user 和 tool 输入，只监督 assistant 生成的
工具调用序列和最终回答。

本例中恰好有两个学习目标：第一次 assistant 轮次的工具提案，以及最后一轮的自然语言答复。

若让所有 token 都参与 loss，模型还会被训练去复述用户、工具结果和 system 文本。

Mask 边界只要错一位，就可能把角色标记当成目标，或者漏掉 assistant 的第一个 token。

不要通过搜索字符串 `<assistant>` 来构造 mask。Tokenizer 可能把空格与相邻文本合并，角色标记也可能是特殊 token。
更可靠的顺序是：

1. 首选模板直接返回的 generation/assistant mask；
2. 模板不支持时，实现明确的 token-level spans；
3. 对少量样本逐位置打印 token、ID、role、mask 和最终 label。

多轮对话还要决定监督所有 assistant turns，还是只监督最后一轮。两者都可以是合法训练目标，
但要写入配置并分别评测，不能让 collator 默认值替产品做决定。

### TRL 配置名不是最终 labels 的证据

截至仓库核对日 2026-08-13，TRL 0.29.1 使用 `assistant_only_loss=True` 时，对话预处理必须能从模板得到
generation mask。另一种路径是提前生成 `input_ids/assistant_masks`，关闭会重复处理对话的开关，保留 mask 列，
再让实际训练使用的 collator 把非监督位置改成 `-100`。

最终要检查的是 collator 输出：

```text
labels[t] == input_ids[t]   when assistant_mask[t] == 1
labels[t] == -100           for other valid tokens and padding
```

### 用固定 Qwen 版本核对最终 labels { #target-qwen-sft-final-label-control }

仓库为指定的 `Qwen/Qwen2.5-0.5B-Instruct` revision 保存了一份运行记录。三条多轮工具对话分别经过原生模板和
审核模板，先比较输入 token IDs 与 assistant masks。

随后，真实 TRL collator 生成最终 labels，目标模型完成一次 CPU FP32、无梯度的 loss 计算。

~~~powershell
python projects/single-gpu-finetuning/run_qwen_target_sft_label_control.py `
  --verify projects/single-gpu-finetuning/qwen2.5-0.5b-sft-label.recorded-report.json
~~~

这次运行只检查记录、模板、collator 和 checkpoint revision 是否正确连接。运行范围到无梯度 forward 为止，
模型质量也没有在这里评测。具体 token 数和 loss 见
[单 GPU 微调项目](../practice/projects/single-gpu-finetuning.md#run)。

## Padding、packing 与 truncation 不能改变标签语义

按长度 bucket 可以减少 padding。Packing 把多条短样本放进同一 sequence，还要明确：

- 样本间 BOS/EOS 或 boundary；
- attention 是否跨样本；
- labels 与 assistant masks 怎样拼接；
- position IDs 是否重置；
- 截断发生在哪个 segment；
- loss denominator 使用哪些 valid objective tokens。

静默右截断可能保留完整用户输入，却删掉 assistant 回答。这样的样本仍占显存，却没有监督 token。
应统计 p50/p95/p99 长度、截断率和每条样本的监督 token 数；过长样本可以裁剪输入、过滤，或进入长上下文路线。

高 packing efficiency 只说明 padding 较少。如果 attention boundary、角色 mask 或 labels 有误，即使利用率达到 100%，
训练目标仍然可能是错的。

## Accumulation 的分母是全窗口有效 labels

每条 SFT 序列包含多少监督 token，取决于 assistant 轮数、工具序列化、补齐和截断。若训练目标是“每个 assistant
token 权重相同”，就要在一次参数更新的整个累积窗口内同时累加 loss sum 和 `labels != -100` 的数量。

设两个小批次分别包含 \(n_1,n_2\) 个监督 token，各自的平均 loss 是 \(L_1,L_2\)。在整个窗口内按监督 token
求平均，结果为：

\[
L_{\text{window}}=\frac{n_1L_1+n_2L_2}{n_1+n_2}.
\]

直接计算 \((L_1+L_2)/2\) 会让监督 token 较少的 micro-batch 获得过高权重。按序列求 mean 或按任务加权也可以是
合理目标，但必须明确命名，不能把它们报告成 token mean。

运行最小反例：

~~~powershell
python projects/single-gpu-finetuning/gradient_accumulation_toy.py
python projects/single-gpu-finetuning/ddp_token_mean_control.py
python projects/single-gpu-finetuning/ddp_accumulation_no_sync_control.py
python projects/single-gpu-finetuning/amp_grad_scaler_control.py
~~~

四条实验分别回答：

1. `[1,3]` 个有效 labels 时，sum/count 如何与 full batch 保持一致；
2. DDP 默认对 rank gradient 求 mean 后，为什么需要 \(D/N\) scaling；
3. `no_sync`、clipping 和 SGD 的正确先后顺序；
4. AMP 为什么必须先 unscale 再 clip，以及 overflow 时哪些状态要一起跳过。

它们都是小型 CPU 实验，范围只到训练机制本身；目标 SFT Trainer、CUDA 和真实数据质量需要另外验证。

迁移到具体框架时，要确认框架是否已经除以累积步数，以及有效 token 数在何时跨 rank 汇总。

发生 overflow 时，还要核对 optimizer、scheduler 和 GradScaler 是否共同保持不变。

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

Trainer 只需要训练文件和 readiness，不读取 validation/test 原文。独立的审计进程先证明训练文件按顺序等于
综合数据工件中的 train subset，再把数据划分、近重复检查和治理决定绑定到 readiness。

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

Readiness 回答“这份训练数据是否经过约定审计”。Tokenizer 和 mask 要在模型 revision 加载后继续检查。
训练入口应拒绝静默截断，保存模板与 mask 报告，并核对实际 collator 产生的最终 labels。

## 训练先从 8–32 条样本过拟合开始

先让 8–32 条样本 overfit，是为了验收训练管道。

预期现象是 loss 明显下降，生成结果能复现目标格式。若做不到，先检查模板、mask、冻结参数、optimizer、数据读取和学习率；
增加 GPU 解决不了管道错误。

仓库的离线 smoke test 会先检查输入记录，再依次贯通 Transformers 模板 mask、TRL collator labels 和一次参数更新：

~~~powershell
python projects/single-gpu-finetuning/smoke_trl_sft.py
~~~

接着运行 100–1000 steps smoke，检查 checkpoint、resume、adapter 加载和 held-out evaluation，
最后才启动完整实验。

训练配置至少保存下面五类状态：

- 模型与 Adapter 身份；
- Optimizer、scheduler、scaler 和 global step；
- Sampler 与各类随机数状态；
- Data manifest、tokenizer、template、mask policy 和 packing 配置；
- 代码 revision 与完整训练参数。

Train loss 下降说明模型更能预测训练 labels，不会自然证明部署任务更好。

## Resume 的位置必须与 optimizer commit 对齐

多 worker DataLoader 会提前把样本 ID 发到 worker 队列，因此同一时刻可能存在三个不同位置：

```text
sampler emitted       = 7  # 已发往 worker
main loop consumed    = 3  # 已交给训练循环
optimizer committed   = 2  # 已反映到参数更新
```

只有第三个位置表示训练状态已经提交。若把 emitted cursor 当成“已训练位置”，恢复时会直接跳过仍在队列中的样本。

Gradient accumulation 崩溃时还可能存在 partial-window gradients。两种恢复策略是：

1. 回到 optimizer-committed boundary，恢复该时刻的随机数状态，并重放尚未提交的样本；
2. 保存 pending sample IDs、累积位置与除数、梯度和崩溃时的 RNG sidecar，然后从窗口中间继续。

仓库的 CPU 实验会把两种恢复路径与不中断运行逐项比较，并用“漏掉梯度”和“恢复错误 RNG”的负例展示参数漂移。
最后写 manifest 可以检测当前 bundle 是否完整，但不会让样本、optimizer 和所有 checkpoint 文件自动成为一个跨故障事务。

真实 SFT 还要一起恢复 assistant mask、有效 token 分母、Adapter 和训练器状态。

随机数状态要覆盖 Python、NumPy、CUDA 与 DataLoader worker。分布式运行还需要各 rank 对恢复决定达成一致。
详细故障矩阵见
[单 GPU 微调项目](../practice/projects/single-gpu-finetuning.md)。

## Validation 与 test 回答不同问题

Validation 用于 early stopping 和超参数选择；最终 test 应限制查看次数。Teacher-forced loss 与真实生成质量不完全一致，
开放生成评测要使用部署时的解码参数、对话模板和工具 schema。

每个 checkpoint 至少比较：

- 目标任务与格式合法率；
- 通用能力与语言切片；
- 安全拒答和 over-refusal；
- 长度、重复与 tool behavior；
- 延迟、显存和每成功任务成本。

领域数据过采样可能提高目标任务，同时造成 catastrophic forgetting。记录通用 instruction、拒答、格式和领域样本比例，
并在真实 validation 分布上评测，不让 validation 跟随训练采样权重变化。

课程顺序、loss weighting 和采样权重都需要一个随机顺序或统一采样 baseline。早停依据应在训练前定义，并同时包含
主指标和不可退化门禁；不要从多个 checkpoint 中挑一个单项分数最好看的版本。

训练后还要检查 Adapter 与底座 revision 是否兼容、合并前后的 logits 和生成是否在容差内、部署模板是否一致。
量化完成后，再重新评测未见 Prompt、长输入、空输入和注入攻击样例。

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
