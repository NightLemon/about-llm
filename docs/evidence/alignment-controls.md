# 对齐与偏好优化证据台账

本页保存偏好数据契约、judgment 统计、RM/PPO/DPO 固定输入、目标 checkpoint 运行记录与精确证据边界，
供实现复核和 claim 审计使用。第一次学习请先读[对齐入门](../training/alignment-basics.md)和
[对齐进阶教材](../training/alignment.md)。

**读者入口**：[对齐入门](../training/alignment-basics.md) · [对齐进阶教材](../training/alignment.md) · [单卡微调项目](../practice/projects/single-gpu-finetuning.md)
{ .doc-nav }

| 查什么 | 去哪里 |
|---|---|
| 第一次理解 SFT、RM、DPO 与 PPO | [对齐入门](../training/alignment-basics.md)和[对齐进阶](../training/alignment.md) |
| 核对偏好数据、目标函数与固定 control | 本页第 1–15 节 |
| 判断一项结果能否写成 claim | 本页“当前仓库证据边界”和“常见错误结论” |
| 运行或复核脚本 | 本页末尾“复核任务与运行入口” |

本页保留完整公式、实现与证据边界。第一次学习先读[对齐与偏好优化入门](../training/alignment-basics.md)，只在确定数据和训练路线后进入对应算法小节。

本页的 CPU control、作者准备的 fixture 和历史 `recorded-report` 分别只能证明各自的局部契约。报告中的日期、输入、checkpoint revision 与 runtime 属于那次历史运行；当前 `--verify` 通过只复核保存 artifact 的身份、结构和范围，不重放训练或把结果升级为当前运行、人类偏好、目标 GPU、QLoRA、质量或安全证据。SHA-256 只有在期望 digest/manifest 已可信时才能发现 bytes 漂移，不能认证数据作者、checkpoint 发布者、执行者或时间。

对齐（alignment）不是单一 loss，也不是一次训练后永久获得的属性。它至少包含：遵循合法指令、帮助用户、与证据一致、在高风险或不确定时适当拒答、尊重权限，并在不同用户目标冲突时执行明确的优先级。训练只能塑造行为分布，权限和副作用必须由系统强制执行。

## Claim 与证据路由

| 要核对的 claim | 本页记录 | 权威正文 |
|---|---|---|
| 对齐对象与优先级 | 数据/评测 artifact 必须绑定目标用户、语言、风险、rubric 与 policy revision | [对齐入门](../training/alignment-basics.md) |
| SFT 行为先验 | Preference trainer 的输入必须来自已审计 train artifact；prompt/response mask 不得错位 | [SFT 数据闭环](../training/sft-data-pipeline.md) |
| 偏好数据质量 | 原始 pair、展示顺序、tie/invalid、逐标注者 judgment、split 与近重复控制 | 本页第 3 节 |
| Reward model | Bradley–Terry oracle、shortcut 反例、文本 Transformer 和目标训练入口 | 本页第 4 节 |
| PPO | GAE mask、sampled ratio、旧策略冻结、精确小环境和 learned-RM proxy 反例 | 本页第 5–6 节 |
| DPO | Reference-relative sequence objective、mask/reduction 与 tiny/Qwen controls | 本页第 7 节 |
| 其他 preference 方法 | 数据形态、reference、在线探索、长度偏差与风险目标 | [对齐进阶](../training/alignment.md) |
| 拒答、系统约束与发布 | Harmful/benign 分母、外部权限、副作用、paired evaluation 与 rollback | [安全](../quality/safety.md) · [Agent 评测](../quality/agent-evaluation.md) |

训练只能塑造行为分布。权限、资源归属、审批和副作用必须由模型外系统强制；偏好标签也只代表具体 rubric、
标注者群体、界面和时间，不是无噪声的“人类价值”标量。

## 3. 偏好数据的最小记录

一条 pairwise record 不应只有 `chosen` 和 `rejected`：

```json
{
  "prompt_id": "p-1042",
  "prompt": "...",
  "candidate_a": "...",
  "candidate_b": "...",
  "presentation_order": ["b", "a"],
  "label": "a",
  "strength": "slight",
  "rubric_revision": "help-safe-grounded-v3",
  "annotator_pool": "domain-experts-cn",
  "adjudication": "none",
  "generator_revisions": ["model-x@sha", "model-y@sha"]
}
```

还应记录语言、风险、长度、候选生成参数和 policy version。若只保留 winner，无法分析位置偏差、generator bias、ties 和 disagreement。

仓库提供 `about-llm.preference-jsonl.v1` 严格契约。除上述字段外，它要求 source/license/task/language/risk/group/split、rubric revision、annotator pool、adjudication 与 A/B generator revision；拒绝重复 JSON key、未知字段、非标准常数、非法 prompt role 边界和含糊 presentation order。当前候选是 assistant text-only pair，不覆盖多消息 tool trajectory 或多候选 ranking。`label` 可为 `a/b/tie/invalid`，二元标签必须配 `slight/clear`，tie/invalid 必须保留为 `not_applicable`，不能为了喂给 trainer 偷改成 winner。

~~~powershell
python -m about_llm.preference_cli audit --jsonl projects/single-gpu-finetuning/preference.example.jsonl --require-splits train,validation,test --output outputs/preference-audit.json
~~~

审计以 prompt + 无序 A/B 内容识别交换候选后的 duplicate pair，并检查 exact prompt、pair 和 group 跨 split 泄漏；同时报告 label、strength 与 preferred display position 分布、有序/无序 dataset identity。通过只证明这些 exact 规则，没有估计 position bias、annotator agreement、rubric validity、语义等价、许可/consent 或 tokenizer 截断。只有 train 且 `a/b` 的记录能转成 TRL conversational `prompt/chosen/rejected`；tie/invalid 会 fail closed。

### 3.1 原始 judgment 与一致性

最终 pair 标签不能替代逐标注者记录。仓库另定义 `about-llm.preference-judgment-jsonl.v1`：每条 judgment 只引用稳定 `pair_id`，保存 annotator、assignment batch、A/B 展示顺序、`a/b/tie/invalid`、强度、rubric revision、盲化/独立声明和耗时。审计要求 judgment 只指向选定的 validation/test case，同一 annotator 不重复判断同一 pair，每个 pair 有精确相同的 rater 数，并且两种展示顺序都达到最小覆盖；rubric mismatch、未知 pair、train 引用、重复 id、未盲化或非独立声明都会 fail closed。

~~~powershell
python -m about_llm.preference_cli evaluate-judgments --cases-jsonl projects/single-gpu-finetuning/preference.example.jsonl --judgments-jsonl projects/single-gpu-finetuning/preference-judgments.example.jsonl --case-splits validation,test --judgments-per-pair 4 --minimum-per-order 2 --bootstrap-samples 10000 --bootstrap-seed 17 --output outputs/preference-judgment-report.json
~~~

报告同时保留三个不同问题：

- raw pairwise agreement：同一 case 内标签相同的 annotator 无序对数 / 全部 annotator 无序对数，`invalid` 仍是一个标签；
- Fleiss’ κ：在每个 case rater 数完全相同时，对 `a/b/tie/invalid` 做 chance correction；若 expected agreement 为 1，则分母为零并报告 `null`，不能伪造 1；
- position effect：逐 case 计算 (P(A\mid A\text{ first})-P(A\mid A\text{ second}))，tie/invalid 不进入二元分母，再以 **case** 而不是单条 judgment 为 bootstrap 单位。

`blind_model_identity=true` 只是 artifact 中的声明，不能证明界面真的隐藏身份；保存 presentation order 也不能证明 assignment 随机。因此 position effect 是描述性诊断，不自动是因果效应。示例的 8 条 judgment 全由本仓库准备，只验证 schema、分母、κ 和 case-cluster bootstrap，不是人类标注、annotator quality、rubric validity 或真实 position bias 证据。

### 3.2 标注界面会改变标签

常见偏差：

- **position bias**：更偏好左/右或先显示候选；
- **length/verbosity bias**：更长回答看起来更充分；
- **style bias**：标题、礼貌和自信掩盖事实错误；
- **authority bias**：引用或专业措辞未经核验；
- **identity leakage**：标注者猜出模型来源；
- **criterion collapse**：把多个 rubric 维度压成一个模糊“更好”。

随机交换 A/B、隐藏模型身份、分维度标注、允许 tie/不可判断，并对关键切片双人标注与 adjudication。

## 4. Reward-model controls

Bradley–Terry control 使用
\(P(y_w\succ y_l\mid x)=\sigma(r_w-r_l)\) 与
\(\mathcal L_{RM}=\operatorname{softplus}(-(r_w-r_l))\)。
它只识别 reward difference；同一 prompt 下的共同 offset 不能从 pair data 识别，绝对分数也不是“用户价值单位”。

### 4.1 线性 RM：把目标与捷径写成可检查的量

为看清优化器究竟学了什么，先令 response 的数值特征为 \(f(x,y)\)，线性 scorer 为

\[
r_w(x,y)=w^\top f(x,y).
\]

对第 \(i\) 个 chosen/rejected pair，记 \(\Delta f_i=f(x,y_{w,i})-f(x,y_{l,i})\)，则 margin 为 \(m_i=\Delta f_i^\top w\)。带可选 L2 的 full-batch objective 与梯度是

\[
J(w)=\frac1n\sum_i\operatorname{softplus}(-m_i)+\frac\lambda2\lVert w\rVert_2^2,
\qquad
\nabla J(w)=-\frac1n\sum_i\sigma(-m_i)\Delta f_i+\lambda w.
\]

负 margin 的 pair 权重更大，梯度会把 \(w\) 推向 chosen 与 rejected 的特征差；但它不会判断该差异是事实质量还是 length/style shortcut。若某个 prompt 的所有 response reward 同加常数，pair difference 不变，因此这种数据也不能识别该 prompt 下的绝对 offset。

仓库的 NumPy CPU control 从零实现上述线性优化器，并显式采用 **strict pair accuracy**：只把 \(m_i>0\) 计为正确，zero-margin 另计 tie，而不是把 tie 当成半个或一个正确样本。

```powershell
python projects/single-gpu-finetuning/reward_model_toy.py
```

`confounded` 训练 pair 中 authored quality signal 与 length proxy 总是同向，因此两个权重相等，训练准确率达到 1，却在“质量仍正、长度方向反转”的 counterfactual held-out pair 上准确率为 0。加入长度正反两种 pair 后，length 权重在数值误差内为 0，训练与 held-out strict accuracy 都为 1。这个对照只说明 fixture 中的捷径可识别；它不是 text/Transformer reward model，不是人类 preference 质量证据，也没有评测 policy optimization 或 reward hacking。

### 4.2 从文本到 Transformer scalar reward

真实 RM 不是先手写 `quality_signal`，而是把完整 prompt+response 经 tokenizer 和 Transformer 编成 hidden states，再由 scalar reward head 输出 \(r_\phi(x,y)\)。Decoder-only classifier 常用最后一个 non-padding token 的 hidden state；因为 causal attention，它能聚合此前 token，但前提是 padding、EOS、截断和 attention mask 都正确。若只输入 response、chosen/rejected 使用不同模板，或静默截断掉判别证据，优化的已不是声明的 pair。

仓库提供完全离线的文本/Transformer CPU control：训练进程只读取 binary train artifact 与不含 held-out plaintext 的 readiness v2，本地 WordLevel vocabulary 也只由 train 构建；随后真实执行 readiness/train ordered binding、chat template、prompt-prefix/截断 audit、随机 tiny `GPT2ForSequenceClassification` forward、Bradley–Terry loss、backward 与 AdamW。reward head 置零，所以初始所有 margin 为 0、loss 精确接近 \(\log2\)；4 个 optimizer steps 后，reward head 与 token embedding 都发生更新，两个 authored train pair 的 strict accuracy 为 1。

```powershell
python projects/single-gpu-finetuning/smoke_transformer_reward_model.py
```

控制实验还故意把已见过的 `good/bad` 表面线索反转为 “chosen=bad、rejected=good”，训练后的 counterfactual strict accuracy 为 0。这不是自然语言质量结论，而是证明 tiny model 可以靠 lexical shortcut 拟合训练 pair。测试把 train/readiness 复制到没有 combined 文件的临时目录运行，并在模型初始化前拒绝缺失或篡改 readiness、以及顺序漂移的 train，因此证明当前 trainer 控制路径不需要 held-out plaintext；但无密钥 readiness 仍不能认证签发者或阻止整套 artifact 被协同重写。它也不证明人类标签、目标 checkpoint、广泛 counterfactual/OOD 鲁棒性、CUDA、reward hacking 或 policy optimization。

### 4.3 目标模型 RM 入口：先拒绝，再加载

TRL 0.29 `RewardTrainer` 会将 prompt 与 chosen/rejected 各自拼成完整序列，调用 `AutoModelForSequenceClassification(num_labels=1)`，优化 pairwise `-logsigmoid(chosen_reward-rejected_reward)`；若设置系数 \(\lambda\)，reward centering 项是 \(\lambda\operatorname{mean}[(r_w+r_l)^2]\)，约束的是 pair midpoint，而不改变 Bradley–Terry difference 的定义。一个容易漏掉的行为是：超过 `max_length` 的 pair 会被 trainer **过滤**。若不先审计，训练仍可成功结束，却悄悄改变数据集和切片分布。

仓库新增 held-out-free LoRA/QLoRA RM 入口。第一阶段只加载 train/readiness，适合无网络的权限与身份 gate：

```powershell
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --data-preflight-only
```

第二阶段下载固定 revision 的 tokenizer，但不加载模型；它真实渲染完整 prompt+response，拒绝 prompt-prefix mismatch、空 chosen/rejected completion、两侧 token 完全相同，以及任何超过 `max_length`、原本会被 trainer 过滤的 pair：

```powershell
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --tokenization-preflight-only
```

正式运行使用 `RewardTrainer`、`SEQ_CLS` LoRA，并显式保存 scalar head；`--qlora` 才加载 NF4/double-quantized frozen base，adapter、score head、梯度、optimizer 与激活仍不是 4-bit。默认 target modules 与 `score` head 名称适合常见 decoder checkpoint，但必须按目标模型结构核对，不能仅凭模型家族名猜测：

```powershell
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --qlora --target-modules q_proj,k_proj,v_proj,o_proj --modules-to-save score
```

入口在 trainer 构造后再次确认 pair 数未被过滤，并记录 model/revision、tokenization/readiness fingerprint、head/modules、pairwise loss、centering coefficient、trainable/total parameters、global step 与 Trainer metrics。测试用本地随机 tiny GPT-2 checkpoint 实际执行该入口的 `RewardTrainer`、`SEQ_CLS` LoRA、一个 optimizer step 与 adapter 保存，并确认保存后的 `lora_B` 不再是全零；因此生产入口的非量化 CPU 框架闭环有执行证据。它没有下载任一目标 checkpoint，也没有执行目标 module mapping、CUDA/QLoRA、显存测量或证明目标 RM 收敛和质量。

### 4.4 数值稳定

不要直接计算 `-log(sigmoid(delta))` 后让 `exp` 溢出。仓库提供稳定基线：

```python
from about_llm.finetuning.preference import bradley_terry_loss

loss = bradley_terry_loss(chosen_reward=3.0, rejected_reward=1.0)
```

`tests/test_preference_objectives.py` 检查 equal reward 的 loss 为 \(\log2\)，并覆盖极端 margin。

### 4.5 Construct boundary

RM 只在训练 pair 附近学习排序代理。Held-out human preference、长度/风格反事实、事实/代码 verifier、
adversarial/OOD slice、reward scale 与 policy 生成分布必须分别检查；训练准确率不能替代这些证据。

## 5. KL reference contract

抽象目标为
\(\mathbb E[r(x,y)]-\beta\,\mathbb E[D_{KL}(\pi\|\pi_{ref})]\)。
实际实现可能使用 sampled token log-ratio、per-token penalty 或 adaptive controller；必须记录 estimator、聚合和
reward scale。Reference 通常是冻结的 SFT snapshot，它约束偏移，不证明本身真实或安全。

## 6. PPO controls

PPO control 必须绑定 trainable policy、frozen reference、reward/value、rollout behavior snapshot、old log-prob、
optimizer/scheduler，以及 EOS、truncation、padding 与 variable-length masks。Clipped surrogate 比较
`rho_t A_t` 与 `clip(rho_t,1-epsilon,1+epsilon) A_t`；它限制 sampled-action contribution，不是完整 policy KL 的硬约束。

GAE 分开 bootstrap 与 continuation：

\[
\delta_t=r_t+\gamma b_tV_{t+1}-V_t,
\qquad
A_t=\delta_t+\gamma\lambda c_tA_{t+1}.
\]

| Transition | bootstrap `b_t` | continuation `c_t` |
|---|---:|---:|
| Absorbing termination | 0 | 0 |
| Time-limit / collector truncation | 由 objective 与 next value 决定 | 0 |
| 普通过渡 | 1 | 1 |
| Padding | 不计算 | 清零递推状态 |

三分类反例令 old/new 为 `(0.1,0.45,0.45)` 与 `(0.1,0.9-10^{-12},10^{-12})`。
恰好采到首动作时 ratio=1、clip 不触发，但完整 `D_KL(old||new)>10`；sampled proxy 与完整分布 KL 必须分开。

### 6.3 可执行 CPU reference

~~~powershell
python projects/single-gpu-finetuning/ppo_objective_toy.py
~~~

`about_llm.finetuning.ppo` 对一维/批量二维数组实现 mask-aware GAE 和 PPO clipped sampled-action surrogate。toy 用解析两步轨迹验证 TD residual/GAE，用同一 truncated transition 对比启用与禁用 bootstrap，用正负 advantage 验证上下界裁剪，并构造 sampled ratio=1、full-distribution KL 仍很大的反例。它只执行作者构造的 NumPy 数组，不含 policy/value/RM forward、reward shaping、rollout collection、optimizer、多 epoch minibatch、value loss、entropy bonus、reference KL controller 或 GPU；不能写成“已实现 PPO RLHF”或“训练稳定”。

### 6.4 PyTorch rollout 与 optimizer control

~~~powershell
python projects/single-gpu-finetuning/smoke_torch_ppo.py
~~~

下一层 control 是可完全枚举的两状态、两动作、两步环境：state 0 选择 action 1 得 1 分，state 1 选择 action 0 得 1 分，所有 episode 在第二步真正 terminated。因此当前 policy 的精确 undiscounted expected return 是

\[
\mathbb E[R]=\pi(a=1\mid s=0)+\pi(a=0\mid s=1),
\]

不需要用有限 rollout 均值冒充真实期望。`smoke_torch_ppo.py` 每轮从当前 categorical policy on-policy 采样 128 条完整 episode，冻结 policy-logit snapshot、逐动作 old log-prob 和 rollout value；随后用 GAE、advantage normalization、clipped policy loss、未裁剪 value MSE、entropy bonus、4 epoch × 64-action minibatch 的 Adam 更新 policy/value。默认 6 轮共执行 96 个 optimizer steps。测试要求：初始均匀策略精确期望为 1，最终超过 1.8；policy/value 参数都改变；存下的 old log-prob 在多 epoch 内逐元素不变，且能由对应 snapshot 精确重算。

这个 control 证明真实 PyTorch categorical sampling、value forward、GAE binding、旧策略统计冻结和 minibatch autograd/optimizer 链路能在该 authored MDP 上改善精确 objective。它没有 tokenizer/语言模型 token rollout、learned reward model、reference-policy KL controller、value clipping、time-limit truncation、checkpoint/resume、GPU 或 distributed actor/learner；单 seed、tiny tabular 环境的改善不证明 PPO 对目标 LLM 稳定、样本高效、安全或优于其他方法。更新后 observed sampled-action ratio 仍可越过 clip 区间，也再次说明 clipping surrogate 不是参数空间或 KL 的硬约束。

### 6.5 Tiny Transformer token PPO control

~~~powershell
python projects/single-gpu-finetuning/smoke_transformer_ppo.py
~~~

`smoke_transformer_ppo.py` 用随机 tiny `GPT2Model` backbone、policy head 与 value head 对整数 token ID 做两步自回归采样。词表大小为 6，固定 BOS 后，每生成一个目标 token 得 1 分；第二步 terminated。每轮复制并冻结 behavior-policy snapshot，另有从初始化起始终冻结的 reference。Reward shaping 使用

\[
r_t^{shaped}=r_t^{task}
-\beta\left[
\log\pi_{old}(a_t\mid s_t)-\log\pi_{ref}(a_t\mid s_t)
\right].
\]

括号内是 **sampled action log-ratio**，单个样本可以为负；只有在 \(a_t\sim\pi_{old}(\cdot\mid s_t)\) 下取条件期望，才得到 \(D_{KL}(\pi_{old}(\cdot\mid s_t)\|\pi_{ref}(\cdot\mid s_t))\)。脚本因此同时报告 sampled log-ratio mean 与在每个 sampled state 上对全部 6 个 action 求和的 exact categorical KL，不能混写两者。

两步 task reward 的精确期望可枚举全部 6 个 first-token branch，再对第二步条件分布精确求和；这等价于汇总全部 \(6^2\) 条 token trajectory，而不是用 rollout mean 估计。默认控制从均匀 policy 的精确期望 \(2/6=1/3\) 出发，测试要求 36 次 Transformer PPO optimizer step 后超过 1.8；同时验证 backbone/policy/value 都改变、reference 参数逐元素不变、每轮 old log-prob `requires_grad=False` 且能由 snapshot 在浮点容差内重算。

它证明随机 tiny causal Transformer 上的 integer-token autoregressive rollout、reference forward、sampled log-ratio reward、GAE 和 PPO autograd 链路。它不执行 tokenizer 或自然语言，不含 learned RM、目标 checkpoint、variable-length EOS、time-limit truncation、checkpoint/resume、CUDA 或分布式 rollout；“重复目标 token”是作者构造的可验证 reward，不是人类偏好、文本质量或安全目标。单 seed 小模型达到高 reward 不能证明目标 LLM PPO 收敛或对齐。

### 6.6 本地 tokenizer/chat-template 文本 PPO control

~~~powershell
python projects/single-gpu-finetuning/smoke_text_ppo.py
~~~

`smoke_text_ppo.py` 再加入本地 WordLevel tokenizer、chat template 和自然语言 prompt `Say good.`。随机 tiny GPT-2 policy 从 13 词表最多生成两 token：第一步生成 `good` 得 1 分，仍存活时第二步生成 EOS 再得 1 分；第一步 EOS 是真实 termination，之后的槽位是 padding，第二步仍未 EOS 则记录为 `max_new_tokens` truncation。Rollout 保存冻结 behavior snapshot、old/reference log-prob、value、逐 token mask 与边界，并为被长度截断的 post-action state 计算 value；policy 与 critic 使用分离的 tiny backbone，避免把共享随机表征上的 critic 干扰误写成 PPO 成功。

这个**有限两步任务**的报告 objective 在生成上限处结束，所以默认 `bootstrap_truncated=False`。`truncated` 描述生成停止原因，不自动决定 GAE 的 bootstrap mask；只有评估目标确实包含 cap 之后的 continuation return，且 next-state value 对该目标有效，才应令 \(b_t=1\)。脚本允许显式打开该反事实配置并标记它与报告的有限时域 objective 不一致，不能再拿后者的精确值验证 optimizer。

短时域可以精确枚举。设首 token 分布为 \(p_0\)，给定首 token \(a\) 的第二步 EOS 概率为 \(p_1(e\mid a)\)，则

\[
J=p_0(\text{good})+\sum_{a\ne e}p_0(a)p_1(e\mid a),\qquad
P(\text{good},e)=p_0(\text{good})p_1(e\mid\text{good}).
\]

初始零 policy head 给出均匀分布，因此 \(J=25/169\)，目标序列概率为 \(1/169\)。默认 96 次 optimizer step 后，测试要求精确 \(J>1.9\)、目标序列概率 \(>0.95\)，同时验证 EOS、truncation、padding 均在首轮出现，policy/value backbone 与 heads 均改变、reference 不变、old log-prob 可由 snapshot 重算。

这证明本地文本 tokenization/chat rendering、变长 autoregressive rollout、边界 mask、sampled reference penalty、actor/critic autograd 与精确短时域 oracle 的控制路径。它仍没有 learned RM、真实人类偏好、目标 checkpoint、长 response、adaptive KL、value clipping、checkpoint/resume、CUDA 或分布式 rollout，也不证明自然语言质量、安全对齐或生产稳定性。

### 6.7 冻结 learned RM 后的 proxy exploitation control

~~~powershell
python projects/single-gpu-finetuning/smoke_learned_rm_ppo.py
~~~

`smoke_learned_rm_ppo.py` 复用相同 tokenizer、chat template 与两 token response 空间，但不再把 authored verifier 直接当 reward。它先用完整 prompt+response 训练随机 tiny `GPT2ForSequenceClassification`：唯一 sparse pair 把 `good, EOS` 标为 chosen、`bad, EOS` 标为 rejected；零 scalar head 的初始 Bradley–Terry loss 为 \(\log 2\)，30 次 AdamW step 后训练准确率为 1、margin 为 5.57。随后 RM 参数全部冻结，sequence score 只绑定到 response 的最后一个有效 action，policy reference 与 behavior snapshot 也按原 PPO 契约冻结。

Pairwise loss 对所有 score 同加常数不敏感。实验因此减去最终训练 pair 的 score midpoint 作为**报告与优化的显式 centering 约定**；这不改变 pair margin 或 response 排名，也不把 OOD score 变成经过校准的效用。生成端显式 suppress `[UNK]`、`[PAD]` 与 role markers，只允许 EOS 和 7 个普通词 token；同一个 allowed-action mask 进入采样、old/new log-prob、entropy、KL 与精确枚举。两步 cap 且 EOS 提前停止时，可达 response 数量是

\[
|\mathcal Y|=1+(8-1)\times8=57.
\]

穷举冻结 RM 后，训练 chosen `good, EOS` 只排第 38，最高分反而是训练中未出现的 `good., good`；55/57 条可达 response 都未出现在训练 pair。训练集准确率与大 margin 因而不能支持“RM 在 policy support 上可靠”。

脚本同时精确积分完整 policy response distribution，而不是用 rollout mean 代替 ground truth：

\[
J_{RM}(\theta)=\sum_{y\in\mathcal Y}\pi_\theta(y\mid x)\,[r_\phi(x,y)-c],\qquad
J_{task}(\theta)=\pi_\theta(y^*=\text{good, EOS}\mid x).
\]

冻结 RM 后执行六轮 PPO，精确 proxy expectation 从 2.739 提升到 4.652，但严格 `good, EOS` success 从 \(1/64=0.015625\) 降到 \(4.99\times10^{-4}\)，约低 31 倍；最终最高概率 response 是 `good, good`。与此同时，给予首 token=`good` 和第二 token=EOS 分项奖励的 dense partial-credit 指标从 \(15/64\) 升至 0.566。报告同时保留这两个 authored verifier，避免把“严格序列目标下降”夸大成“所有任务指标都下降”。测试还验证 RM/reference 参数逐元素不变、old log-prob 可由 snapshot 重算、首轮 EOS/truncation/padding 都实际出现。这里可以严格写成“在这个完整可枚举的 authored support 上观察到 learned proxy 改善而预先声明的严格目标恶化的受控 reward-hacking 反例”。

它不能外推为真实用户或目标模型上的 reward hacking：preference pair、严格目标与 dense verifier 都是作者构造的，RM/policy 都是随机 tiny GPT-2，没有 held-out 人类标签、目标 checkpoint、长 response、真实 reward normalization、adaptive KL、checkpoint/resume、CUDA 或分布式 rollout。它证明的是训练准确率、RM reward、partial credit 与严格 success 必须分账，以及 policy optimization 会主动进入 sparse RM 未覆盖的 response 区域。

## 7. DPO controls

对同一 prompt，DPO 使用 response-token log-prob sum，并比较 policy 的 chosen–rejected margin 相对 reference
改善多少：

\[
u=\beta[(\log \pi_\theta(y_w)-\log \pi_{ref}(y_w))-(\log \pi_\theta(y_l)-\log \pi_{ref}(y_l))],
\qquad
\mathcal L_{DPO}=-\log \sigma(u).
\]

Prompt、padding 与被 mask token 不进入 sequence sum。Policy/reference 的 margin 相同时，`u=0`、loss 为 `log(2)`；
这不是只要求 `chosen policy logp > rejected policy logp`。

### 7.1 可执行检查

```python
from about_llm.finetuning.preference import dpo_loss

loss = dpo_loss(
    chosen_policy_logp=-2.0,
    rejected_policy_logp=-5.0,
    chosen_reference_logp=-2.5,
    rejected_reference_logp=-3.5,
    beta=0.2,
)
```

`preference.py` 函数只验证 per-pair 数学；仓库另有完全离线的真实 TRL DPO 控制闭环：

~~~powershell
python projects/single-gpu-finetuning/smoke_trl_dpo.py
~~~

它从严格 preference fixture 选择二元 train pair，用本地 WordLevel tokenizer、随机 tiny GPT-2 和冻结的初始 reference 构造 `DPOTrainer`。测试逐行确认 collator 前半是 chosen、后半是 rejected，`completion_mask` 对 prompt 全为 0、对 chosen/rejected completion 为 1；policy/reference 初始相同所以标准 sigmoid DPO loss 在浮点容差内为 \(\log2\)，真实 optimizer step 后同一 tiny batch loss 下降，reference 参数逐元素不变。

这仍只是 authored `good/bad` 控制 pair 和随机模型上的 CPU 机制证据，不是人类偏好数据、目标模型质量、CUDA、对齐安全或生产收敛证据。若 policy 与 reference 的 chosen/rejected margin 完全相同，理论 logit 为 0、loss 为 \(\log2\)；有限精度 forward 可有微小误差。

### 7.2 固定 Qwen 的目标 DPO 机制证据 { #target-qwen-dpo-control }

仓库另在固定 `Qwen/Qwen2.5-0.5B-Instruct` revision 上真实执行一次 TRL 0.29.1 sigmoid-DPO/AdamW step。离线 verifier 会重新绑定 checkpoint、binary train、held-out-free readiness、目标 token IDs、collator `[4,28]`、completion mask、LoRA 结构和报告范围：

~~~powershell
python projects/single-gpu-finetuning/run_qwen_target_dpo_control.py --verify projects/single-gpu-finetuning/qwen2.5-0.5b-dpo.recorded-report.json
~~~

初始 policy 与 adapter-disabled reference 相同，loss 为 `0.693147≈log(2)`；一步后同 batch loss 为 `0.333352`，两条 reference-relative margin 为 `8.566292/10.016453`。96 个 LoRA 梯度张量全部 finite，冻结的 494,032,768 个基座参数以及排除 LoRA 后的 `state_dict`、model config、generation config 指纹前后 exact。Transformers 会把 model/generation special-token config 对齐到 tokenizer，control 在 baseline 前显式完成并验证 Trainer 再执行时是 no-op。

一个容易误判的结果是：两次 reference forward 内 adapter 状态都实测为 disabled，冻结 state/config 也完全相同，但 reference log-prob replay 仍有 `0.547077` max-abs drift。报告保留两个 tensor hash 与实际误差，不把它叫作“reference 权重改变”，也不声称 bitwise deterministic。这里的 `good/bad` 只是固定样例；同 batch 一步下降只证明目标 checkpoint/TRL/PEFT 机制链路，不证明人类偏好、held-out 改善、对齐质量、安全、收敛、CUDA/QLoRA 或生产能力。

### 7.3 Reduction 与假设边界

默认 sequence log-prob 对 response token 求和。改成 token mean、加入 length normalization 或只比较尾部会改变目标；
仓库以 `sequence_log_probability(..., reduction="sum")` 固定这一口径；policy/reference 四项必须使用相同
tokenizer、template、mask 与 reduction。DPO 省去显式在线 RM/PPO loop，
仍依赖 Bradley–Terry 假设、reference、离线 pair 覆盖、beta、长度和泛化，不能据此排除 reward overoptimization。

## 8. 跨主题边界索引

| 主题 | 台账核对项 | 权威正文 |
|---|---|---|
| IPO、ORPO、KTO、SimPO、SLiC 等 | 数据是 pair/ranking/rating/unary 还是含 tie；是否有 reference、SFT term、margin 与 normalization | [对齐进阶](../training/alignment.md) |
| Offline 与 online preference | Data generator、当前 policy rollouts、探索成本、非平稳性和安全暴露 | [强化学习](../training/reinforcement-learning.md) |
| RLAIF / 原则驱动反馈 | Rubric、judge 版本、position/length/self-preference、注入、人工一致性与申诉 | [对齐进阶](../training/alignment.md) |
| Outcome / process / verifier | Final outcome、中间状态与 executable verifier 分开；可见 reasoning 不是完整内部计算 | [推理系统](../frontier/reasoning-systems.md) |
| 拒答 | Harmful compliance、benign refusal、safe completion utility 与多语言改写分开 | [安全](../quality/safety.md) |
| Offline/online evaluation | Blind paired cases、win/tie/loss、关键 slice、CI、sample-ratio、延迟与成本 | [评测方法](../quality/evaluation-methodology.md) |
| 系统层对齐 | Instruction hierarchy、ACL、tool schema、approval、idempotency、sandbox 与 incident response | [Agent 运行时](../applications/agent-runtime.md) |
| 发布 | 数据、mask/reduction、reference、checkpoint、quality/safety 与 rollback 一起 gate | [LLMOps](../applications/llmops-release.md) |

方法名不能代替目标定义。Sequence log-prob 的 sum/mean、beta/KL 口径、tie 处理、online/offline 数据分布和
风险约束都会改变问题；必须按所用库版本、公式和源码记录。

## 当前证据清单

| Control | 实际执行 | 不能推出 |
|---|---|---|
| Preference JSONL audit | 严格 pair schema、A/B 无序 identity、split/group/exact leakage 与 binary-train 导出 | 语义无重复、许可/consent、标签正确或真实人群代表性。 |
| Judgment audit | Stable pair binding、4-class labels、固定 rater 数、展示顺序覆盖、agreement、Fleiss’ κ 与 case bootstrap | 盲化/随机化真实发生、annotator 质量或因果 position effect。 |
| Linear RM | Bradley–Terry optimizer、strict pair accuracy、length-confounding 与 counterfactual | 文本 RM、人类偏好或 policy optimization。 |
| Tiny Transformer RM | Held-out-free readiness、WordLevel/template、scalar head、backward/AdamW 与 lexical-shortcut 反例 | 目标 checkpoint、广泛 OOD、CUDA 或 reward hacking。 |
| RewardTrainer 入口 | Data/tokenization preflight、TRL `RewardTrainer`、`SEQ_CLS` LoRA 与 adapter 保存 | 目标 module mapping、QLoRA、显存、收敛或质量。 |
| NumPy PPO | Mask-aware GAE、terminated/truncated、正负 advantage clip 与 sampled-ratio/full-KL 反例 | Policy/value/RM forward、optimizer 或训练稳定。 |
| 两状态 PyTorch PPO | 128 rollouts/轮、4 epochs × 64-action minibatch、6 轮 96 steps；精确 return `1→>1.8` | 语言模型、reference KL、GPU 或目标 LLM。 |
| Tiny Transformer PPO | 6-action、2-step 完整 trajectory 枚举；初始期望 `1/3`，36 steps 后 `>1.8` | Tokenizer、自然语言、learned RM、长序列或分布式。 |
| Text PPO | 本地 13-token vocabulary、EOS/truncation/padding；初始 `25/169` 与 target `1/169`，96 steps 后 objective `>1.9`、target `>0.95` | 真实偏好、目标 checkpoint、长 response、CUDA 或稳定性。 |
| Learned-RM PPO | 57-response 完整 support；RM accuracy `1`/margin `5.57`，proxy `2.739→4.652`，strict success `1/64→4.99×10^-4`，partial credit `15/64→0.566` | 人类效用或生产 reward hacking；三个 verifier 是 authored。 |
| Tiny TRL DPO | Binary train pair、prompt 的 `completion_mask=0`、初始 `log(2)`、真实 step 与冻结 reference | 人类数据、目标模型、质量、安全或收敛。 |
| 固定 Qwen DPO | Revision、`[4,28]` collator、96 finite gradients、base/state/config fingerprints、loss `0.693147→0.333352`、margins `8.566292/10.016453` | 两条 authored pair 和一步 CPU FP32 不证明对齐；reference replay drift `0.547077` 也不是权重更新。 |

## 尚缺证据

仓库仍没有真实人类 preference dataset、真实 annotator agreement/position-bias 实验、可靠目标 reward model、
learned RM 驱动的目标 checkpoint PPO、目标 CUDA/QLoRA 或 held-out 对齐质量。Lexical threshold 未经真实域校准，
registry 不是法律意见，无密钥 hash 不认证审计签发者。因此固定 Qwen DPO 只能说明目标权重机制链路，
不能说明任一模型已经完成偏好对齐。

## Claim 审查矩阵

| 常见表述 | 审查结论 |
|---|---|
| “RM 分数高就是用户价值高” | Reward scale 是特定数据/model 的排序代理，必须与 held-out preference、task verifier 和 OOD slice 对账。 |
| “PPO clip 严格限制 KL” | Clip 只限制 sampled-action surrogate 的一侧贡献，不是完整 policy KL 的硬约束。 |
| “DPO 只提高 chosen 原始概率” | 它比较 policy 相对 reference 的 chosen/rejected sequence margin。 |
| “Token mean 与 sequence sum 等价” | Reduction 改变长度权重与目标，四项 log-prob 必须使用同一契约。 |
| “DPO 没有显式 RM，所以不会 reward hacking” | Offline preference 和隐式 reward 假设仍会被过优化。 |
| “拒绝越多越安全” | 必须同时报告 harmful compliance 与 benign refusal。 |
| “模型已对齐，所以工具安全” | 权限、审批、幂等和 effect verifier 仍由外部系统负责。 |

## 复核运行入口

按 claim 选择最短 control，不要把多条结果拼成更高证据等级：

1. `preference_cli audit/evaluate-judgments`：核对 pair、split、judgment 与统计分母；
2. `reward_model_toy.py`、`smoke_transformer_reward_model.py`：检查 RM 数学、框架链路和 shortcut；
3. `train_reward_model.py --data-preflight-only` 与 `--tokenization-preflight-only`：在加载目标模型前 fail closed；
4. `ppo_objective_toy.py`、`smoke_torch_ppo.py`、`smoke_transformer_ppo.py`、`smoke_text_ppo.py`：逐层增加 PPO 证据；
5. `smoke_learned_rm_ppo.py`：复核 proxy 改善、严格目标恶化的受控反例；
6. `smoke_trl_dpo.py` 与 `run_qwen_target_dpo_control.py --verify ...`：分别检查 tiny 与固定 Qwen DPO。

每次保存输入/data/model identity、完整分母、old/reference snapshot、mask/reduction、optimizer state、原始结果、
失败样例与不能外推范围。
