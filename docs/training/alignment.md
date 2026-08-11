# 对齐、奖励模型与偏好优化

对齐（alignment）不是单一 loss，也不是一次训练后永久获得的属性。它至少包含：遵循合法指令、帮助用户、与证据一致、在高风险或不确定时适当拒答、尊重权限，并在不同用户目标冲突时执行明确的优先级。训练只能塑造行为分布，权限和副作用必须由系统强制执行。

## 1. 先定义“和谁、对什么对齐”

同一个回答可能对终端用户有帮助，却违反系统所有者的数据政策；可能符合多数标注者偏好，却伤害某一语言或群体。项目开始前应写清：

- instruction hierarchy 与不可覆盖约束；
- 目标用户、语言、地区和专业水平；
- helpfulness、truthfulness、harmlessness 等维度怎样权衡；
- 必须拒绝、可以安全替代、应正常帮助的边界；
- 哪些决定必须交给人或外部 policy engine；
- 申诉、纠错和 incident response。

“人类偏好”不是一个无噪声标量。偏好数据反映具体标注协议、标注者群体、界面和时间。

## 2. SFT 建立行为先验

Supervised Fine-Tuning 使用 \((x,y)\) 示范最大化 response token likelihood：

\[
\mathcal L_{SFT}
=-
\sum_{t\in response}
\log\pi_\theta(y_t\mid x,y_{<t}).
\]

高质量 SFT 能建立格式、语气、任务流程和拒答边界。它的局限是：

- 单个 target 把一个多解任务压成一个答案；
- 示范中未出现的 trade-off 没有直接监督；
- 教师文本的冗长、措辞和错误会被模仿；
- 只 mask response 还是同时训练 prompt，改变目标；
- chat template 错位会让 loss 看似正常却训练错误角色。

偏好训练通常从可用 SFT policy 开始；若基础指令跟随很差，pairwise objective 不会自动补齐所有行为。

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

`blind_model_identity=true` 只是 artifact 中的声明，不能证明界面真的隐藏身份；保存 presentation order 也不能证明 assignment 随机。因此 position effect 是描述性诊断，不自动是因果效应。示例的 8 条 judgment 全是 authored fixture，只验证 schema、分母、κ 和 case-cluster bootstrap，不是人类标注、annotator quality、rubric validity 或真实 position bias 证据。

### 3.2 标注界面会改变标签

常见偏差：

- **position bias**：更偏好左/右或先显示候选；
- **length/verbosity bias**：更长回答看起来更充分；
- **style bias**：标题、礼貌和自信掩盖事实错误；
- **authority bias**：引用或专业措辞未经核验；
- **identity leakage**：标注者猜出模型来源；
- **criterion collapse**：把多个 rubric 维度压成一个模糊“更好”。

随机交换 A/B、隐藏模型身份、分维度标注、允许 tie/不可判断，并对关键切片双人标注与 adjudication。

## 4. Bradley–Terry 奖励模型

给定 prompt \(x\)、preferred response \(y_w\) 和 rejected response \(y_l\)，一种常用模型是

\[
P(y_w\succ y_l\mid x)
=
\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
\]

每对样本的 negative log-likelihood 为

\[
\mathcal L_{RM}
=-
\log\sigma(r_w-r_l)
=\operatorname{softplus}(-(r_w-r_l)).
\]

只使用 reward difference，因此给同一 prompt 的所有 reward 加相同常数不改变 pair probability；绝对 reward 不能跨模型/数据集直接解释成“用户价值单位”。

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

### 4.5 奖励模型不是价值测量仪

Reward model 只在训练 pair 分布附近学到排序代理。常见失败：

- policy 生成超出 RM 训练分布的文本；
- length/style shortcut；
- 对事实、代码执行或安全细节缺乏外部验证；
- annotator disagreement 被压成单标签；
- reward scale 随 checkpoint/normalization 变化；
- policy 专门发现 RM 漏洞，即 reward hacking。

应保留 held-out human preference、adversarial pairs、长度匹配 pairs 和真实 task verifier。

## 5. KL-regularized policy objective

一个抽象目标是

\[
\max_\pi
\quad
\mathbb E_{x\sim D,\,y\sim\pi(\cdot\mid x)}
\left[r(x,y)\right]
-\beta
\mathbb E_{x\sim D}
\left[D_{KL}(\pi(\cdot\mid x)\|\pi_{ref}(\cdot\mid x))\right].
\]

这里写的是 sequence-distribution reverse KL。实际 RLHF 常用 sampled token log-ratio 的估计、per-token penalty 或其他近似；“KL”三个字不能证明实现计算了完整分布 KL。

参考策略通常是 SFT checkpoint 的冻结副本。它提供行为锚点，但不保证参考策略本身真实或安全。

### 5.1 \(\beta\) 的含义

在上述理想目标中，较大 \(\beta\) 更强地惩罚偏离 reference。训练实现还受 reward scale、token aggregation、adaptive controller 和 optimizer 影响。不同方法/库的 `beta`、`kl_coef` 或 temperature 参数不能只按名字横比。

## 6. PPO 在 RLHF 中做什么

Policy 先对 prompt 采样 response，reward model/规则给终局 reward，value model 估计 return，advantage 指示动作相对预期的好坏。PPO 的 clipped surrogate 常写成

\[
L^{clip}(\theta)=
\mathbb E_t
\left[
\min\left(
\rho_t(\theta)A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right],
\]

其中

\[
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{old}(a_t\mid s_t)}.
\]

PPO clip 限制的是 sampled action probability ratio 对 surrogate 的影响，不等于严格约束整个新旧策略的 KL，也不保证每次更新都小。

对 \(A_t>0\)，ratio 超过 \(1+\epsilon\) 后继续增大不会提高 clipped surrogate；对 \(A_t<0\)，ratio 低于 \(1-\epsilon\) 后继续减小不会提高它。另一侧并非对称地“把所有 ratio 都夹住”：实现必须先分别算 \(\rho_tA_t\) 与 clipped-ratio objective，再取逐动作最小值。常见 `clip_fraction` 只是样本中 ratio 落在区间外的比例。

训练日志里的 sampled KL proxy 也不是完整分布 KL。例如旧/新三分类分布分别为 \((0.1,0.45,0.45)\) 与 \((0.1,0.9-10^{-12},10^{-12})\)：若恰好采到第一个动作，observed ratio 为 1、不会触发 clip，但未采样动作间的质量重分配使 \(D_{KL}(\pi_{old}\|\pi_{new})>10\)。有限样本、token-wise proxy 和 sequence-distribution KL 必须分开命名。

### 6.1 RLHF 训练状态

典型系统同时维护：

- trainable policy；
- frozen reference policy；
- reward model；
- value model/value head；
- rollout engine 与旧 policy log-prob；
- optimizer、scheduler、advantage/return statistics。

还要处理 variable-length response、EOS、truncation、padding mask、reward whitening、value clipping 和 distributed rollout。把普通 language-model trainer 换一个 loss 并不等于完成 PPO RLHF。

### 6.2 Credit assignment

Sequence-level reward 需要分配到 token/action。给定 transition reward \(r_t\)、value \(V_t\) 与显式提供的下一状态 value \(V_{t+1}\)，仓库采用

\[
\delta_t=r_t+\gamma b_tV_{t+1}-V_t,
\qquad
A_t=\delta_t+\gamma\lambda c_tA_{t+1}.
\]

这里 \(b_t\) 是 **bootstrap mask**，\(c_t\) 是 **continuation mask**，二者不能混为一个 `done`：

- environment `terminated`：吸收终止，\(b_t=0,c_t=0\)；
- time limit/collector `truncated`：若 next state/value 有效，可显式选择 \(b_t=1\)，否则为 0；但新 episode 不能继承旧 advantage，所以总有 \(c_t=0\)；
- 普通 transition：\(b_t=c_t=1\)；
- padding：不计算 residual/advantage、不进入聚合，并把反向递推状态清零。

因此“截断时 bootstrap”不是无条件规则：只有 truncation 语义和 `next_value` 来源允许时才成立。若把 time-limit truncation 当 absorbing termination，return 会有系统偏差；若在 episode boundary 仍令 \(c_t=1\)，则会把下一条轨迹的 reward 泄漏到上一条。Generalized Advantage Estimation 可降低方差但引入 bias 与 \(\lambda\) 超参数；它不会修复错误的 reward、value、EOS 或边界标记。

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

## 7. DPO 的 reference-relative 分类目标

对同一 prompt 的 chosen/rejected response，定义完整 response sequence log-probability：

\[
\log\pi_\theta(y\mid x)
=
\sum_{t\in response}
\log\pi_\theta(y_t\mid x,y_{<t}).
\]

Prompt、padding 与被 mask token 不应计入该和。DPO logit 为

\[
u=\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}
{\pi_{ref}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}
{\pi_{ref}(y_l\mid x)}
\right],
\]

loss 为

\[
\mathcal L_{DPO}=-\log\sigma(u).
\]

等价地，比较 policy 的 chosen–rejected log-prob margin 相对 reference 改善了多少。它不是只要求 `chosen policy logp > rejected policy logp`；若 reference 已经给 chosen 更大优势，policy 需要在 reference-relative margin 上比较。

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

### 7.2 长度与 reduction

标准 sequence log-prob 是 response token log-prob 的**和**，因此长度和每 token 概率共同影响 margin。改为 token mean、加入 length normalization 或只比较尾部会改变 objective，不是无害实现细节。

仓库 `sequence_log_probability(..., reduction="sum")` 默认求和，同时允许显式 `mean` 仅用于演示差别。任何 chosen/reference 四项必须使用相同 tokenizer、chat template、response mask 和 reduction convention。

### 7.3 DPO 没有消除假设

DPO 避免显式训练/在线查询 reward model 和 on-policy RL loop，但仍依赖：

- Bradley–Terry 类偏好模型及 KL-regularized derivation；
- 固定 reference 与离线 preference distribution；
- pair label 质量和覆盖；
- policy 对离线候选以外 response 的泛化；
- beta、length、mask 和 optimization choices。

“无需 reward model”不等于“没有隐式 reward 假设”，也不等于不会 reward overoptimization。

## 8. 其他 preference objectives

IPO、ORPO、KTO、SimPO、SLiC 等名称覆盖不同目标：是否显式 reference、pairwise 还是 unary feedback、是否加入 SFT term、margin/normalization 怎样定义。版本与实现变化很快，应读取所用库版本的公式和源码，不用方法名猜 loss。

选择方法前比较：

- 数据是 pair、ranking、rating、binary desirable 还是含 tie；
- 是否能承担 reference forward 的显存/计算；
- 是否需要在线 exploration；
- length/style bias 如何控制；
- 是否有可执行 verifier；
- 目标是平均偏好、风险约束还是多目标 Pareto。

## 9. Online RL 与 offline preference 的差异

Offline DPO 类方法只在固定候选上学习，训练稳定、工程简单，但不会主动探索当前 policy 的新失败。Online RL/iterative preference 可收集当前 policy rollouts，更贴近更新后分布，却引入成本、非平稳性和安全暴露。

若上线 policy 已显著离开原 preference data generator，旧 pair 的覆盖证据变弱。可以周期性采样新 policy，做人工/可验证任务评测，而不是只观察训练 loss。

## 10. RLAIF 与原则驱动反馈

RLAIF 用模型依据 rubric 生成比较、批评或修订。Constitutional-style workflow 可把原则显式化，改善规模和一致性。它仍需要验证：

- judge 是否理解目标语言/专业域；
- position、length 和 self-preference bias；
- prompt injection 是否能操纵 judge；
- 原则冲突怎样处理；
- 明显优劣、同答案、随机答案等 control items；
- 与盲测人类标签的一致性和分歧切片。

高风险领域不能因“AI 反馈更一致”就删除专家与申诉机制。

## 11. Outcome、Process 与 Verifier

- **Outcome supervision** 只评价最终答案，容易获得，但 credit assignment 弱。
- **Process supervision** 对中间步骤或状态打分，可定位错误，但成本高且标注定义困难。
- **Executable verifier** 用编译器、单测、数学检查、schema 或模拟环境验证结果，在可验证任务上证据更强。

可见 chain-of-thought 不是内部计算的完整忠实日志。过程文本可被后生成、合理化或迎合 rubric。把它作为可检查 artifact，而不是安全证明；产品也要考虑敏感推理文本的存储和暴露。

## 12. 拒答与过度拒答

安全策略至少区分：

1. 明确应拒绝的高风险请求；
2. 表面相似但应正常回答的 benign neighbor；
3. 可以通过缩小范围、脱敏或提供安全替代来帮助的请求；
4. 信息不足，需要澄清而不是拒绝；
5. 用户获授权但系统需要外部权限验证的操作。

只测有害集会奖励“全部拒绝”。同时报告 harmful compliance、benign refusal、safe completion utility 与多语言/改写一致性。

## 13. 对齐评测

### 13.1 Offline paired evaluation

在同一 prompt 上盲测 baseline/candidate，随机交换位置，保留原始 rating 和 disagreement。报告总体 win/tie/loss、关键切片、置信区间和长度分布。若 judge 是 LLM，先对人工集校准并加入控制题。

### 13.2 防止只优化代理

同时监控：

- held-out human preference；
- reward model score 与真实 task success 的 divergence；
- response length、格式和模板化；
- policy–reference log-ratio/KL proxy；
- 通用能力、事实性、安全和拒答回归；
- prompt injection、jailbreak 与工具权限；
- 稀有语言和高风险切片。

Reward 上升而 held-out preference 下降，是 overoptimization 的直接警报。

### 13.3 Online evidence

线上 A/B 需要 guardrail、sample-ratio check、延迟/成本、用户群体切片和停止规则。点击、停留时间和重新提问是含噪代理，并受界面与旧 policy 影响，不能直接当“人类价值”。

## 14. 系统层对齐

训练模型不能强制真实权限。生产系统还需要：

- system/developer/user instruction hierarchy；
- retrieval ACL 与数据最小化；
- tool schema、allowlist、参数验证和 least privilege；
- 高风险动作审批、幂等和 reconciliation；
- rate limit、sandbox、secret isolation；
- trace、异常监控、红队和 incident response；
- 模型、prompt、policy 与评测版本回滚。

Agent 运行时的具体副作用协议见[运行时与副作用](../applications/agent-runtime.md)。

## 15. 发布门禁

### 数据

- pair split 按 prompt/source/user 独立单位，避免改写泄漏；
- A/B 顺序随机且保留原始 presentation；
- tie/disagreement 不被强制变成 winner；
- 各语言、长度、风险和 generator 来源有统计；
- preference data 与最终 eval 隔离。

### 训练

- 打印 chosen/rejected token IDs、response mask 与 sequence log-prob；
- prompt 单独渲染的 token IDs 必须是 prompt+chosen 与 prompt+rejected 的精确前缀；不要忽略模板或 special-token 差异造成的 prompt-prefix mismatch warning；
- 在训练前统计两侧完整长度并显式处理长样本；不能让 `max_length` 截断悄悄删除 prompt 或 completion；
- policy/reference 使用兼容 tokenizer/template；
- 记录 beta、KL 口径、length、EOS/truncation 和 reward scale；
- 极端 margin 下 loss/gradient 有限；
- checkpoint 含 policy/value/optimizer/rollout 所需状态。

### 评测

- 盲测 baseline/candidate 并报告 CI；
- 质量改善不是单纯长度/格式变化；
- harmful compliance 与 benign refusal 同时达标；
- 通用能力、事实性、工具安全和关键语言无回归；
- 线上 rollout 有权限、预算和停止机制。

## 16. 当前仓库证据边界

仓库已提供稳定的 Bradley–Terry/DPO per-pair 数学、mask-aware GAE/PPO clipped-surrogate CPU reference、两状态 PyTorch categorical rollout/optimizer control、随机 tiny GPT-2 integer-token PPO control、带本地 tokenizer/chat template 与精确有限时域 oracle 的文本 PPO control，以及 sparse tiny learned RM 驱动 PPO 后 proxy 上升、独立目标恶化的完整 support 反例；另有 synthetic linear RM optimizer control、随机 tiny GPT-2 上 held-out-free readiness/train binding、真实文本 tokenization/scalar reward head/backbone optimizer control、严格 pairwise preference JSONL/split audit、有序 binary-train/combined binding、prompt↔prompt 与四种跨记录 candidate surface 的字符 n-gram gate、prompt/两侧 candidate 的 source/sensitive governance、不含 held-out 原文的严格 readiness、目标 tokenizer prefix/空 completion/截断 preflight，以及随机 tiny GPT-2 上真实 TRL DPO tokenization、completion mask、冻结 reference 和 optimizer 闭环；另有可选 LoRA/QLoRA DPO 入口。新增 raw judgment binding、agreement/Fleiss’ κ 和 case-cluster position-effect bootstrap，但输入是 authored fixture，不能冒充人类实验。Lexical 阈值和 detector 未经真实域校准，registry 不是法律意见，readiness 也没有验证人类标签质量；无密钥 hash 不认证审计签发者。仓库目前没有真实人类 preference dataset、真实 annotator agreement/position-bias 实证、目标 reward model 训练、learned RM 驱动的目标 checkpoint PPO、目标模型 DPO 或 CUDA 证据。因此当前结果证明公式、artifact/权限/tokenization/统计控制流、tiny MDP/Transformer optimizer、线性/文本 shortcut 与 proxy exploitation 对照及 tiny-pair 优化，不证明任一目标模型已经完成偏好对齐。

## 17. 常见错误结论

- **“RM 分数高就是用户价值高”**：reward scale 只是特定数据和模型上的代理。
- **“PPO clip 就严格限制了 KL”**：clip 是 sampled-ratio surrogate，不是全分布硬约束。
- **“DPO 只提高 chosen 的原始概率”**：目标比较 policy 相对 reference 的 chosen/rejected margin。
- **“把 sequence log-prob 改成 token mean 不影响算法”**：这会改变长度权重与目标。
- **“DPO 没有 reward model，所以没有 reward hacking”**：离线偏好与隐式 reward 假设仍可能被过优化。
- **“拒绝越多越安全”**：benign refusal 会破坏可用性并可能造成不公平。
- **“模型对齐了，所以工具安全”**：权限和副作用必须由外部系统保证。

## 自测与实践

1. 推导 Bradley–Terry equal reward 时 loss 为 \(\log2\)。
2. 构造 policy raw margin 为正、但 reference-relative DPO margin 为负的例子。
3. 两个 response 每 token 平均 log-prob 相同但长度不同，比较 sum 与 mean reduction。
4. 设计含 position、length、style control 的中文 preference 标注协议。
5. 为什么 PPO clip ratio 不能证明整条 sequence policy 的 KL 小？
6. 运行 `smoke_trl_dpo.py`，解释为什么 prompt 的 `completion_mask=0`、reference 参数不变和初始 loss≈\(\log2\)分别验证不同不变量。
7. 运行 `reward_model_toy.py`，解释为什么 confounded 训练准确率为 1 仍不能支持 held-out 质量结论。
8. 运行 `smoke_transformer_reward_model.py`，定位初始 tie、训练 margin 与 authored counterfactual 失败分别证明和没有证明什么。
9. 对比 `train_reward_model.py` 的两个 preflight-only 模式，解释为何 readiness pass 不能代替目标 tokenizer audit。
10. 运行 `ppo_objective_toy.py`，解释为什么 truncated transition 的 bootstrap mask 与 continuation mask 可以不同。
11. 修改 sampled-ratio 反例的未采样尾部概率，观察 sampled proxy 与完整 categorical KL 如何分离。
12. 运行 `smoke_torch_ppo.py`，解释为什么精确 expected return 比单轮 sampled reward mean 更适合做这个环境的 optimizer oracle。
13. 为什么 PPO 多 epoch 更新期间必须保留 rollout policy 的 old log-prob，而不能用当前 policy 重新计算分母？
14. 在 `smoke_transformer_ppo.py` 中，为什么 sampled reference log-ratio 可以为负，而 exact categorical KL 不应为负？
15. 推导两步、6 词表的 exact expected target-token count，并说明它比单轮 rollout reward mean 多证明了什么、仍没证明什么。
16. 运行 `smoke_text_ppo.py`，验证初始 \(25/169\) 与 \(1/169\)，并解释为何 `max_new_tokens` 的 truncation 标签本身不能决定 GAE 是否 bootstrap。
17. 运行 `smoke_learned_rm_ppo.py`，解释为何 RM train accuracy=1、PPO proxy reward 上升与独立 target success 下降可以同时成立，以及 pair-midpoint centering 修复了什么、没有修复什么。
