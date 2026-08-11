# Single-GPU Fine-tuning

目标：在单张消费级 GPU 上完成可比较、可回归的领域 SFT/LoRA/QLoRA 与偏好优化，而不是只得到一个 adapter 文件。

## 已完成的机制基线

src/about_llm/finetuning/lora.py 从零实现 LoRA Linear：

- 基座权重冻结；
- B 零初始化，初始函数与基座一致；
- 只保存 adapter 和必要元数据；
- 合并为普通 Linear 后数值等价；
- 测试证明 optimizer 不会更新基座。

~~~powershell
pytest tests/test_lora.py
~~~

同目录还提供一个与目标 LLM 训练解耦的持续学习控制实验：

~~~powershell
python projects/single-gpu-finetuning/continual_replay_toy.py
~~~

它实际执行 task A→task B 的 PyTorch SGD，并输出完整 accuracy matrix、final ACC、BWT、FWT 与非负 peak-to-final forgetting。任务带显式 task-id feature，模型容量足以联合求解；固定 seed 下 no replay 会严重遗忘 A，而在 B 阶段混入全部 A 样本的 1:1 full replay 可同时拟合两项任务。该结果只验证指标实现、顺序更新与全量 replay control：单 seed、full batch、synthetic CPU fixture 不等于有限 replay buffer、真实 LLM/语料、安全 retention 或生产收益，且全量保存旧样本的隐私、删除、存储和训练成本尚未建模。

`--benchmark` 进一步执行 seed 0–19 的 no replay、64-example uniform reservoir 和 full replay 配对实验，并保存每个 seed 的完整矩阵、有限 buffer 索引、样本呈现量及 seed-level percentile bootstrap：

~~~powershell
python projects/single-gpu-finetuning/continual_replay_toy.py --benchmark
~~~

当前 fixture 中 64-example reservoir 相对 no replay 的旧任务 accuracy 平均增益为 0.4758，95% paired seed-level interval 为 [0.3389, 0.6104]；但新任务差值为 -0.0072，区间 [-0.0102, -0.0045]。三条路径只匹配 Task B 的 100 个 optimizer steps，每步总样本为 256/320/512，不能写成 compute-matched。任务数据没有跨 seed 重采样，因此区间只覆盖初始化与 buffer 选择，不覆盖任务/数据、目标 LLM、硬件、隐私或部署不确定性。

## 实验协议

至少比较四个系统：

1. base + zero/few-shot；
2. base + RAG（若任务依赖事实）；
3. PEFT LoRA/QLoRA；
4. 全参数微调或高预算参考（显存允许时）。

保持同一 chat template、生成参数和测试集。报告任务指标、格式合法率、通用能力回归、训练/峰值显存、耗时和 adapter 大小。

## 单卡数据契约

每条样本严格保留 `id/messages/source/license/task/language/risk/group_id/split`，可选 `metadata`。训练只对 assistant 区域计算 loss；padding、system、user、tool 是否 mask 必须通过 token 级检查。按来源或用户划分测试，禁止近重复跨集合。

先审计包含 train/validation/test 的 combined artifact：

~~~powershell
python -m about_llm.finetuning_cli audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --require-splits train,validation,test --output outputs/sft-split-audit.json
~~~

该命令拒绝宽松 JSON、未知字段和不合法对话，报告重复 id、exact messages、跨 split group/content、切片计数和 canonical fingerprints。它只覆盖 exact/group 规则，不检查 lexical/semantic near duplicate、许可、PII/secret、token 长度或 assistant mask。

将 source/license 标签变成显式决策，并运行有限敏感候选扫描：

~~~powershell
python -m about_llm.finetuning_cli governance-audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --policy projects/single-gpu-finetuning/governance-policy.example.json --evaluated-at 2026-08-06T12:00:00Z --output outputs/sft-governance-audit.json
~~~

policy 对 `source + license` 默认拒绝，显式绑定 training/evaluation purpose、evidence、review time、expiry 和允许的 risk labels。扫描只覆盖固定的 email、若干 key/token/JWT、private-key header 与 Luhn-valid card-like pattern；报告不含命中原文，人工 exception 绑定精确 record/span identity，陈旧 exception 也失败。这只是治理工作流与候选 reference，不是法律意见、consent 证明、完整 PII/secret 检测或 detector 精度证据。

再显式选择 lossy normalization 与阈值，运行字符 n-gram candidate gate：

~~~powershell
python -m about_llm.finetuning_cli near-audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --profile nfc_whitespace --ngram-size 5 --threshold 0.9 --output outputs/sft-near-duplicate-audit.json
~~~

它分别检查 full/user/assistant view，并给出 Jaccard 分子、分母与比较总数。finding 只是 lexical candidate，不是 semantic duplicate；无 finding 也不证明没有改写、翻译或答案片段污染。Readiness gate 是全对 \(O(N^2)\) reference，profile/threshold 尚未替真实领域校准。`audit.example.jsonl` 的有序 train 子集与 `train.example.jsonl` 完全相同；两者都不是有效训练语料。

规模化 candidate reference：

~~~powershell
python projects/single-gpu-finetuning/minhash_lsh_toy.py
~~~

它用稳定 SHA-256 shingle mapping、seeded affine MinHash 和 exact band tuple 生成候选，再逐候选复算精确 Jaccard。默认 64 hashes/16 bands×4 rows 的 authored fixture 将 10 个 pair 缩成 3 个候选；阈值 0.8 下 1 个 true positive、2 个 false positive，snapshot recall=1、precision=1/3。Exhaustive recall audit 为了得到 ground truth 又做了 10 次全对比较，因此不是可规模化验证；单独测试还证明 1-hash 配置会漏掉 Jaccard=2/3 的 pair。LSH 不保证召回、不发现语义/翻译重复，理想 `1-(1-s^r)^b` 只是 banding 调参曲线；当前 core 没有替代 fail-closed readiness gate。无密钥 hash/fingerprint 也不认证数据来源。

## 推荐递进

- 机制：本仓库 LoRALinear + 微型 GPT；
- 实用：Transformers + PEFT，对 0.5B–3B 模型做短序列 LoRA；
- 显存优化：4-bit 基座、gradient checkpointing、paged optimizer；
- 完整实验：数据卡、seed、checkpoint、早停、回归和合并/加载测试。

## PEFT 离线保存、重载与合并验证

`smoke_peft.py` 用随机初始化 tiny GPT-2 实际训练 PEFT LoRA，不下载模型。它先保存 exact base safetensors，再保存 adapter safetensors，从独立重载的 base 加载 adapter，执行 `safe_merge`，保存 merged full weights 并再次从磁盘重载：

~~~powershell
python projects/single-gpu-finetuning/smoke_peft.py
python projects/single-gpu-finetuning/smoke_peft.py --steps 8 --artifact-root artifacts/peft-export-control
~~~

8-step fixture 的 base/adapter/merged weight 文件分别为 110,632/4,608/110,632 bytes；训练前后所有 frozen base 参数 exact，构建时与 verify 后的 adapter reload 最大 logit error 都为 0，merge error 约 $8.94\times10^{-8}$，verified merged reload error 为 0。发布目录还保存 32-token WordLevel tokenizer、special tokens 和 chat template；保存/重载后 `tok5 tok7 <eos> tok9 <eos>` 的 token IDs 都是 `[5,7,2,9,2]`。

`about-llm-export-manifest.json` 是 strict canonical manifest。默认目录共有 13 个被覆盖文件、payload 236,589 bytes、manifest 2,297 bytes；descriptor 按 POSIX relative path 排序并绑定每个文件的 size/SHA-256，再绑定整个 descriptor set。Verifier 在 published-artifact reload 前运行，要求三个 safetensors 均可解析、base/merged 的完整 config payload 与 tensor key/dtype/shape signature 一致，并确认每个 target module 同时存在 LoRA A/B tensor；它拒绝额外或缺失文件、symlink、路径穿越、duplicate/non-canonical manifest、资源上限、size/hash 漂移，以及协同重算 hash 后的 weight/config/adapter/tokenizer 漂移。已有输出目录和已有 manifest 均拒绝覆盖。

这是标准 Transformers/PEFT artifact 加仓库 fail-closed verifier 的 CPU 控制，不是通用 checkpoint。adapter config 使用 immutable base identity string，manifest 另绑定 exact base 文件；但路径或 identity string 不是内容认证，PEFT 自身仍不会自动强制仓库 manifest，调用方必须先执行 verifier。可解析、同 key/dtype/shape 和 LoRA A/B tensor 覆盖都只是结构证据，不证明权重数值正确或确由声明的训练产生。目录没有 optimizer/scheduler/RNG/training-resume state；未执行量化基座 merge、目标 checkpoint 或 CUDA。随机 tiny loss 下降、hash 一致和数值等价不证明 license、runtime 兼容、任务质量、跨版本可移植性、来源认证或断电原子发布；unkeyed SHA-256 可被攻击者协同重算，单文件 exclusive-create+`fsync` 也不构成目录级原子发布。当前 verifier 也没有锁住随后由 Transformers 打开的文件，不能防止 verify 与 load 之间的并发替换（TOCTOU）；生产消费要配合不可变目录、访问控制、lease/content-addressed handle 或等价机制。

## TRL 单卡入口

先运行完全离线的 TRL 闭环。它用随机 tiny GPT-2、本地 WordLevel tokenizer、带 generation 标记的模板和仓库 SFT fixture，实际验证 `messages → assistant_masks → collator labels → optimizer step`；同时断言非 assistant labels 全为 `-100`、assistant labels 保留且 tiny batch loss 下降：

~~~powershell
python projects/single-gpu-finetuning/smoke_trl_sft.py
~~~

这只是 CPU 控制流/label 契约证据，不代表任何目标模型质量、CUDA 兼容、真实数据合法性或生产收敛。

## Reward model CPU shortcut control

`reward_model_toy.py` 用 NumPy 从零训练线性 Bradley–Terry scorer。两个 authored 特征分别表示 quality signal 与 length proxy；`confounded` 训练集让二者总是同向，所以模型能取得 1.0 的训练 strict pair accuracy，却在长度方向反转的 counterfactual held-out pair 上得到 0.0。补入长度正反两种 pair 后，拟合出的 length 权重约为 0，训练与 held-out accuracy 都为 1.0：

~~~powershell
python projects/single-gpu-finetuning/reward_model_toy.py
~~~

输出同时保存初始/最终 Bradley–Terry objective、margin、tie count、偏好概率与权重。strict accuracy 只把正 margin 计为正确，zero-margin 单独计 tie。该实验使用作者构造的数值特征和 preference，不读取文本，不执行 tokenizer/Transformer，也不证明真实人类标签、目标 RM、OOD 鲁棒性、reward hacking 或 policy optimization；训练准确率满分不是上线证据。

### 文本与 tiny Transformer RM 闭环

下一层控制实际执行完整 prompt+response 的 chat-template tokenization、随机 tiny GPT-2 scalar reward head、Bradley–Terry backward 和 AdamW。reward head 从全零开始，因此初始两个 pair 都是 tie、loss≈`log(2)`；4 步后 reward head 与 Transformer token embedding 均改变，authored train pair strict accuracy=1：

~~~powershell
python projects/single-gpu-finetuning/smoke_transformer_reward_model.py
~~~

脚本只读取 `preference.train.example.jsonl` 与 `preference-training-readiness.example.json`，复用 binary train ordered identity、lexical/governance readiness 和 prompt-prefix/截断 audit；tokenizer vocabulary 也只从 train pair 构建。测试在没有 combined 文件的临时目录中运行，并在模型初始化前拒绝缺失/篡改 readiness 与顺序漂移 train。一个故意把已见 `good/bad` 线索反转的 authored counterfactual 得到 strict accuracy=0，说明真实 Transformer optimizer 同样可能学习词面捷径。这个反转标签本身没有自然语言质量含义；无密钥 readiness 也不认证签发者。随机 tiny 权重、本地 tokenizer 和两条 pair 不证明人类 preference、目标 RM、广泛 OOD/counterfactual 鲁棒性、CUDA、reward hacking 或 policy optimization。

### 目标模型 LoRA/QLoRA RM 入口

`train_reward_model.py` 与 DPO 入口使用相同的 train-only 权限边界。`--data-preflight-only` 不导入训练依赖、下载 tokenizer 或加载模型，只验证严格 readiness 与 train ordered identity。`--tokenization-preflight-only` 才加载固定 revision 的目标 tokenizer，真实渲染完整 prompt+chosen/rejected，并在模型加载前拒绝 prefix mismatch、空 completion、两侧相同 token 或任何超过 `max_length` 的 pair。这样避免 TRL `RewardTrainer` 将过长 pair 静默过滤后仍继续训练：

~~~powershell
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --data-preflight-only
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --tokenization-preflight-only
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --qlora --target-modules q_proj,k_proj,v_proj,o_proj --modules-to-save score
~~~

正式入口配置 `AutoModelForSequenceClassification(num_labels=1)`、`SEQ_CLS` LoRA、可选 reward centering，并在 trainer 准备数据后再次断言 pair 数不变。自动化用本地随机 tiny GPT-2 实际贯通正式 `RewardTrainer`、一个 optimizer step、global-step/metrics 工件和 adapter 保存，并验证保存的 `lora_B` 已非全零；这与前面的手写 optimizer control 是两条不同证据。`modules-to-save=score` 和默认 attention module 名不是跨 checkpoint 的普适真理，必须检查目标模型 module tree。当前没有下载目标 checkpoint，也没有执行目标 module mapping、CUDA/QLoRA、真实人类 preference、显存或生产 convergence。

## PPO/GAE CPU objective control

`ppo_objective_toy.py` 是 PPO rollout 之前的数学控制层：它从零计算 masked GAE 与 clipped sampled-action surrogate，并分别输出 TD residual、advantage/return、bootstrap/continuation mask、unclipped/clipped objective、clip fraction、ratio 与 sampled KL proxy：

~~~powershell
python projects/single-gpu-finetuning/ppo_objective_toy.py
~~~

固定两步轨迹解析验证 padding 不进递推或均值，terminated 不 bootstrap；对 truncated transition，脚本显式比较“有可靠 next value 因而 bootstrap”和“不 bootstrap”两种约定，但两种都阻断 advantage 跨 episode 传播。PPO control 同时验证正 advantage 的 ratio 上界和负 advantage 的下界。三分类反例保持已采样动作 probability ratio=1，却把未采样概率质量移到极小尾部，使完整 \(D_{KL}(old\|new)>10\)；因此 clip fraction=0 和 sampled proxy=0 都不是全分布 KL 保证。

这只是作者构造 reward/value/log-prob/distribution 上的 CPU NumPy objective oracle。它没有执行 tokenizer、语言模型、value/reward model、rollout engine、optimizer、reference KL controller、value clipping/entropy loss、真实 EOS/truncation 来源或 CUDA，也不证明稳定 PPO 训练、目标策略质量或安全对齐。

### PyTorch 两状态 PPO 闭环

`smoke_torch_ppo.py` 再向前一步，实际执行 categorical on-policy sampling、policy/value forward、GAE、advantage normalization、clipped policy loss、value MSE、entropy bonus、多 epoch minibatch autograd 与 Adam：

~~~powershell
python projects/single-gpu-finetuning/smoke_torch_ppo.py
~~~

环境只有两个可观察状态，每个 episode 固定两步：正确动作各得 1 分，第二步 terminated。因为精确期望可直接由两个正确动作概率相加，测试不拿有限 rollout 均值代替 ground truth。默认 6 轮、每轮 128 episode、4 epoch、64-action minibatch，共 96 次 optimizer step；固定 seed 下精确 expected return 从均匀策略的 1.0 提升到 1.8 以上，policy/value 参数均改变。每轮 old log-prob 被 detached 保存，多 epoch 后仍逐元素不变，并与 rollout policy snapshot 精确一致。

这证明 tiny tabular MDP 上的 PyTorch rollout/GAE/optimizer 控制路径，不是语言模型 RLHF：没有 token generation、learned RM、reference KL controller、value clipping、time-limit truncation、checkpoint/resume、CUDA 或分布式 actor/learner。单环境单 seed 的改善不能外推到目标 LLM 的稳定性、样本效率、质量或安全。

### Tiny Transformer token PPO

`smoke_transformer_ppo.py` 使用随机 tiny `GPT2Model`、policy/value heads 与冻结 reference，真实执行两步 integer-token autoregressive sampling、sampled reference log-ratio reward、GAE 和多 epoch PPO optimizer：

~~~powershell
python projects/single-gpu-finetuning/smoke_transformer_ppo.py
~~~

词表大小为 6，BOS 固定，生成目标 token 每步得 1 分，第二步 terminated。脚本枚举全部 first-token branch 并精确积分第二步条件概率，得到等价于汇总 \(6^2\) 条 trajectory 的 expected task reward；因此测试可以要求它从均匀 policy 的 \(1/3\) 提升到 1.8 以上，而不是依赖有噪声的 rollout mean。默认 6 轮共 36 个 optimizer steps，验证 GPT‑2 backbone、policy/value heads 都改变，reference 逐参数不变，每轮 old log-prob 与冻结 behavior snapshot 一致。

Reward penalty 中保存的是 sampled action 的 `log π_old - log π_ref`，单条可为负；其在 behavior policy 下的条件期望才是 categorical forward KL。输出另对每个 sampled state 的全部 action 显式求和报告 exact KL。该实验没有 tokenizer/自然语言、learned RM、目标 checkpoint、variable-length EOS、time-limit truncation、checkpoint/resume、CUDA 或 distributed rollout；重复目标 token 只是 authored verifier，不代表真实偏好、质量或安全对齐。

### 本地 tokenizer/chat-template 文本 PPO

`smoke_text_ppo.py` 使用本地 WordLevel tokenizer 与 chat template 渲染 `Say good.`，随机 tiny GPT-2 policy 最多生成两个 token，并真实覆盖第一步 EOS termination、第二步未 EOS 的 `max_new_tokens` truncation，以及 termination 后的 padding：

~~~powershell
python projects/single-gpu-finetuning/smoke_text_ppo.py
~~~

作者构造的 dense task 是首 token 为 `good` 得 1 分、若继续则第二 token 为 EOS 再得 1 分。脚本用分离的 policy/value tiny backbones、冻结 reference 与 behavior snapshot，执行 sampled reference log-ratio shaping、GAE、clipped policy/value/entropy loss 和 96 次 optimizer step；同时为每个被长度截断的 post-action state 计算 value，并逐轮报告 EOS、truncation、padding、exact categorical KL、ratio 和 snapshot 不变量。

这里报告的是**到两 token generation cap 为止**的有限时域 return，因此默认不对 cap 后 continuation 做 value bootstrap。生成 API 的 `length`/truncated finish reason 不能单独推出 GAE 的 \(b_t=1\)：只有训练 objective 定义了 cap 后的 return，且 next value 与它匹配时才能 bootstrap。`bootstrap_truncated=True` 只作为反事实诊断开放，报告会明确标记 optimizer 与有限时域精确 objective 不一致。

13 词表的均匀初始 policy 可精确得到 expected reward \(25/169\) 与 `good, EOS` 概率 \(1/169\)；测试要求训练后分别超过 1.9 与 0.95。该精确枚举比有限 rollout mean 更适合验证这个短任务，但脚本没有 learned RM、真实人类 preference、目标 checkpoint、长 response、checkpoint/resume、CUDA 或 distributed rollout，不证明自然语言质量、目标 LLM PPO 或安全对齐。

### 冻结 learned RM 的 PPO proxy-exploitation 对照

`smoke_learned_rm_ppo.py` 把 RM 与 PPO 真正串成一条 CPU 闭环，并刻意保留独立 authored target verifier：

~~~powershell
python projects/single-gpu-finetuning/smoke_learned_rm_ppo.py
~~~

脚本只用一个 sparse pair 训练随机 tiny Transformer RM：chosen=`good, EOS`、rejected=`bad, EOS`。零 reward head 从 loss \(\log2\) 起步，30 step 后训练准确率为 1、margin 为 5.57；随后逐参数冻结 RM。因为 Bradley–Terry pairwise loss 不识别全局 score offset，脚本明确减去训练 pair midpoint；这个 centering 不改变排序，也不能修复未覆盖 response 的错误外推。

生成端 suppress `[UNK]`、`[PAD]` 与 role markers，只允许 EOS 和 7 个普通词 token；allowed-action mask 同时进入 rollout、old/new log-prob、entropy、KL 与精确枚举。两 token cap 的完整 support 为 \(1+7\times8=57\) 条。穷举发现训练 chosen 只排第 38，最高 RM score 是未见过的 `good., good`，55 条 response 没有训练覆盖。PPO 把精确 centered-RM expectation 从 2.739 提到 4.652，但严格 `good, EOS` 成功率从 \(1/64\) 降至 \(4.99\times10^{-4}\)；dense partial-credit reward 则从 \(15/64\) 升至 0.566。RM/reference 均保持不变，old log-prob/snapshot、EOS/truncation/padding 也有自动化验证。

因此该实验可写成“tiny authored support 上，learned proxy 提升而预先声明的严格 objective 恶化的受控 reward-hacking 反例”，不能写成所有 authored 指标都下降，更不能写成真实人类偏好或目标模型已经发生 reward hacking。它没有 held-out human labels、目标 checkpoint、长 response、真实 reward normalization、checkpoint/resume、CUDA 或 distributed rollout；训练准确率、RM reward、partial credit 和 strict success 必须分别报告。

## Preference 数据与 DPO 离线验证

严格 preference fixture 保留 prompt、A/B 原始候选、展示顺序、winner/tie/invalid、强度、rubric、annotator/adjudication、generator revision 与治理/split 字段。审计会把交换 A/B 后内容相同的 pair 仍视为重复，并阻断 exact prompt/pair/group 跨 split 泄漏：

~~~powershell
python -m about_llm.preference_cli audit --jsonl projects/single-gpu-finetuning/preference.example.jsonl --require-splits train,validation,test --output outputs/preference-audit.json
~~~

生产准备不把 combined 文件直接交给 trainer。审计身份读取 `preference.train.example.jsonl` 与含三种 split 的 combined artifact，要求前者逐记录、顺序敏感地等于后者的 **binary train subset**；train 中的 tie/invalid 会留在审计 artifact，但不会被强制改成 winner。它还要求显式 lexical profile/阈值、source policy 与固定 decision time。生成的 readiness v2 不含 held-out 原文：

~~~powershell
python -m about_llm.preference_cli prepare-training --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --audit-jsonl projects/single-gpu-finetuning/preference.example.jsonl --profile nfc_whitespace --ngram-size 5 --threshold 0.9 --governance-policy projects/single-gpu-finetuning/governance-policy.example.json --governance-evaluated-at 2026-08-06T12:00:00Z --output-dir outputs/preference-prepare
~~~

Preference lexical gate 比较 prompt↔prompt，以及跨记录 candidate A/B 的四种组合，避免候选从 A 换到 B 后漏检；它不比较 prompt↔candidate。报告给出 Unicode code-point character n-gram set 的精确 Jaccard 分子/分母、record pair 与 comparison count。实现仍是 O(N²) 全对 reference，normalization/阈值未按真实域校准，不能发现所有语义改写、翻译或污染。Governance gate 对 prompt 和两侧原始 candidate 执行同一 source/license/purpose/expiry/risk registry 与有限敏感候选扫描，报告不保存命中原文；registry 不是法律意见，未命中不证明无 PII/secret。

原始 judgment fixture 独立于最终 pair 标签。下面的 gate 绑定 validation/test case，要求每个 pair 恰好 4 个不同 annotator、A-first/B-first 各至少 2 个，并阻断未知/train pair、重复 annotator-pair、rubric mismatch 与未盲化/非独立声明。通过后报告 raw pairwise agreement、Fleiss’ κ、明确二元分母的 A-selection position effect，以及按 case 聚类的 percentile bootstrap：

~~~powershell
python -m about_llm.preference_cli evaluate-judgments --cases-jsonl projects/single-gpu-finetuning/preference.example.jsonl --judgments-jsonl projects/single-gpu-finetuning/preference-judgments.example.jsonl --case-splits validation,test --judgments-per-pair 4 --minimum-per-order 2 --bootstrap-samples 10000 --bootstrap-seed 17 --output outputs/preference-judgment-report.json
~~~

这 8 条 judgment 是作者构造的统计控制 fixture，不是人类标注。字段中的 blind/independent 是声明而非外部证明，顺序覆盖也不证明真正随机分配；因此示例 effect 不是因果 position bias，agreement/κ 也不证明 rubric 正确。

`smoke_trl_dpo.py` 使用真实 TRL 0.29 DPOTrainer、本地 tokenizer、随机 tiny GPT-2 和冻结 reference，不下载模型。它验证 train/combined readiness、`label → chosen/rejected` 映射、目标 tokenization prefix gate、collator 的 chosen-first/rejected-second 顺序、prompt/completion mask、相同 policy/reference 的初始 loss≈`log(2)`、optimizer 后 tiny-pair loss 下降以及 reference 参数不变：

~~~powershell
python projects/single-gpu-finetuning/smoke_trl_dpo.py
~~~

fixture 中的 `good/bad` 是作者构造的控制信号，不是人类偏好、对齐质量或安全标签。该闭环不证明目标模型 DPO、CUDA、真实域 length/position bias、annotator agreement 或生产收敛；tie/invalid 只被保留用于审计，不会被静默转成 DPO winner。

真实 LoRA/QLoRA DPO 入口只读取 binary train JSONL 与 preference readiness。加 `--data-preflight-only` 时不会导入训练依赖、下载 tokenizer 或加载模型；正式运行会先下载固定 revision 的 tokenizer，再复现 TRL 0.29 conversational tokenization：prompt 使用 `add_generation_prompt=True`，prompt+chosen/rejected 各自完整渲染。仓库把 TRL 只记录 warning 的 prompt-prefix mismatch 升级为失败，并拒绝空 completion、chosen/rejected token 完全相同以及任何会触发 `max_length` 截断的 pair；通过仍不证明 template 的语义正确或目标模型质量。`--qlora` 使用 NF4/double quant 与单模型 PEFT adapter-disabled reference 路径，当前无 CUDA 环境未实跑：

~~~powershell
python projects/single-gpu-finetuning/train_trl_dpo.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json outputs/preference-prepare/preference-training-readiness.json --output-dir outputs/dpo-run --data-preflight-only
python projects/single-gpu-finetuning/train_trl_dpo.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json outputs/preference-prepare/preference-training-readiness.json --output-dir outputs/dpo-run --qlora
~~~

Preference readiness 绑定 exact/group split audit、binary train identity、声明的 lexical candidate policy 与 source/sensitive governance 决策。正式数据仍需由有权限的人复核候选，并执行 consent、法律许可、语义近重复、position/length bias 和人工一致性审查；不能把 readiness pass 写成“数据已获法律许可”“无敏感信息”或“没有语义污染”。

SFT 数据准备与训练同样是两个权限域。SFT `prepare-training` 接受严格的 train-only JSONL、combined JSONL、显式 near-duplicate policy、governance policy 和固定 decision time，在可读 validation/test 的审计进程验证有序 train 子集绑定并生成 readiness v2；两个 SFT trainer 只接受 train JSONL 与 readiness，不读取 combined 原文。readiness 严格拒绝重复/未知字段、错误版本、失败 gate、指纹篡改和与当前 train 不一致的陈旧 artifact。通过只说明声明 lexical/governance 规则下没有阻断项，不等于语义无重复、法律许可或无敏感信息；未签名 SHA-256 也只能检测意外漂移，不能认证 readiness 的签发者。TRL 0.29 要求所用 SFT 模板能通过 `{% generation %}` / `{% endgeneration %}` 等机制返回 assistant mask；不能假设某个模型族名称自动满足条件。

tokenizer 下载后、模型权重加载前，入口会对每条样本实际调用目标 `apply_chat_template(..., return_assistant_tokens_mask=True)`，拒绝缺失/全零/错长/非二值 mask 以及会被 `max_length` 静默右截断的样本，并写出 `sft-template-mask-audit.json`。报告绑定有序数据身份、model/revision、Transformers 版本、template/special-token ids 以及逐样本 token+mask hash。它证明目标 tokenizer **报告了**结构合法的 mask，不独立证明模板作者标记了语义正确的 token，也没有检查 trainer collator 生成的最终 labels；正式训练仍应抽样可视化 token/mask/label。

若 checkpoint 模板不支持 generation mask，先审核并版本化一个本地 Jinja template，再通过 `--chat-template-path <template.jinja>` 同时交给 preflight 与 trainer；不要在代码里临时拼接另一个格式。自定义模板仍必须匹配该 checkpoint 的 special tokens 和部署格式。

~~~powershell
python -m about_llm.finetuning_cli prepare-training --train-jsonl projects/single-gpu-finetuning/train.example.jsonl --audit-jsonl projects/single-gpu-finetuning/audit.example.jsonl --profile nfc_whitespace --ngram-size 5 --threshold 0.9 --governance-policy projects/single-gpu-finetuning/governance-policy.example.json --governance-evaluated-at 2026-08-06T12:00:00Z --output-dir outputs/sft-prepare
python projects/single-gpu-finetuning/train_trl_sft.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/train.example.jsonl --readiness-json outputs/sft-prepare/sft-training-readiness.json --output-dir outputs/sft-run
~~~

先给第二条命令加 `--data-preflight-only`，可在不导入训练依赖、不下载模型时验证 train/readiness 边界。生产部署应让审计身份可读 combined，让训练身份只读 train 与经受控通道发布的 readiness；若攻击者能整体替换 readiness，它也能重算其中所有无密钥 hash，需用 ACL、签名或独立审计日志补足真实性。

示例数据只验证 schema、exact/binding、未校准 lexical candidate 和 authored-fixture governance 控制流，不能训练出有用模型。真实实验仍必须人工校准/复核 near duplicate 与敏感检测器，由有权限的负责人核验许可、consent、隐私和目标分布。

## 容量与风险

QLoRA 不是全部 4-bit；adapter、梯度、optimizer、激活和部分算子仍是高精度。序列长度与 batch 会显著增加激活。先用极小 batch dry-run，再逐步增大，并记录峰值。

`train_qlora.py` 提供无需 GPU/下载的 `--estimate-only`，拆分量化基座、adapter/optimizer、激活和运行时预留。估算用于筛掉明显不可行配置，不能替代目标 GPU 上的峰值实测：模型结构、attention kernel、词表 logits、bitsandbytes 版本和内存碎片都会改变结果。

~~~powershell
python projects/single-gpu-finetuning/train_qlora.py --model-id <model> --revision <commit> --num-parameters 7000000000 --num-layers 32 --hidden-size 4096 --max-length 1024 --estimate-only
~~~

真实训练去掉 `--estimate-only` 并增加 `--train-jsonl`、`--readiness-json` 与 `--output-dir`；readiness 先由上面的 `prepare-training` 生成。入口固定 NF4、double quant、BF16/FP16 compute、gradient checkpointing、assistant-only loss 和显式 target modules。本仓库当前环境没有 CUDA，因此只验证了估算、参数路径和 CPU 测试；真实 QLoRA 成功与峰值显存仍必须在目标消费级 GPU 上记录。

OOM 降级顺序是：micro-batch 降到 1（用梯度累积保持有效 batch）、启用 checkpoint/高效 attention、基于长度分布缩短序列、减少 target/rank，最后才换小模型。每次变化都要进入实验配置，不能一边降级一边沿用旧基线名称。

微调不能替代最新事实检索，也不能单独保证“无幻觉”。领域提升必须与通用能力、安全拒答和未见模板一起评测。

## MiniGPT 精确训练恢复控制

在接入目标 LoRA/QLoRA checkpoint 前，先运行一个状态面可完整枚举的 CPU control：

~~~powershell
python projects/single-gpu-finetuning/minigpt_resume_toy.py
python projects/single-gpu-finetuning/minigpt_resume_toy.py --artifact-path artifacts/minigpt/training.allmtrn
~~~

默认 fixture 用 7 条×5 token 数据、batch 2、dropout 0.2 和 6 次 AdamW update；一条路径不中断运行，另一条在第 3 次 update 后保存、重载再继续。固定输出的 batch 顺序为 `(6,5),(2,1),(4,0),(1,0),(6,5),(3,4)`，epoch 为 `0,0,0,1,1,1`，LR 为 `0.003,0.0026,0.0022,0.0018,0.0014,0.001`。53,917-byte artifact 包含 11,341-byte manifest、42,520-byte payload 和 51 个 tensor；恢复时和最终态的模型参数、每参数 AdamW step/一阶/二阶矩、当前 LR、permutation/cursor/epoch、data-generator RNG 与 Torch CPU dropout RNG 均逐位相等，loader 也不改变调用者的 Torch RNG。

格式绑定 Byte-BPE payload/config/tied weights、训练 identity 与数据 shape+content fingerprint，但不嵌入 7×5 数据 payload；调用者必须提供完全相同的数据。它只允许 zero-grad optimizer boundary，且只覆盖当前 CPU FP32 MiniGPT、单 AdamW param group、线性 per-update scheduler；因为 AdamW step 是 FP32 tensor，总 update 限制为 $2^{24}$，避免不可精确表示的 step。没有 Python/NumPy/CUDA RNG、AMP scaler、gradient accumulation、DataLoader worker/prefetch、distributed/sharded state、目标 checkpoint 或 CUDA 证据。六个 loss 为 `5.560535, 5.561857, 5.515058, 5.525568, 5.465405, 5.535903`，并不单调下降；bit-exact resume 只证明当前训练状态恢复契约，不证明收敛或质量。SHA-256 未加密钥，不认证来源；exclusive-create 与 file `fsync` 也不证明 crash/断电原子发布。

## 后续里程碑

1. 目标模型 SFT token/mask/label 与 DPO prompt/chosen/rejected 可视化和固定 artifact；
2. 在目标 CUDA 环境记录 QLoRA 实测峰值和 OOM 降级曲线；
3. 把已有 tiny CPU exact-resume 与 PEFT save/reload/merge/export control 扩展到目标 LoRA/QLoRA/CUDA，并完成 tokenizer/runtime 一体化发布；
4. 与 RAG/Prompt 基线的统一评测报告。
