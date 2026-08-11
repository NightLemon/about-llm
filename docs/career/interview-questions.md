# LLM 面试题与回答方法

## 怎样使用

每题先用 30 秒给结论，再用机制、权衡、失败模式和验证实验展开。优秀回答不只复述定义，还能说清 shape、指标、约束以及如何证明。

## Transformer 与生成

### 1. 为什么 attention 要除以根号 head dimension？

**回答骨架**：若 Q/K 分量近似独立且方差稳定，点积方差随维度增长；不缩放会让 softmax 饱和、梯度变小。缩放稳定 score 尺度。进一步说明实际分布不严格独立，但初始化与训练中该归一化仍有用。

**追问**：LayerNorm 已归一化，为什么还需要？温度与这个缩放有何区别？

### 2. causal mask 怎样防止信息泄漏？

训练可并行计算所有位置，但把未来 key 的 score 设为负无穷，softmax 后权重为零。要区分 causal mask、padding mask 和 loss mask。验证实验：只改未来 token，过去位置 logits 必须不变。

### 3. MHA、MQA、GQA 如何取舍？

MHA 每个 query 头有独立 K/V；MQA 共享一组；GQA 多个 query 头共享一组。减少 K/V 头可显著降低 KV Cache 与 decode 带宽，质量可能下降。给出 KV 元素公式并说明 query 计算没有按相同比例消失。

### 3.1 拿到 `config.json`，什么时候可以估 KV Cache，什么时候必须拒绝？

先限定为标准 dense MHA/GQA/MQA layout，并要求层数、query/KV head 数和 head dim 显式且自洽；若 head dim 由 hidden size 推导，要检查整除。理想 payload 是 `2 × L × H_kv × d_head × tokens × batch × bytes(element)`，不含 block/allocator、对齐、scale、workspace 或临时张量。出现 MLA/latent attention 或未知 remote-code architecture 时不能套公式；MoE 本身不改变 attention cache 公式，但总/激活参数也不能从几个 marker 猜出。`max_position_embeddings` 不证明有效上下文，config hash/commit metadata 不证明权重匹配、来源、许可、质量或 runtime layout。高质量回答还应提出固定 immutable revision、检查实际实现/权重 shape，并用目标 runtime 实测峰值。

### 4. Prefill 与 decode 的瓶颈为何不同？

Prefill 对多个 token 做大矩阵乘，较易 compute-bound；decode 每步 batch/序列维小，却反复读权重和 KV，常 memory-bound。用 TTFT、TPOT、GPU 利用率和 memory bandwidth 验证，不能只看总延迟。

### 5. temperature、top-k、top-p 分别做什么？

temperature 缩放整个 logit 分布；top-k 按排名截断，是否“恰好 k 个”取决于并列 tie-break；top-p 保留排序概率累计和首次达到阈值的最小前缀，必须保留 crossing token。说明 repetition/temperature/top-k/top-p 的处理顺序和每次重新归一化时点：若先 top-k 再 top-p，nucleus 用的是 top-k support 上归一化后的概率。`temperature=0` 通常是 API 的 greedy 特例，不能代入除法。即使概率相同，seed、RNG、CDF traversal、并列规则与 GPU 非确定性仍会影响重放；仓库 fixed-uniform CPU oracle 只证明一份显式单步契约，不证明 runtime 默认等价或生成质量。

### 5.1 tokenizer、model config 与 generation config 的 EOS 不同，谁错了？

不能先选赢家。把 BOS/EOS/PAD/decoder-start 归一成 ID 集合，逐对报告 exact/subset/overlap/disjoint，并检查所有 ID 是否落在 tokenizer size 与 model vocab 内。Generation EOS 是 tokenizer EOS 的 superset 可能有意加入 turn-end token；PAD=EOS 也常是明确选择。未解释的 disjoint 或越界是 review signal，但静态一致仍不证明运行时正确。

有效停止行为还取决于模型类 fallback、`generate()` kwargs、`max_new_tokens`/`max_length` precedence、sampling/beam/contrastive mode、stop-string tokenization、Transformers/vLLM/provider 版本和 server request override。normalized `to_dict()` snapshot 可能含库默认值，不是原始 JSON byte hash。高质量回答会保存三方 snapshot、最终请求、模板 token IDs、输出 token trace 与 finish reason，并在目标 runtime 做对照，而不是改一个 JSON 后宣布修复。

可进一步给受控实验：用 logits processor 强制 `[4,3]`，证明 config EOS `{2,3}` 在 3 停；call override EOS=5 后强制 `[3,5]`，证明 3 不停而 5 停；强制 `[4,6]` 配 `max_new_tokens=2`，证明无 EOS 的 length stop。该实验真实走 Transformers generation loop，却不让模型权重选择 token；它不能证明真实 checkpoint、vLLM/provider 或 finish-reason 字段，只能定位当前版本控制流。

### 5.2 Activation patching 能证明什么？怎样避免自欺？

先在 clean/corrupt pair 上固定连续 metric，缓存明确 hook site 的 clean activation，再把它 patch 进 corrupted forward。若 metric 改变，能支持“在这个干预定义、样本和 site 下存在因果作用”，不能自动证明这是唯一、自然或完整回路。报告 (m_{clean},m_{corrupt},m_{patched}) 和未裁剪 recovery；分母接近零时拒绝归一化。负对照至少包括未来/无关位置、随机 clean source、同分布无关样本和多个模板/seed。还要确认 hook 是 pre/post norm/projection 哪个 tensor，shape、batch 对齐、dropout 和缓存路径都正确。仓库随机 MiniGPT fixture 的 joint recovery=1、future control=0 只证明 hook 与 causal mask 控制流；token pair 是 post-hoc 选择，不能当目标模型语言机制证据。

### 5.3 流式 stop string 为什么不能对每个 chunk 单独调用 `endswith`？

因为 stop 可跨 token、event 和 UTF-8 byte chunk；一个 chunk 也可能同时含 stop 前文本、stop 与其后文本。先 strict incremental UTF-8 decode，再暂存“当前 suffix 中也是任一 stop prefix”的最长部分，只 emit 已不可能参与匹配的前缀。必须固定 include/exclude、大小写/normalization 和 overlap 语义；仓库选择逐 character first-completion，同一 character 多个完成按配置顺序。

客户端 matcher 只处理已收到文本。它命中后截断 UI 不等于 provider 返回 stop finish reason，也不证明服务端停止 decode、usage 不再增长、KV 已释放或停止计费。SSE framing、provider event state、应用 stop matching 和 server cancellation 是四层不同证据。

### 5.4 云 API token/费用 hard cap 为什么需要 reservation？

响应后才累加 usage 会让并发请求重复花同一份剩余额度。发送前应以目标 tokenizer 估计 input，从实际 request body 提取唯一 maximum output，并将 request identity/cap/计费 scope 与 receipt 绑定后原子预留 token 和估算费用；成功按 provider-reported usage 结算、释放未用容量。只有确定 transport 没发送才能 cancel；timeout/partial stream 后 usage 未知时保守占用 reservation，并异步对账。若 actual 超过 reservation，调用已经发生，必须先记录超额再 fail closed。

这仍不等于“费用绝不会超”。Estimate 可能不含 provider 隐藏、cache 或 reasoning token，费率会按 model/tier/time 漂移，重试/取消也可能计费。价格快照要绑定 provider/model/revision/checked_at；多 worker 还需要 durable atomic quota，最终用 provider billing export reconciliation。Request hash 只绑定选中的 bytes，不证明 caller、真实发送或保密；micro-USD policy estimate 不能证明发票。

若用 SQLite，`BEGIN IMMEDIATE` 可让同一文件的多 writer 争抢容量时串行化，并让 reservation/event 在进程退出后保留；但不能和远程 HTTP/provider billing 做原子提交。Crash 后 active 记录不能靠 TTL 自动 cancel，因为进程死亡不证明请求未发送；要用 stable call id、attempt/request id 与 billing export 对账，能证明未发送才释放，否则 conservative uncertain。无密钥 fingerprint 只能发现非协同漂移，不能抵抗能改库并重算 hash 的攻击者；单机 SQLite 也不是分布式 quota 或 exactly-once billing。

### 5.5 为什么一次逻辑调用不能只 reserve 一次再自动重试三次？

因为每次 replay 都可能是独立计费调用。只预留一次 maximum output，却允许三次 attempt，最坏 token/费用暴露接近三份；最终成功响应的 usage 只描述它自己，不能证明前两次 timeout/5xx 免费。准确实现要在每个 attempt 发送前用唯一 attempt id 独立 reserve，并分别 settle、确定未发送才 cancel、否则 uncertain；logical call id 只用于聚合，不替代 attempt ledger。若 executor 没有 attempt-start hook，宁可强制 `max_attempts=1`，由外层逐次预留后重放。

收到 HTTP 500 虽然 outcome known，也只说明“服务端返回了 500”，不说明零 usage/零费用；2xx 缺 usage、response parse failure 和 client cancellation 同样不能释放。只有 Pool/Connect 前失败等结构化证据能证明 request 未发送。SQLite 解决本地并发与重开，不消除 provider effect—local commit 的 crash window。

## 训练与微调

### 6. 预训练 loss 下降为何不保证产品质量？

next-token loss 是代理目标，可能改善流畅度却不改善事实、工具、拒答或业务成功。需要域外验证、任务评测、安全与成本共同判断。

### 7. LoRA 的参数量和前向公式是什么？

冻结 W，学习 B 与 A，更新为 W 加 alpha/r 乘 BA。参数量约为 r 乘输入输出维度之和。B 零初始化保证初始函数不变。追问 target modules、rank、合并和多 adapter 服务。工程上 adapter 必须绑定 exact base、target module/fan-in-out/scaling 约定和 tokenizer/template；`merge_and_unload` 成功只说明当前内存路径能合并。可信导出还应从独立 base 重载 adapter，比较 logits，保存 merged full weights 后再次重载比较，并重跑任务/安全评测。

目录 manifest 必须覆盖 config、weights、tokenizer/template 等完整 file set，并在 load 前强制验证 extra/missing/path/size/hash 和关键语义；框架不会因旁边存在 JSON 就自动执行它。路径或 identity string 不是 base 内容认证，unkeyed hash 也不认证来源；FP32 tiny merge 不证明量化基座、目标 checkpoint、跨版本兼容或原子发布。

### 8. QLoRA 为什么不等于“4-bit 训练”？

通常只有冻结基座权重低位存储；反量化计算、LoRA、梯度、optimizer 和激活仍用更高精度。显存还受序列长度和 checkpointing 影响。

### 9. 如何构造 assistant-only loss？

先用真实 chat template 序列化消息，再通过 token 边界生成 label mask；user/system/tool/padding 设 ignore index。用 token 级可视化和特殊 token 单测，不能按字符串长度猜边界。

### 9.1 如何证明 SFT 切分没有泄漏？

不能绝对证明“没有”。先按用户、thread、source document 或 problem family 分组再切分，门禁检查 id、exact content 和 group 跨 split；再用规范化字符 n-gram/MinHash/LSH、embedding 或任务特定规则检查 near duplicate、答案独特片段与时间穿越。Jaccard 回答的是 shingle-set 重叠，不是语义等价；必须报告 normalization、n、阈值、比较分母、人工复核和漏检边界，并隔离 test 权限。exact/lexical gate 通过不代表语义无重复，manifest hash 也不证明许可或隐私合规。

### 9.2 MinHash/LSH 为什么不能直接作为“无泄漏”门禁？

MinHash 签名相等率只是集合 Jaccard 的随机近似；将 \(k=br\) 个分量切成 \(b\) bands×\(r\) rows 后，理想候选概率为 \(1-(1-s^r)^b\)，不是确定召回。增大 bands 提高召回也增加候选，增大 rows 则相反。LSH 命中后仍要 exact Jaccard recheck；未命中可能是假阴性。仓库 64/16×4 authored snapshot 的 10 个 pair 产生 3 candidates、1 true positive、2 false positives，snapshot recall=1/precision=1/3；但 1-hash 反例会漏掉 Jaccard=2/3 的 pair。生产要在目标语言/长度/来源切片抽样 exact ground truth，报告 precision/recall/漏检和区间；exhaustive recall audit 本身仍是 \(O(N^2)\)。这些都是 lexical shingle 证据，不覆盖语义改写、翻译、答案片段或许可隐私。

### 10. 何时用 RAG，何时微调？

易变、私有、需引用事实优先 RAG；行为、格式、风格和稳定领域模式可微调。先做 Prompt 基线，按错误 taxonomy 决策；二者可组合。

### 11. DPO 与 PPO/RLHF 的训练信号有何不同？

DPO 用 chosen/rejected 对和 reference policy，把偏好优化写成分类式目标；PPO 通常先训练 reward model，再对在线采样 response 估 advantage，并用 clipped ratio、KL 等约束更新 policy。DPO 实现简单不等于天然无偏：它仍依赖偏好覆盖、reference、采样分布与超参数。回答时写清 sequence log-prob 是 response token log-prob 的和，prompt/padding 要由 completion mask 排除，length normalization 若使用就是另一个目标口径。工程追问可展示：A/B presentation order、tie 与逐标注者 raw judgment 必须保留，先 gate 未知/train case、重复 annotator-case、rubric、固定 rater 数和双顺序覆盖，再报告 raw agreement 与 Fleiss’ κ；顺序覆盖本身不证明随机化，case 内 position effect 也不自动是因果效应。trainer 只能接收经 combined audit 绑定的 binary train subset；prompt token IDs 必须是两侧完整对话 token IDs 的精确前缀，否则基于长度切片会错位；超过 `max_length` 应显式处理而不是静默截断；policy/reference 初始相同时标准 DPO loss 理论为 \(\log2\)；reference 必须冻结；tiny-pair loss 下降只证明控制流能优化，不证明人类偏好质量或安全对齐。

### 11.1 Reward model 训练准确率很高，为什么仍不可信？

Pair accuracy 只回答 RM 能否排序当前 pair，不能证明它依据了正确理由。若 chosen 在训练集中总是更长、格式更整齐或语气更自信，模型可利用 length/style shortcut 获得满分；同分 margin 还必须明确按 tie、半正确还是错误处理。先保留 train/held-out 分离与 pair binding，再构造事实质量不变但长度、格式、身份线索反转的 counterfactual pairs，报告 loss、margin、strict accuracy、tie count 和按属性切片的结果，并用 held-out human preference 与真实 task verifier 校验。Reward scale 只在当前 checkpoint、normalization 和数据分布内有意义；policy 会产生 OOD response，甚至主动放大 RM 漏洞，所以还要做 adversarial/OOD slice、不同 RM 或人工复核、KL/score 分布监控和 reward-hacking red team。工程上还要防止 `RewardTrainer` 静默过滤超长 pair：在模型加载前用目标 tokenizer 渲染两侧完整序列并拒绝超限，trainer 准备后再次核对 pair 数。仓库的线性 NumPy control 演示 confounding；随机 tiny GPT-2 control 展示 lexical shortcut；本地 checkpoint 又实际贯通正式 RewardTrainer/LoRA optimizer 和非零 adapter 保存。这些证明数学、捷径和框架链路，仍不证明目标 RM 或真实人类偏好质量。

### 11.2 GAE 中 terminated 与 truncated 为什么不能共用一个 done？

GAE 要区分 value bootstrap 与 advantage continuation：

\[
\delta_t=r_t+\gamma b_tV_{t+1}-V_t,\qquad
A_t=\delta_t+\gamma\lambda c_tA_{t+1}.
\]

真正 `terminated` 的吸收状态令 \(b_t=c_t=0\)。time-limit/collector `truncated` 若保留了有效 next state/value，可令 \(b_t=1\)，但它仍是 episode boundary，所以 \(c_t=0\)；否则下一条轨迹的 reward 会泄漏回来。Padding 同样不能进递推或均值。进一步追问 PPO：clipped ratio 只改变已采样 action 对 surrogate 的贡献，`clip_fraction` 或 sampled KL proxy 都不是完整 token/sequence 分布 KL 的硬上界；正 advantage 主要裁 ratio 上界，负 advantage 主要裁下界。

### 11.3 一个可信的 PPO smoke 至少要冻结什么、验证什么？

Rollout 必须绑定生成它的 behavior-policy revision、逐动作 old log-prob、value、reward、terminated/truncated/EOS/padding 与采样配置；多 epoch 更新期间 old log-prob、advantage 和 return 不能随当前 policy 重新计算。最小控制要证明：stored log-prob 可由 rollout snapshot 重算，policy/value 参数确实更新，ratio/clip fraction/KL proxy 分母明确，loss 有限，精确或独立评测 objective 改善。只看训练 reward 不够，因为它可能来自采样噪声、RM 漏洞或数据变化。仓库两状态 control 有可枚举 expected return，适合证明 optimizer 链路；它没有语言模型、RM、reference KL、截断环境或多 seed，因此不能支持“完成 RLHF”或“训练稳定”。

### 11.4 Sampled log-ratio penalty 与完整 KL 有什么区别？

对从 behavior policy 采到的动作，\(\log\pi_{old}(a\mid s)-\log\pi_{ref}(a\mid s)\) 是单样本量，可以为负；在固定 state 下对 \(a\sim\pi_{old}\) 取期望，才等于 \(D_{KL}(\pi_{old}(\cdot\mid s)\|\pi_{ref}(\cdot\mid s))\ge0\)。有限 rollout 的 mean 仍有采样误差，state 分布也由 policy 决定。面试中应分别说明 sampled estimator、对全部 action 求和的 categorical KL、逐 token 聚合和整条 sequence-distribution KL，不能只说“算了 KL”。仓库 tiny Transformer PPO 同时输出 sampled log-ratio mean 与 sampled states 上的 exact categorical KL，但词表仅 6、horizon 仅 2，不证明大词表长序列的估计质量。

### 11.5 `max_new_tokens` 截断后 GAE 一定要 bootstrap 吗？

不一定。`length`/truncated 是 collector 或生成 API 的停止原因，\(b_t\) 则由训练 objective 的 continuation semantics 决定。若任务 return 明确定义为“只评价 generation cap 内的 response”，cap 后没有纳入目标的 reward，默认不应借一个 learned next value 改写这个有限时域目标；若 cap 只是观察窗口，底层 MDP 继续且保存了与同一 objective 匹配的 next state/value，才可以 bootstrap。两种情况都不能让 advantage 跨到下一条 episode。回答时应同时给出 stop reason、reward horizon、next-state 构造、value revision 和 bootstrap/continuation mask，而不是把一个 `done` 布尔值直接取反。

仓库 text PPO control 会计算截断后的 post-action value，但有限两 token oracle 默认不使用它；显式打开 bootstrap 时，报告会标记 optimizer 与该 oracle 不一致。这个反例也说明：训练曲线改善与独立精确 objective 改善必须分开检查。

### 11.6 RM 训练准确率 100%，PPO reward 又上升，能否证明对齐改善？

不能。RM train accuracy 只约束已标注 pair 的相对顺序；pairwise loss 甚至不识别全局 score offset。Policy optimization 会改变 response distribution，主动搜索训练 support 外的高分区域，因此还要报告 held-out/counterfactual preference、完整或抽样的 policy-support coverage、独立 task verifier、score/长度/格式切片、KL 与人工复核。减去 chosen/rejected midpoint 只能固定一个 offset 约定，不改变排序，更不能校准 OOD utility。

仓库 learned-RM PPO control 给出可穷举反例：唯一训练 pair 达到 accuracy=1、margin=5.57，但 chosen 在 57 条 allowed response 中只排第 38，最高分是 `good., good`；PPO 把精确 RM proxy 从 2.739 提到 4.652，同时把严格 `good, EOS` 成功率从 \(1/64\) 降至 \(4.99\times10^{-4}\)。dense partial credit 反而从 \(15/64\) 升至 0.566，所以回答时还必须说明“哪个外部 objective 恶化”，不能宣称所有任务指标都下降。因为 support 完整枚举且 RM/reference 冻结，这能证明该 authored tiny control 中相对严格目标发生 proxy exploitation；它仍不是人类效用、目标模型或生产 reward hacking 的实证。

### 12. Scaling law 能告诉你该训练多大模型吗？

只能在数据、架构、损失和算力口径相近且已验证的区间内做经验外推。`6ND` 是 dense Transformer 训练 FLOPs 的常用一阶口径，不包含所有 attention、embedding、optimizer、通信和失败重跑成本。MoE 要区分 total/active parameters，数据要区分 unique/consumed tokens；最终决策还受推理、数据质量和产品约束影响。

### 13. MoE 为什么可能省计算却不省显存或通信？

每个 token 只激活部分 expert，active compute 可低于同总参数 dense 模型；但所有 expert 权重仍需放在设备群上，路由会引入 dispatch、all-to-all、负载不均、capacity drop 和小 batch 效率问题。比较时同时报告 total/active parameters、每 token FLOPs、显存、通信和端到端吞吐。

### 13.1 MoE capacity 与 token drop 怎样算才不会混淆？

先写 routing group：若有 \(N\) 个非 padding token、\(E\) 个 expert、top-k 为 \(k\)，一种常见教学约定是每 expert \(C=\lceil\phi Nk/E\rceil\)，但真实框架可能有不同 group/minimum/dropless/reroute 语义。再写 expert 内谁优先占 capacity，以及 gate 在 top-k 后、drop 后是否重归一化。

分母必须分开：top-2 的一个 assignment overflow 不等于整个 token 被丢；报告 `dropped assignments / (Nk)` 和 `all-assignments-dropped tokens / N`。Balance auxiliary 也不是跨实现通用公式，需说明使用 pre/post-capacity count、top-1/top-k、probability、stop-gradient 和 reduction。CPU routing/control-flow 通过不证明 all-to-all、GPU grouped GEMM、目标模型质量或吞吐。

### 14. “支持 1M context”是否等于有效使用 1M token？

不等于。API 接受长度、训练长度、位置外推稳定性、检索/推理能力和成本是不同问题。需要位置分层的 needle、multi-hop、干扰、顺序与长输出测试，并与 RAG 或分段方案在相同预算下比较。

### 14.1 怎样评测持续学习中的遗忘？Replay 实验最容易误读什么？

固定 \(R_{i,j}\) 的含义：完成顺序任务 \(i\) 后，在任务 \(j\) 上的同方向指标。报告最终 ACC、旧任务 BWT、逐任务 peak-to-final forgetting，以及相对独立 pretraining baseline 的 FWT；同时给完整矩阵、多个 seed 和置信区间。必须声明 forgetting 是否把最终阶段纳入历史最大值，因为这决定改善时记 0 还是负值；FWT 还要求在学习每个未来任务前就评测它。

Replay 后旧任务不掉，不等于方法普遍有效。先排除任务本身不可联合求解、显式 task-id 泄漏、全量旧数据冒充有限 buffer、更新预算不等、只挑单 seed，以及隐私/删除/计算成本未计入。B 阶段的 replay 不能改变训练 B 之前已经确定的 FWT；把二任务 synthetic toy 外推到真实 LLM、安全行为或长期序列也是证据越界。

多 seed 也不自动解决外推问题：如果每个 seed 只改变初始化和 reservoir 选择，而 task/data 固定，paired bootstrap interval 只描述这些随机源。还要区分“相同 optimizer steps”与“相同样本/FLOPs/时间”；每步把 256 个新样本扩成 320 或 512 个总样本，本身就是额外训练预算。

## RAG

### 15. chunk 越大还是越小？

小块匹配精确但上下文不足；大块语义完整但噪声与 token 成本高。按文档结构、答案跨度和检索模型实验，比较 Recall@k、上下文覆盖、冗余和最终忠实度。

### 16. 为什么 dense retrieval 不能完全替代 BM25？

Embedding 擅长语义相似，BM25 擅长型号、错误码、姓名等精确稀有词。混合召回后用 RRF 或 reranker。向量分数跨 query/模型不可直接用统一阈值。

### 17. RAG 返回正确文档但答案仍错，怎样排查？

依次检查 chunk 是否含完整证据、重排是否截断、上下文是否重复/冲突、Prompt 是否要求引用、模型是否正确使用证据、引用是否真正蕴含主张。分层指标比只看最终答案有效。

### 18. 多租户 RAG 怎样防泄漏？

身份传到检索层，ACL 在召回查询前执行；索引、cache key、trace 和评测也隔离。用交叉租户 canary 做负向测试。不能先全局召回再让模型忽略。

### 19. Recall@k、MRR、nDCG 各测什么？

Recall@k 看相关文档覆盖；MRR 看第一个相关结果位置；nDCG 支持分级相关性并惩罚位置。指标选择取决于生成需要一个证据还是多个证据。

### 19.1 为什么不能把每段分别 tokenize 的长度相加作为 context 上限？

Chat template、special token 和片段边界会改变最终 tokenization，BPE merge 还可能让插入前后的总数非单调。每加入一个候选后渲染完整 prospective prompt，用固定 tokenizer/revision 重新计数，并预留输出；记录 budget、selected/dropped reason 和最终 token ids/数量。字符或 UTF-8 byte 可做透明 fixture，但不能冒充目标模型 token。

### 19.2 怎样证明 RAG 评测的是模型当时看到的证据？

用 request/case identity exact join，绑定 query hash、tenant/principals、按序 chunk id + stable source + version + content hash、实际 rendered context、tokenizer/chat/system/user prompt identity、最终 prompt token IDs、decoding/model/runtime revision、raw output 和 parsed answer fingerprint；从版本化 snapshot 重建 ACL/context，并让运行时把 manifest 写入签名或 append-only 存储。仅有自洽 JSONL 与 SHA-256 不证明真实模型调用：审计器若不重新 tokenize、验证签名或判断 raw-output→claim entailment，就必须逐项声明这些边界。

### 19.3 Reranker 为什么必须再次检查 ACL？怎样重放一次排序？

上游 candidate list 可能来自错误 cache、旧 policy 或错误 tenant；直接交给 cross-encoder 已经泄露正文，即使最终 top-k 再过滤也来不及。Scorer 前重查 tenant/principals，拒绝重复 id 与 rank 漂移，只把 visible candidates 交给模型。

重放时保存 query hash、ordered candidate id/rank/source/score、每段 content hash、scorer/model revision、tokenizer/max length/truncation、逐候选 rerank score 与最终 rank。Recorded score 能验证 identity 和排序代码，但 unsigned `scorer_identity` 不认证模型执行；没有 held-out qrels、延迟和成本对照，也不能声称相关性提升。

### 19.4 为什么先做 extractive RAG baseline？它证明什么？

它把检索、ACL、packing、引用、拒答和评测串成可执行控制组，且逐字 span 能机械验证没有从 packed source 外新增 claim。它不等于生成质量或 entailment：lexical overlap 会选错句，来源可能错，答案可能不完整。阈值要在独立校准集上选；qrels 只在生成后用于评测，不能传进在线 answer API。真实生成还需目标 tokenizer packing、raw-output trace 和独立 claim judge。

### 19.5 怎样公平比较原生 RAG、LangChain 与 LlamaIndex？

先固定 canonical corpus/chunk/query/security context/top-k/qrels，让同一个 authorization-first retriever 作为排名权威；框架只承接 Retriever、Prompt 或 orchestration adapter。逐项比较 ordered document ID、正文、score/rank、metadata、最终 Prompt、模型输入/输出和同一评测，不允许三套实现各自换 embedding、切分和 Prompt 后再把差异归因给框架。

框架对象转换通过只证明字段没有相对 supplied canonical results 漂移，不证明框架默认 ACL，也不认证 supplied results 来源。LlamaIndex node 的控制面 metadata 应明确排除出默认 embedding/LLM content，但自定义 formatter 仍可能读取；LangChain prompt 也只有显式选择 `page_content` 才不会自动引入 metadata。完整结论还要加入相同 provider/checkpoint、tokenizer/template、decoding、重试、费用、延迟和并发。仓库 control 使用 deterministic extractive non-LLM answer，因此只能证明离线接口、ACL 和 identity parity，不能证明 learned retrieval/generation 质量或生产性能。

### 19.6 FastAPI 返回 504 后，为什么 RAG work 可能仍在运行？

`asyncio.wait_for(asyncio.to_thread(sync_query), timeout)` 只能取消等待它的 coroutine，不能强制终止已经运行的 Python thread、SQLite 调用或模型 kernel。如果 504 时立刻释放 semaphore，后续请求会进入，而旧 work 仍占 CPU/连接/GPU，实际并发就超过声明上限。Reference 用 `shield` 保留 task，并把 permit 延迟到后台 work 真正结束后释放；client cancellation 也遵循同一规则。

这只是诚实记账，不是强制终止：卡死 thread 会长期占位。生产上要给下游传 cooperative deadline/cancel token，使用支持取消的 driver，或在可回收进程/worker 中隔离；多 Uvicorn worker 和多副本还需要全局 admission。还要区分 queue timeout、execution timeout、下游 timeout 与 client disconnect，分别记录 all-attempt denominator，不能把 504 当作服务端 work 已停止或费用为零的证据。

## Agent

### 20. Agent 和 workflow 的边界？

分支确定、风险高的流程用代码状态机；模型只处理开放语义决策。开放性越高，越需要预算、审批、幂等、恢复和审计。

### 21. 如何避免重复转账或重复发消息？

模型生成稳定 call_id 不够；执行层用幂等键、参数指纹、数据库唯一约束和业务事务。崩溃恢复先查 ledger/外部状态，不能盲目重放。

### 22. 提示注入为什么不能靠 system prompt 解决？

外部内容与高优先级指令共享模型上下文，模型可能错误服从。真正边界在最小权限、秘密隔离、参数/ACL 校验、域名限制和人工审批。

### 23. Agent 怎样判断停止？

模型输出 `finish` 只是 proposal；确定性 verifier 通过才是完成。最大 decision step、模型 token、wall time、费用和外部资源预算是硬边界；连续相同 action fingerprint、`A/B/A/B`、同类错误与状态无进展是不同停止信号。审批暂停、升级、预算耗尽和安全停止不算成功。还要说明 token/费用通常在模型响应后才能记账，同步调用越过 deadline 也不能保证抢占。

### 23.1 为什么 handler 返回 completed 仍不能证明邮件已发一次？

本地 completed 只证明 runtime 记录了 handler 返回；远端可能已成功但本地落账失败，也可能 provider 接收请求后异步拒绝。反过来，handler timeout 也不证明动作没发生。要用 provider idempotency key、outbox/业务状态或 provider audit 做 effect verifier，并把 `handler_attempted`、`effect_applied` 与 unresolved pending 分开记录。

### 23.1.1 Transactional outbox 能否保证 exactly-once？

不能。它只让本地业务状态与待投递 row 原子提交。worker 用 lease 并发 claim，但 provider 成功、本地 ack 前崩溃仍会重投；lease 不是远端 exactly-once。若 provider honor 稳定 `effect_id` idempotency key，at-least-once request 才可能折叠成一个 effect；否则需查询 provider/业务状态、reconciliation 或补偿。receipt 也只是 supplied artifact，dead letter 需要 operator runbook，不能无限自动重试。

### 23.2 Agent 评测为什么不能只报一个平均成功率？

任务完成不能抵消一次越权或重复付款。task success 只在有 final-state verifier judgment 的 case 上计算；policy violation、policy over-refusal、未审批副作用 attempt、重复 applied effect 与 pending 分别保留 numerator/denominator，并作为独立发布 guardrail。分母为零是 N/A，不是 0% 风险。

### 23.3 参数 fingerprint 做成 SHA-256 后是否已经保护秘密并证明审批安全？

没有。hash 只绑定所选 canonical bytes；它不证明业务语义、权限或 payload 无秘密，低熵值仍可能被枚举。审批还要绑定主体、task、call id、scope、资源版本和过期时间，原始敏感 payload 应最小化、隔离加密并受 retention 控制。

### 23.4 为什么 cache hit 也要重新授权？proposal fingerprint 与 execution fingerprint 有何区别？

用户权限可能在第一次执行后被撤销；先查 cache 会把旧敏感结果返回给当前已无权主体。proposal fingerprint 只标识模型提出的 tool + arguments；execution fingerprint 还绑定可信 task/subject/tenant、tool contract、server-resolved resource revision 与 policy decision。每次 replay 先重新授权，同 call id 但 execution identity 改变应冲突，不能静默复用。

### 23.5 为什么 handler 返回的 dict 仍要 JSON snapshot？

handler 可能返回仍被它或调用方持有的嵌套可变对象；直接放入 cache 会产生“审批/执行后结果漂移”。先做严格 JSON 编码与 round-trip，再递归冻结；NaN、set 和自定义对象失败并保持 pending。这样只证明结果值域稳定，不证明内容真实、业务正确或远端 effect 已发生。

### 23.6 approval pause 后怎样安全恢复？

checkpoint 绑定可信 task/subject/tenant、原 loop/handler cap、累计 token/cost/active time 与 handler counter、事件/action 历史、pending decision 和 execution fingerprint。恢复先重新授权并执行同一 decision，不重新调用 planner 或重复计 usage；旧身份、清零 counter、扩大 cap、过期/漂移 grant 都拒绝。还要指出 checkpoint SHA-256 没有密钥，不防攻击者重算；文件与 ledger 需事务/一致性协议，等待审批要另设绝对 deadline，并用签名/MAC、ACL、加密、lease 和一次性 approval store 补齐生产控制面。

### 23.7 怎样把模型自由文本安全地接到 typed Agent loop？

先构造 canonical request identity，绑定 prompt/prompt revision、task state、剩余预算、tool catalog、输出 cap 与预期 model revision；transport 返回时要求 exact model revision、provider request id、input/output usage、cost 和 finish reason。然后用拒绝 duplicate key、non-finite number、Markdown fence、未知字段/工具的 closed JSON parser，只生成 `tool/finish/escalate` typed proposal。最后仍由模型外 runtime 重新执行 schema validator、server-owned resource resolution、policy、approval、预算与幂等，`finish` 也必须过 verifier。

Request/response/decision fingerprint 只能证明所列 canonical bytes 一致，不认证 provider，也不证明安全或语义正确。Tool observation 即使在 system prompt 中标为 untrusted，仍与指令共享上下文；真正安全边界是最小权限和外部控制面。Provider usage 缺失不能猜，output usage 超 request cap 要拒绝；整次 input+output 仍可能事后越过总预算，此时记账但不执行 action。Recorded response control 只证明 parser/runtime 控制流，不能说成目标模型、线上 API、账单或生产 IAM 实测。

若 Prompt 展示 JSON Schema，还要回答怎样防“文档 schema 与执行 callback 漂移”：用一个版本化 contract 同时派生 Planner catalog 与 runtime Tool，并把 schema/validator revision 纳入 request identity。显式固定 draft；限制 schema/instance bytes；不允许运行时从模型 URL 取 remote `$ref`；说明 `format` 是 annotation 还是 enforced。标准 validator 不做 coercion、不应用 `default`，schema violation 错误不能回显 secret value。即使 schema 通过，resource resolver、tenant/policy、approval 和业务 cross-field check 仍必须独立执行。

## 评测与实验

### 24. LLM-as-judge 有哪些偏差？

位置、长度、风格、自我偏好、知识和提示敏感。交换顺序、匿名、结构化 rubric、多 judge，并用专家集校准一致性。不能只保存 adjudicated winner：逐判断绑定 case、annotator/batch、presentation order、tie/invalid、rubric revision 和盲化/独立声明。Raw agreement 的分母是同 case 内 annotator 无序对；Fleiss’ κ 还要求每 case 固定 rater 数。位置诊断应逐 case 比较 \(P(A\mid A\ first)-P(A\mid A\ second)\)，再以 case bootstrap；它不因保存顺序或区间就成为随机实验的因果估计。

### 25. 为什么用 paired bootstrap？

基线和候选回答同一批 case，差值天然配对。对 case 级差值重采样得到均值差置信区间，比独立均值更贴近问题。但用户/文档内多个 case 相关时应按 cluster 重采样；逐行 bootstrap 会虚增有效样本量。Bootstrap 改善概率是重采样统计量的比例，不是“候选真实更好的后验概率”。Percentile interval 也不自动具有小样本、偏斜分布或重尾下的标称 coverage。

### 25.1 Cluster bootstrap 的 case-weighted 与 equal-cluster 路径怎样算？

每次有放回抽 \(G\) 个完整 cluster。Case-weighted target 用抽中 cluster 的 difference sums 作分子、抽中 cluster sizes 总和作分母；分母必须随 resample 改变。Equal-cluster target 则平均抽中 cluster 的 mean differences。前者回答平均 case，后者回答平均 cluster，不能看 outcome 后切换。

小 \(G\) 时可枚举 \(G^G\) 个 ordered resample 来消除 Monte Carlo 误差，但这不会制造更多独立 cluster，也不保证 percentile coverage。报告 cluster 数与 size 分布、weighting、quantile method、resample 数/seed、最大 cluster sensitivity 和 interval；cluster 定义、独立/代表性、interference 与抽样设计仍需外部证据。

工程门禁还必须把这些选择写进不可静默漂移的 artifact。仓库 comparison v2 在 root 绑定 `unit`、cluster metadata key、weighting、exact threshold、requested Monte Carlo samples/seed；每个 overall/slice result 单独保存 cluster sizes、estimand、`exact|monte_carlo`、实际 resample 数和有效 seed。因为不同 slice 的 cluster 数不同，method 不能只在 root 写一次。Artifact-only 验证不重开 cases，也不会证明 metadata key 真的是正确 sampling unit。

### 25.2 Paired randomization/sign-flip test 与 bootstrap 有什么区别？

Sign-flip 在 sharp null 与 pair-label exchangeability 下翻转每个非零差值符号，计算 observed mean difference 的尾部概率；\(m\) 个非零 pair 可 exact 枚举 \(2^m\) 项。零差值留在 mean denominator，但无需把相同 assignment 加倍。大样本 Monte Carlo p-value 用 `(extreme+1)/(draws+1)`，不能报伪精确 0。

它给 p-value，不给 effect-size 置信区间；p-value 不是 null 为真概率，也不保证业务重要。单/双侧需预注册，cluster correlation、多个 metric/slice、反复改 Prompt 和数据选择分别要求正确随机化单位与 multiplicity/selection 处理。仓库 authored CPU fixture 没有证明真实 case 抽样、交换性、因果或模型改善。

### 25.3 同一用户有多条 case，怎样做 cluster sign flip？

若用户才是可交换单位，同一用户内所有 case difference 必须共享一个正负符号，不能逐行独立翻转。这样不要求用户内 case 独立，但仍要求 cluster-level label exchangeability、cluster 间可独立组合，并需要足够多且有代表性的 cluster。

还要先选 estimand：case-weighted 用每个 cluster 的 difference sum、总 case 数作分母，大用户权重更高；equal-cluster 用每个 cluster 的 mean difference、cluster 数作分母，每个用户等权。它们不是两种“标准误算法”，而是两个问题。仓库反例中 5 个 `+1` 属于用户 A、1 个 `-1` 属于用户 B：逐 case greater p=7/64，case-weighted cluster-joint p=2/4，而 equal-cluster observed difference=0。不能看完三项后挑最好看的一项；报告 cluster 定义、size 分布、weighting、干扰/抽样假设与 effect size。

### 25.4 同时评测多个 metric/slice 时，Holm correction 怎样算、不能解决什么？

先在看结果前定义 family，把 \(m\) 个有效 p-value 升序排列。第 \(i\) 项先乘 \(m-i+1\)，再对这些 scaled value 做前缀最大值并 cap 到 1；最后映回原 hypothesis。即

\[
\tilde p_{(i)}=\min(1,\max_{j\le i}(m-j+1)p_{(j)}).
\]

在每个 component p-value 有效的前提下，Holm 对任意依赖结构控制 FWER。不能漏掉 running maximum，也不能把 adjusted p-value 解释为 null posterior。它不修复事后挑指标/family、可选停止、测试集调参、错误的 cluster unit 或本来就无效的 p-value，也不回答效果大小和业务重要性。面试中还应说明为何这些检验属于同一 family、报告全部原始/adjusted p-value，而不是只展示显著项。

### 26. 测试集反复调 Prompt 有什么问题？

测试集变成事实上的开发集，结果乐观。保留隐藏集、滚动新鲜集和时间切片，记录试验次数与版本。

### 27. 线上总体提升但中文用户下降怎么办？

分层结果必须显式报告；关键群体设 guardrail，不能用总体均值抵消。检查数据、路由、tokenization、Prompt 和 judge 语言偏差，再决定阻断或局部发布。

### 28. pass@k 测的是什么？为什么不是线上单次成功率？

当同一任务有 \(n\) 个采样、其中 \(c\) 个通过测试时，无放回选择 \(k\) 个至少一个成功的估计是：

\[
\operatorname{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},\quad 1\le k\le n
\]

它描述给同一任务最多看 \(k\) 个候选的能力；线上只生成一次时更接近 pass@1，还受采样、测试覆盖、任务分布和选择策略影响。测试通过也不证明补丁安全或满足未编码需求。

### 29. 怎样校准 LLM judge，而不是直接相信它？

先定义 rubric 和专家 gold subset；测 pairwise/分类的一致性、各 slice 混淆、位置交换与重复运行稳定性。judge model、prompt/template 和解析器都要版本化。高一致性仍只说明与该专家协议接近，不证明真实用户价值。

### 29.1 baseline 与 candidate 的 case id 完全一致，是否足以做可信配对比较？

不足。同一 ID 下的 input、gold/rubric、slice、metadata 可能已变，metric 同名实现也可能升级。run manifest 应绑定 ordered case 全语义、ordered result、recorded output、metric revision、scorer 与 system revision；compare 在统计前重算并 fail closed。canonical SHA-256 只能检测已列内容相对可信 manifest 的变化，unsigned manifest 和自报 `system_id` 不证明输出来自该模型、评测集代表真实分布或指标有效。

最终 gate report 还要绑定 bootstrap seed/sample/confidence、质量/安全/延迟与 slice 阈值、两侧 manifest、统计结果和全部失败原因。否则同一组分数可在事后换阈值或改 `passed`。comparison fingerprint 能发现相对可信锚点的局部漂移；若 JSON 与 hash 一起被重写，仍需签名或 append-only 发布记录发现。

### 29.2 HMAC 链通过，是否已经证明评测发布历史没有被回滚？

没有。先用每条记录声明的 `key_id` 从可信 resolver 取 key，按固定 canonical/domain-separated 规则重算 MAC，并验证连续 sequence 与前序 MAC；这只证明相对这些共享密钥的链。再用 artifact-id→path 的精确映射重读 size/hash，才证明当前引用 bytes 一致。最后必须从 ledger 外取得 trusted head（至少 sequence+MAC）并匹配，才能发现删除尾部或换回旧的合法 snapshot；否则任何有效前缀都能通过。

还要说明 HMAC 不提供公钥签名的不可否认性，key custody/轮换/吊销要靠 KMS/HSM 与权限协议；被 MAC 的 timestamp 只是 caller 字符串，不是可信时间。Exclusive-create/file-fsync 也不证明 parent directory durability、目录原子发布或 verify 后文件不变。面试中的完整方案应包含外部 transparency/object-lock anchor、消费端强制验证和 TOCTOU 边界，而不是只说“用了 hash chain”。

### 29.3 Artifact-only 校验与完整证据图复算有什么区别？

Artifact-only 只检查最终 comparison 的 schema、内部算术/判定和无密钥 fingerprint，不知道引用的 manifests/results/cases 是否存在或匹配。完整本地复算还要重开 cases、两侧 answers/results/manifests；按 case 顺序重算 answer/case identity，要求 metric revision 可由当前实现精确执行，重新评分，并从记录的 resampling/gate 配置重跑统计与最终 decision。这样能发现单文件自洽但跨文件错误的 score 或 summary。

边界仍要说完整：复算没有重新请求模型/provider，不认证 `system_id` 或本地文件来源；能改全套 bytes 的攻击者仍可协同重写。它也不验证 sampling/cluster 假设、指标 construct validity 或线上效果。可信链路需要把“计算可复现”“artifact 被认证”“外部 head 未回滚”“真实 runner/model 执行”和“统计/业务有效”作为五个不同命题。

### 29.4 评测 HTML 报告怎样避免成为安全漏洞或“证据升级”？

只从严格加载的 artifact 派生；system id、slice、reason 等所有动态文本统一 HTML escape，不拼接可信标签。页面不需要 JavaScript 或外部资源，使用 restrictive CSP，并以文字而非只靠红绿颜色表达判定；cluster/case 结果按 schema 分支展示，不能把字段缺失默认为 0。对闭合 table cell、`script/img/onerror`、超长文本和 Unicode 做回归。

更重要的是 scope：报告 receipt 应写 `artifact_only_render`、`statistics_recomputed=false`、`artifact_authentication_verified=false`。HTML 是可覆盖的展示物，不是 canonical decision artifact；消费方必须回到 JSON verifier、完整 evidence recomputation 和独立认证链。XSS-safe 只证明渲染边界，不证明数据、统计或发布决定正确。

## 系统与故障

### 30. LLM 服务 p95 TTFT 突然升高如何排查？

先确认计时起点：负载生成器 `offered_at`、HTTP dispatch、网关进入、engine 入队、prefill 开始不能混成一个 timestamp。若请求在 client concurrency semaphore 前等待、取得槽位后才开始 TTFT，测试工具本身会隐藏排队。再分解 client/gateway/engine queue、prefill、外部检索与网络；看输入长度/并发分布、batch 调度、长 prompt、GPU 利用率、KV 容量和 preemption。p50 正常而 p95 高通常提示排队或长请求干扰，但只有 client trace 不能定位服务端根因。

### 30.1 Closed-loop 与 open-loop 压测有什么差别？

Closed-loop worker 通常等上一个请求完成再发下一个，服务变慢时 offered load 会下降，容易漏掉饱和后的排队。open-loop 按外生 constant/Poisson schedule 到达，completion 不改变后续时刻，更适合扫容量 knee；但还要证明 generator 跟得上。将 scheduled timestamp 记为 `offered_at` 可把 event-loop 迟到计入 client queue，不代表请求真的准时 dispatch，也不能把 client queue 当 server queue。有限 seeded schedule 只证明该轮到达工件可复现，不证明生产流量分布。

### 30.2 Paged KV 为什么仍会碎片化？Prefix sharing 怎样安全追加？

固定 block 避免每请求预留最大连续区，但每条序列 tail 仍可能未填满，block size 是 metadata/调度开销与内部碎片的权衡。Prefix fork 让多张 logical block table 引用同一 physical block；full shared tail 后追加只分配新块，partial shared tail 必须 copy-on-write，否则父子序列互相污染。

Append 应先算出并预留“COW replacement + 填满 tail 后仍需的新块”，容量不足则在 occupancy/length 改变前整体失败。指标分开 logical tokens、physical materialized token positions、logical references、allocated blocks 和 internal fragmentation；logical tokens 会重复共享 prefix，不能直接当物理使用量。CPU metadata 模拟通过不等于真实 KV copy、GPU page table、preemption 或性能通过。

### 30.3 Continuous batching 中 prompt+output 为什么不等于模型 forward token work？

标准 causal LM 的 prefill 最后一个 prompt position 已产生首个输出 token 的 logits；若请求 \(i\) 有 \(P_i\) 个 prompt token、实际发出 \(O_i\ge1\) 个输出 token，且没有 prefix reuse、speculative verification 或 beam，一阶 forward positions 是 \(\sum_i(P_i+O_i-1)\)。API usage 仍可报告 prompt+completion token，不能因计算口径减一就修改计费/输出分母；padding、prefix cache、speculative、kernel 和 scheduler 还会让物理工作不同。

回答调度题时继续固定 arrival/admission boundary、sequence cap、token budget、prefill chunk、decode/prefill priority、preemption 与首 token boundary。离散 CPU state machine 可证明某份 policy 的 queue/TTFT/TPOT step 和 work conservation，但 step 不是秒、token-slot utilization 不是 GPU utilization，也不证明某版 vLLM scheduler 或吞吐。

### 30.4 KV 抢占为什么会让实际 forward work 大于 `prompt+output-request_count`？

无 prefix reuse/speculation/beam 时，后者是 logical causal positions。Recompute preemption 释放某条 sequence 的 KV；恢复时要重跑已经处理的 prompt/历史输出 positions，所以 `executed = logical + recomputed`。重建 KV 不应重复向用户发 token。仓库反例把 logical 9、recomputed 2、executed 11 分账，B 的输出仍只有 boundary 2/6。回答还要说明 victim/priority、当轮保护、完成释放、单请求能否独占放下、swap 与 recompute 的区别；CPU block trace 不证明目标 vLLM、VRAM 或时延。

### 30.4 Prefix cache 为什么不能只用 prompt 的 SHA-256？

因为“同 hash”既不是授权，也不表达完整执行 identity。候选必须绑定可信 tenant/visibility domain、authorization/policy revision、model/tokenizer/template/adapter revision、RoPE/position config 与 KV dtype；cached token ids 还必须逐项等于请求 token ids 的某个前缀，多个候选取最长。Fingerprint 只能缩小候选桶，命中前仍做 full identity/token comparison，因此碰撞不能导致复用。

并发系统还需 lease/refcount pin 使用中的 entry，LRU 只能淘汰未 leased 项；满容量且全 leased 应在 mutation 前失败。Unkeyed hash 不隐藏低熵 prompt，cache timing 也可能泄露前缀是否存在。仓库 collision fixture 只证明 metadata 状态机，不证明真实 K/V、vLLM policy、VRAM、prefill savings 或 timing-channel mitigation。

### 30.5 Beam search 为什么不保证找到全局最高概率序列？length penalty 又怎样改变答案？

有限 beam 只按当前 prefix 累计 log probability 保留 (B) 条，较差 prefix 以后可能接上高概率后缀，但已经无法恢复。反例：root 的 `A=0.6,B=0.4`；下一步 `A→EOS=0.51`、`B→EOS=1`。beam 1 返回 `A,EOS`，概率 0.306；beam 2 才保留并返回 `B,EOS`，概率 0.4。因此增加宽度只扩大搜索，不构成任意生成树上的全局最优证明。

若最终分数约定为 (s=\log p/T^\alpha)，由于 log probability 为负，正的 \(\alpha\) 会让较长候选的分数更接近 0，可能翻转 raw-probability 排名。答题时必须先定义 (T) 是否含 prompt/EOS，再说明 active pruning 用 raw 分数还是 normalized score、EOS 何时进入 finished set、finished-candidate cap、tie-break 和 early stopping。仓库 oracle 的 (T) 只计生成 token、含 EOS、不含 prompt，保留所有从 active prefix 产生的 EOS且无 heuristic early stopping；不能把这份教学契约说成 Transformers、vLLM 或云 API 默认。

### 30.6 约束解码为什么不能只检查 token 的第一个字符？

Tokenizer token 可能一次解码成多个字符或 bytes。若 grammar 在当前状态允许 `1`，token `1]` 的首字符检查会通过，但完整转移可能在 `]` 处失败；必须计算 token 完整片段的 (delta^*) 并据此屏蔽。屏蔽后对合法 token 的原始概率质量重新归一化；合法集合质量为零要显式 constraint error，不能解除约束。EOS 只在 accepting state 合法，而“状态已接受”与“请求已 EOS 完成”也要分开，length 截断仍是不同 finish reason。

工程上每条 beam/sequence 都携带 parser state，缓存 allowed-token mask 时绑定 grammar 与 tokenizer revision。JSON Schema/CFG 约束最多保证被编码的语法属性，不保证字段事实、数据库 ID、权限或副作用安全。仓库 toy 只对 supplied Unicode text fragment 和有限 literal trie 做 CPU 验证，不执行真实 tokenizer bytes、完整 JSON Schema、模型、GPU 或 provider runtime。

### 31. 4-bit 模型为何不一定更快？

取决于硬件内核、反量化、group size、未量化层和 batch。若计算不是权重带宽瓶颈或 kernel 不成熟，文件更小不代表端到端更快。还应区分三个数字：裸 code 的理论 bit 数、含 scale/zero point/alignment 的 artifact 或 resident bytes，以及运行时峰值显存。

对 \(R\times C\) 权重、bit width \(b\)、每个 contiguous row group \(G\) 保存一个 FP32 scale，理想下界是 `ceil(bRC/8) + 4R*ceil(C/G)` bytes；它仍没含容器、未量化层和 workspace。真正达到 code 下界还需固定 signed-code 映射、bit/endian 顺序、padding 与 alignment；完成 CPU dense bit packing 也不代表 runtime 采用同一 layout。若单矩阵格式再加 32-byte header 和 32-byte digest，文件大小就是 raw payload + 64 bytes；小 fixture 的相对开销会很大。小 group 通常降低局部量化误差，却增加 metadata。CPU 上 strict reload/unpack/反量化后做 FP32 matmul 只能验证 artifact 和数值 plumbing，不能声称执行了 int4 kernel 或得到加速；unkeyed hash 也不是签名。

### 31.1 多矩阵量化 bundle 为什么仍不等于完整模型 checkpoint？

“能按 tensor name 重载多个量化矩阵”只解决 state serialization 的一个子集。完整 checkpoint/runtime artifact 还要定义并验证 architecture/config 语义、tokenizer vocab/merges/chat template、embedding/bias/norm 等未量化 state、tied weights、dtype、shard index、device/runtime layout 与 model forward compatibility。只有 tokenizer id/revision 不能恢复 tokenizer；只有 architecture JSON identity 也不会自动提供可执行 forward。

还要区分三层完整性：每个 tensor digest 检测 tensor 漂移，outer digest 检测 bundle body 漂移，签名或受控发布链才可能认证来源。攻击者能协同改内容并重算 unkeyed SHA-256。Exclusive create 防覆盖，但写入中崩溃可留下 partial target；file `fsync` 也不等于 parent-directory durability 或断电原子发布。仓库 two-layer NumPy fixture 只证明严格 manifest、多 tensor round trip 和 control flow，不证明 GGUF/safetensors 兼容、LLM 质量、resident VRAM、fused kernel 或加速。

仓库进一步给出的 repo-native MiniGPT checkpoint 展示了怎样跨过这条边界：保存 Byte-BPE merges、严格 config、全部唯一二维/一维参数和 tied-weight contract，并由固定 architecture revision loader 恢复 causal forward。但“完整”仍是相对契约：base-byte mapping 与 forward code 来自 trusted repo；normalizer/special/chat template、训练状态、sharding/device layout 和外部 runtime compatibility 均不在格式内。它只对当前 tiny inference model self-contained，不能据此宣称支持任意 Llama/Qwen、训练 resume 或低位 GPU 执行。

### 31.2 INT8 KV cache 为什么通常达不到 FP32 的 4× 压缩？

若 K/V head dim 都是 (D)，per-token/per-KV-head 独立保存 K/V 两个 FP32 scale，则 FP32 是 (8BH_{kv}TD) bytes，INT8 payload 是 (2BH_{kv}T(D+4))，比率为 (4D/(D+4))，只在大 (D) 时接近 4×。实际还要计 paged block、alignment、allocator、workspace 和临时 dequant buffer；GQA 公式使用 (H_{kv}) 而不是 (H_q)。容量变小也不代表更快：若先 dequantize 再用浮点 attention，只证明数值路径，不能声称执行了融合 KV kernel。质量需分 K→logits/softmax 与 V→output 误差，并测长位置、检索、多跳和安全切片。

### 31.3 Speculative decoding 为什么能保持 target sampling distribution？

proposal \(x\sim q\) 以 `min(1,p(x)/q(x))` 接受；接受路径给 token \(i\) 的质量是 `min(p_i,q_i)`。总拒绝概率是 `1-sum(min(p,q)) = TV(p,q) = sum((p-q)_+)`，拒绝后从 normalized positive `(p-q)` residual 采样，所以两条路径相加恰为 \(p_i\)。一步接受率因此是 `1-TV(p,q)`。

block 顺序验证，只保留首个拒绝之前的 draft；拒绝位置发 residual token并丢弃后续 proposal，全部接受才发一个 bonus target token。必须强调相同 vocabulary/prefix 和采样变换后的真实 \(p,q\)。Greedy prefix verification 是另一算法；概率恒等式不等于速度保证，后者还取决于 draft 成本、verification kernel、接受长度、KV 管理和 batch。

### 31.4 为什么只保存 model、optimizer 和 global step 仍未必能精确恢复训练？

下一批样本和下一次随机算子还依赖 sampler/permutation/cursor、数据 loader worker/prefetch 与 Python/NumPy/Torch CPU/CUDA RNG；AMP 依赖 scaler，gradient accumulation 依赖窗口位置和未提交梯度，scheduler 可能按 update、token 或 metric 前进，FSDP/ZeRO 还要绑定 shard/world topology。数据本身若漂移，即使路径和 shape 相同也不是同一 run。正确做法是先定义只允许保存的一致性边界，再把所有实际消费的状态纳入 snapshot，并用 uninterrupted-vs-kill/reload split run 比较逐步 sample/LR/loss 与最终参数/optimizer/stream/RNG。近似 loss 或能继续跑只说明 warm restart，不是 bit-exact resume。

仓库 CPU FP32 MiniGPT control 在 zero-grad AdamW boundary 保存 16 个参数 tensor、32 个 moment tensor、每参数 step、线性 schedule、数据 fingerprint、permutation/cursor/epoch、data-generator RNG 与 dropout 的 Torch CPU RNG，6 步的 split run 与不中断路径逐位相同。它没有使用或保存 Python/NumPy/CUDA RNG，也不支持 AMP、accumulation、worker、distributed/sharded 或目标 LoRA/QLoRA；这说明 checkpoint 的“完整”总是相对具体训练契约，而不是文件里字段越多越完整。

### 32. 如何做一次可信消融？

固定数据、token 预算、模型、seed、调参预算和评测；只改变目标组件，多次运行并报告方差。若计算预算不同，明确回答的是“同成本谁更好”还是“最高质量谁更好”。

### 33. 模型版本相同，为什么仍不能保证请求可重放？

模型名称不足以重放。还需要精确 revision、tokenizer/chat template、system/developer/user 消息、工具 schema、检索索引与 ACL 版本、generation 参数、随机性、runtime/kernel 和外部工具状态。配置 fingerprint 只能证明所序列化字段的 canonical bytes 相同，不能证明遗漏字段、外部状态、语义等价或位级确定性。

### 34. 对话摘要能否当作长期记忆的真实状态？

不能。摘要是有损、由模型生成的派生表示，可能遗漏否定、时间、权限和未决事项。事实状态应有 typed schema、来源、时间和用户修正；摘要用于节省上下文，并需要从原始事件重建或审计。

### 35. Prompt delimiter 为什么不是安全边界？

XML tag、Markdown fence 或“以下是不可信内容”能帮助模型区分结构，但外部文本仍在同一推理上下文中。权限、秘密、工具参数、网络访问和副作用必须由模型外执行层校验。结构化输出 schema-valid 也不等于语义正确或授权通过。

### 36. 为什么云模型调用不能“429/5xx 一律重试”？

HTTP class 不足以决定重试：`400/401/403/404` 通常需要修正请求或权限；`501/505` 也不会因属于 5xx 自动变成瞬时故障。先按固定 provider/endpoint/version 建 allowlist，再同时检查请求 replay-safe、远端 outcome 是否确定、attempt/deadline/费用预算和 `Retry-After`。有效 `Retry-After` 超过预算时应停止或上抛，不能提前轰击；timeout 后若远端可能已完成，自动重放可能重复副作用或计费。Exponential backoff、jitter 和 idempotency key 都不能替代对 provider 语义的验证。

实现时至少区分 pool/connect 前失败与 write/read/protocol 后失败：后者仅凭客户端异常不能证明服务端没执行。使用 monotonic deadline 而不是墙上时钟计算预算；exact origin allowlist、HTTPS 和 no-redirect 防止凭据被带到意外 endpoint。非流式客户端在完整 body 已缓冲后检查长度，只是 acceptance cap，不能回答峰值接收内存。

## 代码题建议

能够现场写并测试：

- stable softmax 与 causal attention；
- top-k/top-p sampling；
- BM25 或 RRF；
- Recall@k、MRR、token F1；
- LoRA Linear；
- 有并发上限、monotonic deadline、bounded retry、Retry-After 与 replay guard 的异步调用；
- 幂等工具执行与参数 schema；
- KV Cache 容量估算。
