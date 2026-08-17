# 预训练：从 token 目标到可恢复系统

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：预训练方案、训练平台和恢复机制设计者。
- **先修**：[Transformer](../core/transformer.md)、tokenization、损失与优化基础。
- **首次阅读**：训练目标 → 数据混合 → token/计算预算 → 稳定性 → checkpoint。
- **完成信号**：能写训练预算和包含数据、优化器、RNG 的恢复契约。
- **卡住时**：回到[机器学习与深度学习](../foundations/ml-dl.md)的训练闭环。

</div>

## 学习目标与证据边界

读完本章应能解释 decoder-only 预训练目标、数据混合与 token 记账，估算理想化 dense 训练计算量，设计 AdamW/schedule 与稳定性监控，并写出可精确恢复的 checkpoint manifest。还应能判断 loss 下降究竟证明了什么、哪些能力结论仍需独立评测。

本仓库提供 NumPy/PyTorch/JAX tiny Transformer 与 CPU overfit 证据，不包含大规模集群预训练结果。文中的 FLOPs、显存和扩展规律都是带假设的模型，不是本仓库实测成本；具体训练配方必须绑定模型、数据、tokenizer、硬件和软件版本。

## 预训练目标到底优化什么

对 token 序列 \(x_{1:T}\)，decoder-only 模型学习：

\[
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
\]

带 mask 的平均负对数似然为：

\[
\mathcal L(\theta)=
-\frac{\sum_{i,t}m_{i,t}\log p_\theta(x_{i,t}\mid x_{i,<t})}
{\sum_{i,t}m_{i,t}}
\]

\(m_{i,t}\in\{0,1\}\) 用于忽略 padding、跨样本边界后不应训练的位置或其他无效 token。分母应是**有效 target token 数**；若直接按固定 batch × length 平均，不同 padding 比例会改变梯度尺度。

训练时使用 teacher forcing：真实前缀已知，可以对所有位置并行计算 logits；推理时才逐 token 把生成结果反馈为新输入。这一目标奖励“训练分布中下一个 token 概率更高”，不直接优化事实实时性、无害性、工具授权或用户任务成功率。

### Shift、mask 与 packing 是一个契约

常见实现令输入为 `tokens[:, :-1]`、target 为 `tokens[:, 1:]`，也有模型内部完成 shift。必须用一个微型序列手工检查：位置 \(t\) 的 target 是否真是下一个 token、BOS/EOS 是否按设计参与 loss、padding 和文档边界是否被 mask。

Sequence packing 把多个短文档装入固定窗口以减少 padding。若允许后一个样本注意前一个样本，模型会学习不存在的跨文档关系；若使用 block-diagonal attention，则 position ids、attention mask 和 loss mask 必须共同隔离。只在 loss 上 mask，仍可能让后一个样本读到前文信息。

## 数据系统比“抓很多文本”更难

### 每个样本要能追溯

原始对象至少保留：

```text
source id / URI / snapshot time
license or usage basis / consent / retention policy
language / domain / quality / risk labels
raw hash / normalized hash / parser version
PII-secret scan result / deletion lineage
split and shard assignment
tokenizer id + revision
```

训练 shard 只是派生产物。若无法从 token shard 追回 source，就无法可靠处理许可审计、删除请求、污染调查或 parser 勘误。

### 过滤、去重与污染

过滤器常处理乱码、模板页、低信息文本、恶意内容、隐私与重复。每个规则都有误杀：语言识别可能伤害代码混排和低资源语言；质量模型可能偏向主流写作风格；安全过滤可能删去研究风险所需的负样本。

去重至少区分：

- exact duplicate：相同字节或规范化文本；
- near duplicate：模板、转载、轻微改写；
- semantic overlap：答案、摘要或翻译泄露测试内容；
- source-group leakage：同一文档的不同切片进入 train/test。

公开 benchmark 的题面、答案、解释和改写都可能污染。污染检测只能提供风险证据，不能证明“完全没见过”；最终报告应同时给出时间切片、私有集、扰动题和过程指标。

## 数据混合与采样分布

设域 \(d\) 的可用 token 数为 \(n_d\)。直接按 token 比例采样会让大域支配训练；对小域过采样又会增加重复和过拟合。常见温度式权重可写为：

\[
q_d=\frac{n_d^\alpha}{\sum_j n_j^\alpha},\quad 0\le\alpha\le1
\]

\(\alpha=1\) 接近原始 token 比例，\(\alpha<1\) 提高小域相对权重。这只是采样族，不存在对所有任务通用的最佳 \(\alpha\)。还需记录每个域的重复次数、有效 token、loss、梯度贡献与下游指标。

“代码占 20%”必须说明是原始字节、文档数、tokenizer 后 token、batch 槽位还是 loss 权重。真正影响优化的是进入 loss 的有效 token 和梯度。

### 动态混合

训练中改变混合比例相当于改变目标分布。课程学习、后期高质量数据或领域加权可能有用，但每次切换都应版本化：触发 step/token、旧新权重、数据 snapshot 和 schedule。否则 loss 跳变无法区分数据变化与数值故障。

## Token 预算、Batch 与计算估算

### 用 token 而不是 epoch 描述进度

大规模混合语料常重复采样、动态更新或没有自然 epoch。报告至少包含：

\[
N_{\text{step}}=B_{\text{micro}}\times A\times D_{\text{data}}
\times T_{\text{effective}}
\]

其中 \(A\) 是 gradient accumulation，\(D_{data}\) 是数据并行副本数，\(T_{effective}\) 是每样本平均有效 target token。TP/PP/EP 不复制独立数据，不能乘入 global batch。

当样本长度不同，`examples/step × max_length` 会高估训练 token；应直接累计 loss mask。

### 可变 token 下的梯度累积

Gradient accumulation 只在 reduction 契约正确时才近似一个大 batch。设一个 optimizer update 内第 \(i\) 个 micro-batch 的有效 target token 数为 \(n_i\)，对应 loss sum 为 \(S_i\)。若训练目标是所有有效 token 同权，应使用

\[
L=\frac{\sum_i S_i}{\sum_i n_i},
\]

而不是把每批的 \(S_i/n_i\) 再除以 accumulation steps。后者在 padding、assistant-only mask、packing 或截断造成 \(n_i\) 不等时，会让短批中的每个 token 获得更大系数。最稳妥的审计方式是同时记录每批 numerator/count，在整个 accumulation window 得到全局 count 后缩放 sum-gradient，并在 clip、optimizer step 与 scheduler step 前完成归一化。DDP 默认 gradient averaging 还会引入 world-size 因子，必须按实际 reducer 语义调整；详见[高效与分布式训练](../systems/distributed-training.md#mean-or-sum)。

仓库的 `gradient_accumulation_toy.py` 用 `Fraction` oracle 与真实 PyTorch Float64 `cross_entropy.backward()` 对照：有效 token `[1,3]` 时 full batch 与 sum/count 路径逐元素相同，等权 micro-batch mean 改变梯度。它没有执行 optimizer、dropout/BatchNorm、AMP、DDP/FSDP/ZeRO、CUDA 或目标模型，因此不能把局部 reduction 等价升级成训练等价、吞吐或质量结论。

独立的 `ddp_token_mean_control.py` 把同一反例分到两个 rank，并真实运行双进程 CPU/Gloo。默认 DDP 对 rank gradient 取 mean，所以 `D=2,N=4` 时 local loss sum 的正确 backward scale 是 `D/N=1/2`：同步梯度为 `(0.575,-0.575)`，与单进程 full batch 在 `1e-15` 内一致；错误的 `1/N=1/4` 得到 `(0.2875,-0.2875)`，rank-local mean 得到 `(0.35,-0.35)`。它证明当前 PyTorch/Gloo/default reducer 固定路径，不证明 gradient accumulation + `no_sync`、AMP/scaler、optimizer update、FSDP/ZeRO、GPU、多节点或真实预训练行为。

后续 `ddp_accumulation_no_sync_control.py` 已真实覆盖同机两个 rank、每 rank 两个 micro-batch、首批 `no_sync`、末批同步、同步后 global-norm clipping 与一次 plain SGD step。`[[1,2],[3,1]]` counts 给出 `N=7,D/N=2/7` 和精确 pre-clip gradient `(+19/35,-19/35)`；built-in 路径的 gradient、clip 后 gradient、参数更新都与 full batch 一致。官方 reference all-reduce hook 的独立计数对照显示 forward+backward 均在 `no_sync` 时 1 次 hook，只包 backward 时 2 次；后者没有节省通信，但在本线性 fixture 上仍得到相同数值。该 control 没有 AMP、dropout/BatchNorm、多 bucket、AdamW、checkpoint resume、FSDP/ZeRO、GPU、多节点、目标模型或吞吐证据。

AMP 必须另按实际 update window 验证。`amp_grad_scaler_control.py` 在真实 CPU FP16 autocast/GradScaler 上得到 scaled accumulated gradient 24；正确 `unscale_→clip` 是 `3→约 0.5`，与 full batch 相同，错误 `clip→unscale_` 则变为约 0.0625。一个窗口中任一 micro-batch 产生 non-finite gradient，`scaler.step` 会跳过整个 AdamW update；当前 fixture 连续观察 scale `8→4→2→1`，而 parameter、step 与 moments 都保持不变。进程内 split-run 还证明漏恢复 scale=1 的 scaler 会让下一条 10000 边界梯度在 fresh scale=8 下溢出并跳步。这个控制不包含 scheduler/RNG、磁盘 checkpoint、真实进程重启、DDP/CUDA、目标模型或质量证据。

进一步的 `checkpoint_resume_control.py` 把这些状态放进同一条真实 split-run：1 次有限 update 后的 3 次 overflow 只回退 scale，不推进 AdamW 或 `StepLR`；phase-1 进程写入 model/optimizer/scheduler/scaler、Torch CPU/Python RNG、stateful shuffle permutation/cursor/epoch 与 dataset hash 后退出，另一个 PID 恢复。8 个 attempt 的后半段 trace 和最终全部状态与 uninterrupted 进程 bit-exact；错误地在 overflow 上推进 scheduler，以及分别漏 scheduler/scaler/RNG/data state 都有独立反例。它仍只是 6 参数 CPU FP16 authored fixture、custom stream 与 zero-grad boundary；没有 accumulation 中间态、worker/prefetch、distributed checkpoint、CUDA/目标模型、crash recovery、性能或质量证据。

`ddp_amp_overflow_consensus_control.py` 则补 DDP 与 AMP 的同路径状态机，而不借用 resume 结论。双进程 CPU/Gloo 中，rank 0 在首个 `no_sync` micro-batch 产生 Inf 后，末批 built-in DDP reduction 让两 rank 都观察 non-finite，AdamW/StepLR 共同 skip、scale `8→4`。负对照先让 DDP 在两边得到 finite scaled grad=8，再于 rank 0 的 `unscale_` 前人为改成 Inf；rank-local GradScaler 于是让 rank 0 保持 step=1、rank 1 前进到 step=2，参数、moments、scheduler、LR 与 scaler 分叉。optimizer 前对 unscaled local flag 做 `all_reduce(MAX)` 可让两边共同 skip；但示例的 `update(new_scale=...)` 是显式 scale policy，growth tracker 保持 1，并不等同于 native found-inf transition。该 control 只有单参数/单 bucket、authored fault 和 CPU/Gloo，无 clipping、自然 overflow、custom hook、checkpoint、CUDA/NCCL、多节点、目标模型、性能或质量证据。

### \(6ND\) 只是理想化 dense 近似

对 \(N\) 个 dense 非 embedding 参数、\(D\) 个训练 token，常用：

\[
C_{train}\approx6ND
\]

直觉是一次矩阵乘前向约 \(2ND\)，反向对输入与权重约再付两倍前向，总计约 \(6ND\)。它省略或粗化：attention 随长度的项、embedding/logit head、激活函数、norm、重计算、稀疏专家、路由、padding、通信和 kernel 效率。

MoE 应区分总参数、每 token 激活参数与通信；长上下文下 attention FLOPs 不能忽略。估算值不能直接等同于 GPU wall time、用电或费用。

### Scaling law 怎么用

公开 scaling law 说明在特定模型族、数据和训练范围内，loss 随参数、数据和计算呈可预测趋势，并可讨论 compute-optimal 配比。它们不是永久常数：tokenizer、数据质量、重复、架构、优化器和目标任务变化都会改变外推。

实践流程是先做多组小规模 pilot，保持训练/评测协议一致，拟合本项目曲线与不确定性，再用于预算决策；不能只把论文中的参数/token 比例套到另一种模型和数据。

## 初始化与信号传播

初始化要让前向激活和反向梯度在深度上保持可用尺度。线性权重常按 fan-in 或隐藏维缩放；残差分支可能额外缩小，具体规则依 norm 位置、层数和架构。

必须核对：

- pre-norm/post-norm 与 RMSNorm/LayerNorm epsilon；
- attention logits 是否按 head dimension 缩放；
- residual projection 的初始化；
- tied embedding/lm head 是否共享同一 storage；
- RoPE/position 参数和词表新增行；
- 实际 parameter dtype 与计算 dtype。

“能跑一轮前向”不够。初始化 smoke test 应统计各层 activation RMS/最大值、logits 分布、初始 loss、gradient norm 和 NaN/Inf。

## AdamW、学习率与更新频率

Adam 的矩估计可写为：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

偏差修正后，AdamW 更新的简化形式是：

\[
\theta_{t+1}=\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta_t\lambda\theta_t
\]

实现细节包括 epsilon 放置、moment dtype、是否有 master weight、weight decay mask 和 fused kernel，必须按框架版本核对。norm scale、bias-like 参数是否衰减是实验配置，不应被默认隐藏。

### Warmup 与衰减

Warmup 在随机初始化和 optimizer moment 尚不稳定时逐渐提高学习率。之后可用 cosine、linear 或其他 schedule；若计划继续训练，衰减到严格 0 可能不利于无缝续训。

学习率应与 global batch、模型规模、初始化和 optimizer 联合调节。“batch 加倍，学习率必然线性加倍”只在有限条件和范围内近似成立。超大 batch 减少每 token 参数更新次数，也可能改变泛化与稳定性。

### Gradient clipping

全局 norm 裁剪：

\[
g' = g\min\left(1,\frac{c}{\lVert g\rVert_2+\epsilon}\right)
\]

它限制单步尖峰，不修复持续的数据错误或数值溢出。日志应记录裁剪前 norm、是否触发和按层/参数组统计；若大部分步骤都被裁剪，应调查学习率、loss scale、异常 batch 或实现错误。

## 上下文长度与数据课程

短序列的 attention 便宜，先短后长可能节约算力；但它改变 position 分布、每更新 token 数、packing 和数据选择。最终模型要在目标长度上看到足够 token，并评测：

- 不同位置的检索；
- 跨段多跳与冲突；
- 全局聚合与顺序；
- 长输出约束保持；
- 短上下文能力是否退化。

只把配置中的 max position 调大，不代表模型学会利用长上下文。RoPE scaling/插值、继续训练数据和 runtime kernel 是不同层面的工作。

## 数值精度与稳定性

BF16 的指数范围接近 FP32，常比 FP16 更易训练；FP16 往往需要 loss scaling。混合精度不等于所有状态同 dtype：参数、master weight、梯度、归约、optimizer moments、norm/softmax 与 loss 可能不同。

监控至少包括：

| 类别 | 信号 | 常见根因 |
|---|---|---|
| loss | spike、NaN、突然下降 | 坏 batch、mask/shift、overflow、数据切换 |
| gradient | global/per-layer norm、裁剪率 | 学习率、异常 token、通信或 scaler |
| activation | RMS、最大值、NaN/Inf | 初始化、norm、残差、低精度 |
| optimizer | update/weight norm、moment | epsilon、state 恢复、weight decay |
| data | 域/语言/长度、重复率、有效 token | sampler、shard、iterator 恢复 |
| system | step time、通信、straggler、重试 | 网络、存储、热降频、编译 |

全局平均 loss 可能掩盖小语言或高风险域崩溃。按域、语言、长度和 source bucket 切片，同时保留 batch/sample id 以便回放。

## Checkpoint 与精确恢复

完整 checkpoint 不只是 model weights：

```text
model parameters
optimizer moments / master weights / scaler
scheduler and global step / consumed effective tokens
all RNG streams
data shard, iterator and shuffle-buffer position
parallel topology and sharding metadata
model/tokenizer/data/config/code versions
```

原子发布流程通常是：写临时目录 → 每个 shard 完成并校验 → 写 manifest/hash → 独立加载 smoke test → 原子标记为 complete。只有 manifest 完整的 checkpoint 才可恢复；不能因为目录存在就认为保存成功。

DataLoader 的“sampler 已产生 index”不等于“训练已消费 sample”，更不等于“包含该 sample 的 optimizer update 已提交”。仓库 `dataloader_prefetch_resume_control.py` 用真实双 worker、prefetch factor 2、spawn 和 batch 1 固定出三层边界：训练只接收 permutation 前 3 条时，sampler emitted cursor 已到 7，queue 中还有 `[7,0,9,4]`；从 7 恢复会跳过它们，从应用记录的 consumed cursor 3 重建则恢复完整 ID 顺序。当前观察到的 ahead=4 与安装版 PyTorch 的 loader 调度一致，但 control 未读写私有 queue 字段，不能把具体 prefetch 深度当跨版本公开 API。

同一实验还证明只恢复顺序不恢复 worker-local random stream：fresh worker tail 的 `torch.rand` 与 uninterrupted 不同，而用 sample ID 派生的局部 RNG tail exact。Stateless key 在真实多 epoch/重复采样中还需加入 dataset/transform revision、epoch/visit 等语义；否则每次见到同一 ID 都产生同一增强。该控制无模型、optimizer 或分布式 sampler，也未保存 worker/queue state，因此只能指导数据交付协议，不能替代训练 checkpoint 的完整一致性边界。

后续 `optimizer_commit_resume_control.py` 用 main-process inverted-Bernoulli mask、真实 backward、SGD momentum、StepLR 和两步 gradient accumulation 固定出 emitted/consumed/committed=`7/3/2` 的崩溃窗口。base checkpoint 不保存第三条 sample `1` 的 `.grad`，但保存 commit-boundary model/optimizer/scheduler/Torch RNG；从 committed=2 同时恢复 RNG 并重放后，与 uninterrupted 的 ledger 和四类终态 bit-exact。第一个隔离负例从 consumed=3 起步、恢复正确 crash RNG 却漏 gradients：未来 mask 与终态 RNG 相同，ledger 漏 `1`；即使两边仍各有 5 次 optimizer/StepLR step、LR 同为 `0.0125`，参数最大差仍为 `0.005767858566116724`。

同一 phase 另写绑定 base digest 的 gradient sidecar，保存 pending `[1]`、accumulation position/divisor、两个 finite gradients与 crash-observed Torch RNG；最后发布的 canonical manifest 再绑定数据 identity、base/sidecar name/schema/size/hash 与完成顺序。第五个 PID 先验 manifest、后按声明 hash 重查实际反序列化 bytes，从 consumed=3 把 `[1,7]` 完成成一个窗口，终态也与 uninterrupted bit-exact。第六个 PID 保留完整 gradients/ledger，却故意使用 commit-boundary RNG；它同样执行 5 steps、LR 仍为 `0.0125`，但参数最大差为 `0.017878893573032573` 且终态 RNG 不同。这证明“step 数、ledger、sidecar 都完整”仍不能替代相关 RNG，也证明完整半窗口 state 可以恢复当前 stochastic fixture。

故障矩阵还证明 sidecar 协议会拒绝 base-only、两 payload 无 manifest、manifest 缺 sidecar 与 sidecar 被篡改四种快照；其中 base-only 并非坏 checkpoint，仍可从 committed=2 replay。manifest-last 是 completeness gate，不是 base+sidecar+manifest、sample 与 optimizer 的原子事务；无目录 `fsync`、power-loss/filesystem fault、原子目录/远程存储或来源认证证据。实验覆盖 main-process Torch RNG 与 StepLR，但仍无 worker/Python/NumPy/CUDA RNG、原生随机层、GradScaler、distributed shard、CUDA 或目标模型证据。

### 恢复等价性测试

在小规模上比较：

1. 连续训练 \(K+M\) 步；
2. 训练 \(K\) 步保存，重启后再训 \(M\) 步；
3. 对比 batch id、loss、参数/optimizer state 和下一随机数。

硬件/collective/浮点顺序可能让 bitwise 相等不可得，但差异应在定义的容差和来源内。只比较加载后的第一条生成文本太弱。

## 评测、记忆与停止决策

训练/validation NLL 衡量 token 预测；下游还要测知识、推理、代码、多语言、安全、记忆、偏见和长上下文。公开基准可能被训练污染，应增加时间切片、私有集与扰动题。

停止训练不是只看 loss：

- validation 与目标任务的边际收益；
- 各域是否出现过拟合或遗忘；
- 新增 token 的计算/资金机会成本；
- 推理预算和部署模型尺寸；
- 数据是否开始高重复；
- 安全/记忆风险是否恶化。

检查点选择也不能在最终 test 上反复试；应由 validation 和预注册规则决定，最终集只用于有限确认。

## 继续预训练与领域适配

Domain-Adaptive Pretraining（DAPT）用领域无标注文本继续 next-token training，适合术语、文体和领域分布适配。它可能带来通用能力退化、灾难性遗忘、重复记忆和 chat 行为漂移。

可信实验至少比较：

1. 原基座 + Prompt/RAG；
2. 纯领域继续训练；
3. 领域 + 通用 replay 的不同比例；
4. 不同学习率/token 预算；
5. 领域、通用、安全和记忆切片。

如果改变 tokenizer，原 token id 与 embedding/lm head 对应会破坏。扩词表需要保留旧 id、初始化新行、决定是否训练 tied head，并用足够新 token 数据验证；“直接换一个中文 tokenizer”不是兼容操作。

## 可运行的最小验收

在投入集群前，依次通过：

1. 单 batch forward：shape、mask、初始 loss；
2. 单 batch overfit：loss 明显下降；
3. 多 batch 小数据：sampler/validation 正确；
4. save/resume 等价：optimizer/RNG/data 位置恢复；
5. 单卡到多卡等价：global batch 与 loss reduction；
6. 短运行故障注入：坏 batch、NaN、checkpoint 中断、worker 重启；
7. 目标规模 pilot：吞吐、内存、通信与成本外推。

本仓库 JAX/Optax 项目覆盖第 1–2 项的一条 CPU 路径；其余不能因单元测试通过而写成已验证。

## 常见错误

- loss shift 一位或只 mask loss、不隔离 packed attention；
- 把原始字节/文档比例当作实际 loss token 混合；
- 用 `examples × max_length` 冒充有效训练 token；
- 把 \(6ND\) 当作精确 wall time、能耗或 MoE 公式；
- 大 batch 机械线性放大学习率；
- 裁剪后 norm 很平稳就认为训练没有尖峰；
- checkpoint 只保存权重，却称为可恢复训练；
- 在最终 test 上选 checkpoint；
- 继续预训练只报领域收益，不报通用/安全/记忆退化；
- 从小规模 scaling law 无误差地外推超出观测范围。

## 面试追问

1. teacher forcing 为什么允许训练并行，和推理分布差异在哪里？
2. packing 时 attention mask 与 loss mask 分别阻止什么泄漏？
3. \(6ND\) 从何而来，在哪些模型/上下文下会严重失真？
4. AdamW 与把 L2 penalty 加到 loss 有何实现差别？
5. global batch 增大时，更新频率、学习率与泛化怎样共同变化？
6. 怎样证明 checkpoint 恢复了 RNG 与 data iterator，而不只是权重？
7. 继续预训练如何区分领域学习、记忆和评测污染？
8. loss 下降但下游退化时，如何用数据/数值/系统三类证据定位？

## 一手资料

- Kaplan 等，[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)，特定设置下的规模规律。
- Hoffmann 等，[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，参数/数据 compute-optimal 研究。
- Loshchilov 与 Hutter，[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)，AdamW。
- Megatron-LM 与 PyTorch FSDP 等目标 runtime 官方文档；实际分布式配方应按固定版本核对。
- 本仓库 JAX/PyTorch tiny GPT、测试与运行 manifest；本仓库已执行结论的最高优先级证据。
