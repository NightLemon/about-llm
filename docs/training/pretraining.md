# 预训练：从下一个 Token 到一场可以恢复的训练

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想理解预训练目标、数据系统、计算预算和 checkpoint 的工程师。
- **先修**：[Transformer](../core/transformer.md)、tokenization、loss 与梯度基础。
- **首次阅读**：一条序列 → 数据混合 → token/compute 预算 → 稳定训练 → crash recovery。
- **完成信号**：能写出有效 token 口径，并说明 checkpoint 必须恢复哪些状态。
- **卡住时**：先回到[机器学习与深度学习](../foundations/ml-dl.md)的训练闭环。

</div>

想象你正在训练一个小型 decoder-only 模型。第 20,000 步 loss 正常，下一步突然 NaN；重启最近的 checkpoint 后，
loss 曲线又与不中断运行不同。问题可能来自坏数据、mask、FP16 overflow、学习率，也可能只是恢复时漏了 RNG 或
DataLoader cursor。

预训练不是“把很多文本喂给 Transformer”。它是一条从 source data 到 token loss，再到 optimizer update 和
durable checkpoint 的长协议。本章沿一条训练序列走完这条链。

## 一条 Token 序列究竟在优化什么

对序列 (x_{1:T})，decoder-only 模型分解联合概率：

\[
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
\]

带 mask 的平均负对数似然是：

\[
\mathcal L(\theta)=
-\frac{\sum_{i,t}m_{i,t}\log p_\theta(x_{i,t}\mid x_{i,<t})}
{\sum_{i,t}m_{i,t}},
\qquad m_{i,t}\in\{0,1\}.
\]

分母是有效 target token 数。Padding、跨文档边界后不应训练的位置和其他无效 token 不进入分子或分母。
按固定 `batch × max_length` 平均，会让不同 padding 比例悄悄改变梯度尺度。

训练使用 teacher forcing：真实前缀已知，所以所有位置可以并行算 logits。推理才把模型刚生成的 token 反馈到下一步。
这个目标奖励“训练分布中下一个 token 概率更高”，没有直接优化事实时效、无害性、工具授权或用户任务成功。

### 用四个 Token 检查 Shift

设一条序列为：

```text
[BOS, A, B, EOS]
```

常见训练对齐是：

```text
input : BOS  A    B
target: A    B    EOS
```

有些模型在 loss 内部 shift，有些 data collator 预先 shift。两处同时做会错开一位。正式训练前用一个微型序列打印
input IDs、targets 与 loss positions，确认 BOS/EOS、padding 和 ignore index 的实际处理。

### Packing 需要同时管 Visibility 和 Loss

把多个短文档放进一个窗口可以减少 padding。若普通 causal attention 允许后一篇读取前一篇，模型会学到不存在的
跨文档关系。Block-diagonal/document-aware attention 控制“能看谁”，loss mask 控制“哪些位置计分”；二者不能
互相替代。Position IDs 与 EOS 语义也要和 packing 协议一致。

## 数据系统先回答“这条 Token 从哪里来”

训练 shard 是派生产物。原始对象至少要能追溯：

```text
source ID / URI / snapshot time
license or usage basis / consent / retention
language / domain / quality / risk labels
raw + normalized hash / parser revision
PII-secret findings / deletion lineage
split + shard assignment
tokenizer ID + revision
```

若 token shard 无法追回 source，团队就很难处理删除请求、许可审计、benchmark 污染和 parser 勘误。

过滤器也不是无损清洁：语言识别可能伤害代码混排和低资源语言；质量模型可能偏好主流写作风格；安全过滤过强
会删除研究危险行为所需的负样本。每条规则应记录输入量、删除量、抽样复核和已知偏差。

### 去重与污染分四层看

| 层级 | 例子 |
|---|---|
| Exact duplicate | 完全相同字节或规范化文本 |
| Near duplicate | 模板页、转载、轻微改写 |
| Semantic overlap | 答案、摘要、翻译或解释泄露评测题 |
| Source-group leakage | 同一文档的不同切片跨 train/test |

污染检测会有漏报和误报。更可信的做法是同时保留时间切片、私有评测、扰动题与过程记录，而不是宣称“模型从未
见过任何等价内容”。

## 数据混合就是训练目标的一部分

设域 (d) 有 (n_d) 个可用 tokens，一种温度式采样权重是：

\[
q_d=\frac{n_d^\alpha}{\sum_j n_j^\alpha},
\qquad 0\le\alpha\le1.
\]

`alpha=1` 接近原始 token 比例；减小 alpha 会提高小域相对权重，同时增加重复采样和过拟合风险。
不存在跨项目通用的最佳 alpha。

“代码占 20%”还不够明确。它可能指 bytes、documents、tokenizer 后 tokens、batch slots 或 loss weight。
真正作用到参数的是进入 loss 的有效 tokens 与它们产生的 gradients。

训练中途改变 mix，相当于改变目标分布。课程学习或后期高质量数据可能有价值，但必须记录切换 token/step、
旧新权重、snapshot 与 schedule，否则 loss 跳变无法和数值故障区分。

## 用有效 Token 描述进度

一次 optimizer update 的近似有效 tokens 是：

\[
N_{update}=B_{micro}\times A\times D_{data}\times T_{effective}.
\]

其中 (A) 是 accumulation steps，(D_{data}) 是 data-parallel replicas，(T_{effective}) 是每样本平均有效
target tokens。Tensor/pipeline/expert parallel 没有产生新的独立数据副本，不能再次乘入 global batch。

大规模混合语料不一定有自然 epoch。报告 consumed effective tokens、optimizer updates 和各域 exposure，比只写
“训练了三轮”更可比较。

## 可变长度下怎样保持 Token Mean

一个 update window 内，第 (i) 个 micro-batch 有 loss sum (S_i) 与有效 token 数 (n_i)。如果每个 token 同权：

\[
L=\frac{\sum_i S_i}{\sum_i n_i}.
\]

先算每批 mean 再等权平均，会给短 batch 的 token 更大权重。正确实现要保留 numerator/count，在整个 window
得到全局 count 后缩放 sum-gradient，再做 clip、optimizer step 与 scheduler step。

DDP 默认对 rank gradients 求 mean，还会引入 world-size 因子。AMP 又要求先 unscale，再进行 global-norm clipping；
任何 micro-batch 出现 non-finite 时，整个 update 的 step/scheduler 行为必须跨 rank 一致。

仓库提供四个逐层推进的小实验：

| 实验 | 回答的问题 |
|---|---|
| `gradient_accumulation_toy.py` | Batch mean 与 token mean 是否给出同一梯度 |
| `ddp_token_mean_control.py` | DDP rank mean 下 world-size 怎样进入缩放 |
| `ddp_accumulation_no_sync_control.py` | `no_sync`、clip 与一步 update 是否对账 |
| `ddp_amp_overflow_consensus_control.py` | 一个 rank overflow 时所有 rank 是否共同 skip |

精确 tensors、fractions 和负例放在[分布式训练](../systems/distributed-training.md#global-batch-loss-normalization)。
这些小型 CPU 实验只检查局部机制；大型预训练是否收敛、GPU 吞吐如何，仍需目标训练任务回答。

## (6ND) 能估什么，不能估什么

对 (N) 个 dense non-embedding 参数和 (D) 个训练 tokens，常用粗略估算：

\[
C_{train}\approx6ND.
\]

直觉是矩阵前向约 (2ND)，反向对输入与权重再付约两倍前向。它省略或粗化 attention 的长度项、embedding/logit
head、activation、norm、recompute、padding、communication 和 kernel efficiency。

MoE 还要区分总参数与每 token activated parameters，并加入 routing/all-to-all；长上下文下 attention FLOPs 也不能
忽略。`6ND` 是预算量级，不是 GPU wall time、费用或能耗公式。

## Scaling law 应从小型 Pilot 里学

Scaling-law 研究说明，在固定模型族、数据和训练范围内，loss 往往随参数、数据和 compute 呈可拟合趋势。
Tokenizer、data quality、repetition、architecture 与 optimizer 变化都会改变曲线。

实务上先跑多组小规模 pilots，保持训练和 validation 协议一致，拟合本项目曲线及不确定性，再做预算决策。
论文中的 compute-optimal ratio 不应被原样套到另一种模型、数据和目标任务上。

## 初始化 Smoke Test 看哪些信号

训练前向能跑通还不够。初始化需要让 activation 与 gradient 在深度上保持可用尺度。核对：

- Pre/Post-Norm、RMSNorm/LayerNorm 与 epsilon；
- Attention score 的 head-dimension scaling；
- Residual projection 初始化；
- Tied embedding/LM head 是否共享 storage；
- RoPE/position config 与新增 vocabulary rows；
- Parameter、compute、reduction 与 optimizer dtypes。

用固定 batch 记录每层 activation RMS/max、logits distribution、initial loss、gradient norm 和 NaN/Inf。
如果某层从第一步就爆炸，先修初始化或数值路径，不要靠 gradient clipping 长期掩盖。

## AdamW、Warmup 与更新频率

Adam 的 moments：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

偏差修正后，AdamW 的简化更新是：

\[
\theta_{t+1}=\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta_t\lambda\theta_t.
\]

Epsilon 放置、moment dtype、master weights、weight-decay mask 与 fused kernel 都属于实现契约。
Warmup 在随机初始化和 moments 尚不稳定时逐步增加学习率，之后再按 cosine/linear 等 schedule 衰减。

Global batch 增大通常减少 gradient noise，也减少固定 token 预算下的 update 次数。“Batch 加倍，LR 必然线性加倍”
只在有限范围内是经验近似。

全局 norm clipping：

\[
g'=g\min\left(1,\frac{c}{\lVert g\rVert_2+\epsilon}\right).
\]

它限制一次尖峰，不会修复长期错标签或 overflow。若大多数 steps 都触发 clipping，应调查数据、LR、loss scaling
与实现，而不是只欣赏一条平滑的 clipped-norm 曲线。

## 数值稳定要和数据、系统一起监控

BF16 的指数范围接近 FP32，通常比 FP16 更不易 overflow；FP16 常需要 loss scaling。Mixed precision 不表示所有
状态都用同一 dtype。

| 层级 | 需要记录 | 常见问题 |
|---|---|---|
| Loss | Global + domain/language/length slices | 坏 batch、mask/shift、mix change |
| Gradient | Global/per-layer norm、clip rate、non-finite | LR、overflow、communication |
| Activation | RMS、max、NaN/Inf | Init、norm、residual、precision |
| Optimizer | Update/weight norm、moments、scale | Epsilon、decay、resume state |
| Data | Source/shard、有效 tokens、重复率 | Sampler、iterator、filter drift |
| System | Step time、throughput、collective、retry | Network、storage、thermal throttling |

全局平均 loss 可能掩盖一个小语言或高风险域已经崩溃。保留 batch/sample IDs，异常时才能重放具体输入。

## Crash 后到底从哪里继续

完整 checkpoint 至少包含：

```text
model parameters
optimizer moments / master weights / GradScaler
scheduler + global step + consumed/committed effective tokens
all relevant RNG streams
data shard / permutation / iterator / shuffle-buffer position
parallel topology + sharding metadata
model / tokenizer / data / config / code identities
```

“Sampler 已 emitted index”“Main loop 已 consumed sample”“包含该 sample 的 optimizer update 已 committed”是三个状态。
多 worker prefetch 会让 emitted cursor 领先 consumed；gradient accumulation 又会让 consumed 领先 committed。

恢复协议可以选择：

1. 只在 zero-grad/optimizer-committed boundary 保存，恢复后重放尚未提交 samples；
2. 保存半窗口 gradients、position、divisor 与 crash-time RNG，再从 consumed cursor 继续。

两种都要把 model、optimizer、scheduler、RNG 与 data state 绑定为同一发布身份。Manifest-last 只能标记文件集完整，
仍要考虑目录原子发布、`fsync`、远程对象存储和来源认证。

仓库的恢复实验分别演示：

- 漏 Scheduler/GradScaler/RNG/data state 会怎样分叉；
- DataLoader 从 emitted cursor 恢复会漏掉预取队列中的样本；
- 恢复正确 RNG 却漏掉半窗口 gradients，step 数相同也会得到不同参数；
- 保存 gradients 但使用错误 RNG，同样无法 exact replay。

运行方法与具体结果见[Single-GPU Finetuning 恢复实验](../practice/projects/single-gpu-finetuning.md#controls)。
这些都是小型 CPU 故障注入；跨节点预训练 checkpoint 需要在目标集群上重新验收。

### 怎样做恢复等价性测试

在小规模上比较：

1. Uninterrupted 运行 (K+M) steps；
2. 运行 (K) steps，写盘并终止进程；
3. 新进程加载后再运行 (M) steps；
4. 比较 batch IDs、LR、loss、parameters、optimizer state 和下一随机数。

不同硬件/collective 的浮点顺序可能使 bitwise 相等不可得，但允许误差和来源要预先定义。只比较恢复后的第一条
生成文本太弱，无法发现 optimizer 或 data cursor 漂移。

## 长上下文课程会改变训练分布

先短后长可以节省 attention compute，但会改变 position distribution、packing 和每 update token 数。
最终模型要在目标长度上看到足够训练 tokens，并分别评测 evidence position、跨段多跳、冲突版本、全局聚合、
长输出约束与短上下文回归。

只增大 `max_position_embeddings` 不代表模型学会利用长上下文。RoPE scaling、继续训练数据和 runtime kernel 是三层
不同工作。

## 继续预训练与领域适配

Domain-Adaptive Pretraining（DAPT）继续使用领域无标注文本做 next-token training，适合术语、文体和领域分布适配。
它也可能造成通用能力退化、灾难性遗忘、重复记忆与 chat behavior drift。

至少比较：

1. 原 base + Prompt/RAG；
2. 纯领域继续训练；
3. 领域 + 通用 replay 的多种比例；
4. 不同 LR/token budget；
5. 领域、通用、安全与记忆 slices。

更换 tokenizer 会破坏原 token IDs 与 embedding/LM head 对应。扩词表必须保留旧 IDs、初始化新 rows、决定是否
训练 tied head，并用足够新-token 数据验证；它不是一个可随意替换的前处理组件。

## 投入大规模计算前的七道门

1. 单 batch forward：Shape、mask、initial loss；
2. 单 batch overfit：模型与 labels 能对齐；
3. 小数据多 batch：Sampler 与 validation 不泄漏；
4. Save/resume：Optimizer、RNG、data position 能恢复；
5. 单卡/多卡对账：Global batch 与 reduction 相同；
6. 故障注入：Bad batch、NaN、partial checkpoint 与 worker restart；
7. 目标规模 pilot：Throughput、memory、communication 与 cost extrapolation。

本仓库 JAX/Optax 与 PyTorch tiny GPT 覆盖部分 CPU 机制。它们没有大规模集群预训练结果，文中的 FLOPs、显存和
scaling 关系也都是带假设的模型。

## 自测

1. Teacher forcing 为什么允许并行训练，而生成仍需逐 token decode？
2. Packing 中 attention mask 与 loss mask 分别阻止什么泄漏？
3. 为什么 `examples × max_length` 会高估有效训练 tokens？
4. (6ND) 在 MoE 和长上下文场景下遗漏哪些主要项？
5. 一个 sample 已 consumed、但 optimizer update 未 committed 时，checkpoint 有哪两种恢复策略？
6. Loss 下降而下游退化时，怎样从数据、数值和系统三层定位？
