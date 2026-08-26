# 机器学习与深度学习：从一批工单到一个可验收模型

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：第一次系统理解训练、泛化和评测的工程师。
- **先修**：基本代数、均值与概率直觉；不要求先学完整微积分。
- **首次阅读**：任务定义 → 数据切分 → 损失 → 梯度更新 → 测试与漂移。
- **完成信号**：能为一个真实任务设计 train/validation/test，并解释训练 loss 与业务结果的差别。
- **卡住时**：回到[数学基础](math.md)查看概率、导数或矩阵形状。

</div>

假设你要训练一个售后工单分类器，把新工单路由到 `refund`、`logistics` 或 `fraud_review`。
第一版模型在随机切分的测试集上有 94% accuracy，上线后一周却频繁把欺诈投诉送进普通退款队列。

应该先换更大的模型吗？未必。你得先知道：一条样本是什么、标签从哪里来、测试集是否真的代表未来、
训练目标怎样给不同错误定价。机器学习的主线就在这几个问题之间，不在模型名字的数量里。

## 先用五分钟跑完一个最小闭环 { #ml-minimal-loop }

下面的脚本不下载数据或模型。它把同一个工单分类例子从数据切分一直追踪到业务分指标，让你先看到整张地图，
再逐节理解公式：

```powershell
python projects/transformers-basics/ticket_classification_walkthrough.py
```

输出依次展示四件事：

1. 逐行切分会让 `thread-100` 和 `thread-200` 同时出现在训练集与测试集；按 `thread_id` 切分则不会。
2. logits `[2.0, 0.5, -1.0]` 经过 softmax 后，真实类别 `fraud_review` 的概率约为 `0.0391`，
   对应的 negative log-likelihood（NLL）约为 `3.2413`。
3. 交叉熵对 logits 的梯度是 `p-y`。沿负梯度走一步后，真实类别概率上升，NLL 下降。
4. 只错一条工单可以得到 `99% accuracy`，但如果错的正是唯一一条欺诈工单，`fraud_review recall` 仍然是 0。

第三步为了把计算压缩到一屏，直接把三个 logits 当作可以更新的参数。真实模型更新的是网络权重，梯度还要通过
线性层和前面的网络继续反向传播。因此，这个局部步骤只验证交叉熵推动当前样本的方向；它没有训练真实模型，
也不能证明模型能够泛化或改善业务结果。

带着这四个观察，下面再把任务定义、切分、loss、反向传播和评测逐一拆开。

## 第一步：把问题写成一次可观察的预测

对这套工单系统，可以先写：

> 当一条新会话首次进入路由服务时，只使用当时可见的用户文本和账户公开状态，预测负责处理它的队列；
> 系统按会话计错，并把漏掉 `fraud_review` 视为更高成本的错误。

这句话固定了预测时点、输入、目标、样本单位和错误成本。少写其中任何一个，后面都可能出现“离线很准、
线上不可用”：例如训练数据包含了客服最终处理备注，而线上预测时这段备注还没有产生，这就是 target leakage。

设输入与目标来自部署分布 (P(X,Y))，模型 (f_\theta) 的总体风险是：

\[
R(\theta)=\mathbb{E}_{(X,Y)\sim P}
\left[\ell(f_\theta(X),Y)\right].
\]

我们无法枚举未来全部工单，只能在训练集 (D=\{(x_i,y_i)\}_{i=1}^{n}) 上最小化经验风险：

\[
\hat R_D(\theta)=\frac{1}{n}\sum_{i=1}^{n}
\ell(f_\theta(x_i),y_i).
\]

训练做的是第二件事，产品需要的是第一件事。两者之间的差距就是 **generalization（泛化）** 问题。

## 为什么随机逐行切分会给出虚假的 94%

回看数据后发现，同一会话被拆成了多行：用户追问、客服回复和最终处理结果。随机逐行切分把同一会话的相邻
文本分到了训练集和测试集。模型不需要学会处理新会话，只要认出已经见过的措辞即可。

修复方法是按真正近似独立的单位切分：

| 数据结构 | 切分单位 | 想阻断的泄漏 |
|---|---|---|
| 多轮工单 | 完整 `thread_id` | 同一上下文跨集合 |
| 同一用户的重复咨询 | `user_id` | 记住个人习惯或账户特征 |
| 一份文档的多个 chunk | `document_id` | 重叠段落跨集合 |
| 模板改写或镜像网页 | 去重簇/来源簇 | 只换表述的近重复 |
| 会随时间变化的业务 | 时间窗口 | 用未来规则预测过去 |

训练集用于拟合参数；validation 用于选模型、阈值和超参数；锁定 test 用于最后估计。每看一次 test 再修改系统，
都在把它逐渐变成 validation。

LLM 评测还要检查 benchmark 是否出现在预训练、SFT、few-shot 示例或 RAG 语料中。Exact match 只能发现完全相同
文本；近重复和语义改写需要额外检测，但检测器也会误报，所以结果应进入污染证据，而不是被当作绝对判决。

## 标签不是天然真相

工单的“负责队列”可能来自客服最终选择，也可能由事后审计员重标。前者便宜，却会继承历史团队的路由习惯；
后者更接近目标，却可能有分歧和覆盖不足。

在采集数据前至少问：

- 标签由谁、在什么时间产生？
- 标注者当时看到了哪些模型不可见的信息？
- 无法判断、多人协作和多标签工单怎样编码？
- 模型输出会不会改变未来能收集到的标签？

例如上线后只有“模型判为欺诈”的工单被专家复核，那么可见标签已经被旧模型筛选。直接拿这些日志重训，
会不断强化旧模型看得见的区域。

## 三类学习信号在 LLM 中怎样出现

学习范式的区别在于监督信号从哪里来：

| 范式 | 信号来源 | LLM 中的例子 |
|---|---|---|
| Supervised learning | 外部提供输入—目标对 | 工单分类、SFT response |
| Self-supervised learning | 从原始数据本身构造目标 | Next-token prediction |
| Reinforcement learning | 动作改变状态，随后获得回报 | Policy 根据 reward/value 更新 |

偏好 pair 可以直接进入 DPO 一类目标，也可以先训练 reward model 再用于强化学习。它们都属于“后训练”，
但数据分布、损失和失败方式不同，不能只因为阶段名称相同就混为一种算法。

## Loss 是训练的方向盘，不是产品成绩单

工单分类常用交叉熵。模型给真实类别 (y) 的概率为 (p_\theta(y\mid x)) 时：

\[
\ell=-\log p_\theta(y\mid x).
\]

它鼓励提高真实类别概率，却没有直接优化客服等待时间、欺诈损失或用户满意度。若漏判欺诈比错送普通退款贵，
可以调整类别权重、重采样或决策阈值；每种做法都会改变训练或部署时的有效问题，需要重新检查概率校准。

不同任务使用的代理目标也不同：

| 任务 | 常见 Loss | 需要额外注意 |
|---|---|---|
| 回归 | MSE、MAE | 噪声与离群值 |
| 分类 | Cross-entropy | 类别成本与阈值 |
| 检索/排序 | Pairwise、listwise、InfoNCE | 负样本是否太简单或是假负例 |
| 语言建模 | 有效 token 的 NLL | Shift、padding、prompt 与文档边界 |
| 偏好学习 | Pairwise preference objective | 选择偏差与 reference policy |

训练 loss 下降说明优化器更好地拟合了这份代理目标。它没有自动回答“新用户是否更满意”或“模型是否更安全”。

## 从线性模型走到神经网络

一个线性分类器直接计算 (z=Wx+b)，再把 logits 变成类别概率。它易于调试，也是一条重要 baseline：如果一个大型
模型无法稳定超过它，应先检查数据和评测，而不是继续堆参数。

神经网络把多层可微函数串起来：

\[
h_{l+1}=\sigma(W_lh_l+b_l).
\]

非线性 (\sigma) 让多层网络能够表达一层线性映射做不到的关系。若拿掉所有非线性，多层矩阵乘法仍可合并成
一个线性层。

模型结构还带有 **inductive bias（归纳偏置）**：卷积利用局部和平移结构；Transformer 用 attention 建立 token
间的内容相关交互，并通过位置机制表达顺序。归纳偏置决定模型更容易学到哪类规律，但不保证它只学到人希望的规律。

参数量也不是有效容量的唯一尺度。训练步数、数据多样性、参数共享、优化器和正则化都会改变可拟合的函数。
LoRA 只训练低秩增量，并不把底座的前向表示能力缩成同样的低秩大小。

## 反向传播到底在算什么

前向计算得到 loss 后，自动微分沿计算图反向应用链式法则。若标量 (L) 经过中间量 (h)：

\[
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial h}
\frac{\partial h}{\partial x}.
\]

框架能生成梯度代码，并不代表你的训练目标正确。意外 `detach`、原地修改、错误 mask、错误 reduction 和混合精度
溢出，都可能让程序继续运行却更新了错误方向。

一个可靠的第一步是让小模型过拟合一个极小 batch。如果做不到，先打印输入、labels、mask、有效监督数和梯度，
再考虑扩大数据或模型。

残差块 (x+F(x)) 为梯度提供较短路径，也让子层学习对输入的更新。LayerNorm 同时使用均值和方差，RMSNorm
按均方根缩放；两者是不同函数，不能在 checkpoint 上任意互换。Pre-Norm 与 Post-Norm 的位置差异同样会改变
训练动力学和模型函数。

## 优化器怎样把梯度变成更新

最基本的随机梯度更新是：

\[
\theta_{t+1}=\theta_t-\eta_t\hat g_t.
\]

Mini-batch 梯度是总体梯度的带噪估计。增大 batch 往往降低估计方差，也会改变显存、并行效率和优化行为；
“学习率随 batch 线性放大”只是特定范围内的经验规则。

Adam 维护梯度的一阶和二阶滑动统计。AdamW 另行执行 decoupled weight decay：

\[
\theta_{t+1}
=
(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

在自适应优化器中，这通常不等同于把 (\lambda\theta) 加进 loss gradient。Bias 和 norm 参数是否衰减，
也要由参数分组明确记录。

训练循环中几个常见部件各有职责：

- Warmup 缓解初期统计量尚不稳定时的大步更新；
- Learning-rate schedule 控制后期更新幅度；
- Global-norm clipping 在一次更新过大时整体缩放梯度；
- Gradient accumulation 用多个 micro-batch 构成一个 optimizer step；
- Mixed precision 减少部分计算/存储成本，同时引入 scale、overflow 与恢复状态。

Clipping 能限制一次异常更新，却修不好持续的脏数据或错标签。

## 为什么 Gradient Accumulation 容易悄悄换目标

设第 (i) 个 micro-batch 有 (n_i) 个有效 token，token loss 为 (\ell_{ij})。若训练目标是整个 update window 的
token mean：

\[
L_{token}=\frac{1}{N}\sum_i\sum_{j=1}^{n_i}\ell_{ij},
\qquad N=\sum_i n_i.
\]

直接平均每个 micro-batch 的 mean，会让短 batch 中的单个 token 获得更大权重。正确实现要累计 loss sum 与有效
token count，再按全窗口分母缩放；`ignore_index` 位置既不进分子，也不进分母。

分布式训练还要确认框架对各 rank 梯度做 sum 还是 mean。DDP 默认取 rank mean 时，全局 token mean 的缩放需要
显式包含 world size。AMP 则要求先 unscale 再做 global-norm clipping，checkpoint 也要保存 scaler。

这些不是只靠文字记忆的细节。下面三个小实验分别改变 reduction、`no_sync` 的作用范围和 AMP 顺序，
你可以直接观察梯度怎样随之变化：

```powershell
python projects/single-gpu-finetuning/gradient_accumulation_toy.py
python projects/single-gpu-finetuning/ddp_token_mean_control.py
python projects/single-gpu-finetuning/ddp_accumulation_no_sync_control.py
python projects/single-gpu-finetuning/amp_grad_scaler_control.py
```

完整推导与适用边界见[分布式训练](../systems/distributed-training.md#global-batch-loss-normalization)。

## 正则化是在表达偏好

过拟合表现为训练误差继续下降，而 validation 或目标分布误差恶化。欠拟合则连训练集都学不好，原因可能是
容量不足，也可能是优化失败、标签错位或特征不够。

常见正则化手段并不是可以全部拉满的开关：

| 手段 | 引入的偏好或变化 |
|---|---|
| Weight decay | 偏好较小的部分参数 |
| Dropout | 训练时随机屏蔽激活 |
| 数据增强 | 希望模型对保持标签语义的变换不敏感 |
| Early stopping | 用 validation 选择 checkpoint |
| Label smoothing | 让目标分布不再完全尖锐 |
| 去重 | 降低依赖近重复记忆，也改变来源权重 |

模型规模与过拟合不是简单单调关系。现代深度模型可能进入插值区域并出现 double descent；更大的模型也可能获得
更好的迁移能力。最终仍要把结论绑定到数据量、训练预算、正则化和目标分布。

## 怎样判断模型真的变好

工单系统的总体 accuracy 可能掩盖 `fraud_review` 全部漏判。评测要从业务决策反推指标：

| 决策问题 | 常看什么 | 容易遗漏什么 |
|---|---|---|
| 普通多分类是否改善 | Accuracy、macro-F1、confusion | 稀有类和高成本错误 |
| 稀有风险能否检出 | Precision、recall、PR-AUC | 阈值对应的人工处理量 |
| 概率能否用于路由 | Log loss、Brier、calibration curve | 分桶依赖与分布漂移 |
| 检索结果是否有用 | Recall@k、MRR、nDCG | 候选池与 relevance 标注 |
| 生成系统是否完成任务 | Task outcome、事实/引用、人工偏好 | 长度偏差、失败分母和成本 |

阈值在 validation 上按错误成本选择，再到锁定 test 报告。对同一批 case 比较两个系统时，保留逐 case difference
并使用 paired 分析；只比较两个独立均值会丢掉问题难度的配对信息。

概率校准也有边界。一组 80% 置信度的预测若约 80% 正确，可以称为校准；它仍可能没有区分能力。
LLM 的 token probability 更不是“整段回答为真”的直接概率。

## 上线后，分布还会继续变化

第二周 fraud 类别突然增加，可能是输入 (P(X)) 变了；业务规则调整后，同样文本对应的正确队列变了，
则更接近 (P(Y\mid X)) 变化。真实系统经常同时出现多种 shift，监控一项 embedding 距离不能自动证明质量下降。

生产监控要把输入变化与延迟标签、人工审计或可验证业务结果连接起来。还要关注反馈回路：路由模型决定哪些工单
进入专家队列，专家队列又决定下一轮训练能看到哪些高质量标签。

## 训练记录要支持“为什么坏了”的调查

一次可恢复、可比较的训练至少保存：

- 代码、依赖、模型配置、tokenizer 与模板 revision；
- 数据快照、过滤/去重规则、切分单位与混合权重；
- Seed 以及仍不能保证确定性的算子；
- Optimizer、参数分组、scheduler、precision 与 global batch 口径；
- Checkpoint 中的模型、optimizer、scheduler、RNG、scaler 和数据游标；
- Loss、梯度范数、吞吐、validation slice 与非有限值。

复现也有不同强度：bitwise replay、指标在随机波动内一致、以及最终工程结论一致。不同 GPU、kernel 和并行归约
可能产生细小浮点差异，因此先说明需要哪一种复现。

Loss 异常时按下面顺序缩小问题：

1. 在一个极小 batch 上过拟合。
2. 检查 token/feature、shift 后 labels、mask 和有效分母。
3. 确认关键参数收到有限梯度，并位于 optimizer 参数组。
4. 关闭 AMP、compile 和 distributed，在最小环境复现。
5. 固定数据顺序，对比单步更新与 checkpoint 恢复前后状态。

## 把这条主线映射到 LLM

| 工单分类中的问题 | LLM 对应问题 |
|---|---|
| 一条样本是什么 | Token 序列、SFT conversation、preference pair、tool trajectory |
| 标签从哪里来 | Next token、assistant response、chosen/rejected、reward 或 verifier |
| 怎样防止泄漏 | 文档/用户/时间 split，benchmark contamination，RAG corpus 边界 |
| 训练优化什么 | Token NLL、preference loss、reward proxy |
| 怎样验收 | 新任务/语言/时间段、风险 slice、事实与工具结果 |
| 上线怎样漂移 | Prompt、知识、tool/API、用户行为和路由变化 |

从小分类器到大语言模型，参数规模和系统复杂度变化很大，但判断顺序没有变：先固定任务与数据，再理解 loss 与
更新，最后用未参与选择的证据检查泛化。

## 自测与实践

1. 为什么把同一工单 thread 的多轮消息随机分到训练集和测试集会高估泛化？
2. 对有效 token 数为 10 和 100 的两个 micro-batch，比较 batch-mean 与 token-mean 下单个 token 的权重。
3. 解释 AdamW 的 decoupled weight decay 为什么通常不等同于在 Adam loss 中加入 L2。
4. 构造一个 accuracy 为 99%、风险类 recall 为 0 的分类器。
5. 为客服路由任务设计 group split、time split 和三个关键 slice，并说明每项防什么风险。
6. 在 MiniGPT 上尝试过拟合一个小 batch；若失败，按本章排查顺序记录第一个不变量破坏的位置。
