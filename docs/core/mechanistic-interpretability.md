# 机制可解释性：从可解码信息到因果干预

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备设计 probe、归因、activation patching、SAE 或模型编辑实验的工程师与研究者。
- **先修**：[架构谱系与运行时依赖](architectures-interpretability.md)、残差流、logit 与基本因果实验直觉。
- **首次阅读**：证据阶梯 → 连续 metric → probe/归因 → patching/path patching → SAE/editing → 验收。
- **完成信号**：能区分行为相关、可解码信息、归因敏感性与干预证据，并为每类结论设计负对照。
- **卡住时**：先固定一组 clean/corrupt 输入和一个方向不变的 logit difference。

</div>

用一组贯穿本页的反事实开始：

```text
clean:   The capital of France is
corrupt: The capital of Germany is
```

模型分别更偏向 `Paris` 和 `Berlin`。这里要问的不是信息在计算图上能否传播，而是在这个具体行为中，
哪些表示、组件或路径真的改变了输出。架构层的前置判断见[架构谱系与运行时依赖](architectures-interpretability.md)。

## 从行为到机制：先看证据阶梯

| 证据层 | 怎样做 | 可以回答 | 仍不能直接回答 |
|---|---|---|---|
| 行为 | 最小对照或反事实 Prompt | 改输入后，输出是否随之改变 | 内部哪一部分造成变化 |
| 可解码性 | 训练线性 probe 读取 hidden state | 某项信息能否从表示中读出 | 模型生成答案时是否使用它 |
| 归因 | Gradient、Integrated Gradients、遮挡、attention 分析 | 输出对哪些输入或组件敏感 | 该组件是否必要，归因是否唯一 |
| 组件干预 | Ablation、activation patching | 改动指定组件后，metric 是否改变 | 它是否是唯一机制或完整 circuit |
| 路径干预 | Path patching、causal scrubbing | 限制传播路径后，预先预测的行为是否成立 | 结论能否推广到未测行为和输入 |

这不是一张“做得越往下就一定越正确”的排行榜。每一层都需要自己的对照和假设，只是回答的问题不同。
Probe 准确率很高，只说明 probe 能取到信息；activation patching 恢复一个指标，也不等于发现了唯一 circuit。

## 行为实验先定义连续 metric

先用本章开头的 France/Germany 对照定义一个从头到尾不变的读数。两个 Prompt 只改变国家 token，
模板、长度和 tokenization 尽量保持一致。然后固定比较 `Paris` 与 `Berlin` 的 logit：

\[
m(x)=z_{\mathrm{Paris}}(x)-z_{\mathrm{Berlin}}(x).
\]

这个方向在所有运行中都不变：France 输入应得到较大的 \(m\)，Germany 输入应得到较小的 \(m\)。
如果每个输入都把“正确答案”放在减号前，两个指标的方向会同时为正，反而无法计算后面的恢复程度。

连续的 logit difference 比只看最终输出更敏感，可以在 top-1 翻转前观察变化。不过，它只衡量这两个
单 token 候选。答案包含多个 token 时，应预先定义整个序列的得分，不能看完结果再挑最有利的首 token。

负对照至少包括：

- 替换无关位置或未来位置；
- 随机打乱作为来源的 activation；
- 使用分布相近、但与任务无关的来源；
- 检查变化是否只来自 activation norm；
- 在多个模板、位置、语言和随机种子上复现。

## Probe 检查“能否读出”，不是“是否使用”

线性 probe 是一个额外训练的小模型。它从 frozen hidden state \(h\) 预测我们关心的属性：

\[
\hat y=Wh+b.
\]

如果同一实体或模板同时出现在训练集和测试集，probe 可能只记住词面。数据应按实体和模板分组切分，
同时控制类别、长度和 token 位置。还要拿简单的输入特征作基线，而不只是和随机猜测比较。

Probe 太强时，可能自己学会任务，而不是揭示原模型已有的简单表示。反复尝试 layer、feature 和
超参数后继续汇报同一个测试集，也会把选择偏差伪装成泛化能力。

更进一步，可以先预测“沿 probe 找到的方向做干预后，目标行为会怎样变化”，再检查预测是否成立，
以及无关能力是否保持。

## Attribution 描述 sensitivity，仍要小心输入分布

Gradient 描述当前点附近的局部敏感度：输入轻微变化时，输出分数朝哪个方向变化。遇到 saturation，
gradient 可能接近零，即使离开当前区域后该 feature 仍然重要。

Integrated Gradients 从 baseline \(x'\) 沿路径积分：

\[
IG_i(x)=(x_i-x_i')
\int_0^1
\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}
d\alpha.
\]

结果会随 baseline、积分路径和目标函数改变。特别是对 token embedding 做线性插值时，
路径中间点通常不对应任何自然语言输入。

遮挡法（occlusion）会删除、mask 或置零某项输入。这很直观，却可能制造训练分布之外的样本；
多个相关 feature 共同起作用时，贡献也不会自动公平分摊。

注意力权重（attention weight）只表示某层某个 head 怎样加权 value。它还没有包含 value 的内容、
输出投影、残差、MLP 和后续层。

因此，attention 图可以提示模型“正在读哪里”，却不是最终 token 重要性的完整答案。

## Activation patching 直接干预中间表示

Probe 问“这里能不能读出国家信息”，activation patching 问得更直接：如果把 France 运行中的某个
中间表示搬到 Germany 运行里，模型会不会重新偏向 `Paris`？基本实验分四步：

1. 运行 clean input `France`，缓存指定位置的 activation，并记录 \(m_{clean}\)；
2. 运行 corrupt input `Germany`，记录 \(m_{corrupt}\)；
3. 再运行 `Germany`，但在预先选定的位置换入 `France` 的 activation；
4. 记录干预后的 \(m_{patched}\)，其余计算保持不变。

常用恢复分数：

\[
R=\frac{m_{patched}-m_{corrupt}}{m_{clean}-m_{corrupt}}.
\]

当 \(R\) 接近 1，说明这次干预把选定指标拉回了 clean 水平；接近 0，说明它几乎没有缩小差距。
分母接近零时，比例会非常不稳定。此时应停止计算恢复率，只报告三个原始 metric。
\(R>1\) 和 \(R<0\) 都可能出现，不能为了画图方便而裁剪到 `[0,1]`。

这项干预可能制造模型训练时从未见过的 activation 组合。即使恢复率很高，严谨结论仍然只是：
“在这组输入、这个位置和这项替换下，指定 metric 发生了变化。”冗余组件、备用路径和纠缠在一起的
features，都可能让一次 ablation 或 patching 给出不完整答案。

### 先用随机 MiniGPT 检查 hook 和因果负例

第一次不要急着解释自然语言知识。仓库先在固定随机种子的两层 MiniGPT 上执行真实 PyTorch hook：

~~~powershell
python projects/transformers-basics/activation_patching.py
python -m pytest tests/test_activation_patching.py -q
~~~

输入 `[1,2,3,4]` 与 `[5,2,3,4]` 只改首 token。程序分别替换第 0 层残差输出中的来源位置、
读取位置和完整 causal prefix，确认 metric 会按构造改变；再替换未来位置，确认它不能影响过去的读取位置。
这个负对照同时检查 causal mask 和 hook 位置是否接对。

先把 hook 的选择写成一个 shape 合同。这个脚本的残差输出是 `[batch, sequence, hidden] = [1,4,16]`；
`changed_source_position` 从 clean run 取 `activation[0, 0, :]`，并替换 corrupt run 的同一向量，
metric 则读 `logits[0, 1, 27] - logits[0, 1, 19]`。`future_position_negative_control` 替换位置 2，
却仍读取位置 1，所以它应保持为负对照。这里的位置从 0 开始计数。先写清 batch、位置、hidden 维和
logit 读数，才能区分“替换了来源”与“误把读取位置或未来位置接进 hook”。

这一步只检查张量、hook 和因果方向。模型没有训练过；比较哪两个输出 token，也是在看过两次运行的差异后
才为这个样例选择的。完整 prefix 恢复 clean 输出更是特意构造的正例，所以这里不能声称发现了自然语言 circuit。

### 在固定 Qwen checkpoint 上做一次单事实干预

确认实验方法能工作后，再把同一思路用于真实权重。这里使用
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`。
套用 chat template 后，两条输入都是 26 tokens，只有位置 19 从单 token ` France` 变成 ` Germany`；
程序在位置 25 始终计算 `logit(Paris) - logit(Berlin)`。

这也是从 0 开始计数：残差 site 的形状为 `[1,26,896]`，source slice 是 `[0,19,:]`，metric 读取
`logits[0,25,:]`。未来位置负对照（future negative control）会先追加一个第 26 位 token，再替换这个未来切片，同时仍读
位置 25；它检查因果方向，不是在原来 26 token 的张量上访问不存在的位置。

~~~powershell
python projects/transformers-basics/run_qwen_activation_patching_control.py `
  --local-files-only
~~~

这是选修的本地 snapshot 实验。`--local-files-only` 会在缺少该 immutable revision 或运行所需依赖时停止，
不会下载或改用其他 checkpoint；此时先完成上面的 MiniGPT hook 实验即可。它验证的是 hook、shape 和因果
负对照，仍不是这个 Qwen 实验的替代证据。

实验在执行前按“第一层、中间层、最后一层”的规则选定 layers 0、11、23，没有从全部层里挑最好结果。
Clean metric 是 `9.210311`，corrupt metric 是 `-7.700302`，两者相差 `16.910613`。录制结果如下：

| 替换的位置 | Recovery | 这一步实际说明什么 |
|---|---:|---|
| Layer 0，来源位置 19 | 约 1.000 | 替换整条 896 维 residual 后，这个指标回到 clean 附近 |
| Layer 11，来源位置 19 | 约 0.992 | 对这组输入和干预接近恢复，尚未定位具体 head、MLP 或 feature |
| Layer 23，来源位置 19 | 0 | 该 hook 之后没有让来源位置再次影响读取位置的跨 token mixing |
| Layer 0，全部 causal prefix | 1 | 完整搬入 clean prefix 的构造性正对照 |
| Layer 23，读取位置 25 | 1 | 直接换入最终读取 residual 的构造性正对照 |
| Layer 0，未来位置 26 | 0 | 未来位置不能改变过去读取位置的负对照 |

这些数字来自真实 Qwen 权重，但它们不表示“第 11 层存储了法国首都”。样本只有一个英文事实、一个模板，
batch size 为 1；一次替换 896 维 residual，可能同时搬运实体、词形、位置信息和其他 features。

要形成机制研究，还要加入随机或无关来源、改写、跨语言对照、多个事实、更细的组件或路径干预，
并在未参与选择方案的数据上复现。报告自带的无密钥 hash 能发现内容是否变化，却不能认证是谁执行了实验。

## Path patching 把问题缩到组件之间的边

普通 activation patching 替换一个节点的全部输出，无法区分它通过哪条下游路径起作用。Path patching
尝试只改变“来源组件 → 指定下游组件”这条边，同时控制来源组件通往其他位置的影响。

在动手之前，应把 circuit 假设写成可以失败的预测：

```text
节点：哪些 heads、MLPs、features 或 residual sites
边：哪个投影、从哪个 token 位置传到哪里
行为：研究哪组输入，用什么 metric
必要性：移除这条 circuit，目标行为应下降多少
充分性：只保留这条 circuit，目标行为应保留多少
忠实性：干预不应制造哪些无关错误
```

即使这些预测在几十个模板上成立，结论范围仍然是这些行为、组件和干预，不能直接升级成
“整个模型都使用这套算法”。

## Sparse Autoencoder 尝试拆解 superposition

一个 neuron 同时对城市名、代码符号和否定词有反应时，很难给它贴上唯一标签。

Superposition hypothesis 提供了一种解释：模型可能在有限维 residual space 中，用彼此不正交的方向
表示更多稀疏 features，于是单个 neuron 看起来具有多种语义。这是一种待检验的表示模型，
不是已经适用于所有网络的定理。

Sparse autoencoder（SAE）常学习：

\[
f=\operatorname{ReLU}(W_{enc}x+b_{enc}),
\qquad
\hat x=W_{dec}f+b_{dec},
\]

并优化：

\[
\lVert x-\hat x\rVert_2^2+\lambda\lVert f\rVert_1.
\]

实际 SAE 也可能只保留 top-k activations、约束 decoder 向量的范数，或采用不同的归一化。不能只展示几个
“看起来很会说话”的 feature 名称，至少要从下面几方面评价：

| 问题 | 可以观察的指标 |
|---|---|
| 原表示保留了多少 | Reconstruction error、下游 loss 变化 |
| 表示是否真的稀疏 | L0/L1、每条样本激活的 feature 数 |
| 字典是否健康 | Activation frequency、dead features、不同宽度下的稳定性 |
| 人类解释能否泛化 | 留出样本上的精确率和召回率 |
| Feature 是否影响行为 | 预先设计的 feature intervention |

一个 feature 容易命名，不代表它对应真实且唯一的概念。随机种子、字典宽度或 regularization 改变后，
feature 可能分裂、合并或消失；这些稳定性本身就是实验结果。

## Model editing 检查改变是否局部而持久

Model editing 的目标不是重新训练整套模型，而是有针对性地改变某项行为。做法可以是更新少量参数或层、
施加 low-rank update、编辑中间表示，也可以把新信息放进 adapter、外部 memory 或检索系统。

至少从七个方向评估：

| 维度 | 问题 |
|---|---|
| 生效（Efficacy） | 目标 Prompt 是否更新 |
| 改写（Paraphrase） | 换一种问法是否一致 |
| 迁移（Portability） | 相关推论和其他语言是否同步 |
| 局部性（Locality） | 无关事实与能力是否保持 |
| 特异性（Specificity） | 相似实体是否被误改 |
| 持久性（Persistence） | 保存、merge 或继续训练后是否仍存在 |
| 冲突（Conflict） | 参数记忆与新的 RAG 证据冲突时怎样处理 |

把一条问答改对，不代表相关知识已经全局一致。经常变化、需要删除或追溯来源的事实，通常更适合放在
数据库或 RAG 中，因为来源、回滚和访问权限更容易审计。

## 可解释性工具也需要自己的验收

可解释性代码也可能“图画得很漂亮，但 hook 根本接错了”。至少检查：

- 随机化权重或标签后，attribution/probe 是否仍几乎不变；
- 更换 baseline、metric 或 Prompt 后，结论是否稳定；
- Hook 拿到的是 norm 前后、投影前后，还是完整 module output；
- KV cache、tensor parallel 和 quantization 是否改变 site；
- Patch 后 forward 是否真正消费新 activation；
- 更换 batch、dtype 或重复运行后是否复现。

Module 名称不是可靠的概念边界。名为 `attention` 的 hook 可能拿到每个 head 的 value、合并后的结果，
也可能已经经过输出投影。要结合源码、张量形状和一个结果可预测的最小干预来确认。

## 从 checkpoint 开始的一次机制审计

1. 记录模型与 tokenizer 的具体 revision，以及代码、dtype 和执行后端；
2. 读取配置与张量形状，画出真实计算图、推理状态和可干预的 residual sites；
3. 建立行为数据，预先定义连续 metric 和 clean/corrupt 对照；
4. 先跑行为基线，再做 probe 或 attribution，避免直接从内部图猜故事；
5. 写下因果预测，然后用 ablation 或 patching 尝试推翻它；
6. 加入无关来源、未来位置和随机权重等负对照；
7. 在不同模板、语言、位置、事实和随机种子上报告结果分布；
8. 把结论限定在实际测过的行为、metric、site 和 intervention 内。

仓库目前先用随机 MiniGPT 检查 hook，再在上述 Qwen checkpoint 上完成一次单事实干预。
这说明实验流程和局部 intervention 可以运行。要研究目标模型的普遍机制，还需要多样本、事先写定的分析方案
和留出数据上的验证。

## 安全结论不能从一个 circuit 外推

内部分析可以帮助发现记忆现象、router collapse、异常 features 或拒答路径，但覆盖范围有限。
Features 会随上下文变化，量化或分布式执行也可能改变 hook 对应的实际计算。

即使一条 circuit 在当前 Prompt 上非常稳定，也只解释了模型内部的一小部分行为。系统风险还可能来自
RAG、工具、权限、cache 和 runtime。

高风险部署仍需要行为评测、red teaming、访问控制、审批、监控和事件响应。

## 自测与实践

1. 为一个 probe 设计按实体和模板分组的切分，怎样避免词面泄漏？
2. Activation patching 的 clean-corrupt 分母接近零时应怎样报告？
3. Layer 11 的 source patch 恢复 0.99，为什么仍不能说“事实存储在第 11 层”？
4. 为一个 SAE 报告设计重建、稀疏度、稳定性和因果干预四类指标。
5. 选择公开小模型，在运行 patching 前写下行为、metric、hook shape、负对照和结论边界。
