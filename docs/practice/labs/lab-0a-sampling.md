# 实验 0A：从 logits 到采样

**定位**：入门必做，预计 30–60 分钟，只用 CPU 和固定数据，不访问模型或网络。

**实验导航**：[返回总览](../labs.md#lab-0) · [生成与解码](../../core/generation.md) · [进入 0B](lab-0b-generation-protocol.md)
{ .doc-nav }

## 开始前

**先修知识**：知道 token 是模型处理的离散 id；知道概率总和为 1。看不懂 logit、softmax 或条件概率时，先回到[数学基础](../../foundations/math.md)和 [NLP 与语言建模](../../foundations/nlp.md)。

**本页完成后**：你应该能从固定 logits 复算 temperature、top-k、top-p 和一次 CDF 采样，并解释处理顺序为什么属于算法契约。

## 最小运行

~~~powershell
python projects/inference-serving/sampling_toy.py
~~~

运行前先预测：top-k 会保留几个候选，top-p 是否包含刚好跨过阈值的 token，固定 uniform 落在哪个 CDF 区间。

## 预期现象

- temperature 改变 logits 的相对尺度，但不直接指定“创造力”。
- top-k 先限制候选数量，top-p 再按累计概率保留 crossing token。
- 交换 processor 顺序或改变 tie-break，可能得到不同 support；这不是无关紧要的实现细节。

## 最低通过

1. 手算 top-k 后的归一化概率。
2. 指出 top-p crossing token，并复算最终概率和为 1。
3. 用固定 uniform 找到被采样 token 的 CDF 区间。
4. 故意交换 top-k/top-p 顺序或制造同分 token，解释输出为什么改变。

## 推荐扩展：观察真实模型

1. 准备 20 个问题，覆盖事实、抽取、创作、歧义和无法回答。
2. 固定 Prompt，分别用 greedy、temperature 0.7、top-p 0.9 各运行多次。
3. 记录答案差异、正确率、输出长度和 token 数。
4. 将一个关键条件从 Prompt 开头移到中间，比较结果。

真实模型实验可以使用本地模型或云 API，但必须记录模型 revision、采样参数、输入、原始输出和费用；没有这些信息时，结果不可复查。

## 常见失败

- 只看最终 token，不保存每一步候选和概率，导致无法定位 processor 顺序。
- 把 top-p 理解为“固定保留前 90% 的 token 数量”。它限制的是累计概率质量，不是候选数量比例。
- 每次同时改变 Prompt、temperature 和 top-p，导致无法归因。

## 交付与结论边界

最低交付物：一张手算表、一份脚本原始输出、一个故意失败的反例和不超过五行的结论。

固定 logits 的 CPU 结果用于核对当前采样契约是否与手算一致。目标 checkpoint、Transformers、vLLM 或云
provider 可能采用不同默认顺序；配置是否提高真实任务质量还要另做评测。

下一步：进入[实验 0B](lab-0b-generation-protocol.md)，学习 beam、约束、EOS 和流式 stop；只想完成入门路径时，也可以先回到[实验 1](../labs.md#lab-1)。
