# 第 5 天：注意力与第一周收口

**今日目标**：把注意力从公式变成三份可对账的实现，理解 causal mask、GQA 和 online softmax，
然后把第一周的五天连成一条完整链路。

**导航**：[上一天](day-04.md) · [返回速成总览](index.md) · [下一天](day-06.md)
{ .doc-nav }

## 时间盒

| 时段 | 内容 | 产出 |
|---|---|---|
| 上午 2.5h | [Transformer 结构](../../core/transformer.md) | 一张手画的 decoder layer 数据流图 |
| 上午 1.5h | notebook `01_attention_three_ways.ipynb` | 三份实现的数值一致性结果 |
| 下午 2h | notebook `02_minigpt_forward.ipynb` | 未训练模型 loss 约等于 `ln(vocab_size)` 的验证 |
| 下午 2h | 第一周整体复盘 | 一张全链路时序图 |

## 今天最重要的一个断言

notebook 01 里有一个值得反复琢磨的设计。判断 causal mask 是否正确，
**不能**断言"未来位置的概率质量大于 0"——softmax 输出恒为正，
这个条件即使 mask 完全正确也成立，抓不到任何 bug。

真正有效的对照是两条一起看：

```python
masked_future_mass = float(numpy_probabilities[0][np.triu_indices(4, k=1)].sum())
assert masked_future_mass == 0.0   # 有 mask：严格为 0
assert future_mass > 0.1           # 无 mask：明显不为 0
```

这就是"一个能失败的断言"和"一个永远通过的断言"的区别。
把这条思路记住——它在后面每一周都会再出现。

## notebook 02 的 sanity check

未训练模型的初始 loss 应该接近 `ln(vocab_size)`，因为模型此时在均匀猜测。
如果你的初始 loss 明显偏离这个值，说明初始化、mask 或 loss 计算有问题。
这是一个成本极低、命中率极高的调试锚点。

## 第一周收口：画一张时序图

用一页纸画出：一个用户问题从进入系统到返回答案，经过了哪些阶段。
至少标出：分词 → 前向 → logits → processor 链 → 采样 → 停止判定 → 解码 → 流式发出。

在每个阶段旁边写一句：**这一步可能怎样失败**。这张图是你第二周的地基。

## 必答题

1. GQA 相对 MHA 省了什么？省的是显存还是算力？
2. online softmax 为什么能在不实体化完整注意力矩阵的情况下算出正确结果？
3. causal mask 是加在 logits 上还是 probability 上？为什么？

## 今日交付

```text
三份注意力实现的最大绝对误差
未训练 minigpt 的初始 loss 与 ln(vocab_size) 的对比
一张标注了失败模式的全链路时序图
```

## 可以跳过

- [`core/scaling.md`](../../core/scaling.md)：知道结论即可，不用推导
- [`core/architectures-interpretability.md`](../../core/architectures-interpretability.md)
- `models/` 下的模型家族横向对比

## 明天接什么

第二周开始进应用工程。[第 6 天](day-06.md) 追踪一次完整的 RAG 请求。
