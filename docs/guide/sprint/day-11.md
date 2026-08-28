# 第 11 天：推理基础——prefill、decode 与 KV Cache

**今日目标**：理解 LLM 推理为什么分成两个性质完全不同的阶段，
以及 KV Cache 为什么是显存的主要消耗者。

**导航**：[上一天](day-10.md) · [返回速成总览](index.md) · [下一天](day-12.md)
{ .doc-nav }

## 时间盒

| 时段 | 内容 | 产出 |
|---|---|---|
| 上午 3h | [推理基础](../../systems/inference.md) | prefill / decode 对比表 |
| 下午 2h | KV Cache 显存估算 | 一个具体模型的显存账 |
| 下午 3h | 复盘 + 与第一周注意力知识对接 | 必答题答案 |

## 今天要建立的核心对比

| 维度 | Prefill | Decode |
|---|---|---|
| 处理的 token 数 | 整个 prompt | 每步一个 |
| 瓶颈 | 算力（compute-bound） | 显存带宽（memory-bound） |
| 并行度 | 高 | 低 |
| 对应的用户指标 | TTFT（首 token 延迟） | TPOT（每 token 延迟） |

这张表是第三周所有内容的地基。**几乎每一个推理优化技术，都是在改善其中一格。**
后面看到任何优化手段，先问：它优化的是 prefill 还是 decode？

## 自己算一次显存账

挑一个具体模型（比如 7B、32 层、GQA 8 个 KV head、head_dim 128、FP16），
算出：

```text
每 token 每层的 KV 大小 = 2 (K和V) × kv_heads × head_dim × 2 bytes
每 token 总大小 = 上面 × 层数
2048 token 的一条序列 = ?
并发 32 条 = ?
```

算完之后对照模型权重本身的大小。你会发现：**在高并发场景下，KV Cache 可以超过权重。**
这个数字直接解释了为什么第 13 天要学 paged KV。

顺便回答：为什么 GQA 能大幅降低这个数字？（提示：看公式里的 `kv_heads`。）

## 必答题

1. 为什么 decode 阶段是 memory-bound 而不是 compute-bound？
2. batch size 增大，对 TTFT 和 TPOT 的影响方向一样吗？
3. KV Cache 能不能不缓存、每步重算？代价是什么？

## 今日交付

```text
prefill / decode 对比表（自己填一遍）
一个具体模型在具体并发下的 KV Cache 显存数字
GQA 对显存的影响倍数
```

## 可以跳过

- 各家推理框架的功能横向对比——第 4 周会直接上手一个
- 训练侧的显存分析

## 明天接什么

[第 12 天](day-12.md) 把今天的两个阶段放进一条完整的请求生命周期。
