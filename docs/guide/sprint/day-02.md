# 第 2 天：采样——从 logits 到一个 token

**今日目标**：能说出 temperature、top-k、top-p、repetition penalty 各自改变了什么，
以及**它们的执行顺序为什么是一条契约而不是实现细节**。

**导航**：[上一天](day-01.md) · [返回速成总览](index.md) · [下一天](day-03.md)
{ .doc-nav }

## 时间盒

| 时段 | 内容 | 产出 |
|---|---|---|
| 上午 1.5h | [生成基础](../../core/generation-basics.md) | 一句话说清 logits 与 probability 的区别 |
| 上午 2h | [实验 0A](../../practice/labs/lab-0a-sampling.md) 前半：温度与截断 | 填好的预测表 |
| 下午 2.5h | 实验 0A 后半：penalty 与 processor 顺序 | 一个顺序颠倒后结果不同的具体例子 |
| 下午 2h | 复盘 + 日志 | 必答题答案 |

## 动手前先预测

打开实验 0A，**在运行任何命令之前**，先在纸上填完预测表。这是这条速成路线的第一条铁律：

> 先看到答案，实验的价值会损失八成。你要买的是"预测错"那一刻的记忆，不是结论本身。

重点预测这几件事：

1. temperature 趋近 0 时，分布形状怎么变？趋近无穷大呢？
2. top-k=1 与 temperature=0，产生的是不是同一个结果？
3. 先做 top-p 再做 temperature，和先做 temperature 再做 top-p，结果一样吗？

第 3 题是今天的核心。

## 读 `scope` 字段

运行完脚本后，**不要只看主输出**。找到 JSON 里的 `scope` 段落，逐条读那些 `false`：

```text
xxx_executed: false
xxx_proved: false
```

这些"我没有验证什么"的声明，是本仓库最有工程价值的部分。真实工作中，
90% 的事故来自把一次局部验证外推成全局承诺。今天开始，每个实验都要读 `scope`。

## 必答题

1. repetition penalty 作用在 logits 上还是 probability 上？为什么这个区别重要？
2. top-p=0.9 时，候选集大小是固定的吗？
3. 为什么 processor 的执行顺序必须写进文档，而不能"反正差不多"？

## 今日交付

```text
预测表（预测列 + 实际列 + 差异解释）
一个 processor 顺序颠倒导致输出不同的最小例子
从 scope 字段里抄下来的三条「本实验没有证明什么」
```

## 可以跳过

- 教学版 Byte BPE 实现（实验 1A）——你不需要自己写 BPE
- 任何 sampling 算法的复杂度证明

## 明天接什么

[第 3 天](day-03.md) 把单步采样扩展成完整的生成协议：beam search、约束解码和流式停止。
