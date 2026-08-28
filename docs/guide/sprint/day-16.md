# 第 16 天：上 GPU——环境、真实 checkpoint 与 CUDA 路径

**今日目标**：把前三周积累的 CPU 结论放到真实 GPU 上验证，
并把第 15 天那张"未验证"支持卡的空缺项一格一格填上。

**导航**：[上一天](day-15.md) · [返回速成总览](index.md) · [下一天](day-17.md)
{ .doc-nav }

## 时间盒

| 时段 | 内容 | 产出 |
|---|---|---|
| 上午 2h | 环境搭建与版本记录 | 环境 identity 快照 |
| 上午 2h | [实验 2D](../../practice/labs/lab-2d-operator-stack.md) 第四步（CUDA） | 三组 JSON |
| 下午 2h | 填完支持卡 | 完整支持卡 |
| 下午 2h | 拉真实模型权重，跑通一次前向 | 一次成功的 GPU 推理 |

## 先记环境，再跑实验

这是一个纪律问题。任何 GPU 结果，脱离环境记录都无法复现：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())"
```

把输出粘进日志。第 20 天写报告时，这段是必需的。

## 实验 2D 第四步：三次受控变化

按实验给的顺序跑三次，每次**只改一个变量**：

1. `--device cuda --dtype float32`
2. 只改 dtype → `float16`
3. 只改 shape → `--batch-size 4 --sequence-length 128 --hidden-size 1024`

实验文档明确说了：这些命令**用于检查路径和契约，不用于比较速度**——
每种配置只运行一次，没有独立 warm-up、重复采样或计时同步协议。

所以不要从这三次运行里读出任何性能结论。这个自律是今天的重点：
**手上有数字，不代表这个数字能支持你想说的话。**

如果某个 dtype 或 shape 失败了，**保留错误**。它正是当前版本支持面的证据。

## 3070 的显存现实

3070 是 8GB。这意味着很多默认配置跑不动，你会真实地撞到容量墙。
这是好事——第 13 天算的那笔 KV Cache 显存账，今天可以对着 `nvidia-smi` 验证。

提前想好降级路径：更小的模型、量化、更短的上下文、更小的并发。
把可行配置记下来，第 17、18 天要用。

## 必答题

1. FP16 和 BF16 在 3070（Ampere）上都支持吗？它们的取舍是什么？
2. 为什么"跑了一次没报错"不能写成"支持这个配置"？
3. 8GB 显存下，权重和 KV Cache 各占多少？哪个先成为瓶颈？

## 今日交付

```text
完整环境 identity 快照
实验 2D 三次运行的 JSON（含任何失败的错误）
填完的支持卡（CPU 部分 + CUDA 部分 + 仍然未验证的部分）
3070 上的可行配置清单
```

## 明天接什么

[第 17 天](day-17.md) 把一个真实模型送进一个真实推理引擎。
