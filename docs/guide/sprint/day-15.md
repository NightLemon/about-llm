# 第 15 天：算子栈、服务形态与第三周收口

**今日目标**：往下看清"一行模型代码怎样变成设备上的执行"，
往上看清"一个服务要对外承诺什么"。今天全部内容仍然只需 CPU。

**导航**：[上一天](day-14.md) · [返回速成总览](index.md) · [下一天](day-16.md)
{ .doc-nav }

## 时间盒

| 时段 | 内容 | 产出 |
|---|---|---|
| 上午 3h | [实验 2D](../../practice/labs/lab-2d-operator-stack.md) 第一到第三步（CPU） | FX 图与 export 图对比 |
| 下午 2h | [推理服务](../../systems/serving.md) | 服务契约清单 |
| 下午 2h | [硬件与边缘](../../systems/hardware-edge.md) | 硬件账本 |
| 下午 1h | 第三周收口 | GPU 任务清单 |

## 实验 2D：只做前三步

第四步需要 CUDA，留到[第 16 天](day-16.md)。今天做完前三步就好。

核心观察是一个 RMSNorm Module 展开成 **6 个**节点：

```text
FX:           operator.mul → mean → operator.add → torch.rsqrt → operator.mul → operator.mul
torch.export: aten.mul.Tensor → aten.mean.dim → aten.add.Tensor → aten.rsqrt.default
              → aten.mul.Tensor → aten.mul.Tensor
```

但要注意报告里的 `scope.kernel_count_inferred_from_fx_or_export` 是 `false`——
**节点数不等于 kernel 数**。编译器会融合，库实现会在一个事件内部启动多个 kernel。

这一层区分（模型表达 / 框架算子 / 运行事件 / 设备 kernel）是很多性能讨论跑偏的根源。
有人说"这个算子很慢"，你要先问：他说的是哪一层？

## 支持卡：把"我没验证"写下来

实验 2D 第五步要求填一张支持卡。今天你只有 CPU 数据，
所以 Dtype、Device、Performance 几行都要写"未验证"。

**这不是缺点，是今天要练的能力。**「3070 支持 RMSNorm」是一句没有信息量的话；
「在 torch 2.x + CUDA 12.x、FP32/FP16、这三组 shape 上前向对账通过，
未验证 BF16、动态 shape、torch.compile 和性能」才是一句工程语言。

## 服务契约清单

从 serving 文档里提炼出：一个推理服务对外要承诺什么？至少包括：

```text
支持的模型与版本
上下文长度上限与超限行为
并发上限与超限行为（拒绝还是排队？排多久？）
延迟目标（P50/P95/P99，分 TTFT 和 TPOT）
流式协议与断连语义
计费口径（断连和取消怎么算）
```

最后一条回到第 12 天的那两个问号。

## 必答题

1. FX 图和 export 图的区别是什么？哪个更接近设备执行？
2. 为什么"节点数不等于 kernel 数"？
3. 边缘部署相对云端，最主要的约束是什么？

## 今日交付

```text
FX 与 export 两张图的节点对照
一张只填了 CPU 部分、其余明确标注「未验证」的支持卡
服务契约清单
第 4 周 GPU 任务清单（按优先级排序）
```

## 明天接什么

第四周上 3070。[第 16 天](day-16.md) 先把环境和 CUDA 路径打通。
