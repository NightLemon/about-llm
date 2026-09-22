# 集合通信与数据中心网络：从张量结果到网络关键路径

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要排查多 GPU 训练、分布式推理或 MoE 通信瓶颈的工程师。
- **先修**：[分布式训练](distributed-training.md)、张量切分和字节单位。
- **首次阅读**：结果布局 → ring 账本 → ready time → rail 与割集 → MoE metadata。
- **完成信号**：能为一次通信写出输入、结果布局、载荷、就绪时刻、拓扑与测量清单。
- **卡住时**：先回到[全局 batch 与 loss](distributed-training.md#global-batch-loss-normalization)，只追踪两个 rank 的一个分片。

</div>

两台各有八张 GPU 的训练机器正在做 data parallel。每个 rank 已经算完自己的梯度，
某个 step 却比前一 step 慢得多。团队查看网卡峰值后仍解释不了问题：哪个张量经过了哪条链路，
以及最慢 rank 何时准备好，都还没有进入账本。

排查应从 collective（集合通信）的结果语义开始。先确认结束时每个 rank 持有完整结果还是一个分片，
再计算算法载荷、到齐时间和物理路径。本页公式描述固定算法与固定拓扑，不预测任意通信库或集群的性能。

## 先区分结果布局

设 \(n\) 个 rank 分别持有同形状张量 \(x_0,\ldots,x_{n-1}\)，归约算子为逐元素求和。
AllReduce 让每个 rank 得到完整结果：

\[
y=\sum_{i=0}^{n-1}x_i.
\]

ReduceScatter 也完成归约，但把 \(y\) 切成 \(n\) 个分片。rank \(j\) 只保留 \(y_j\)。
AllGather 不求和；它收集各 rank 已持有的不同分片并按约定布局拼接。

| 操作 | 每个 rank 的输入 | 每个 rank 的结果 | 是否归约 |
|---|---|---|---|
| AllReduce | 完整 \(x_i\) | 完整 \(y\) | 是 |
| ReduceScatter | 完整 \(x_i\) | 一个 \(y_j\) 分片 | 是 |
| AllGather | 一个不同分片 | 拼接后的完整张量 | 否 |
| All-to-All | 发往多个目的地的分段 | 来自多个来源的分段 | 否 |

结果布局兼容时，ReduceScatter 后接 AllGather 可以实现 AllReduce。这个分解说明数据去了哪里，
没有规定通信库必须使用哪种线路或算法。把 All-to-All 的返回顺序直接当成原 token 顺序，
则会遗漏 MoE 的身份恢复与 gate 合并。

## 用 ring 建立载荷和轮次直觉

考虑等分分段的 ring AllReduce。每个 rank 的输入大小为 \(M\) bytes，参与者数量为 \(n\)。
ReduceScatter 和 AllGather 各有 \(n-1\) 轮，每轮发送一个 \(M/n\) bytes 的分片。
因此每个 rank 的发送载荷与总轮次为：

\[
V_{\text{rank}}=2(n-1)\frac{M}{n},
\qquad
R_{\text{ring}}=2(n-1).
\]

这里的 \(M\) 是单个 rank 的完整输入，不是全组输入之和；\(V_{\text{rank}}\) 也是单个 rank 的发送量。
当 \(n=1\) 时，结果为零轮和零字节。只有一个参与者时不存在跨 rank 通信。

当 \(n=4\)、\(M=4\) MiB 时，每个分片为 1 MiB。每个阶段 3 轮，合计 6 轮；
每个 rank 共发送 \(2\times3\times1=6\) MiB。二项树和分层算法使用不同的轮次与载荷模型，
所以这组数字不能写成所有 AllReduce 的固定成本。

## 把载荷放到 ready-time 关键路径

字节数除以带宽只描述数据开始传输后的理想搬运时间。令 \(a_i\) 为 rank \(i\) 的当前 bucket
可以通信的时刻；若该 bucket 必须等待全组，则启动时刻为：

\[
t_{\text{start}}=\max_i a_i.
\]

在无争用、均匀链路的简化 ring 中，令每轮启动开销为 \(\alpha\)，单向有效带宽为 \(B\)，则：

\[
t_{\text{finish}}
=
\max_i a_i
+2(n-1)\left(\alpha+\frac{M/n}{B}\right).
\]

一个 rank 晚到两毫秒会把第一项整体后移。提高链路带宽只能缩短第二项。
梯度 bucket 可以让较早完成的层提前通信；bucket 太大推迟 ready time，太小又让更多 \(\alpha\) 占主导。
判断时要同时查看每个 bucket 的 ready time、collective trace 和逐 rank timeline。

## 分层 collective 先画域，再算跨域字节

多张 GPU 往往分属不同的节点内互联、PCIe switch、NUMA root 和 NIC 路径。
下面的固定账本只用来解释分层通信：两台服务器各有 8 个 rank，每个 rank 的输入为 192 MiB。
节点内先做 ReduceScatter 后，每个跨节点分片为：

\[
192/8=24\ \text{MiB}.
\]

若八对对应 rank 分别使用八条对齐的 rail 做两 rank AllReduce，每对双向合计发送：

\[
2\times24=48\ \text{MiB}.
\]

八对跨节点的双向总量为 \(8\times48=384\) MiB。它们可能并行经过八条 rail，
因此 384 MiB 不能当作一张 NIC 上的串行 payload。

这个账本依赖“两节点、每节点 8 rank、192 MiB 输入、节点内先切分、每卡一条对齐 rail”。
换成不同 NIC 数量、NUMA 映射或分组方式后，需要重新画路径和计数。

## 割集与超售限制并发流

所有离开某个叶交换机的流量都要经过它的上联。若每卡端口带宽为 50 GB/s，
超售比 \(r\) 定义为下联总带宽与上联总带宽之比，则均分后的每卡上联份额为 \(50/r\) GB/s。

令 \(f\) 为一张卡发出字节中必须跨叶的比例。跨叶需求为 \(50f\) GB/s，均匀账本要求：

\[
50f\le\frac{50}{r}
\quad\Longrightarrow\quad
f\le\frac{1}{r}.
\]

当 \(r=1\) 时，该模型允许 \(f\le1\)；当 \(r=3\) 时，均分上联只能容纳不超过三分之一的跨叶比例。
ECMP 哈希碰撞、incast、链路故障和流量不均仍可能让满足不等式的运行产生排队，
所以割集计算是容量门，而不是 tail-latency 保证。

TP、EP 和 DP 共享 NIC 或上联时，应分别记录它们跨越的域。TP 的逐层通信通常对启动延迟敏感，
EP 的 All-to-All 会产生多个目的地，DP 的梯度归约往往覆盖更多节点。单项带宽测试无法证明组合负载没有争用。

## MoE All-to-All 还要恢复身份

Expert Parallel（专家并行，EP）把专家放到不同 rank 后，All-to-All 只把分段送到目的地。
它不自动完成 capacity、gate 加权或原 token 顺序恢复。每个有效条目至少要关联：

| Metadata | 它回答的问题 |
|---|---|
| source rank | token 来自哪个发送 rank？ |
| source token 位置 | 它在原输入中的哪个位置？ |
| executed expert | 哪个专家实际处理了它？ |
| gate 或 assignment slot | 多专家结果怎样加权？ |
| validity / padding / capacity | 该条目有效、填充，还是被丢弃？ |

combine 阶段按 metadata 把结果放回原位置，再执行路由规则规定的聚合。
远端按 expert 分组返回时，tensor shape 可能完全正确，前向也没有报错，但按到达顺序写回会交换两个 token 的输出。

仓库现有 MoE 小实验能检查 Float64 数据流、split、逆向返回和梯度账本。
它们没有测 CUDA/NCCL、多节点 exactly-once、DDP/FSDP 收敛或目标 checkpoint 质量。

## 目标工作负载的测量清单

声称某个拓扑或 collective 更快前，至少保存：

- 模型 revision、parallel degrees、group membership、rank mapping 和通信库版本；
- GPU、驱动、节点内链路、NIC、NUMA、交换机路径、rail 和方向性端口速率；
- collective 类型、tensor dtype、payload、bucket、算法选择和通信组；
- 序列长度、micro/global batch、MoE top-k、capacity 和每专家 token 分布；
- 每个 rank 的 ready/start/end、overlap、idle 与 step-time 分位数；
- 方向性发送/接收字节、重传/错误、同机背景流量、warm-up 和重复次数；
- 原始 profiler/trace 路径，以及失败、超时、取消和未完成的完整分母。

对照实验一次只改变一个关键因素。例如固定模型、batch、dtype 和 rank mapping，只改变 rail 映射。
同时改变通信库、bucket、并行度和路由负载时，结果无法归因。

## 目前能得出什么

独立算术支持 collective 结果布局、等分 ring 的轮次与每 rank 载荷，以及固定两节点和固定超售账本。
这些结果没有回答通信库实际选择、目标拓扑有效带宽、TP/DP/EP 争用和端到端训练吞吐。
后一组问题需要目标 runtime、硬件、workload 和原始 trace。

## 自测

1. ReduceScatter 结束后每个 rank 持有什么？AllGather 又改变了什么？
2. 对 \(n=4\)、\(M=4\) MiB 的 ring，写出轮次与每 rank 发送量；为什么还要检查 \(n=1\)？
3. 一个 rank 晚于其他 rank 完成 bucket，它影响 ready-time 公式的哪一项？
4. 在 \(r=3\) 的账本中，为什么 \(f=1/2\) 会超过均分上联？
5. 为什么 MoE combine 不能相信 All-to-All 到达顺序？列出两项必要 metadata。
6. 设计一组只改变 rail 映射的对照实验，并写出必须保存的原始 trace。

## 延伸阅读

- NVIDIA，[NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)：核对 collective API 与运行时行为。
- Thakur、Rabenseifner、Gropp，*Optimization of Collective Communication Operations in MPICH*，2005，
  [DOI](https://doi.org/10.1177/1094342005051521)：比较 collective 成本模型。
- [分布式训练](distributed-training.md)：继续学习并行维度与 global loss；
  [分布式训练正确性](distributed-training-correctness.md)负责 checkpoint、恢复和验收。
- [MoE 系统](../frontier/moe-systems.md)：追踪 router、capacity、dispatch 与 combine。
