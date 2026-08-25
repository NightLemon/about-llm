# MoE 系统：跟一批 token 走完路由、跨卡与反向传播

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想理解 MoE 训练、路由、通信和推理部署的算法与系统工程师。
- **先修**：[Transformer](../core/transformer.md)、softmax、MLP 与分布式集体通信基础。
- **首次阅读**：4 个词元的路由 → 容量筛选 → 跨设备发送 → 合并 → 反向传播。
- **完成信号**：能解释一个词元为什么被某个专家接收，又怎样从专家所在设备回到原位置。
- **卡住时**：先只看 t3，沿着它的两条专家路径画一遍去程和回程。

</div>

**专题导航**：[前沿总览](reasoning-long-context-moe.md) · [DeepSeek](../models/deepseek.md) · [分布式训练](../systems/distributed-training.md) · [证据台账](../evidence/frontier-controls.md)
{ .doc-nav }

稠密 Transformer 会让每个词元经过同一套前馈网络。混合专家（Mixture of Experts，MoE）准备多套
专家 MLP，再由路由器为每个词元选择其中少数几套。模型因此可以增加总参数，而不让每个词元执行所有参数。

这句话只解释了架构动机。真正运行一层 MoE，还要回答一连串系统问题：某个 expert 收到太多 token
怎么办？expert 不在当前设备上怎么办？输出乱序返回后怎样找回原 token？router 和远端 expert 又怎样获得
梯度？

本章不从术语清单开始。我们先让 4 个 token 走完一次可复算的 top-2 路由，再把幸存的计算任务放到
两张设备上。你会看到 MoE 的基本计算单位不是“4 个 token”，而是 token 与 expert 组成的
**assignment（分配任务）**。

## 先认识这批 token {#running-batch}

假设一层有 3 个路由 expert，记为 e0、e1、e2。Router 为 4 个 token 产生下面的 logits：

```text
t0  [4, 1, 0]
t1  [4, 2, 0]
t2  [4, 3, 0]
t3  [0, 4, 3]
```

每个 token 选择分数最高的两个 expert：

```text
t0 → e0, e1
t1 → e0, e1
t2 → e0, e1
t3 → e1, e2
```

这一步一共产生 \(4\times2=8\) 个 assignments。按 expert 计数，e0 收到 3 个，e1 收到 4 个，
e2 收到 1 个。Router 明显更偏爱前两个 expert。

你可以直接运行同一组输入：

```powershell
python projects/transformers-basics/moe_routing.py
```

脚本会打印 softmax 概率、top-2 结果、容量前后的计数、合并权重和线性 expert 输出。下面先手算其中最重要的
部分，再解释真实跨设备运行时还要补什么。

## 第一步：路由器把词元变成专家任务

对 token hidden state \(x\)，最简单的 router 先计算 logits，再转成概率：

\[
z=W_rx,\qquad p=\operatorname{softmax}(z).
\]

Top-k 选择的是 expert 下标，gate weight 则来自选中 expert 的概率。以 t3 为例，完整概率约为
`[0.0132, 0.7214, 0.2654]`，所以它选择 e1 和 e2。

如果暂时不考虑容量与共享 expert，一层输出可以写成：

\[
y(x)=\sum_{e\in S(x)}\tilde p_e(x)f_e(x).
\]

\(S(x)\) 是 top-k 选出的专家集合，\(f_e\) 是第 e 个专家网络，\(\tilde p_e\) 是最终合并权重。

某些架构还会让所有词元经过 shared expert（共享专家）。这份样例不包含共享路径；目标架构若有，需要把它
另外加到输出中。

Top-k 给出的下标是离散选择。被选中的 softmax 概率仍参与加权，因此主任务损失能沿合并权重回到路由器。
若代码误将该权重 `detach`，专家仍可能获得梯度，路由器会失去这条主任务梯度。

## 第二步：capacity 决定哪些任务真的执行

如果所有 token 都涌向同一个 expert，单个设备会出现大缓冲、长尾计算甚至内存不足。Capacity（容量）策略
给每个 expert 设置本轮最多接受多少 assignments。

本仓库样例使用：

\[
C=\left\lceil
\text{capacity factor}\times\frac{Nk}{E}
\right\rceil.
\]

这里 \(N=4\)、\(k=2\)、\(E=3\)，capacity factor 为 0.75，因此：

\[
C=\left\lceil0.75\times\frac{4\times2}{3}\right\rceil=2.
\]

每个 expert 最多接受两个 assignments。样例在同一 expert 内按 router probability 从高到低保留；概率相同
时，再按 token 序号和 top-k 位置决定。于是：

```text
e0：保留 t0、t1；丢弃 t2
e1：保留 t3、t2；丢弃 t1、t0
e2：保留 t3
```

把结果重新放回 token 视角：

```text
t0 → e0 保留，e1 丢弃
t1 → e0 保留，e1 丢弃
t2 → e0 丢弃，e1 保留
t3 → e1 保留，e2 保留
```

8 个 assignments 中有 5 个被接受、3 个被丢弃，但没有任何 token 的两条路径同时被丢掉。所以准确说法是
“丢了 3/8 个 assignments”，不是“丢了 3 个 token”。

这也解释了为什么下面五个状态必须分开记账：

```text
top-k 选中 selected
→ capacity 接受 accepted
→ 发往 owner dispatched
→ expert 完成 executed
→ 返回并合并 returned/combined
```

如果只记录 top-k histogram，就看不见 capacity drop；如果只记录 dispatch 数量，又可能漏掉通信或 expert
执行失败。

### Drop、reroute 与 dropless 是三种不同选择

上面的样例直接丢弃 overflow assignment。真实系统也可以把它改送给排名更低且仍有空位的 expert，或者采用
dropless 策略，让 hot expert 暂时超过名义容量。

- **Drop** 控制了 expert 工作量，但可能让某些 token 少算一条甚至所有路由路径。
- **Reroute** 尽量保留计算，却改变了实际执行的 expert，并需要完整候选排序与确定的去重规则。
- **Dropless** 不丢任务，但负载倾斜和尾延迟仍然存在。

容量也可能按序列、微批次（micro-batch）、设备或全局路由组计算。公式里的 \(N\) 究竟包含哪些词元，是
算法约定的一部分；两个实现即使使用相同的容量系数，也可能产生不同结果。

## 第三步：专家计算后按路由权重合并

容量筛选后，本仓库样例重新归一化幸存 gate，使每个仍有 assignment 的 token 权重和为 1：

```text
t0  [1, 0]
t1  [1, 0]
t2  [0, 1]
t3  [0.7311, 0.2689]
```

t0、t1 和 t2 都只剩一个 expert，所以幸存路径权重变为 1。t3 的两条路径都保留，e1 与 e2 的相对权重
仍由原概率决定。

样例让 e0 执行恒等线性变换、e1 执行 \(2I\)，e2 执行另一组固定线性权重。最终输出为：

```text
t0  [1.0000,  2.0000]
t1  [2.0000,  1.0000]
t2  [2.0000, -2.0000]
t3  [4.9242,  2.5379]
```

以 t3 为例，结果来自 e1 与 e2 输出的加权和，而不是任选其中一个。

Drop 后也可以保留原始权重。这样能留下被丢路径原本占有的权重质量，但幸存权重之和可能小于 1。
重新归一化与保留原始权重会产生不同输出，模型或运行时必须明确选择哪一种。

若一个词元的所有路由任务都被丢弃，这份样例的专家分支输出为零。上层的残差连接或共享专家仍可能让
整个模块输出非零。

## 第四步：把幸存任务发到专家所在设备 {#expert-parallel-path}

到目前为止，4 个词元都在一个进程内完成。多卡 expert parallel（专家并行）会把不同专家放在不同设备上。
为了看清数据流，假设：

- GPU 0 保存 e0、e2，并拥有源 token t0、t2；
- GPU 1 保存 e1，并拥有源 token t1、t3。

这是用来解释通信的放置方案，不是 `moe_routing.py` 实际启动的 GPU 任务。把前面 5 个幸存任务放进这张图
后，会得到：

```text
t0 → e0：GPU 0 本地执行
t1 → e0：GPU 1 发往 GPU 0
t2 → e1：GPU 0 发往 GPU 1
t3 → e1：GPU 1 本地执行
t3 → e2：GPU 1 发往 GPU 0
```

每个来源进程先按目标设备打包隐藏状态、合并权重与元数据，随后进行全互连通信（all-to-all）。
专家所属进程分组执行 MLP，再把结果与元数据送回来源进程。最后一步才是恢复原词元顺序并加权合并：

```text
来源进程完成路由与容量筛选
→ 按专家所在设备打包
→ all-to-all 去程
→ 专家所在设备执行 MLP
→ all-to-all 回程
→ 来源进程按元数据复位并合并
```

### 为什么返回顺序不能直接当作 token 顺序

通信通常按目的 rank 和 expert 重排。一个 source 发出的第二行，可能比第一行更早从另一 owner 返回。
因此 metadata 至少要能恢复：

- source rank 与 source token 位置；
- 实际执行的 expert；
- gate weight 或对应 assignment slot；
- padding、capacity 与有效性状态。

Tensor shape 完全正确，也可能把 e1 的输出加到错误 token。仅凭“前向没有报错”发现不了这种错序。

仓库另有一条真实启动两个 CPU/Gloo 进程的实验。两个进程各拥有一个专家，用可变行数的
`all_to_all_single` 交换行数、隐藏状态和元数据，再完成专家计算与结果回程。实验故意让 0 号进程的返回
顺序变成 `[1,0,2]`。如果忽略元数据、直接按到达顺序合并，与正确结果的最大绝对差约为 0.8959。

这条实验使用另一组 top-1 输入，不能与前面的 4-token top-2 数字拼成一次运行。它的作用是补上前面
NumPy 样例没有执行的真实进程间通信。精确命令与结果见[证据台账](../evidence/frontier-controls.md#moe-all-to-all-control)。

### 全局容量统计还不是专家分发

用 `all_gather` 汇总所有进程的路由分数，再用 `all_reduce` 对账全局选中数量，可以让各进程基于同一批
输入做容量竞争。此时词元仍可能只是被复制到所有进程，并没有发往真正的专家所属设备。

判断一段代码是否完成专家并行，要看它是否真的把词元发给所属设备、只在那里执行专家计算，并把结果
送回来源位置。仅仅调用一次分布式 collective 还不够。

## 第五步：反向传播沿原路返回 {#backward-path}

训练时，专家所属进程需要自己的参数梯度，来源进程上的路由器需要合并权重对主任务的梯度。
若全互连去程与回程要留在计算图内，反向传播还必须按相反的分段方式交换梯度。

可以沿 t3 的两条路径理解：

1. Source 把 t3 发给 e1 与 e2，并保存路由、gate 和 source metadata。
2. 两个 owner 计算 expert 输出，source 将它们按 gate 合并。
3. Loss 对合并结果求导，梯度拆回 e1 与 e2 的输出和 gate。
4. 反向全互连把专家输出的梯度送回所属进程，把隐藏状态与合并权重的梯度送回来源进程。
5. e1、e2 在各自 owner 上累积参数梯度；复制在多个 source rank 上的 router gradient 再做 SUM 归约。

不同进程接受的词元数往往不同。若每个进程先算自己的平均损失，再对这些平均数做简单平均，词元少的进程
会获得过大权重。应先明确全局分子和分母，例如按全部有效词元数归一化，再与单进程处理同一批输入的结果
对账。

仓库的 CPU/Gloo 训练实验先验证了无容量限制时的反向通信。另一个实验把全局容量淘汰、只发送幸存任务、
反向通信和一步参数更新放进同一计算图。

后一个样例还包含某个来源进程发送零行的情况。即使它本轮没有词元可发，也必须按相同顺序参加集体通信，
否则其他进程会一直等待。

这些实验说明的是小规模 Float64 数据流和梯度账本能够对齐；它们没有测 CUDA/NCCL 性能，也没有使用
DDP、FSDP 或目标模型 checkpoint。

## 路由失衡时应该看什么

回到最初的计数 `(3,4,1)`：e1 比 e2 忙得多。训练常加入 load-balancing loss、router z-loss、噪声或
动态偏置，让路由不要过早坍缩到少数 experts。

但一个辅助 loss 下降，不等于系统已经均衡。至少一起观察：

- 每个 expert 被选中、被容量接受和真正执行的 assignments；
- overflow、drop、reroute 与整 token 全丢比例；
- expert 负载的 max/mean、变异系数和尾部计算时间；
- gate entropy、辅助 loss 与主任务质量；
- all-to-all 的数据量、耗时和消息大小。

本仓库用下面的可复算诊断描述容量前分配：

\[
f_e=\frac{n_e}{Nk},\qquad
p_e=\frac1N\sum_i\operatorname{softmax}(r_i)_e,
\qquad
L_{bal}^{ref}=E\sum_e f_ep_e.
\]

这里 \(f_e\) 是专家 e 的任务比例，\(p_e\) 是平均路由概率。它只是一种教学诊断。不同论文和框架可能
使用 top-1 计数、不同分组、停止梯度或缩放方式。

即使选中数量完全均匀，也不代表专家已经形成了有用分工。路由统计只能说明相关性。

若要声称某个专家“学会了代码或中文”，还要检查不同随机种子和训练检查点下是否稳定，并通过屏蔽或替换
专家观察因果影响。词元频率、位置和长度等混杂因素也需要控制。

## 三本参数账解释 MoE 为什么不等于免费加速

谈 MoE 模型大小时，先分开三种参数口径：

- **Total parameters（总参数）**：所有 experts、router、attention、embedding 等唯一权重。它影响权重文件、
  集群总存储和总体分片需求。
- **Active parameters（激活参数）**：一个 token 实际经过的 routed/shared experts 与公共模块。它描述条件计算，
  但不等于真实 FLOPs 或延迟。
- **Resident parameters（驻留参数）**：当前 device/rank 实际持有的权重。Expert parallel 可以只放部分 experts，
  attention、router 或 shared weights 仍可能复制。

所以“总参数很大、每 token 激活很少”不能推出单卡能加载，也不能用 active/total 比例直接估算显存。
实际速度还受 packing/unpacking、all-to-all、小矩阵利用率、负载倾斜和最慢 expert 影响。

## 模型、运行时与底层库分别负责什么 {#runtime-stack}

一条 MoE 路径通常跨过下面几层：

| 层次 | 在 MoE 中负责什么 |
|---|---|
| 模型架构与权重 | 规定专家数、路由形式、top-k、共享专家以及训练得到的参数 |
| 模型实现 | 把路由、专家 MLP、残差与辅助损失写成张量计算 |
| 训练或推理系统 | 管理批次、容量、跨设备分发、并行组、故障和指标 |
| PyTorch/JAX 等框架 | 提供自动微分、张量、模块与分布式 API |
| NCCL/Gloo 等通信后端 | 执行设备或进程间的集体通信；它不负责决定该发送哪些词元 |
| CUDA 与计算内核 | 执行路由、排序、打包、分组矩阵乘法与结果合并 |

配置文件中出现 `num_experts_per_tok` 或 `top_k`，只能说明发布者声明了某些字段。Tie-break、分组选择、
capacity、drop/reroute、gate normalization 与训练梯度，仍要核对同版本模型代码和 runtime。

通信后端支持 all-to-all，只表示它提供了全互连搬运能力。生产实现还要降低打包开销，并把进入同一专家的
词元组织成分组矩阵乘法（grouped GEMM）。数值类型、GPU 和网络不同，适合的计算内核也会变化。

## 到推理服务时，prefill 与 decode 要分开看

推理不再计算 backward，但总权重加载、路由、跨卡通信和负载倾斜仍然存在。

预填充（prefill）一次处理较多提示词词元，同一专家往往能凑出更大的批量。逐词元解码（decode）时，
每个活跃序列每轮通常只新增一个词元。单个专家收到的批量更小，通信与计算内核的启动开销更容易占主导。

因此服务评测至少分开记录：

- prefill 与 decode 的 expert batch size、通信时间和吞吐；
- 每个 expert 的负载与尾延迟；
- 总权重、KV Cache、通信 buffer 和 workspace 显存；
- 请求并发变化时的质量、tokens/s 与端到端延迟。

模型层的专家容量与服务层的接入容量位于不同层级。前者决定一层内部怎样处理路由任务，后者决定请求能否
进入队列。即使模型采用不丢任务的路由，服务仍可能因为显存、截止时间或并发上限而拒绝请求。

## 用仓库里的实验逐级验证

不要从纸笔路由直接跳到“生产 MoE 已验证”。本仓库把证据拆成递进的几层：

1. **NumPy 路由与容量**：复算本页 4-token top-2 案例，以及线性 expert 的 dispatch/combine。
2. **单进程 PyTorch 训练**：对账稀疏计算与稠密参考计算的前向、梯度、容量和溢出策略。
3. **两进程全局容量**：用真实 collective 建立跨 rank 的 capacity competition，但不做 owner dispatch。
4. **两进程 expert dispatch**：让 token 真正往返 owner，并用 metadata 恢复顺序。
5. **两进程训练**：验证 reverse all-to-all、router gradient 归约、owner expert 更新与一步 optimizer。

常用入口：

```powershell
python projects/transformers-basics/moe_routing.py
python projects/transformers-basics/moe_training_control.py
python -m pytest tests/test_moe_routing.py tests/test_moe_training.py -q
```

Gloo/all-to-all 实验的完整命令、固定输入和结果在[证据台账](../evidence/frontier-controls.md)。这些 CPU 样例
适合检查当前实现的数据流、公式和梯度是否对齐。

真实模型性能属于下一层证据。它需要加载目标 checkpoint，并在 GPU 上运行分组矩阵乘法、NCCL 通信和目标
工作负载，再分别测量吞吐、显存、收敛或模型质量。

## 面试时怎样回答

面对“解释 MoE”时，沿一个 assignment 的生命周期回答：

1. 路由器为词元打分，top-k 产生词元—专家任务。
2. 容量策略决定每个任务是接受、重路由还是继续超额执行。
3. 运行时按专家所属设备打包隐藏状态、合并权重与来源元数据。
4. All-to-all 把任务送到专家所在设备，执行 MLP 后再把输出送回来源进程。
5. 来源进程按元数据恢复词元顺序，并按路由权重合并。
6. 训练时梯度沿通信反向返回；各设备更新本地专家，复制的路由器梯度还要正确归约。
7. 评测同时看总/激活/驻留参数、质量、负载、通信、显存和尾延迟。

如果继续追问，你应该能解释：为什么 selected 与 accepted 的数量不同；为什么 `all_gather` 全局计数不等于
token-to-owner dispatch；以及为什么 active parameters 较少仍不保证低延迟。

## 自测

1. 本页 t2 为什么没有执行 e0？它最终的 combine weight 是多少？
2. 8 个 assignments 丢 3 个，为什么不能说“丢了 3 个 token”？
3. t3 的 e2 输出跨设备返回时，最少需要哪些 metadata 才能放回正确位置？
4. Top-k 下标不可导时，router 为什么仍可能从主任务获得梯度？
5. 为什么全局 capacity collective 与真正的 expert parallel 是两份不同证据？
6. Decode 阶段为什么比 prefill 更容易形成很小的 expert batch？
