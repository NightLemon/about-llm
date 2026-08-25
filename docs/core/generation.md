# 生成与解码

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已完成[生成入门](generation-basics.md)，需要完整协议与工程边界的开发者。
- **先修**：条件概率、logits、softmax 与 [Transformer](transformer.md) 前向。
- **首次阅读**：生成循环 → temperature/top-k/top-p → 停止 → 约束解码。
- **完成信号**：能复算一步采样，并为每个输出说明停止原因。
- **卡住时**：先完成[实验 0A](../practice/labs/lab-0a-sampling.md)。

</div>

本页是完整参考章节。第一次学习先读[生成与解码入门](generation-basics.md)并完成实验 0A；需要 beam、约束、
流式协议或生产验收时，再回来查对应小节。

先看一次具体请求。客服系统要求模型返回：

```json
{"action":"refund","order_id":"1001"}
```

模型不会一次吐出整段 JSON。它每次前向只为词表中的每个 token 给出一个分数，也就是 logits。
推理框架还要决定哪些候选目前合法、怎样调整分数、选择哪一个，以及何时结束。

因此，**decoding（解码）是把条件分布变成一连串受协议约束的决定**。Temperature 只是其中一个环节。
先看这条 JSON 在系统里经历的完整路径：

| 阶段 | 退款请求中发生什么 | 需要更新的状态 |
|---|---|---|
| 模型前向 | 根据已有前缀给出下一 token 的 logits | Token 序列、每层 KV cache |
| 语法约束 | 只保留仍能组成目标 JSON 的 token | JSON parser 或有限状态机 |
| 分数处理 | 应用 penalty、temperature、top-k/top-p | 处理后的候选分布 |
| 选择 | Greedy 取最大值，或按概率采样 | 随机数状态与新 token |
| 文本输出 | 增量解码 token，并暂存可能构成 stop 的后缀 | UTF-8 decoder、stop matcher |
| 完成 | 记录 EOS、stop、length、cancel 或 error | Finish reason、usage、最终 JSON |

本章会沿这六步向下展开。最后得到合法 JSON 仍只是“解码完成”；订单是否存在、退款是否获批，
还要由应用层验证。

## 1. 生成循环的最小契约

给定前缀 \(x_{1:t}\)，模型最后一层输出词表 logits \(z\in\mathbb R^V\)。生成下一个 token 时，
一次典型的处理顺序是：

1. 根据当前 JSON 状态屏蔽不合法的 token；
2. 应用 repetition、presence、frequency 等分数调整；
3. 按约定顺序应用 temperature 与 top-k/top-p；
4. 重新归一化为概率并选择下一 token；
5. 更新序列、KV cache、语法状态、停止匹配和流式输出；
6. 直到 EOS、stop 条件、token 预算、上下文预算、取消或错误终止。

这份顺序用于建立心智模型，不是所有框架的共同标准。框架可能交换前 3 步，或把某些操作合并。
即使都写着 `temperature=0.8, top_p=0.9`，最终候选也可能不同。复现实验时应记录实现、版本和
完整的 logits processor 顺序。

### 1.1 Temperature

当 \(T>0\) 时：

\[
p_i(T)=
\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.
\]

- \(T<1\)：放大 logit 差异，分布更尖锐；
- \(T>1\)：缩小差异，分布更平坦；
- \(T\to 0^+\)：概率质量趋向最大 logit，但数值上不能把 \(T=0\) 代入除法。

API 中的 `temperature=0` 通常表示 greedy，或是服务商定义的近似 greedy 特殊路径，具体以契约为准。
当 \(T>0\) 时，temperature 不改变 logit 排名，却会改变相对概率。因此，后续 top-p 看到的候选集合也会变化。

实现 softmax 时应先减最大值，避免指数溢出；成熟框架通常已处理这一点。

## 2. 确定性选择与随机采样

### 2.1 Greedy decoding

每步选择

\[
x_{t+1}=\arg\max_i z_i.
\]

Greedy 快、无需随机数，适合作为退款 JSON 的第一条可复现基线。不过，“每一步都选概率最大”不保证
整段序列的联合概率最大，也不保证业务质量最好。早期一个局部选择会改变后续所有条件分布。

### 2.2 Categorical sampling

Categorical sampling 按 \(p_i\) 抽取一个 token。即使分布不变，不同的随机数也会产生不同序列。
可控实验应使用显式随机数生成器和 seed，并区分：

- **相同 seed、相同 kernel/调用顺序**能否重放；
- **不同 seed 的质量分布**是否稳定；
- **线上服务**是否真的暴露并遵守 seed 契约。

仅报告一个“幸运 seed”会高估采样策略。

## 3. 截断采样

完整词表的低概率尾部可能包含大量不合适 token。截断采样先缩小候选集合，再在集合内重新归一化。
它调整的是随机选择范围，并不能代替退款 JSON 的语法约束。

### 3.1 Top-k

Top-k 保留 logit 最大的 \(k\) 个 token，其余设为 \(-\infty\)。如果第 \(k\) 位出现并列，
实现还要规定谁先被保留。

有些阈值实现会留下全部同分 token，使候选数超过 \(k\)。本仓库的 NumPy 实现按 token ID 升序处理并列，
因此总是恰好保留 \(k\) 个。

- 优点：在 exact-k 约定下候选数量有上界，直观；
- 局限：无论分布很尖还是很平都使用同一个 \(k\)。

`k=1` 在无其他 processor 且没有并列处理差异时退化为 greedy。若 \(k>V\)，实现通常截到 \(V\)，但也可能报错。

### 3.2 Top-p / nucleus sampling

先按概率从高到低排序为 \(p_{(1)},p_{(2)},\ldots\)，选择满足

\[
\sum_{i=1}^{m}p_{(i)}\ge p
\]

的最小前缀，并在其中采样。关键实现边界是：**必须保留第一个让累计概率达到或超过阈值的 token**，因此候选集合不会为空。

当分布尖锐时候选很少，分布平坦时候选变多。`top_p=1` 通常等于不做 nucleus 截断；参数范围应是 \(0<p\le1\)。

### 3.3 Min-p 与 typical sampling

- **min-p** 常按“相对于当前最大概率的阈值”删除过小 token；不同库对参数和最小保留数定义可能不同。
- **typical sampling** 偏好信息量接近当前分布熵的 token，而不只取最高概率前缀。

它们不是统一标准。跨框架实验前应检查源码或官方契约，不能仅按参数名假设等价。

### 3.4 多种过滤器的组合

同时设置 top-k 与 top-p 时，一种常见顺序是先取 top-k，再在剩余分布上计算 top-p。
温度缩放放在 top-p 前后，也会改变 nucleus 集合。

服务实现可能选择另一种顺序。所以实验记录不能只写“使用 top-p”，还要保存所有非默认参数和应用顺序。

仓库的 `sample_next_token` 给出一条可以手算的单步流水线：先做考虑正负号的重复惩罚，再依次应用
temperature、精确 top-k 和 top-p，最后重新归一化并用逆 CDF 抽样。同分时，token ID 较小者优先。运行：

~~~powershell
python projects/inference-serving/sampling_toy.py
~~~

Toy 的原始概率是 `[0.4,0.3,0.2,0.1]`。Top-k=3 后重新归一化为 `[4/9,3/9,2/9,0]`；
再做 top-p=0.7，只保留 token 0 和 1，最终得到 `[4/7,3/7,0,0]`。固定 uniform=0.6 时选中 token 1。

这里 top-p 看到的是 top-k **之后**的概率。交换 processor 顺序会得到另一份合法但不同的生成契约。

直接固定一个 \([0,1)\) 均匀随机数，比只给 seed 更容易逐项验算。真实推理框架还会受到随机数算法、
CDF 遍历、浮点归约和并列处理的影响。因此，相同 seed 不足以保证不同实现逐 token 重放。

仓库 `MiniGPT.generate` 明确定义：temperature → top-k → top-p → softmax → multinomial，并测试了极小 top-p 只保留 argmax 的边界：

```python
sample = model.generate(
    prompt,
    max_new_tokens=32,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    generator=torch.Generator().manual_seed(7),
)
```

这是教学实现，不包含 EOS、批内独立停止、KV Cache 或流式 UTF-8 解码。

`generation_runtime_control.py` 会真实调用 Transformers 的 `generate()`。它分别设置模型配置和本次调用的
EOS、`max_new_tokens` 与停止条件，用来观察哪一层设置最终生效。

脚本通过本仓库提供的 logits processor 生成固定 token 轨迹，因此能检查 Transformers 的调用路径。
它没有评价 checkpoint 质量，也不能代表 vLLM 或云服务的行为。完整轨迹见
[Transformers 控制台账](../evidence/transformers-controls.md)。

## 4. Logits processor 与惩罚

### 4.1 Repetition penalty

“重复惩罚”不是单一公式。仓库采用一种常见的、考虑 logit 正负号的约定。对每个已经出现过的唯一 token：

\[
z_i'=
\begin{cases}
r z_i, & z_i<0,\\
z_i/r, & z_i\ge 0,
\end{cases}
\qquad r>0.
\]

同一 token 即使出现多次，也只处理一次。

其他实现可能直接减分、按出现次数处理，或只统计生成部分。惩罚器不知道重复是否合理。
例如退款 JSON 中的引号和字段片段可能本来就要重复，过强惩罚反而会破坏结构。

### 4.2 Presence 与 frequency penalty

一种常见形式是

\[
z_i' = z_i
-\lambda_{presence}\mathbf 1[c_i>0]
-\lambda_{frequency}c_i,
\]

其中 \(c_i\) 是 token \(i\) 已经出现的次数。存在惩罚（presence penalty）只关心“出现过没有”，
频率惩罚（frequency penalty）还关心“出现了几次”。

具体 API 是否统计 Prompt、按 token 还是文本片段计数，以及参数符号和范围，都可能不同。

### 4.3 Token bias 与禁用集合

Logit bias 可以提高或降低指定 token 的分数，也常用 \(-\infty\) 将它屏蔽。
可见字符串和 token ID 并非一一对应：字符串可能被拆成多个 token，token 也可能包含前导空格或一段 UTF-8 字节。

因此，应使用真实 tokenizer 检查映射。禁止少量 token 也不是内容安全方案，改写、拆分和其他语言都可能绕过。

## 5. Beam search 与序列级分数

采样每一步只沿一条路径继续，beam search 则同时保留累计分数最高的 \(B\) 条部分序列。序列对数概率为

\[
\log p(x_{1:T})=
\sum_{t=1}^{T}\log p(x_t\mid x_{<t}).
\]

每一项通常不大于 0，所以序列越长，累计值往往越负。直接比较未归一化分数，会系统性偏向较短的
已完成序列。Length penalty、coverage penalty 和任务约束可以改变这种偏好，但也同时改变了搜索目标。

一个显式但并非通用标准的最终分数可写为

\[
s(x_{1:T})=\frac{\log p(x_{1:T})}{T^\alpha},\qquad \alpha\ge 0.
\]

这里必须定义 \(T\) 是否包含 Prompt、EOS 和其他特殊 token。本仓库只数生成 token，其中包括已经发出的 EOS，
不数 Prompt。

由于 log probability 为负，除以更大的 \(T^\alpha\) 会让分数更接近 0。因此，这个公式并不是字面上的
“惩罚长文本”。其他推理框架可能使用不同归一化、完成候选上限和提前停止规则。

Beam search 不是全局穷举。考虑一个两步反例：第一步 `A=0.6, B=0.4`；第二步
`A→EOS=0.51, B→EOS=1`。

宽度为 1 时，第一步已经剪掉 B，最终只得到概率 \(0.6\times0.51=0.306\) 的 `A,EOS`。
宽度为 2 才能保留概率 0.4 的 `B,EOS`。加宽 beam 修复了这个例子，却不能保证任意有限宽度都找到全局最优序列。

一条候选发出 EOS 后已经完成，不应再送入模型。未完成候选用尽 `max_new_tokens` 时，则以 length 结束。
能否提前停止还取决于未完成候选的分数上界、长度归一化和框架契约，不能只看当前累计分数。

Beam 常用于翻译、语音识别等输出空间较明确的任务。对退款 JSON，语法约束通常比盲目加宽 beam 更直接；
开放对话使用大 beam，还可能产生更通用、更重复的文本。

运行这个可以手算结果的确定性示例：

~~~powershell
python projects/inference-serving/beam_search_toy.py
~~~

这个小实验保存每一步仍在搜索的 beam、候选扩展、已完成序列和最终排序，也演示 length penalty
怎样改变长短序列的排名。它只检查搜索规则，没有执行模型、tokenizer、KV cache 或 GPU 算子。

复现实验时至少记录：

- Beam width 和每个输入返回的序列数；
- 原始累计 log probability；
- Length penalty 公式，以及长度是否包含 EOS；
- 提前停止规则与最多保留多少条完成候选；
- 分数并列时怎样排序；
- 已完成序列与用尽长度的未完成序列怎样比较。

## 6. 停止条件是协议，不只是 EOS

### 6.1 EOS 与最大长度

- **EOS token** 是模型词表中的特殊 token，由模型预测；
- **`max_new_tokens`** 限制新生成 token；
- **`max_length`** 在一些库中限制 prompt + generation 总长度；
- **上下文上限**还包括系统提示、工具 schema、检索证据和隐藏模板。

这些预算不能混用。请求若已接近上下文上限，服务可能截断 prompt、缩短输出预算或拒绝请求；静默截断最危险，因为它会改变任务语义。

对退款请求而言，`length` 结束时即使已经得到 `{"action":"refund"`，它仍是不完整 JSON，不能按成功处理。

### 6.2 Stop token 与 stop string

Stop token 在 token ID 层判断，stop string 则在解码后的文本层判断。
一个 stop string 可能跨越多个 token，也可能跨越多次流式事件。网络读取甚至会把一个 UTF-8 多字节字符切开。

因此，实现需要增量解码，不能假设“一个数据块等于一个 token”或“一个 token 等于一个字符”。

边匹配边向用户发送文本时，末尾的 `<EN` 还可能在下一次输入后变成 `<END>`，所以暂时不能发送。
一种有界做法是：只保留“既是当前文本后缀、又是某个 stop 前缀”的最长部分，其余文本可以安全发出。

UTF-8 decoder 也要跨字节 chunk 保留尚未完成的 code point。如果数据流结束时字符仍被截断，应返回解码错误。

仓库的 `IncrementalStopMatcher` 先把任意字节切片还原为同一条 UTF-8 文本流，再逐字符匹配。
因此，网络怎样切 chunk 不会改变结果。若同一个字符同时完成多个 stop，程序按配置顺序选择；
默认不返回 stop 本身，匹配区分大小写，也不做 Unicode normalization。运行：

~~~powershell
python projects/inference-serving/stop_matching_toy.py
~~~

这个样例把 `甲🙂乙<END>尾` 中的 emoji 字节和 `<END>` 分别切开，最终仍只返回 `甲🙂乙`。
它还检查重叠 stop：`BC` 与 `ABC` 在同一字符完成时，按配置优先级选择。其他服务可能采用最长匹配，
所以重叠规则必须写入契约。

这个 matcher 从 UTF-8 字节流开始工作，不负责把 token ID 解码成字节，也不知道服务端怎样计算 usage。
如果只在客户端截断已收到的文本，远端模型可能仍在 decode。

因此，客户端不能自行声称 GPU 已释放或计费已停止，也不能改写服务端提供的 finish reason。
这些结论需要服务端协议和 trace。

一个完整的停止契约至少要回答：

| 问题 | 为什么影响调用方 |
|---|---|
| 返回文本是否包含 stop | 决定解析器最终看到什么 |
| 多个 stop 同时命中时选哪个 | 决定完成原因和截断位置 |
| Stop token 是否计入 output usage | 决定计费和长度统计 |
| 已流式发送的前缀能否撤回 | 决定客户端能否安全增量展示 |
| Finish reason 是 EOS、stop、length、content filter、cancel 还是 error | 决定结果能否继续进入业务流程 |

### 6.3 批内独立完成

批量生成时，不同序列会在不同步数结束。已完成序列应退出活跃集合，或只追加不会影响结果的 padding。
一个样本的 EOS 不能结束整批，完成后的 padding 也不能计入有效输出或 usage。

## 7. 约束解码与结构化输出

有限状态机、正则语法、上下文无关文法（CFG）或 JSON Schema，可以在每一步屏蔽会破坏合法前缀的 token。
对退款 JSON，已经生成 `{"action":` 时，引号、合法枚举值和后续分隔符受当前 parser 状态约束。

Tokenizer 的一个 token 可能同时包含多个字符。约束器必须把完整 token 送入状态机，检查全部字符后的状态，
不能只看第一个字符是否合法。

令当前语法状态为 \(q\)，token \(i\) 对应完整解码片段 \(s_i\)。如果转移 \(\delta^*(q,s_i)\) 存在，
这个 token 才合法。屏蔽后重新归一化：

\[
\tilde p_i=
\frac{p_i\mathbf 1[\delta^*(q,s_i)\text{ exists}]}
{\sum_j p_j\mathbf 1[\delta^*(q,s_j)\text{ exists}]}.
\]

分母为 0 表示当前候选集合中没有合法概率质量，程序应返回 constraint dead end。常见原因是语法与
tokenizer 无法继续，或 top-k/top-p 已经先删掉全部合法 token。实现需要规定约束与截断的顺序，
不能遇到死路就临时解除语法。

只有当前状态已经接受完整输出时，EOS 才能成为合法候选。不过，进入接受状态只说明 JSON 语法完整。
请求仍可能在 EOS 之前因 length、取消或错误结束。

Beam search 的每条分支还需要自己的语法状态，分叉后不能共享一份可变 parser state。

运行仓库提供的 trie/DFA 对照示例：

~~~powershell
python projects/inference-serving/constrained_decoding_toy.py
~~~

这个固定样例只接受 `{"x":1}` 或 `{"x":2}`。Token `1]` 的第一个字符虽然合法，完整片段却无法到达有效状态，
所以整个 token 都要屏蔽。`1}` 与 `2}` 则在合法候选的概率质量内重新归一化；EOS 只在接受状态开放。

这个小实验直接拼接预先给出的 token 文本，没有运行真实 tokenizer 的字节解码和 normalization。
生产约束器必须按照真实 token bytes 与 decoder state 检查状态转移。

实验中的有限字符串集合只用来解释原理，并不等同于完整 JSON Schema 或 CFG 引擎。

约束解码可以保证它实际编码的**语法性质**。对退款请求，它仍然无法判断：

- 字段值事实正确；
- ID 在数据库中存在；
- 金额、日期和枚举满足业务关系；
- 工具调用经过授权；
- 引用真的支持结论；
- schema 本身没有过度授权。

因此，生成结束后还要执行 schema validation、业务语义校验、权限检查、幂等控制、审批和结果核对。
Agent 工具调用的完整路径见[运行时与副作用](../applications/agent-runtime.md)。

### 7.1 无合法 token 的失败状态

约束解码会在两种情况下走进死路：当前 parser state 找不到任何可完成转移的 tokenizer token；
或者合法 token 存在，但经过其他 processor 后都变成了 \(-\infty\) 或零概率质量。

此时应返回 constraint error，或切换到事先验证过的安全模板并重新开始请求。继续生成前必须显式记录回退；
静默解除约束会把“保证输出结构”的契约变成一句空话。

## 8. KV Cache、上下文与生成成本

Decoder-only Transformer 通常把历史 token 的 K/V 缓存在每层，避免每步重复计算全部历史。于是：

- **prefill** 处理 prompt，可并行计算多个位置；
- **decode** 通常每个活跃序列每步产生一个 token，受内存带宽、KV 读取和调度影响；
- 输出越长，decode 步数和累计 KV 访问越多；
- beam search 和 `best_of` 会增加活跃序列或内部候选，显著增加成本。

生成退款 JSON 时，系统实际上同时维护几份状态：

| 状态 | 它记录什么 |
|---|---|
| KV cache | 模型前向需要的历史 key/value |
| JSON parser state | 当前前缀在语法中的位置 |
| Stop matcher | 尚不能安全发送的文本后缀 |
| Finish reason | 请求为何结束，结果能否继续使用 |

`n=4`、`best_of=4`、并行采样 4 次和顺序调用 4 次，可能采用不同的调度、cache 共享与计费方式。
返回内容也未必相同。

服务基准应报告真实输入/输出 token、请求并发、TTFT、TPOT 和完成原因。
指标定义见[推理与服务指标](../systems/inference.md)。

## 9. 流式生成

Server-Sent Events（SSE）和其他流协议发送的是**传输数据块（chunk）**，不是模型 token。
一个数据块可以不含文本，也可以包含一个或多个 token 对应的片段。某个 token 还可能因为 UTF-8 字符
尚未解码完整而延后显示。

流式客户端要处理：

- 心跳、空事件和服务商自定义的事件类型；
- 增量文本、工具参数片段和最终汇总事件；
- 网络中断、重复事件与部分结果；
- Usage 只在结束事件给出，或根本不提供；
- 客户端取消后服务端是否仍继续计费/执行。

数据块数量无法换算成输出 token。仓库的服务基准在缺少 token usage 时会明确失败，
不会用数据块数量伪造 TPOT。

仓库的 `SSEDecoder` 用不同字节边界检查 BOM、换行、多行 `data`、截断 EOF 和大小上限。
Cloud streaming executor 再检查断流、timeout、取消和关闭 response。

这些实验验证的是客户端 framing 与资源清理。它们没有证明真实服务商已经收到取消、释放 GPU 或停止计费。
TCP EOF 也不能代替协议定义的完成事件。精确控制见
[推理服务证据页](../evidence/inference-serving-controls.md)。

## 10. 确定性与可复现边界

即使 `temperature=0`，以下因素仍可能改变输出：

- 模型或 tokenizer 修订；
- chat template、系统提示或工具 schema 变化；
- 浮点精度、kernel、量化与并行归约顺序；
- dynamic batching、专家路由或服务端调度；
- 最大值并列时的 tie-breaking；
- provider 在同一模型别名后更新权重或服务栈。

需要审计级重放时，至少保存：

- 模型的具体 revision 或 hash，以及 tokenizer 版本；
- 模板渲染后的实际输入和 token IDs；
- 完整 generation config、seed、框架和硬件；
- 输出 token IDs、finish reason 与原始响应。

即使这些信息齐全，第三方封闭 API 也可能只支持“再次发送同一请求”，无法保证 bitwise replay。

## 11. 按任务选择策略

下面是实验起点，不是通用最优值：

| 任务 | 合理起点 | 必须同时验证 |
| --- | --- | --- |
| 抽取/分类 | greedy 或低随机性、约束 schema | 字段语义、漏抽、校准、拒答 |
| 基于证据问答 | 低到中随机性、citation schema | 引用支持、不可回答、检索失败 |
| 创意写作 | temperature/top-p、多候选 | 多样性、连贯性、安全与成本 |
| 代码生成 | greedy/采样多候选 + tests | 编译、单测、沙箱与依赖风险 |
| 数学/规划 | 多候选、搜索或 verifier | verifier 偏差、共享错误、预算 |
| 工具调用 | 结构约束、低随机性 | 权限、参数、幂等、审批和回执 |

“多采样再投票”只有在候选错误不完全相关、选择器确有区分能力时才可能提高质量。让同一模型生成并评判可能共享盲区。

对开头的退款请求，一个合理起点是：低随机性或 greedy、JSON 语法约束、明确输出预算。
返回后再验证 `order_id` 是否存在、动作是否获授权，并根据 finish reason 决定是否进入执行流程。

## 12. 解码实验设计

比较 generation config 时，应固定 Prompt 集、模型 revision、模板和最大输出预算，并让不同策略处理同一批 Prompt。

### 12.1 建议记录

- 任务成功率、事实/引用正确性与 schema validity；
- 输出长度、EOS/stop/length 比例；
- 每请求输入/输出 token 与成本；
- TTFT、TPOT、端到端延迟；
- 多 seed 均值、置信区间和最坏切片；
- 重复率、多样性指标及人工偏好；
- 安全、拒答和工具副作用失败。

长度同时影响质量、成本和 judge 偏好。比较两个策略时，应使用相同预算并报告长度分布，
再人工检查评分是否只是偏爱更长回答。

### 12.2 区分两类非确定性

1. 固定服务条件和 seed，多次发送完全相同请求，测量**系统非确定性**；
2. 固定服务条件但改变 seed，测量**采样分布方差**；
3. 锁定候选 token 序列，在离线 scorer 中复算 log probability，定位 processor/服务差异。

封闭 API 若不保证 seed 或返回 logits，第 3 步可能不可做，应明确记录证据边界。

## 13. 生产验收清单

### 请求契约

- 固定 model revision 或记录时间敏感 alias；
- 保存渲染后的 prompt、template 与 generation config；
- 区分 `max_new_tokens`、总上下文上限和服务端硬限制；
- 明确 top-k/top-p/penalty 的支持范围和组合顺序。

### 响应契约

- 解析所有 finish reason，不能把 length truncation 当正常完成；
- usage 缺失时标记未知，不用 chunk/字符数冒充 token；
- 结构化输出继续做类型、业务、权限与引用验证；
- 保存 request ID、模型版本和原始错误，密钥与敏感 prompt 需脱敏。

### 运行时

- 为超时、取消和重试定义幂等语义；
- 监控输出长度、停止原因、拒答率、格式失败和成本漂移；
- 对 streaming UTF-8、跨 token stop、批内独立 EOS 做测试；
- 升级模型、模板、tokenizer 或 serving engine 后重放固定评测集。

## 14. 常见错误结论

- **“temperature=0 就能跨服务 bitwise 复现”**：模型、kernel、批调度和 tie-breaking 仍可能变化。
- **“top-p=0.9 就保留概率大于 0.9 的 token”**：它保留累计概率首次达到 0.9 的最小高概率前缀。
- **“beam search 找到全局最优序列”**：有限 beam 会剪枝，length/coverage penalty 也改变目标。
- **“一个 SSE chunk 就是一个 token”**：chunk 是传输单位，不能用于 token 计数。
- **“JSON Schema 保证工具调用正确”**：它只保证约束覆盖的结构性质。
- **“重复惩罚越大越好”**：精确字符串、代码、数字和引用可能被破坏。

## 自测与实践

1. 给定概率 `[0.55, 0.25, 0.10, 0.06, 0.04]`，分别写出 top-k=3 与 top-p=0.8 的候选集合。
2. 解释为什么 temperature 会改变 top-p 集合，却不改变 top-k 的排名。
3. 实现 top-p 时，为什么要保留第一个让累计概率越过阈值的 token？
4. 设计一个跨 token stop string 与 UTF-8 流式分片测试。
5. 为工具调用列出“语法合法但语义/权限错误”的三个案例。
6. 在 `MiniGPT.generate` 上以 20 个 seed 比较 top-k、top-p 和 greedy；报告任务指标、长度与置信区间，不只展示最好样本。
7. 运行 `sampling_toy.py`，手算为什么 top-k 后的 top-p=0.7 最终得到 `[4/7,3/7,0,0]`，并构造一个 threshold tie 让“exact k”与“保留全部并列”产生不同 support。
