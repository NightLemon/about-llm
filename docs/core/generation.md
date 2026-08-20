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

模型不会一次吐出整段 JSON。它在每一步只给出整个词表的 logits。Runtime 需要决定哪些 token 目前合法、怎样
改变分布、选中哪一个、何时停止，以及流式文本何时可以安全交给客户端。

因此，**decoding（解码）是把条件分布变成一连串受协议约束的决策**。Temperature 只是这条链中的一个环节。

## 1. 生成循环的最小契约

给定前缀 \(x_{1:t}\)，模型最后一层输出词表 logits \(z\in\mathbb R^V\)。以上面的退款 JSON 为例，
生成一个 token 的完整步骤是：

1. 对 logits 应用允许/禁止 token 的硬约束；
2. 应用 repetition、presence、frequency 等分数变换；
3. 按约定顺序做 temperature 与 top-k/top-p 等过滤；
4. 重新归一化为概率并选择下一 token；
5. 更新序列、KV Cache、停止状态和流式输出；
6. 直到 EOS、stop 条件、token 预算、上下文预算、取消或错误终止。

不同框架会交换第 1—3 步的部分顺序。即使配置里都写着 `temperature=0.8, top_p=0.9`，最终分布也可能不同。
要复现实验，必须记录实现、版本和完整 processor 顺序。

### 1.1 Temperature

当 \(T>0\) 时：

\[
p_i(T)=
\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.
\]

- \(T<1\)：放大 logit 差异，分布更尖锐；
- \(T>1\)：缩小差异，分布更平坦；
- \(T\to 0^+\)：概率质量趋向最大 logit，但数值上不能把 \(T=0\) 代入除法。

API 中的 `temperature=0` 通常是 greedy 或接近 greedy 的特殊约定，具体以服务契约为准。temperature 不改变 token 的 logit 排名，但会改变相对概率，因此也会影响后续 top-p 候选集合。

实现 softmax 时应先减最大值，避免指数溢出；成熟框架通常已处理这一点。

## 2. 确定性选择与随机采样

### 2.1 Greedy decoding

每步选择

\[
x_{t+1}=\arg\max_i z_i.
\]

Greedy 快、无需随机数，适合作为可复现基线。但“每一步概率最大”不保证整段序列联合概率最大，也不保证任务质量最好。早期一个局部选择会改变以后所有条件分布。

### 2.2 Categorical sampling

按 \(p_i\) 从 categorical distribution 抽样。即使分布不变，不同随机样本也会产生不同序列。可控实验应使用显式随机数生成器和 seed，并区分：

- **相同 seed、相同 kernel/调用顺序**能否重放；
- **不同 seed 的质量分布**是否稳定；
- **线上服务**是否真的暴露并遵守 seed 契约。

仅报告一个“幸运 seed”会高估采样策略。

## 3. 截断采样

完整词表的低概率尾部可能包含大量不合适 token。截断采样先构造候选集合，再在集合内重新归一化。

### 3.1 Top-k

保留 logit 最大的 \(k\) 个 token，其余设为 \(-\infty\)。若第 \(k\) 位存在并列，必须再定义 tie-break：有的 threshold 实现会把同分 token 全部留下，候选数可能大于 \(k\)；仓库 NumPy oracle 则按 token id 升序打破并列，恰好保留 \(k\) 个。

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

若同时设置 top-k 与 top-p，常见做法是先 top-k 再在剩余分布上 top-p，但服务实现可能不同。temperature 放在 top-p 前后也会改变 nucleus 集合。实验报告不能只写“使用 top-p”，应记录所有非默认参数及顺序。

仓库 `sample_next_token` 固定了一份可手算的单步策略：sign-aware repetition penalty → temperature →
exact top-k → top-p → renormalize → categorical inverse CDF。同分时按 token id 打破 tie。运行：

~~~powershell
python projects/inference-serving/sampling_toy.py
~~~

Toy 的原始概率是 `[0.4,0.3,0.2,0.1]`。Top-k=3 后重新归一化为 `[4/9,3/9,2/9,0]`；
再做 top-p=0.7，只保留 token 0 和 1，最终得到 `[4/7,3/7,0,0]`。固定 uniform=0.6 时选中 token 1。

这里 top-p 看到的是 top-k **之后**的概率。交换 processor 顺序会得到另一份合法但不同的生成契约。

固定 uniform 比只给 seed 更容易逐项验算。真实 runtime 还会受到 RNG、CDF traversal、浮点归约和 tie-break
影响，所以相同 seed 不是跨实现逐 token 重放的充分条件。

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

`generation_runtime_control.py` 进一步真实调用 Transformers `generate()`，验证 config-level 与 call-level EOS、
`max_new_tokens` 和 stopping loop 的覆盖顺序。它使用 authored logits processor 固定 token trace，因此证明协议路径，
不代表任意 checkpoint 的生成质量或 vLLM/云 Provider 语义。精确 trace 见
[Transformers 控制台账](../evidence/transformers-controls.md)。

## 4. Logits processor 与惩罚

### 4.1 Repetition penalty

“重复惩罚”不是单一公式。仓库采用一种常见 sign-aware 约定：对每个出现过的唯一 token，若 \(z_i<0\)，
令 \(z_i'=r z_i\)；否则令 \(z_i'=z_i/r\)，其中 \(r>0\)。同一 token 出现多次仍只处理一次。

其他实现可能直接减分、按出现次数处理，或只统计 generated tokens。惩罚器也不知道重复是否合理：JSON key、
代码变量、数字和引用都可能需要原样重复。

### 4.2 Presence 与 frequency penalty

一种常见形式是

\[
z_i' = z_i
-\lambda_{presence}\mathbf 1[c_i>0]
-\lambda_{frequency}c_i,
\]

其中 \(c_i\) 是 token \(i\) 在已生成文本中的计数。但具体 API 的符号、范围、是否统计 prompt、是否按 token 还是文本片段计算都可能不同。

### 4.3 Token bias 与禁用集合

logit bias 可鼓励或禁止某些 token。由于一个可见字符串可能对应多个 token，一个 token 也可能带前导空格或只是 UTF-8 字节片段，按字符串猜 token ID 很危险。安全过滤不能只靠禁止若干 token：改写、拆分和其他语言都可绕过。

## 5. Beam search 与序列级分数

Beam search 每一步保留累计分数最高的 \(B\) 个部分序列。序列对数概率为

\[
\log p(x_{1:T})=
\sum_{t=1}^{T}\log p(x_t\mid x_{<t}).
\]

因为每项通常不大于 0，未归一化累计分数偏好更短的已完成序列。实现会使用 length penalty、归一化、coverage penalty 或任务约束，但这些会改变搜索目标。

一个显式但并非通用标准的最终分数可写为

\[
s(x_{1:T})=\frac{\log p(x_{1:T})}{T^\alpha},\qquad \alpha\ge 0.
\]

这里必须定义 \(T\) 是否包含 prompt、EOS 和特殊 token。仓库 oracle 只计生成 token，包含已发出的 EOS，
不计 prompt。由于 log probability 为负，增大正的 \(\alpha\) 会让长序列分数更靠近 0；它并非简单的“惩罚长文本”。
不同 runtime 还可能采用不同 normalization、finished-candidate cap 和 early-stopping 语义。

Beam search 不是全局穷举。考虑第一步 `A=0.6, B=0.4`，随后 `A→EOS=0.51`、`B→EOS=1`。
Beam width 1 会先剪掉 B，得到概率 \(0.6\times0.51=0.306\) 的 `A,EOS`；width 2 才能保留概率 0.4
的 `B,EOS`。加宽 beam 修复了这个反例，却不能让任意有限 beam 都变成全局最优搜索。

EOS 候选完成后不再送入模型；未完成 prefix 到达 `max_new_tokens` 时标记为 length。能否提前停止取决于
分数上界、length normalization 和实现契约，不能只看当前 raw score。Beam 常用于翻译、语音识别等输出空间
较明确的任务；开放对话中，大 beam 可能产生更通用、更重复的文本。

运行可手算的 deterministic oracle：

~~~powershell
python projects/inference-serving/beam_search_toy.py
~~~

Toy 会保存每一步 active beam、扩展、已完成 EOS 与最终排序，还演示 length penalty 如何改变短/长序列排名。
它用于检查搜索契约，不执行模型、tokenizer、KV Cache 或 GPU kernel。精确 fixture 留在实验输出中。

需要记录：beam width、raw cumulative log probability、length penalty 公式与长度口径、EOS 语义、early stopping、finished-candidate cap、tie-break、每个输入返回几个序列，以及完成序列与未完成序列如何比较。

## 6. 停止条件是协议，不只是 EOS

### 6.1 EOS 与最大长度

- **EOS token** 是模型词表中的特殊 token，由模型预测；
- **`max_new_tokens`** 限制新生成 token；
- **`max_length`** 在一些库中限制 prompt + generation 总长度；
- **上下文上限**还包括系统提示、工具 schema、检索证据和隐藏模板。

这些预算不能混用。请求若已接近上下文上限，服务可能截断 prompt、缩短输出预算或拒绝请求；静默截断最危险，因为它会改变任务语义。

### 6.2 Stop token 与 stop string

Stop string 可能跨 token 边界和流式事件边界；更底层的网络字节读取还可能切开 UTF-8 多字节字符。正确实现需要增量解码并保留足够后缀用于匹配，不能假设“一 chunk 等于一 token”或“一 token 等于一个字符”。

若边匹配边把文本交给用户，不能立刻释放所有已解码字符：末尾的 `<EN` 可能在下一 chunk 变成 `<END>`。一个 bounded 做法是只保留“当前文本后缀中、同时也是任一 stop 的前缀”的最长部分，其余才安全 emit。UTF-8 decoder 也必须跨 byte chunk 保留未完成 code point，并在 EOF 严格拒绝截断序列。

仓库 `IncrementalStopMatcher` 把输入视为同一条 UTF-8 text stream，并逐 decoded character 匹配。因此网络怎样切
byte chunk 不会改变结果。若同一字符同时完成多个 stop，按配置顺序选择；默认不返回 stop，匹配区分大小写，
也不做 Unicode normalization。运行：

~~~powershell
python projects/inference-serving/stop_matching_toy.py
~~~

Fixture 把 `甲🙂乙<END>尾` 同时切开 emoji bytes 和 `<END>`，最终仍只返回 `甲🙂乙`。它还覆盖重叠 stop：
若 `BC` 与 `ABC` 在同一字符完成，按配置优先级选择。其他服务可能采用 longest-match 或不同 priority，
所以 overlap 规则必须写入契约。

该 matcher 处理的是已经形成的 UTF-8 文本，不把 token id 解码成文本，也不知道 provider 是否把 stop token 计入 usage。若它只在客户端截断已收到文本，不能节省远端 decode、改变可信 finish reason 或证明取消已释放 GPU/停止计费；这些仍需服务端协议和 trace 验证。

必须定义：

- stop 内容是否包含在返回文本中；
- 命中多个 stop 时返回哪一个；
- stop token 是否计入 output token usage；
- 流式输出已经发送的前缀能否撤回；
- finish reason 如何区分 EOS、stop、length、content filter、cancel 和 error。

### 6.3 批内独立完成

批量生成时，不同序列会在不同步数结束。已完成序列应停止增长，或只追加不会影响结果的 padding；不能让一个样本的 EOS 强制结束整批，也不能把结束后的 padding 计入有效输出。

## 7. 约束解码与结构化输出

有限状态机、正则语法、context-free grammar 或 JSON Schema 可在每一步屏蔽不能形成合法前缀的 token。若 tokenizer 的一个 token 同时包含多个字符，约束器必须按 token 对语法状态的完整转移判断，不能只检查 token 的第一个字符。

令当前语法状态为 (q)，token (i) 对应完整解码片段 (s_i)，若转移 (delta^*(q,s_i)) 存在，则它合法。屏蔽后重新归一化：

\[
\tilde p_i=
\frac{p_i\mathbf 1[\delta^*(q,s_i)\text{ exists}]}
{\sum_j p_j\mathbf 1[\delta^*(q,s_j)\text{ exists}]}.
\]

分母为 0 是明确的 constraint dead end，不能偷偷解除约束。EOS 只有在当前状态接受完整输出时才能被允许；反过来，到达接受状态也不等于请求已经以 EOS 正常完成，仍可能因 length、取消或错误终止。Beam search 下每条 prefix 还要携带自己的语法状态，分叉时不能共享一份可变 parser state。

运行仓库有限字符串集合的 trie/DFA oracle：

~~~powershell
python projects/inference-serving/constrained_decoding_toy.py
~~~

Fixture 只接受 `{"x":1}` 或 `{"x":2}`。Token `1]` 的第一个字符虽然合法，完整片段却无法到达有效状态，
所以必须整 token 屏蔽；`1}` 与 `2}` 在 allowed mass 内重新归一化。EOS 只在 accepting state 开放。

Toy 直接拼接 supplied token text，没有执行真实 tokenizer 的 byte decoder 或 normalization。生产约束器必须按实际
token bytes 与 decoder state 验证完整转移；有限字符串集合也不代表完整 JSON Schema 或 CFG 引擎。

约束解码可以保证某种**语法性质**，但不能保证：

- 字段值事实正确；
- ID 在数据库中存在；
- 金额、日期和枚举满足业务关系；
- 工具调用经过授权；
- 引用真的支持结论；
- schema 本身没有过度授权。

因此还需要应用层 schema validation、语义校验、权限检查、幂等键、审批和执行结果核对。Agent 工具调用的完整路径见[运行时与副作用](../applications/agent-runtime.md)。

### 7.1 无合法 token 的失败状态

约束状态机可能仍有语法出边，但模型给所有合法 token 零概率，或者 tokenizer 根本没有可完成转移的 token。
此时应返回 constraint error、使用已验证的安全模板或重构请求，不能静默解除约束。还要固定约束与 top-k/top-p
的先后顺序：先截断可能删掉全部合法 token。

## 8. KV Cache、上下文与生成成本

Decoder-only Transformer 通常把历史 token 的 K/V 缓存在每层，避免每步重复计算全部历史。于是：

- **prefill** 处理 prompt，可并行计算多个位置；
- **decode** 通常每个活跃序列每步产生一个 token，受内存带宽、KV 读取和调度影响；
- 输出越长，decode 步数和累计 KV 访问越多；
- beam search 和 `best_of` 会增加活跃序列或内部候选，显著增加成本。

`n=4`、`best_of=4`、“并行采样 4 次”和“顺序调用 4 次”在计费、调度、缓存共享及返回内容上不一定等价。服务基准应报告实际输入/输出 token、请求并发、TTFT、TPOT 和完成原因，详见[推理与服务指标](../systems/inference.md)。

## 9. 流式生成

Server-Sent Events 或其他流协议发送的是**传输 chunk**，不是 token。一个 chunk 可含零个、一个或多个 token/文本片段；token 也可能因增量 UTF-8 解码而延后显示。

流式客户端要处理：

- 心跳、空事件和 provider-specific event type；
- 增量文本、工具参数 delta 和最终汇总事件；
- 网络中断、重复事件与部分结果；
- usage 只在结束事件给出或根本不提供；
- 客户端取消后服务端是否仍继续计费/执行。

不能用 chunk 数估算 output token。仓库的 serving benchmark 对缺失 token usage 选择显式失败，而不是用 chunk 数伪造 TPOT。

仓库 `SSEDecoder` 用不同字节边界覆盖 BOM、换行、多行 data、截断 EOF 与大小上限；Cloud streaming executor
再覆盖断流、timeout、取消和 response close。它们验证客户端 framing 与资源清理，不代表真实 Provider 已观察到
取消、释放 GPU 或停止计费。看到 TCP EOF 也不能替代协议中的完成事件。精确控制见
[推理服务证据页](../evidence/inference-serving-controls.md)。

## 10. 确定性与可复现边界

即使 `temperature=0`，以下因素仍可能改变输出：

- 模型或 tokenizer 修订；
- chat template、系统提示或工具 schema 变化；
- 浮点精度、kernel、量化与并行归约顺序；
- dynamic batching、专家路由或服务端调度；
- 最大值并列时的 tie-breaking；
- provider 在同一模型别名后更新权重或服务栈。

需要审计级重放时保存：模型不可变 revision/hash、tokenizer、模板渲染后的实际输入、token IDs、全部 generation config、seed、框架/硬件、输出 token IDs、finish reason 和原始响应。即便如此，第三方封闭 API 也可能只提供“请求可重放”而非 bitwise replay。

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

## 12. 解码实验设计

比较 generation config 时应固定 prompt 集、模型 revision、模板和最大输出预算，对同一 prompt 做 paired 比较。

### 12.1 建议记录

- 任务成功率、事实/引用正确性与 schema validity；
- 输出长度、EOS/stop/length 比例；
- 每请求输入/输出 token 与成本；
- TTFT、TPOT、端到端延迟；
- 多 seed 均值、置信区间和最坏切片；
- 重复率、多样性指标及人工偏好；
- 安全、拒答和工具副作用失败。

长度既影响质量又影响成本，也会影响 judge 偏好。比较两个策略时最好限制预算、报告长度分布，并人工检查“更长所以得分高”的偏差。

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
