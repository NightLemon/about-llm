# 生成与解码

模型给出的是下一 token 的分数，产品需要的是一段在质量、延迟、成本、格式与安全约束下完成任务的输出。**解码（decoding）是把条件分布变成决策的过程**，并不是给模型加一个“创造力旋钮”那么简单。

## 1. 生成循环的最小契约

给定前缀 \(x_{1:t}\)，模型最后一层输出词表 logits \(z\in\mathbb R^V\)。一次生成步包括：

1. 对 logits 应用允许/禁止 token 的硬约束；
2. 应用 repetition、presence、frequency 等分数变换；
3. 按约定顺序做 temperature 与 top-k/top-p 等过滤；
4. 重新归一化为概率并选择下一 token；
5. 更新序列、KV Cache、停止状态和流式输出；
6. 直到 EOS、stop 条件、token 预算、上下文预算、取消或错误终止。

不同框架会交换第 1—3 步的部分顺序，公式名称相同也可能得到不同分布。要复现实验，必须记录**实现、版本和完整 processor 顺序**。

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

仓库 `sample_next_token` 固定一份可手算的单步策略：sign-aware repetition penalty → temperature → exact top-k（同分 token id 小者优先）→ 在 top-k 重新归一化后的概率上取 top-p → 再归一化 → 按 token id 升序执行 categorical inverse CDF。运行：

~~~powershell
python projects/inference-serving/sampling_toy.py
~~~

固定 logits 对应原始概率 `[0.4,0.3,0.2,0.1]`。Top-k=3 后分布是 `[4/9,3/9,2/9,0]`；top-p=0.7 的最小 crossing prefix 是 token 0、1，最终概率为 `[4/7,3/7,0,0]`，固定 uniform=0.6 选 token 1。注意 top-p 用的是 **top-k 后重新归一化**的概率，不是原始 `[0.4,0.3,...]`；更换 processor 顺序会得到另一份合法但不同的契约。

固定 uniform 比只给 seed 更容易逐项验算，但不同 runtime 的 RNG、CDF traversal、浮点归约或 tie-break 即使面对同一最终分布，也未必把“同一个 seed/uniform”映射为同一 token。该 CPU oracle 不执行模型/tokenizer，也不含多 token 循环、EOS/stop、KV、batch、质量或性能证据。

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

这是教学实现，不包含 EOS、批内独立停止、KV Cache 或流式 UTF-8 解码，不能据此宣称生产吞吐。

仓库另有 `generation_runtime_control.py`，在随机 tiny GPT-2 上真实调用 Transformers `generate()`，再用 authored logits processor 覆盖全部 next-token scores。固定三条 trace 分别得到 `[4,3]`（config EOS set 在 3 停止）、`[3,5]`（call-level EOS=5 覆盖 config 的 `{2,3}`）和 `[4,6]`（call-level `max_new_tokens=2` 截断）。这比只检查输出 shape 更强，因为实际执行了模型 forward、processor 与 stopping loop；但 token 由测试 processor 强制，不证明随机权重或任何 checkpoint 的分布、质量、正常 processor 组合、vLLM/provider 语义或性能。Transformers 返回中没有 provider 风格 finish reason，报告只能由受控路径推断。

## 4. Logits processor 与惩罚

### 4.1 Repetition penalty

“重复惩罚”不是单一公式。仓库 oracle 采用一种常见 sign-aware 约定：对每个曾出现的唯一 token，若 \(z_i<0\) 则令 \(z_i'=r z_i\)，否则令 \(z_i'=z_i/r\)，其中 \(r>0\)；同一 token 重复出现多次仍只处理一次。于是 \(r>1\) 会把正 logit 拉低、负 logit 变得更负，但这不是 frequency penalty。其他实现可能直接减分、按出现次数处理，或只统计 generated tokens。它通常不区分“无意义复读”和“任务必须重复”：代码变量、JSON key、引用、数字、诗歌副歌都可能被误伤。

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

这里必须定义 (T) 是否包含 prompt、EOS 和特殊 token。仓库 oracle 规定 (T) **只计生成 token，包含已发出的 EOS，不计 prompt**。因为 log probability 为负，增大正的 \(\alpha\) 会让较长序列的分数向 0 靠近；这不是“简单惩罚长序列”，也不能把该公式套到所有框架。Transformers、vLLM 或云服务可能使用不同 normalization、finished-candidate cap、EOS/early-stopping 语义。

Beam search 不是对所有可能序列的精确全局搜索；有限 beam 仍会剪掉以后可能变好的前缀。考虑 root 上 `A=0.6,B=0.4`，随后 `A→EOS=0.51`、`B→EOS=1`：beam width 1 会先剪掉 B，返回概率 (0.6\times0.51=0.306) 的 `A,EOS`；width 2 才能返回概率 0.4 的 `B,EOS`。这说明更宽 beam 能修复某个反例，却不保证有限宽度在任意树上全局最优。

EOS 候选一旦完成就不应再次送入模型；未完成 prefix 到 `max_new_tokens` 时应明确标记 length。是否在“最好完成候选已经胜过 active candidates”时提前停止，取决于分数上界、length normalization 与实现契约，不能凭当前 raw score 随意停止。它常用于翻译、语音识别和有明确输出空间的任务。开放式对话中，大 beam 可能产生更通用、重复或缺少多样性的文本。

运行可手算的 deterministic oracle：

~~~powershell
python projects/inference-serving/beam_search_toy.py
~~~

它逐 step 保存 active beam、全部正概率 expansion、立即完成的 EOS 与最终排序。除上面的 pruning 反例外，第二个 fixture 固定短序列概率 0.6、长度 2，长序列概率 0.4、长度 3：\(\alpha=0\) 选短序列，\(\alpha=2\) 选长序列。Oracle 会保存所有从 active prefix 产生的 EOS，不模拟某个 runtime 可能采用的 top-\(2B\) candidate cap；也没有模型/tokenizer/KV/GPU、生成质量或性能证据。

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

仓库 `IncrementalStopMatcher` 固定以下语义：输入是同一 UTF-8 text stream 的任意 byte chunks；逐 decoded character 处理，所以 byte chunking 不改变结果；某个 character 使一个或多个 stop 完成时立即终止，同一 character 同时完成多个 stop 则按配置顺序选择；默认不返回 stop，自选 `include_stop` 可返回；匹配区分大小写且不做 Unicode normalization。运行：

~~~powershell
python projects/inference-serving/stop_matching_toy.py
~~~

Fixture 把 `甲🙂乙<END>尾` 同时切开 emoji 的 UTF-8 bytes 和 `<END>`，最终只返回 `甲🙂乙`；另用 `("BC","ABC")` 处理 `ABCZ`，两者在字符 `C` 同时完成，配置顺序选 `BC`，因此返回前缀 `A` 并丢弃 stop 后的 `Z`。这种“first completion”还意味着 stops `("END","E")` 面对 `END` 会在 `E` 完成时停止，不等待更长候选。其他服务可能采用不同 overlap/priority 规则，必须写入契约。

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

Fixture 只接受 `{"x":1}` 或 `{"x":2}`。在关键步，token `1]` 的首字符 `1` 可以转移、第二个字符 `]` 不可以，因此其原始概率 0.65 被完整屏蔽；合法 `1}`/`2}` 的原始质量是 0.25/0.10，总 allowed mass 0.35，重归一化后为 `5/7`/`2/7`。EOS 在输出进入 accepting state 前始终被禁用，最终 token 序列为 `(0,1,2,EOS)`。

这份 oracle 的单位是 supplied token text 中的 Python Unicode code point，并假设各 fragment 直接拼接就是 decoded text。它不执行真实 tokenizer 的 byte/incremental decode 或 normalization，只接受 authored finite literal set，不是 JSON Schema、CFG、正则引擎、模型或 provider runtime。对真实 tokenizer，必须按实际 token bytes/decoder state 验证完整转移；字符串级 toy 不能证明所有 schema 都可表达或高效解码。

约束解码可以保证某种**语法性质**，但不能保证：

- 字段值事实正确；
- ID 在数据库中存在；
- 金额、日期和枚举满足业务关系；
- 工具调用经过授权；
- 引用真的支持结论；
- schema 本身没有过度授权。

因此还需要应用层 schema validation、语义校验、权限检查、幂等键、审批和执行结果核对。Agent 工具调用的完整路径见[运行时与副作用](../applications/agent-runtime.md)。

### 7.1 无合法 token 的失败状态

约束状态机可能仍有语法出边，但模型在所有合法 token 上给出零概率，或 tokenizer 根本没有能完成该转移的 token。生产实现应显式返回 constraint error、回退到经过验证的安全模板或重新构造请求，不能悄悄解除约束继续生成。若采用浮点阈值、top-k/top-p 或其他 processor，还要固定“先约束还是先截断”：先截断可能把所有合法 token 删除，交换顺序会改变条件分布。

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

本仓库 `SSEDecoder` 的 CPU fixture 会把 UTF-8 字节逐个或按不同边界送入 parser，验证 BOM、CR/LF/CRLF、多行 data、截断 EOF 与 line/event/total byte 上限。它只证明 framing 状态机，不证明真实网络 chunk 分布、provider event schema、backpressure、取消已传播到服务端或停止计费。Provider state machine 必须另外验证完成事件；看到 TCP EOF 不等于模型成功完成。

Cloud streaming executor 进一步用 MockTransport 验证 response 在成功、超限、断流、timeout 与取消后关闭，并把 callback 时间计入 monotonic deadline。它在 2xx body 开始后禁止自动重试，因为已经交付的 partial text 与远端计费都可能发生。MockTransport 的 close 事件仍不能证明真实服务端观察到 RST/取消或及时释放 GPU 工作。

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
