# 推理基础与优化

本章建立生成计算的基础。性能与部署继续阅读[从算子到 KV Cache 的推理优化](inference-optimization.md)和[vLLM/OpenAI-compatible 单卡服务](vllm-serving.md)。

## 两个阶段

自回归推理分为：

1. **Prefill**：一次并行处理整个输入，生成每层 K/V 并存入缓存；通常计算密集。
2. **Decode**：每步只输入新 token，读取此前 KV Cache 生成下一个 token；通常受内存带宽和调度影响。

TTFT（Time To First Token）主要受排队与 prefill 影响；TPOT（Time Per Output Token）描述稳态解码速度；端到端延迟还包含网络、tokenize、后处理和工具调用。

标准 causal LM 的 prefill 最后位置已经产生首个输出分布。若 prompt 为 \(P\)、实际输出 \(O\ge1\)，且没有 prefix reuse、speculative verification 或 beam，模型 forward positions 的一阶账本是 \(P+O-1\)：后续 \(O-1\) 个输出才各需要一次 decode forward。它不是 API 计费口径，也没有包含 padding、cache 命中或 kernel 实际工作。

## KV Cache

因果注意力中，旧 token 的 K/V 不随新 token 改变，因此可缓存，避免每步重算整个前缀。粗略元素数：

\[
2\times L\times T\times H_{kv}\times D
\]

其中 2 表示 K 和 V，\(L\) 为层数，\(T\) 为已缓存长度，\(H_{kv}\) 为 K/V 头数，\(D\) 为头维；再乘每元素字节数和 batch/并发序列数。GQA/MQA 通过减少 \(H_{kv}\) 显著降低缓存。

长上下文会线性增加每请求 KV 内存。PagedAttention 将 KV 划为固定页，按需映射，减少连续大块分配和内部碎片，并支持更灵活共享前缀。

固定页并不消灭碎片：每条序列的最后一页仍可能空置；共享 partial tail 后追加还需要 copy-on-write。容量不足必须在写 tail 前整体拒绝 append。仓库 metadata-only allocator 可验证 block table/refcount/COW 与碎片账本，但没有真实 K/V tensor 或 GPU kernel。

## 批处理与调度

静态批处理等待一组请求一起运行，简单但长度差异导致 padding 和空闲。连续批处理允许每个 decode step 动态加入新请求、移除完成请求，提高利用率。调度必须在吞吐、公平、TTFT 和长请求饥饿之间权衡。

Chunked prefill 把很长 prompt 分块，与 decode 请求交错，避免长 prefill 阻塞所有生成；过小分块会增加调度开销。

仓库的离散 continuous-batching oracle 固定 FCFS admission、sequence/token cap、decode-first 和 per-request prefill chunk，逐 boundary 记录首 token与完成，并验证 `sum(P+O-1)` work conservation。它是 CPU state machine；step 不是秒、slot utilization 不是 GPU utilization，也不等于特定 vLLM 版本的 scheduler。

## 采样分布也是服务契约

只记录 `temperature=0.8, top_p=0.9` 不足以重放：还要固定 repetition/presence/frequency processor、执行顺序、top-k/top-p 边界、并列 token 的 tie-break、重新归一化时点、RNG/CDF 映射和框架版本。仓库单步 NumPy oracle 固定 repetition → temperature → exact top-k → post-top-k top-p → token-id-order inverse CDF，并用 `[0.4,0.3,0.2,0.1]` fixture 精确得到最终 `[4/7,3/7,0,0]`。它用于核对概率账本，不代表 provider/vLLM/Transformers 默认值，也没有多 token、模型质量或性能证据。

停止条件还要区分 config 与 call override。仓库 Transformers runtime control 用强制 token plan 真实执行三条生成循环：config EOS `{2,3}` 在 3 停止；call EOS=5 后 3 不再停止、5 才停止；call `max_new_tokens=2` 在无 EOS 时按长度停。它验证当前依赖版本的受控 `GenerationMixin` 路径，不执行真实 tokenizer、正常模型 token 选择、vLLM/provider 或 GPU。该返回对象没有 provider 风格 finish reason，因此报告的 stop 分类是依据已知 plan 与条件推断，而不是服务端 receipt。

Beam search 还要单独固定 active-prefix 的累计 log probability、最终 length normalization、EOS 是否计入生成长度、prompt 是否计入长度、finished-candidate cap、early stopping 与并列规则。仓库 table-driven oracle 使用 \(s=\log p/T^\alpha\)，其中 (T) 只计生成 token、包含 EOS、不含 prompt；EOS 立即完成且不再展开，不做 heuristic early stopping，并保留所有从 active prefix 产生的 EOS。它用 beam 1 返回概率 0.306、beam 2 返回概率 0.4 的反例证明有限 beam 可剪掉更优路径，但不等价于 Transformers、vLLM 或 provider 的实现。

约束解码还要固定 grammar state 与 tokenization 的组合语义：每个 token 的完整 decoded fragment 都必须能穿过状态机，EOS 只在 accepting state 开放，屏蔽后的 allowed probability mass 必须重新归一化，质量为零则 typed failure。仓库 finite-literal trie oracle 用高概率 `1]` 反例证明只检查 token 首字符会放过非法输出；它假设 supplied text fragment 可直接拼接，不覆盖真实 tokenizer byte state、JSON Schema/CFG 或 runtime 性能。

## 量化

- **权重量化**：把 FP16/BF16 权重压到 INT8、INT4 等，减少容量与带宽。
- **激活量化**：进一步加速矩阵乘，但动态范围和 outlier 更难处理。
- **KV Cache 量化**：降低长上下文和高并发内存，可能影响注意力精度。
- **PTQ**：训练后用校准数据确定尺度/舍入。
- **QAT**：训练时模拟量化误差，成本高但可能恢复质量。

常见粒度有 per-tensor、per-channel、per-group；group 越小通常越准，但尺度元数据与内核开销更高。权重大小约为“参数量 × 每参数位数/8”，实际还包含 scale、zero point、未量化层、运行时缓冲和 KV Cache。

量化是否加速取决于硬件和内核。一个 4-bit 文件更小，不代表端到端一定更快；若频繁反量化或缺少优化 kernel，可能反而变慢。

## 投机解码

小型 draft 模型先提出多个 token，大模型一次并行验证，按拒绝采样规则接受前缀，从而在保持目标模型分布的条件下减少串行大模型步数。加速取决于接受率、draft 成本、验证长度和内核效率。自投机方法也可用目标模型的早退层或多个 head。

“按规则”是必要条件：sampling 版本以 `min(1, p/q)` 接受，首次拒绝从 normalized positive `(p-q)` residual 采样；全部 proposal 接受时才从额外 target position 发一个 bonus token。一步接受率是 `1 - TV(p,q)`。Greedy 前缀一致算法是另一份契约；任意“大模型挑小模型输出”不保证 target sampling distribution。仓库的概率级 CPU oracle 只证明该数学和 block 控制流，不证明模型/kernel 实现或速度。

## 前缀缓存

多个请求共享完全相同的系统提示或文档前缀时，可复用其 KV。命中必须同时满足可信 tenant/security domain、authorization/policy revision、model/tokenizer/chat template/adapter revision、RoPE/position config、KV dtype 相等，以及 cached token ids 是请求 token ids 的 exact prefix；不能只比较 raw text 或 hash。若有多个候选，选择最长 exact prefix。

Hash 只适合索引候选，命中仍做完整 identity/token comparison；未加密 hash 也不隐藏低熵 prompt。使用中的 entry 由 lease/refcount pin 住，不能被 LRU 淘汰。跨安全域共享、命中时延和删除生命周期都可能泄漏信息，需隔离、审计并按威胁模型处理。仓库 metadata oracle 用强制 hash collision 验证这些状态机边界，但没有真实 K/V、GPU、vLLM 或性能证据。

## 内存预算

推理显存包括：权重、KV Cache、临时激活、CUDA graph/工作区、量化元数据和框架开销。容量规划不能只把显存除以模型文件大小。保留峰值余量，测试最长输入/输出和最大并发。

## 优化顺序

1. 固定质量与流量分布基线。
2. 分解排队、prefill、decode、网络和外部工具耗时。
3. 判断是算力、带宽、容量还是调度瓶颈。
4. 一次改变一个变量：batch、量化、并行度、缓存或模型。
5. 同时回归质量、尾延迟、OOM 和成本。

## 自测

1. 为什么长输入主要影响 TTFT，而长输出显著影响总延迟和服务占用？
2. GQA 如何改变 KV Cache 公式？
3. 4-bit 权重模型为什么可能没有 2× 的 FP8 推理速度？
