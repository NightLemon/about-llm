# 推理基础：模型怎样逐个生成 token

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：第一次系统学习 LLM 推理，或准备进入推理服务与容量规划的开发者。
- **先修**：[Transformer](../core/transformer.md)与[生成基础](../core/generation-basics.md)。
- **首次阅读**：一次生成 → prefill/decode → KV Cache → 多请求调度 → 指标。
- **完成信号**：能用一条请求解释 TTFT、TPOT、KV 容量和 (P+O-1) 的工作量账本。
- **卡住时**：先只看“请求 A 的三轮模型工作”，暂时忽略分页、量化和服务协议。

</div>

用户看到的是一段连续文字，模型看到的却是一串“预测下一个 token”的重复工作。
推理系统的任务，就是在显存和时间预算内，把这些重复工作安排好，并把结果可靠地送回客户端。

本章先回答“模型在算什么”。请求怎样进入调度器、KV block 怎样变化、流式响应怎样结束，
将在[端到端请求生命周期](inference-request-lifecycle.md)中串起来。

## 从一次最小生成开始

假设 prompt 被 tokenizer 编码成四个 token：

```text
[x1, x2, x3, x4]
```

模型不是一次写出完整答案，而是计算下一个 token 的概率分布，选出 `y1`，再根据
`[x1,x2,x3,x4,y1]` 预测 `y2`，如此继续。

```text
prompt                         -> y1
prompt + y1                    -> y2
prompt + y1 + y2               -> y3
```

如果每次都重算全部历史，前面四个 prompt token 的 K/V 会被反复计算。
KV Cache 的作用，就是保存这些以后仍会使用的 K/V。

## Prefill 与 decode 是同一生成过程的两个阶段

### Prefill：先读完整个输入

Prefill 对整段 prompt 做 causal forward。Prompt 内多个位置可以并行计算，
每层的 K/V 被写入缓存，最后一个位置的 logits 用来选择第一个输出 token。

```text
输入模型: x1 x2 x3 x4
写入 KV:  x1 x2 x3 x4
产生输出:             y1
```

长 prompt 的 prefill 通常包含较大的矩阵计算，常更偏向计算受限；
但是否真正 compute-bound 仍取决于模型、batch、序列长度、kernel 和硬件。

### Decode：一次让每条序列前进一步

下一轮只把新 token `y1` 输入模型。它读取已有 KV、追加 `y1` 的 K/V，再产生 `y2`。

```text
输入模型: y1
读取 KV:  x1 x2 x3 x4
追加 KV:              y1
产生输出:                 y2
```

Decode 必须串行等待前一个输出，单步工作又相对小。小 batch 时常反复读取大量权重和 KV，
因此通常更受显存带宽、kernel launch 和调度影响。Batch 增大后，这一判断可能改变。

### 为什么工作量是 (P+O-1)

若 prompt 长度 (P=4)，模型输出 (O=3) 个 token：

| Forward | 处理的位置数 | 产生的输出 |
|---|---:|---|
| Prefill | 4 | `y1` |
| Decode 1 | 1 | `y2` |
| Decode 2 | 1 | `y3` |

总 forward positions 是 (4+3-1=6)。最后一个 prompt position 已经产生 `y1`，
所以后续只需 (O-1) 次普通 decode。

这个等式只适用于标准 decoder-only causal generation，且没有 prefix reuse、speculative verification 或 beam。
它是模型工作量的教学账本，不是 API usage、计费 token 或 GPU kernel work 的通用定义。

## KV Cache 保存了什么

在一层 attention 中，历史 token 的 K/V 在生成过程中不会因新 token 到来而改变。
缓存它们后，新 query 可以直接和历史 key 做 attention，再用得到的权重组合 value。

标准布局下，每个已缓存 token 的理想化 K/V 字节数约为：

\[
M_{KV/token}=2\times L\times H_{kv}\times D\times bytes(dtype),
\]

其中：

- 2 表示 K 与 V；
- (L) 是层数；
- (H_{kv}) 是 K/V head 数；
- (D) 是 head dimension；
- `bytes(dtype)` 是每个元素的字节数。

例如 32 层、8 个 KV heads、head dimension 128、BF16 每元素 2 bytes：

```text
2 * 32 * 8 * 128 * 2 = 131072 bytes = 128 KiB / token
```

8192 个缓存位置的理想化单序列 KV 就是 1 GiB。真实 runtime 还需要 block metadata、对齐、workspace，
也可能使用与标准 MHA 不同的布局，所以不能直接把公式结果当作峰值 VRAM。

### GQA 与 MQA 为什么能降低 KV

MHA 通常让 query heads 和 K/V heads 数量相同。GQA 让一组 query heads 共享一个 K/V head，
MQA 更进一步让所有 query heads 共享同一组 K/V。

公式中决定缓存大小的是 (H_{kv})，不是 query head 数。因此在其他条件相同时，减少 K/V heads
可以显著降低长上下文和高并发时的缓存容量。

## 为什么 KV 需要分页管理

若每个请求一开始就预留最大长度的连续 KV，短请求会浪费大量空间；请求不断到达和结束时，
寻找足够大的连续区域也很困难。

Paged KV 把物理缓存切成固定大小的 block。每条序列用 block table 记录自己的逻辑第 0、1、2 块
分别落在哪些物理 block 上。它类似虚拟内存分页，但只是帮助理解映射关系，两者实现并不相同。

```text
逻辑 block:   0   1   2
物理 block:   5   1   8
block table: [5, 1, 8]
```

物理 block 不必连续。固定 block 也不会消灭所有碎片：每条序列最后一个 block 仍可能没有填满。
共享未满尾块后继续 append，还需要 copy-on-write。

具体状态变化见[端到端请求生命周期](inference-request-lifecycle.md#kv-block-table)和
[Paged KV 引导实验](../practice/labs/lab-7a-paged-kv.md)。

## 一个请求变成多个请求

在线服务里，请求会在不同时间到达，prompt 和输出长度也不同。

静态 batching 先组成一批再执行。若一条请求很长，已经完成的短请求可能留下空位。
Continuous batching 在调度边界加入新请求、移除完成请求，让 batch 成员动态变化。

Scheduler 每轮要在四个目标之间权衡：

- 新请求的首 token 不应等待太久；
- 正在 decode 的请求不应频繁停顿；
- GPU 应获得足够工作，避免 batch 太小；
- KV 容量、token budget 和公平性不能被突破。

Chunked prefill 会把长 prompt 分成片段，使它有机会与 decode 工作交错。
KV 不足时，有的系统会拒绝新请求，有的会抢占并在以后重算 context，也可能使用其他策略。

Continuous batching 只表示引擎会在请求执行期间持续重组 batch；prefill-first、decode-first 和抢占规则仍由
具体 scheduler 决定。
这些行为必须针对具体 runtime 和版本确认。

## 用户体验由哪些指标描述

只报告“每秒 token 数”无法解释用户等在哪里。

| 指标 | 回答的问题 | 常见影响因素 |
|---|---|---|
| Queue time | 请求在执行前等了多久？ | admission、并发上限、过载 |
| TTFT | 多久看到第一个 token？ | queue、tokenize、prefill、网络 |
| TPOT / ITL | 首 token 后，token 以多快速度出现？ | decode batch、带宽、调度、kernel |
| E2E latency | 请求多久得到终态？ | 排队、输入输出长度、错误和重试 |
| Throughput | 系统单位时间完成多少工作？ | batch、容量、硬件、失败分母 |

提高 batch 往往能增加吞吐，却可能让单请求排队更久。没有一个脱离 SLO 的“最大吞吐最优值”。

TPOT 常用首 token 后的生成区间除以 `output_tokens - 1`。只有一个输出 token 时分母为零，
因此 TPOT 应记为未定义，而不是 0。

## 其他优化在解决什么

先建立 prefill、decode、KV 和调度的心智模型，再看优化名词会简单很多：

| 技术 | 主要想减少什么 | 不能自动保证什么 |
|---|---|---|
| FlashAttention | Attention 中间结果的 HBM 往返 | KV 持久容量或端到端一定加速 |
| Weight quantization | 权重容量与读取带宽 | 质量不变、已有高效低位 kernel |
| KV quantization | 长上下文和并发的 KV payload | Attention 误差可接受或 runtime 支持 |
| Prefix cache | 重复前缀的 prefill 工作 | 后续 decode 变快或跨权限安全共享 |
| Speculative decoding | 串行 target decode 次数 | 接受率足够高或一定加速 |
| CUDA Graph | 重复 decode 的 CPU launch 开销 | 动态 shape 与所有控制流都可捕获 |
| Tensor parallelism | 单设备放不下的权重与计算 | 通信免费或单卡小模型更快 |

下一章[推理优化](inference-optimization.md)会从症状出发，说明何时选择这些技术，而不是逐项堆术语。

## 第一次读推理代码的顺序

不要从最底层 kernel 随机跳读。沿一条请求走：

1. 请求怎样保存 prompt、长度上限和 sampling config。
2. Scheduler 如何把 waiting 请求变成当前 batch。
3. Model Runner 如何准备 token、position、block table 和张量边界。
4. Attention 如何读写 KV。
5. Sampler 如何从 logits 产生 token。
6. Postprocess 如何判断 EOS、长度、stop、取消和资源释放。

仓库中可以先读[端到端请求生命周期](inference-request-lifecycle.md)，再运行
[Inference Serving 项目](../practice/projects/inference-serving.md)。
用于手算对照的实现与测试范围位于[推理服务证据页](../evidence/inference-serving-controls.md)。

## 自测

1. Prompt 有 5 个 token、输出 4 个 token时，普通生成需要处理多少 forward positions？逐轮写出来。
2. Prefill 产生首 token 后，首 token 的 K/V 是否已经写入缓存？它在什么时候写入？
3. 为什么 GQA 改变 KV 容量，却不要求 query head 数和 KV head 数相同？
4. Paged KV 为什么仍会有内部碎片？
5. TTFT 很高时，为什么不能直接断言 prefill kernel 很慢？

下一步：[一次请求如何穿过 LLM 推理引擎](inference-request-lifecycle.md)。
