# 实验 7B：Qwen3-0.6B 如何穿过 nano-vLLM

这个实验把你正在使用的 `Qwen3-0.6B + nano-vLLM + RTX 3070 Laptop` 放回一条完整推理链路。
我们不从零散类名开始，而是跟着一次 768-token 请求，看它怎样从 waiting 变成 running，怎样命中
256-token KV blocks，怎样完成 prefill 与 decode，最后又怎样释放全部 block 引用。

**相关教材**：[请求生命周期](../../systems/inference-request-lifecycle.md) ·
[Paged KV 实验](lab-7a-paged-kv.md) ·
[Qwen](../../models/qwen.md) ·
[Inference Serving 项目](../projects/inference-serving.md)
{ .doc-nav }

## 完成标准

做完后，你应该能只看报告回答六个问题：

1. 当前 step 为什么是 `prefill`、`chunked_prefill` 或 `decode`？
2. 为什么相同前缀命中两个 block，而第 257 个 token 漂移后只命中一个？
3. 为什么 prefill 产生第一个输出，8 个输出 token 只需要 7 次 decode？
4. 为什么 CUDA Graph 只出现在当前实验的 decode 路径？
5. 一条 sequence 的 waiting、running、finished 与 block table 怎样同步变化？
6. 哪个数字证明请求结束后没有活动 KV 引用，哪个数字只表示 prefix metadata 仍可复用？

这是一项真实 GPU 长任务：四个独立进程会分别加载模型，CUDA Graph 组还要先 capture；32 个消融 case
各运行一次 warmup 和五次 measurement。3070 Laptop 的实际耗时取决于功耗、散热、驱动和依赖构建，
因此本页不会预先承诺运行时间。

## 本实验使用的版本和参数

为了让实验结果可以复现，本实验使用以下版本和参数。它们也写在 Manifest 中，运行脚本会据此核对环境。

| 对象 | 本实验使用的版本或设置 |
|---|---|
| nano-vLLM | `GeeeekExplorer/nano-vllm@bb823b3e06983d71485a8e1f23715ebd87d98ef8` |
| 模型 | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| 报告 schema | `about-llm.nano-vllm-study.v1` |
| KV block size | 256 tokens |
| 输入 / 输出 | 768 synthetic token IDs / 8 sampled tokens |
| 对照 | eager/CUDA Graph、exact/drift、batch budget 256/1024、并发 1/2/4/8 |

加载模型以前，Collector 会核对 nano-vLLM 的 Git commit 和远端地址，还会确认源码目录没有本地改动。
随后，它逐个检查 config、tokenizer、generation config 和权重文件的大小与 SHA-256。只要版本或文件内容
与表中记录不同，脚本就会停止，并告诉你具体是哪一项不一致。源码目录里有未跟踪文件时也会停止。

推荐在 WSL 的独立环境准备依赖。安装时避免用 editable install 向被检查的源码目录写入构建产物：

~~~bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git ~/src/nano-vllm
git -C ~/src/nano-vllm checkout --detach \
  bb823b3e06983d71485a8e1f23715ebd87d98ef8
git -C ~/src/nano-vllm status --short

hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir ~/models/Qwen3-0.6B-c1899de
~~~

依赖安装以这个 nano-vLLM 版本的 `pyproject.toml` 为准。先确认 WSL 中的 `nvidia-smi`、PyTorch CUDA、
FlashAttention 和 Triton 能工作，再启动长实验。Windows CPU Python 不能代替这项检查。

## 先看一次请求的源码路线

nano-vLLM 的公开 `LLM` 直接继承 `LLMEngine`。普通 `generate()` 先逐条调用 `add_request()`，
再循环调用 `step()`，直到 scheduler 中既没有 waiting，也没有 running sequence：

```text
LLM.generate
  -> LLMEngine.add_request
     -> Sequence(status=WAITING)
     -> Scheduler.add
  -> LLMEngine.step
     -> Scheduler.schedule
     -> ModelRunner.call("run", ...)
        -> prepare_prefill / prepare_decode
        -> Qwen3ForCausalLM
        -> Sampler
     -> Scheduler.postprocess
  -> tokenizer.decode
```

Collector 不复制一套 scheduler，也不伪造 model output。它临时包装 `Scheduler.schedule()` 读取
本轮选择与 KV 账本，然后仍调用 upstream `LLMEngine.step()` 完成真实 model runner、sampler 和
postprocess；step 返回后再记录状态。这样 trace 与被测控制流来自同一次执行。

### 谁负责模型，谁负责推理运行时

Transformers 参与读取配置和 tokenizer；真正执行这次 forward 的模型类来自 nano-vLLM。

| 部件 | 在本实验中承担的职责 | 它不负责什么 |
|---|---|---|
| Transformers | `AutoConfig`、`AutoTokenizer`、Qwen3 config 类型 | `AutoModel.generate()` 在生成 |
| nano-vLLM | Sequence、Scheduler、BlockManager、ModelRunner、Qwen3 模型实现和 Sampler | 官方 vLLM 的精简配置 |
| Qwen3 checkpoint | config、tokenizer 与 safetensors 权重 | 自带 scheduler 或 CUDA kernel |
| PyTorch | Module/tensor、CUDA allocator、distributed、`torch.compile` 与 CUDA Graph API | 单独决定 batching policy |
| FlashAttention | varlen prefill 与带 KV cache 的 decode attention | 管理 sequence 生命周期 |
| Triton | 把本轮 K/V 写入 block slot 的自定义 kernel | 完整实现所有 attention |
| xxhash | 计算链式 token-block cache key；命中后仍比较 token IDs | 权限校验或密码学认证 |
| NCCL | 初始化 tensor-parallel process group；本实验 world size 为 1 | 已执行多卡通信实验 |

这里使用的 Qwen3-0.6B 是 text-only dense decoder。nano-vLLM 在这里实现 RMSNorm、RoPE、GQA、
gated SiLU MLP 和 causal LM head；它不需要多模态 processor，也没有运行视觉 encoder。

## 768 个 prompt token 怎样进入三个 block

Block size 是 256，所以 prompt 的逻辑 block 是：

```text
block 0: token positions   0..255
block 1: token positions 256..511
block 2: token positions 512..767
```

每个样本先运行一条不计入性能的 primer。Primer 完成后，活动引用已经释放，但完整 block 的 hash/token
metadata 可以留在 free blocks 中供 prefix lookup 使用。

这个版本的 `can_allocate()` 只扫描 `num_blocks - 1` 个 prefix blocks，不把 sequence 的最后一个 block
作为可复用前缀。因此，对于这组 768-token 输入，后来的请求应该命中两个 block，而不是三个。

### Exact prefix

测量请求与 primer 的 768 个 token 全部相同：

```text
命中 block 0、1 -> cached_tokens=512 -> 还需 prefill 256 tokens
```

当 `max_num_batched_tokens=256` 时，单请求一次 prefill 正好完成 prompt，并从最后位置的 logits
采样第一个输出 token。

### One-token drift

负例只修改 position 256，也就是第二个 block 的第一个 token：

```text
block 0 相同 -> 命中
block 1 token 不同 -> miss
后续 hash 依赖前一个 block hash -> 后面的链也不能继续命中
cached_tokens=256 -> 还需 prefill 512 tokens
```

在 256-token budget 下，第一次 `chunked_prefill` 只把 cached frontier 从 256 推到 512。
Model Runner 虽然产生了 logits 和 sampled token，`Scheduler.postprocess()` 发现 prompt 尚未完成，会丢弃这次
sample，不把它计作用户输出。第二个 prefill chunk 完成 prompt 后，首 token 才真正 commit。

这就是报告中 `phase=chunked_prefill` 与 `committed_token_count=0` 必须同时出现的原因。

## Prefill 之后为什么还要七次 decode

Prefill 完成 768 个 prompt positions，并生成输出 `y1`。此时缓存里只有 prompt 的 K/V，`y1` 自己的 K/V
尚未写入。下一轮 decode 输入 `y1`、写入它的 K/V，再采样 `y2`。

```text
prefill(prompt 768) -> y1
decode(y1)          -> y2
decode(y2)          -> y3
...
decode(y7)          -> y8 -> FINISHED
```

因此输出 8 个 token 对应 1 次完成 prompt 的 prefill 和 7 次 decode。Prompt 原本恰好占满三个 block；
append `y1` 后逻辑长度变为 769。第一次 decode 调度发现需要新的尾块，才分配第四个物理 block。

报告每个 step 保存三份 KV 快照：schedule 前、schedule 后、postprocess 后。重点核对：

```text
free_blocks + used_blocks == total_blocks
sum(block.ref_count) == 所有活动 block-table references
完成后 used_blocks == 0
完成后 ref_count_total == 0
```

`cached_hash_entries` 在完成后可以大于 0。那表示 free block 仍带有可验证的 prefix metadata，
不表示仍有 sequence 持有 KV lease，也不表示显存已归还给 CUDA driver。

## Sequence 状态怎样变化

把一条 drift + 256 budget 的请求压缩成下面四行：

| 时刻 | status | cached tokens | scheduled tokens | 发生了什么 |
|---|---|---:|---:|---|
| add_request 后 | waiting | 0 | 0 | 尚未分配 block table |
| 第一次 chunk 后 | waiting | 512 | 0 | 已有 block table，但 prompt 未完成 |
| 第二次 prefill 后 | running | 768 | 0 | 首 token 已 commit，可进入 decode |
| 第七次 decode 后 | finished | 0 | 0 | 达到 max tokens，block table 已清空 |

这里的 `Sequence.is_prefill` 是另一个字段：它在首次 decode 被调度时才切成 false。
判断当前 model runner 路径时，需要同时查看 status、本轮 `is_prefill` 和 scheduled work。

## CUDA Graph 究竟覆盖哪一段

这个版本的 `ModelRunner.run_model()` 按下面的条件选择执行路径：

```text
is_prefill == true       -> eager model call
enforce_eager == true    -> eager model call
decode batch > 512       -> eager model call
其他 decode              -> captured CUDA Graph replay
```

本实验最多并发 8，所以 `cuda_graph` 组的 decode 符合 replay 条件；prefill 无论在哪一组都走 eager。
`enforce_eager=True` 只控制这里的 model CUDA Graph 分支，不表示程序完全禁用了所有编译：本实验使用的 Sampler
仍带有 `torch.compile`。

CUDA Graph capture 发生在 engine 初始化，不在 measurement window 内；但 capture 后保留的 graph pool、模型、
KV arena 和 allocator reserved memory 仍会影响测得的显存基线。`reset_peak_memory_stats()` 只重置峰值统计，
不会把这些已经保留的显存变成零。

## 运行完整消融

从 about-llm 仓库根目录执行：

~~~bash
python projects/inference-serving/nano_vllm_study.py collect \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --source-root ~/src/nano-vllm \
  --model-snapshot ~/models/Qwen3-0.6B-c1899de \
  --output artifacts/inference/nano-vllm-study.json
~~~

Collector 为 `(eager|cuda_graph) × (256|1024)` 启动四个独立子进程，避免前一组 graph、allocator、
prefix cache 或 process group 污染后一组。每个 case 内运行一次 warmup 和五次 measurement，并保留原始值、
中位数与失败终态。

运行完成后先离线验证：

~~~bash
python projects/inference-serving/nano_vllm_study.py verify \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --report artifacts/inference/nano-vllm-study.json
~~~

Verifier 不需要 GPU。它会重新检查 JSON 结构、版本信息、时间顺序、性能指标计算和 KV block 数量。
例如，报告中出现重复字段、`NaN/Infinity`、未知字段、与 Manifest 不同的版本、倒退的时间戳、
超出预算的 prefill，或者一次 decode 调度了多个 token，验证都会失败。

报告里的 fingerprint 是内容摘要：内容改变时摘要也会改变，所以它可以帮助发现报告是否被修改。
它没有使用密钥，无法证明报告由谁生成。

## 怎样读四组消融，而不是只找最快数字

| 对照 | 先看机制字段 | 再看指标 | 可以提出的解释 |
|---|---|---|---|
| eager vs CUDA Graph | decode 的 `execution_path` | TPOT、吞吐、reserved memory | replay 是否减少当前小 batch 的 launch overhead |
| exact vs drift | `prefix_hits`、prefill scheduled tokens | TTFT、prefill steps | 少算了多少 prompt positions |
| budget 256 vs 1024 | chunk 数与每步 scheduled tokens | TTFT、吞吐、峰值显存 | chunking 与 batch packing 如何改变 |
| concurrency 1/2/4/8 | 每步 sequence 数、block refs | 吞吐、TTFT、TPOT、失败 | batching 收益何时被队列或容量反噬 |

这些方向都不是预先保证。一次 warmup 加五个样本不足以建立普适置信区间；Laptop 的 Dynamic Boost、温度、
后台进程和降频都可能改变结果。先保留原始值和失败，再决定是否增加重复、随机化运行顺序或记录功耗温度。

这里的 TTFT 从 measured request 加入 engine 到首 token 在 postprocess 中 commit；E2E 到 sequence finished；
TPOT 使用首 token 之后的七个 token 间隔。它们不包含 HTTP、网关、tokenization、网络或客户端排队，
不能与在线服务指标不加说明地放在同一列。

## 实验记录模板

```text
运行环境：source/model revision，Python/Torch/CUDA/driver，GPU
预测：exact/drift 各命中几块，256/1024 各需几次 prefill
单请求 trace：waiting/running/finished，cached/scheduled token，KV before/after
消融：五个原始样本、中位数、失败终态
解释：变化对应哪个 scheduler/block/model-runner 分支
边界：没有证明 HTTP SLO、模型质量、跨引擎排名或另一台 3070 的性能
```

报告通过验证并完成脱敏审查后，我们才会把其中数字记录为本仓库的 3070 实测结果。
在这之前，本页只介绍运行脚本、报告格式和预期观察到的机制，不提供任何 3070 性能数字。

## 自测

1. 为什么 768-token primer 留下三个 hash entry，却只允许后续请求命中两个 prefix blocks？
2. Drift 发生在 position 256 时，为什么不是“只多算一个 token”？
3. Chunked prefill 中 sampler 已运行，为什么 `committed_token_count` 仍是 0？
4. `used_blocks=0`、`cached_hash_entries>0` 分别说明什么？
5. 为什么 CUDA Graph 对照必须与 eager 放在独立进程，而不能只切一个布尔值后继续复用同一 engine？
6. 报告中的 engine TTFT 为什么不能直接当成 HTTP 用户 TTFT？
