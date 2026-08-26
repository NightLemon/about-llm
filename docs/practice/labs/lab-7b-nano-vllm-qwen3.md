# 实验 7B：Qwen3-0.6B 如何穿过 nano-vLLM

这个实验使用你正在学习的 `Qwen3-0.6B + nano-vLLM + RTX 3070 Laptop`。我们跟随一次 768-token 请求，
把源码里的类名还原成一条可以观察的推理链路。

请求刚加入队列时处于等待（waiting）状态；被调度后进入运行（running），生成完 8 个 token 后变成完成（finished）。

沿途还会看到它怎样复用 256-token 的 KV block，怎样经历预填充（prefill）与逐 token 解码（decode），以及完成后
怎样释放所有活动引用。

**相关教材**：[请求生命周期](../../systems/inference-request-lifecycle.md) ·
[Qwen3 tokenizer 实验](../labs.md#lab-1b) ·
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

这是一项真实 GPU 长任务。脚本会启动四个独立引擎进程；使用 CUDA Graph 的进程还要先完成图捕获。
四组配置一共展开为 32 个对照 case，每个 case 先预热一次，再测量五次。

3070 Laptop 的实际耗时会受到功耗、散热、驱动和依赖构建影响，所以只能在你的机器上实测。

## 本实验使用的版本和参数

为了让实验结果可以复现，本实验使用以下版本和参数。它们也写在实验清单（Manifest）中，运行脚本会据此核对环境。

| 对象 | 本实验使用的版本或设置 |
|---|---|
| nano-vLLM | `GeeeekExplorer/nano-vllm@bb823b3e06983d71485a8e1f23715ebd87d98ef8` |
| 模型 | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| 报告 schema | `about-llm.nano-vllm-study.v1` |
| KV block size | 256 tokens |
| 输入 / 输出 | 768 synthetic token IDs / 8 sampled tokens |
| 对照 | eager/CUDA Graph、exact/drift、batch budget 256/1024、并发 1/2/4/8 |

这里的 768 个 ID 由实验程序固定生成。它与页面开头的中文请求是两组不同输入。

长实验开始前，先运行[实验 1B](../labs.md#lab-1b)，亲眼看一次真实 message 怎样变成 Qwen3 的 29 个输入 ID。
回到本页后，只研究输入长度、前缀变化、调度和 KV block。前一个实验解释文本编码，本实验解释 runtime 执行。

加载模型以前，收集程序先核对 nano-vLLM 的 Git commit 和远端地址，并确认源码目录没有本地修改或未跟踪文件。

接着，它逐个检查模型配置、tokenizer、生成配置和权重文件的大小与 SHA-256 摘要。
只要版本或文件内容与表中记录不同，程序就会停止，并指出不一致的项目。

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

依赖安装以这个 nano-vLLM 版本的 `pyproject.toml` 为准。在 WSL 中依次确认：

- `nvidia-smi` 能看到目标 GPU；
- PyTorch 可以访问 CUDA；
- FlashAttention 可以导入；
- Triton kernel 可以执行。

Windows 下的 CPU Python 不能替代这次 GPU 预检。

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

收集程序没有重写一套 scheduler，也没有伪造模型输出。它只临时包装 `Scheduler.schedule()`，读取本轮调度结果和
KV 账本；真正的计算仍由原版 `LLMEngine.step()` 完成。

一次 step 返回后，程序再记录 model runner、sampler 和 postprocess 留下的状态。因此，报告中的 trace 和模型输出
来自同一次真实执行。

### 谁负责模型，谁负责推理运行时

Transformers 参与读取配置和 tokenizer；真正执行这次 forward 的模型类来自 nano-vLLM。

| 部件 | 在本实验中承担的职责 | 它不负责什么 |
|---|---|---|
| Transformers | 读取 `AutoConfig`、`AutoTokenizer` 和 Qwen3 配置类型 | 调用 `AutoModel.generate()` 生成 |
| nano-vLLM | 实现 Sequence、Scheduler、BlockManager、ModelRunner、Qwen3 模型和 Sampler | 充当官方 vLLM 的完整替代品 |
| Qwen3 checkpoint | 提供 config、tokenizer 和 safetensors 权重 | 自带 scheduler 或 CUDA kernel |
| PyTorch | 提供 Module、tensor、CUDA allocator、distributed、`torch.compile` 和 CUDA Graph API | 决定怎样组 batch |
| FlashAttention | 执行变长 prefill 和带 KV Cache 的 decode attention | 管理 sequence 生命周期 |
| Triton | 用自定义 kernel 把本轮 K/V 写入 block slot | 实现全部 attention 逻辑 |
| xxhash | 计算链式 token-block cache key；命中后仍比较 token ID | 做权限校验或密码学认证 |
| NCCL | 初始化 tensor parallel 进程组；本实验 world size 为 1 | 证明多卡通信已经测试 |

这里使用的 Qwen3-0.6B 是纯文本稠密解码器（text-only dense decoder）。它需要的主要模型部件由 nano-vLLM 自己实现：

- RMSNorm；
- 旋转位置编码（RoPE）；
- 分组查询注意力（GQA）；
- gated SiLU MLP；
- 因果语言模型输出头。

这条模型路径不需要多模态 processor，也不会启动视觉 encoder。

## 768 个 prompt token 怎样进入三个 block

Block size 是 256，所以 prompt 的逻辑 block 是：

```text
block 0: token positions   0..255
block 1: token positions 256..511
block 2: token positions 512..767
```

每次测量前，脚本先运行一条不计入性能的预热请求（primer），用它填充前缀缓存。
Primer 完成后，活动引用已经释放；完整 block 的哈希和 token 元数据仍可留在空闲 block 中，供后续前缀查找。

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

当调度预算只有 256 token 时，第一次 `chunked_prefill` 只能把已缓存边界从 256 推到 512。

Model Runner 仍然会算出 logits，并采样一个候选 token。随后，`Scheduler.postprocess()` 发现 prompt 尚未完成，
便丢弃这个中间结果，不把它计入用户输出。

第二个 prefill chunk 完成剩余 prompt 后，首个输出 token 才真正提交。

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

因此，8 个输出 token 对应 1 次完成 prompt 的 prefill 和 7 次 decode。

Prompt 原本恰好占满三个 block。追加 `y1` 后，逻辑长度变为 769；第一次 decode 调度这才发现需要一个新的尾块，
并分配第四个物理 block。

报告每个 step 保存三份 KV 快照：schedule 前、schedule 后、postprocess 后。重点核对：

```text
free_blocks + used_blocks == total_blocks
sum(block.ref_count) == 所有活动 block-table references
完成后 used_blocks == 0
完成后 ref_count_total == 0
```

请求完成后，`cached_hash_entries` 仍可能大于 0。这表示空闲 block 还保留着可验证的前缀元数据，后续请求可以复用。

判断活动引用是否清零，要看 `used_blocks` 和 `ref_count_total`。这些计数清零后，KV arena 本身仍由引擎持有，
显存也没有因此归还给 CUDA driver。

## Sequence 状态怎样变化

把一条 drift + 256 budget 的请求压缩成下面四行：

| 时刻 | 状态 | 已缓存 token | 本轮调度 token | 发生了什么 |
|---|---|---:|---:|---|
| add_request 后 | waiting | 0 | 0 | 尚未分配 block table |
| 第一次 chunk 后 | waiting | 512 | 0 | 已有 block table，但 prompt 未完成 |
| 第二次 prefill 后 | running | 768 | 0 | 首 token 已 commit，可进入 decode |
| 第七次 decode 后 | finished | 0 | 0 | 达到 max tokens，block table 已清空 |

`Sequence.is_prefill` 是另一个状态字段，首次调度 decode 时才会变成 false。
判断 Model Runner 当前走哪条路径，要同时查看 sequence 状态、本轮 `is_prefill` 和实际调度工作量。

## CUDA Graph 究竟覆盖哪一段

这个版本的 `ModelRunner.run_model()` 按下面的条件选择执行路径：

```text
is_prefill == true       -> eager model call
enforce_eager == true    -> eager model call
decode batch > 512       -> eager model call
其他 decode              -> captured CUDA Graph replay
```

本实验最多并发 8，请求数没有超过图回放的 batch 上限。因此，`cuda_graph` 组的 decode 会使用已捕获图；
prefill 在所有组里都按普通 eager 路径执行。

`enforce_eager=True` 只关闭 Model Runner 的 CUDA Graph 分支，并不等于关闭进程中的全部编译。
例如，这个版本的 Sampler 仍使用 `torch.compile`。

CUDA Graph 在引擎初始化时捕获，捕获耗时不计入单次测量。捕获后保留的 graph pool、模型、KV arena 和 allocator
预留内存仍构成测量开始时的显存基线。

`reset_peak_memory_stats()` 只重置峰值计数器，不会释放已经保留的显存。

## 运行完整消融

从 about-llm 仓库根目录执行：

~~~bash
python projects/inference-serving/nano_vllm_study.py collect \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --source-root ~/src/nano-vllm \
  --model-snapshot ~/models/Qwen3-0.6B-c1899de \
  --output artifacts/inference/nano-vllm-study.json
~~~

收集程序为 `(eager|cuda_graph) × (256|1024)` 启动四个独立子进程。这样，前一组留下的 CUDA Graph、allocator、
前缀缓存和进程组不会进入后一组。

每个 case 先预热一次，再测量五次。报告同时保留五个原始值、中位数和失败终态。

这四个进程只隔离“执行模式 × token 预算”。同一进程内，两个前缀版本和四档并发仍按固定顺序复用一个引擎，
所以 allocator 和前缀缓存的历史会延续到后面的 case。若两组结果只差很小，先把它视为可能的顺序效应；
更严格的比较还需要逐 case 进程隔离或随机化运行顺序。

运行完成后先离线验证：

~~~bash
python projects/inference-serving/nano_vllm_study.py verify \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --report artifacts/inference/nano-vllm-study.json
~~~

验证程序不需要 GPU。它会重新检查 JSON 结构、版本、时间顺序、性能指标计算和 KV block 账本。

下面这些情况都会让验证失败：重复或未知字段、`NaN/Infinity`、与 Manifest 不同的版本、倒退的时间戳、
超出预算的 prefill，以及单条 sequence 在一次 decode 中调度多个 token。

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

本实验使用以下计时边界：

- **首 token 延迟（TTFT）**：从请求加入 engine，到首个输出在 postprocess 中提交；
- **端到端延迟（E2E）**：从请求加入 engine，到 sequence 完成；
- **每个输出 token 时间（TPOT）**：首 token 之后七个生成间隔的平均值。

这三个指标都不包含 HTTP、网关、tokenization、网络和客户端排队。因此，它们不能直接和在线服务指标并列比较。

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
