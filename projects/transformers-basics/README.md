# Transformers Basics：从 bytes 到真实 checkpoint

这个项目把语言模型最容易混在一起的四层拆开：从零实现的数学与算法、框架中的 tiny model、公开 checkpoint 的
配置与协议，以及真实权重运行。你可以先建立机制直觉，再判断 Transformers、模型仓库和推理 runtime 各自负责什么。

第一次学习请从[项目教学页](../../docs/practice/projects/transformers-basics.md)开始。它从原始字节和 token IDs 出发，
依次讲训练目标、注意力、微型模型、生成配置和 checkpoint。本页只保留运行入口、脚本索引和排错方法。

## 第一次运行

下面七条命令都不下载公开模型：

```powershell
python projects/transformers-basics/ticket_classification_walkthrough.py
python projects/transformers-basics/train_byte_bpe.py --vocab-size 280
python projects/transformers-basics/trace_language_model_sample.py
python projects/transformers-basics/online_softmax_demo.py
python projects/transformers-basics/trace_minigpt_training_step.py
python projects/transformers-basics/smoke_tiny.py
python projects/transformers-basics/generation_runtime_control.py
```

它们依次回答七个问题：

1. 数据泄漏、交叉熵、一次梯度更新和分指标评测怎样构成最小机器学习闭环？
2. Byte-level BPE 怎样从 256 个 byte 开始学习 merge？
3. 一段文字怎样变成 input IDs、labels、causal mask 和 loss mask？
4. Online softmax 为什么不需要保存完整 attention score matrix？
5. 同一个样本怎样经过 MiniGPT 的 forward、masked loss、backward 和参数更新？
6. Transformers 的 tiny GPT-2 能否完成训练与生成接线？
7. EOS、最大生成长度和调用参数覆盖怎样共同决定停止？

先解释输出，再进入真实 checkpoint。完整的 90 分钟 CPU 路线见[推荐运行顺序](../../docs/practice/projects/transformers-basics.md#run)。

## 从零实现、Transformers 与模型仓库的分工

| 层次 | 在这个项目中负责什么 | 不能替代什么 |
|---|---|---|
| NumPy/PyTorch/JAX 参考实现 | 暴露 token、mask、attention、cache 和 gradient 的计算 | 生产级 kernel、调度与服务 |
| Transformers | 加载 config/tokenizer/model，提供 forward 与 generation loop | 模型发布身份、业务评测与服务容量 |
| 模型仓库 snapshot | 提供固定 revision 的配置、tokenizer、模板和权重文件 | 运行时默认值、硬件兼容与部署策略 |
| vLLM/nano-vLLM 等 runtime | 调度请求、管理 KV Cache、执行高效推理 | Tokenizer/template 正确性和任务质量 |

同一个模型名称不足以固定实验。要复现真实运行，还需要绑定 checkpoint commit、tokenizer、chat template、
generation config、库版本和执行硬件。

## 根据当前问题选择脚本

| 你想理解什么 | 入口 |
|---|---|
| 数据切分、NLL、梯度方向和分类指标为什么必须一起看 | `ticket_classification_walkthrough.py` |
| Byte-level BPE 的计数、merge 与编码 | `train_byte_bpe.py` |
| 文本 token 怎样变成因果 LM 的逐位置训练目标 | `trace_language_model_sample.py` |
| Causal attention、mask、cache 与 online softmax | `online_softmax_demo.py` 和 `tests/test_attention_numpy.py` |
| 同一训练目标怎样产生 logits、NLL、梯度和参数更新 | `trace_minigpt_training_step.py` |
| Tiny Transformers 的训练与生成接线 | `smoke_tiny.py` |
| EOS、length cap 与 generation 参数优先级 | `generation_runtime_control.py`、`inspect_generation_protocol.py` |
| 本地 config 能否使用标准 GQA/KV 公式 | `inspect_config.py` |
| 一条中文对话怎样变成 Qwen3 的真实输入 IDs | `trace_qwen3_tokenizer.py` |
| 公开 checkpoint 的 config、tokenizer 与模板 | `inspect_checkpoint.py` |
| Llama、Qwen、DeepSeek 发布资料如何固定 | `verify_release_evidence.py` |
| 固定 Qwen 权重的 forward、KV cache 与 generate | `run_target_checkpoint.py` |
| 单个 Qwen weight 的 packed INT4 误差 | `run_qwen_weight_quantization_control.py` |
| Activation patching 能说明什么因果关系 | `activation_patching.py`、`run_qwen_activation_patching_control.py` |
| MoE top-k、capacity、drop 与 sparse combine | `moe_routing.py`、`moe_training_control.py` |
| Expert parallel 的 dispatch、return 与 backward | `moe_distributed_capacity_control.py`、`moe_all_to_all_*_control.py` |

MoE、activation patching 和真实权重量化是机制选修，不应阻塞第一次理解 Transformer。精确输入和运行结果见
[Transformers 证据页](../../docs/evidence/transformers-controls.md)，不要把多个独立实验拼成某个发布模型的完整能力结论。

## 追踪一条 Qwen3 对话输入

本地已经缓存固定版本时，运行：

```powershell
python projects/transformers-basics/trace_qwen3_tokenizer.py --local-files-only
```

模型放在单独目录时，可以传入 `--model-snapshot <path>`。第一次没有缓存时去掉 `--local-files-only`；脚本请求的仍是
Qwen3-0.6B 的完整 commit `c1899de289a04d12100db370d81485cdf75e47ca`。

默认输出从一条中文用户消息开始。程序先展示对话模板组成的文本，再列出 29 个 token IDs、可读片段和词表 token。

随后，它会比较“模板直接返回的 IDs”和“先渲染文本再编码的 IDs”。这一步只加载 tokenizer，不需要模型权重、
nano-vLLM 或 GPU。

加载后的类名是 `Qwen2TokenizerFast`，因为 Qwen3 复用了兼容的 tokenizer 实现。判断模型家族要看 checkpoint
配置和模型类，不能只看 tokenizer 的 Python 类名。

## 检查公开 checkpoint

先只读取 config、tokenizer 和可用的 generation config，不加载权重：

```powershell
python projects/transformers-basics/inspect_checkpoint.py `
  Qwen/Qwen2.5-0.5B-Instruct `
  --revision <full-commit-hash>
```

这里应填写完整 commit，而不是可能移动的 branch 或 tag。检查结果需要回答：

- 实际 model class 和 `model_type` 是什么；
- attention heads 与 KV heads 是否形成 GQA/MQA；
- tokenizer、模型和 generation config 的特殊 token 是否一致；
- chat template 是否存在，怎样序列化 role 和 tool；
- 当前 Transformers 版本能否直接加载，是否要求 remote code。

静态检查通过后，再决定是否下载和运行权重。仓库保存的 Qwen2.5 运行入口是：

```powershell
python projects/transformers-basics/run_target_checkpoint.py --local-files-only
```

它需要本机已有固定 snapshot。这个运行检查 forward、KV-cache 下一步和框架 `generate()` 是否对齐；单个 prompt
通过不代表总体质量、长上下文效果或 GPU 性能。

## Generation 是多方协议

生成行为至少由下面几部分共同决定：

```text
tokenizer special tokens
+ model config
+ generation config
+ generate() call arguments
+ logits processor / stopping criteria
+ serving runtime overrides
```

先用本地快照检查 token ID 关系：

```powershell
python projects/transformers-basics/inspect_generation_protocol.py `
  --snapshot projects/transformers-basics/protocols/aligned-superset-eos.example.json

python projects/transformers-basics/generation_runtime_control.py
```

静态 ID 一致只能证明配置关系；运行实验才会观察实际停止路径。换到 vLLM、云 API 或其他 runtime 时，需要重新确认
参数名、默认值、停止原因和 usage 口径。

## 主要输入与输出

| 文件或目录 | 用途 |
|---|---|
| `configs/*.json` | GQA、MoE 与 MLA 的本地架构配置样例 |
| `protocols/*.json` | Special token 与 generation 协议样例 |
| `release-evidence/*.json` | 固定发布资料、上游文件 hash 和预期字段 |
| `target-checkpoints/*.json` | 固定 Qwen snapshot 与录制报告 |
| `artifacts/`、`outputs/` | 本机生成的 tokenizer、配置快照和运行报告 |

配置样例只用于验证公式和检查器。名称中写有 Qwen、Llama 或 DeepSeek 的证据，只有在明确绑定官方 revision 和文件时，
才能支持对应模型的静态结论。

## 常见故障

| 现象 | 先检查 |
|---|---|
| BPE merge 顺序与预期不同 | 计数是否跨 document、同频 tie-break 和 merge rank 是否固定 |
| 模型看似在复制当前 token | Input 与 labels 是否正确错开一位，shift 是否重复或遗漏 |
| Attention 输出出现未来 token 信息 | Causal mask 的方向、广播形状和 softmax 维度 |
| Cached logits 与 full recompute 不一致 | Position IDs、mask、past length 和比较位置 |
| Loss 不下降 | Labels shift、padding mask、optimizer 参数和 `train()`/`eval()` 状态 |
| Generate 过早结束或不停止 | EOS 集合、PAD/BOS、length cap、调用参数覆盖和 template |
| Config 计算出的 KV 数字不合理 | 架构是否为标准 attention；MLA 等结构不能套用 GQA 公式 |
| Tokenizer 能渲染对话，但 SFT mask 全零 | Template 是否提供 assistant generation span |
| Recorded report 通过，新运行却不同 | Snapshot、库版本、dtype、设备、输入 token 和执行模式是否一致 |
| Remote code 才能加载 | 先审查固定 revision，在隔离环境显式决定是否信任 |

## 运行检查

核心算法与 tiny model：

```powershell
python -m pytest `
  tests/test_ticket_classification_walkthrough.py `
  tests/test_tokenizer.py `
  tests/test_language_model_sample.py `
  tests/test_qwen3_tokenizer_trace.py `
  tests/test_minigpt_training_trace.py `
  tests/test_attention_numpy.py `
  tests/test_gpt_torch.py `
  tests/test_gpt_jax.py `
  tests/test_generation_contract.py -q
```

Config、checkpoint 与机制实验：

```powershell
python -m pytest `
  tests/test_model_config.py `
  tests/test_model_release_evidence.py `
  tests/test_moe_routing.py `
  tests/test_moe_training.py `
  tests/test_activation_patching.py -q

python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

默认 CPU 检查只覆盖参考实现和微型模型。真实权重与 GPU kernel 需要单独运行。

完成这里后，可以进入 [Inference Serving 项目](../../docs/practice/projects/inference-serving.md)，把同一个 Qwen
请求放进 nano-vLLM 或 vLLM，继续观察调度和容量。
