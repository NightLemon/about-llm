# Transformers：离线机制与真实 checkpoint

## 从零训练 byte-level BPE

`train_byte_bpe.py` 不下载语料或模型，使用 `src/about_llm/from_scratch/tokenizer.py` 中的确定性 reference：基础词表为 256 个 raw byte；pair 频次只在每篇 document 内统计；同频时按 id pair 字典序打破平局；编码按已学习的 merge rank 执行。

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --vocab-size 280
python projects/transformers-basics/train_byte_bpe.py `
  --text "banana bandana" --text "banana" --sample "bandana"
~~~

输出包含实际词表大小、每条 merge 的 byte expansion、样本 token 数与 UTF-8 round-trip。它用于验证 BPE 机制，不包含 normalization、pre-tokenizer、special token、offset map 或 checkpoint chat template，不能替代真实模型 tokenizer，也不能由小型 authored corpus 推断多语言压缩率。

## 现代 attention correctness oracle

NumPy reference 不依赖 PyTorch kernel，提供 stable softmax、past-aware causal mask、RMSNorm、interleaved RoPE 与显式 K/V repeat 的 GQA。测试验证：fully masked row 会失败；RoPE 保持向量范数及共同 position shift 下的 Q/K dot product；GQA 等于对应的显式 MHA 展开；逐 token cache attention 等于完整 causal attention。

~~~powershell
python -m pytest tests/test_attention_numpy.py -q
python -m pytest tests/test_gpt_torch.py tests/test_gpt_jax.py -q
~~~

显式 repeat 和 float64 累积用于解释数学，不是生产 kernel；这些小数组测试不证明 FlashAttention/vLLM backend、目标 dtype、cache allocator、GPU 性能或三套完整模型逐层等价。

## MoE top-k、capacity 与 sparse combine oracle

~~~powershell
python projects/transformers-basics/moe_routing.py
python -m pytest tests/test_moe_routing.py -q
~~~

NumPy reference 对 `[tokens, experts]` logits 做稳定 softmax 与 deterministic top-k，padding token 不计 capacity；per-expert capacity 为 `ceil(factor * active_tokens * top_k / experts)`。同分 expert id 小者优先，expert 内按 probability 降序、token/rank 升序保留。输出同时保存 pre/post capacity counts、dropped assignment、整 token drop、gate weights、router entropy、z-loss 和本仓库明确命名的 generalized balance diagnostic。

固定 4-token/3-expert/top-2 fixture 的 capacity=2，count 从 `(3,4,1)` 变为 `(2,2,1)`，8 个 assignment 丢 3 个但没有整 token 全丢。脚本还真实执行 kept assignment 的 bias-free linear expert 与 weighted combine。它没有训练 router/MLP，不做 backward、all-to-all 或 GPU kernel，也不复现 DeepSeek/Qwen checkpoint；auxiliary loss、capacity group、drop/reroute 和归一化语义必须按目标实现重建。

## Activation patching 因果控制实验

`activation_patching.py` 在固定 seed 的随机两层 MiniGPT 上运行真实 forward hooks：缓存第 0 层 post-residual tensor，把 clean activation 按 batch/position patch 到 corrupted forward，并报告 raw logit difference 与不裁剪的 normalized recovery。

~~~powershell
python projects/transformers-basics/activation_patching.py
python -m pytest tests/test_activation_patching.py -q
~~~

fixture 同时包含正干预、联合 causal-prefix patch 和未来位置负对照；测试还检查 hook 移除、detach/clone、shape/device/token 边界与小分母拒绝。联合 patch 的 recovery=1、未来位置 control=0，证明这个固定计算图中的干预管线和 causal visibility 按定义工作。

模型没有训练，token 27/19 是根据当前 fixture 的 clean-corrupt 差异事后选择，batch 也只有 1；因此结果不能解释成语言机制、目标 checkpoint circuit、跨 prompt 稳定性或安全解释。完整 prefix activation 被替换后恢复 clean metric 也不惊人：它一次带入该位置的全部纠缠特征。真正研究必须先固定行为和 metric，再扩展多样本、模板、seed、随机 source、无关 site、component/path patch 与 held-out replication。

## 离线 smoke test

smoke_tiny.py 不下载任何模型。它从 GPT2Config 建立微型模型，在固定 batch 上训练 12 步并执行 greedy generation，用于验证：

- config 到模型的映射；
- labels 触发 causal LM shift/loss；
- optimizer 与 train/eval 状态；
- generate 的输入输出 shape；
- 参数量和纯权重存储量。

~~~powershell
python -m pip install "transformers>=4.48,<5"
python projects/transformers-basics/smoke_tiny.py
~~~

纯参数存储不是实际训练或推理显存；后者还包含梯度、optimizer、激活、KV cache 和工作区。

## 检查真实模型

`inspect_checkpoint.py` 只下载 config、tokenizer 与可用的 generation config，不加载权重。命令行强制填写 revision；为了真正固定内容，应传完整 immutable commit hash，而不是仍可能移动的 branch/tag：

~~~powershell
python projects/transformers-basics/inspect_checkpoint.py Qwen/Qwen2.5-0.5B-Instruct --revision <commit-hash>
~~~

脚本输出 requested revision、Transformers 在 config 上记录的 resolved commit metadata、`AutoConfig.to_dict()` 规范化快照的 canonical fingerprint、保守归一化的 attention/MoE/MLA contract，以及 tokenizer 模板的示例文本与模板直接产生的 token IDs。它还尝试读取 `GenerationConfig`，逐项比较 tokenizer、model config、generation config 的 BOS/EOS/PAD/decoder-start ID，并标出集合 disjoint、tokenizer/model vocab 越界和同时存在的 `max_length`/`max_new_tokens`。config 或 generation snapshot 可能包含库补出的默认值和运行时 metadata，不是远端原始 JSON bytes 的哈希。

Base tokenizer 没有 chat template、checkpoint 没有独立 generation config 都可能是合法状态；前者输出模板字段 null，后者输出 `unavailable_or_load_error`，这个状态也可能表示认证/网络/缓存错误，不能武断写成“文件不存在”。脚本不会猜模板或有效 generation defaults，也不会把渲染后的字符串再次 `encode`，从而避免重复添加 special tokens。

默认 `trust_remote_code=False`。若某 checkpoint 必须执行远程代码，先审查对应 revision，再在隔离环境显式开启。resolved commit 是库对象暴露的 metadata，不是签名或来源认证；三方 special-token 完全一致也不证明 `generate()` kwargs、模型类 fallback、vLLM/provider defaults、stop-string tokenization 或实际停止行为一致。config/tokenizer/generation config 成功读取仍不证明权重匹配、许可、有效上下文、质量或 runtime 支持。

## 离线 config contract 与 KV 账本

`inspect_config.py` 对本地 JSON 做 strict load，拒绝 duplicate key、`NaN`/`Infinity` 和非 object 根节点。它只在 `num_hidden_layers`、`num_attention_heads`、`num_key_value_heads` 与 head dimension 足够明确，且没有已知 MLA marker 时，计算标准 dense K/V 的理想 tensor payload：

~~~powershell
python projects/transformers-basics/inspect_config.py `
  projects/transformers-basics/configs/standard-gqa.example.json `
  --tokens 4096 --batch-size 1 --element-bytes 2
python projects/transformers-basics/inspect_config.py `
  projects/transformers-basics/configs/mla-moe.example.json --tokens 4096
~~~

三个 `configs/*.example.json` 的 `model_type` 和 `architectures` 都以 `authored` 开头，是本仓库自编的公式回归 fixture，不是 Llama、Qwen、DeepSeek 或任何发布 checkpoint 的配置快照。标准 GQA fixture 的 4096-token、batch 1、2-byte element 结果为 536,870,912 bytes；MoE-GQA fixture 同条件、batch 2 为 402,653,184 bytes。MoE marker 不改变标准 attention 的 K/V 公式，但也不能据此推断 total/active parameters。MLA fixture 必须拒绝标准公式；它只测试 fail-closed 分支，不能反推出某个 MLA runtime 的真实 cache layout。

这些数字不含 allocator metadata、block 对齐、量化 scale、workspace、临时张量和权重，也不证明显存峰值或吞吐。`max_position_embeddings` 只是被记录的 config 字段，不是有效上下文或质量证据。

## 离线 generation protocol 对账

`inspect_generation_protocol.py` 对一个明确的三方快照做 strict JSON 与 token-ID 对账：

~~~powershell
python projects/transformers-basics/inspect_generation_protocol.py `
  projects/transformers-basics/protocols/aligned-superset-eos.example.json
python projects/transformers-basics/inspect_generation_protocol.py `
  projects/transformers-basics/protocols/drift-out-of-range.example.json
~~~

第一份 authored fixture 中 tokenizer/model EOS 为 `{2}`，generation EOS 为 `{2,3}`，因此只报告 strict subset，不判错：额外停止 token 可能是 checkpoint 的有意协议。第二份把 generation BOS/EOS/PAD 改成与另外两方 disjoint 的 `{4}/{5}/{9}`，其中 9 超出 tokenizer/model 的 8-token 空间，检查器必须逐项暴露。两份文件都不是任何发布模型快照，也没有执行 `generate()`；`PAD=EOS` 只会得到“可能有意”的 observation。

检查器不裁决谁正确，不推断 `max_new_tokens` 与 `max_length` 在目标版本的优先级，也不把 `do_sample`、beam/contrastive search 或 stop strings 拼成所谓“有效配置”。最终部署仍应显式传参，并对 Transformers/vLLM/provider 分别保存请求、token trace、finish reason 与版本。

## Transformers `generate()` 停止控制实验

静态对账之后运行真实框架控制：

~~~powershell
python projects/transformers-basics/generation_runtime_control.py
~~~

脚本在随机、未训练的 3,824-parameter tiny GPT-2 上真实执行 forward 与 `GenerationMixin.generate()`，但一个 authored `LogitsProcessor` 会把每步全部 next-token scores 覆盖成单个确定 token。因此权重不决定输出，三条路径可以精确审计：

1. GenerationConfig EOS `{2,3}`，计划 `[4,3]`，token 3 后停止；
2. 同一 config 在调用时传 `eos_token_id=5`，计划 `[3,5]`，token 3 不停止、token 5 停止；
3. Config 的 `max_new_tokens=5`，调用时传 2，计划 `[4,6]`，未遇 EOS 也恰好生成两个 token。

实验还验证 caller-owned `GenerationConfig` 没被 mutation。报告中的 finish reason 是根据受控 token plan 与 EOS/length 条件**推断**的；Transformers 的该返回对象没有 provider 风格 finish reason。它证明当前安装 Transformers 版本的这三条控制流，不使用真实 tokenizer/chat template，不加载公开 checkpoint，也不证明正常模型 logits、vLLM/provider precedence、质量、性能或 GPU 行为。

## 加载权重时的检查顺序

1. 阅读 model card、许可和用途限制；
2. 固定 model revision 与 tokenizer revision；
3. 核对 chat template、BOS/EOS/PAD 和 generation config；
4. 选择 torch_dtype，不用字符串“auto”掩盖实际 dtype；
5. 单张 GPU 先用 device_map 明确 placement；
6. 打印参数量、权重存储和实际峰值显存；
7. 用小型质量集确认量化/模板没有破坏输出；
8. 再做 batch、上下文和并发扫描。

## 家族差异

AutoModel 统一接口不代表模型协议相同。Llama、Qwen 和 DeepSeek checkpoint 可能在 GQA/MoE、RoPE、special tokens、tool template 和 generation config 上不同。GPT、Claude、Gemini 等云 API 也不能假设与 Transformers messages 一一等价。
