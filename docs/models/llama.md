# Llama 家族

## 学习目标与证据边界

读完本章应能从 checkpoint 的配置而不是品牌名推断架构，解释 Llama 系模型中 RMSNorm、RoPE、SwiGLU 与 GQA 对训练/推理的影响，并能为一个固定 revision 设计单卡量化或 LoRA 实验。

**先修知识**：decoder-only Transformer、KV Cache、量化、LoRA/QLoRA、Hugging Face checkpoint 结构。

Llama 是公开权重生态的重要基线，但“Llama”不是一个固定架构。不同代际、尺寸、Base/Instruct 和多模态版本可能有不同词表、head 数、GQA 配置、上下文、RoPE scaling、chat template 与许可。所有具体结论以所选 checkpoint 的 `config.json`、tokenizer、generation config、model card、权重 revision 和许可为准。

## 从 checkpoint 读取架构

加载权重前先读取配置；至少记录：

| 配置 | 决定什么 | 常见误读 |
|---|---|---|
| `hidden_size` | residual stream 宽度 | 不等于 head dim 或 MLP 宽度 |
| `num_hidden_layers` | block 数与 KV 层数 | 不能只由参数量猜 |
| `num_attention_heads` | query heads | 不一定等于 KV heads |
| `num_key_value_heads` | MHA/GQA/MQA 与 KV 容量 | 忽略它会严重错估长上下文显存 |
| `intermediate_size` | MLP 中间宽度 | 门控 MLP 参数公式不同于普通两层 MLP |
| `vocab_size` | embedding/lm head 维度 | tokenizer 文件实际特殊 token 仍需检查 |
| `max_position_embeddings` | 配置允许的长度 | 不证明目标任务在该长度可靠 |
| `rope_*` | 旋转频率与扩展方式 | 不能跨代复制 scaling 配置 |
| `tie_word_embeddings` | 输入/输出权重是否共享 | 影响参数量、扩词表和 adapter 保存 |

本仓库 `inspect_checkpoint.py` 要求显式 revision，只读取 config/tokenizer/可用的 generation config，不加载大权重：

```powershell
python projects/transformers-basics/inspect_checkpoint.py <model-id> --revision <commit-hash>
```

应传完整 immutable commit hash；参数“必填”本身不能阻止用户传入可移动 branch/tag。输出的 normalized config/generation snapshots、requested/resolved revision metadata、模板 token IDs 与三方 special-token 对账应进入实验 manifest；不要只把模型短名写在报告标题里。snapshot fingerprint 可能绑定库默认值/metadata，并不是原始 JSON byte hash。该工具不加载权重，也不能证明有效 `generate()` kwargs、许可或 runtime kernel，因此这些仍需独立记录。

若要先离线练习，可运行：

```powershell
python projects/transformers-basics/inspect_config.py `
  projects/transformers-basics/configs/standard-gqa.example.json `
  --tokens 4096 --element-bytes 2
```

这个 `authored_standard_gqa` 配置是公式测试夹具，不是任何 Llama checkpoint。它得到的 536,870,912 bytes 只表示 32 层、8 KV heads、head dim 128、4096 tokens 下的 BF16/FP16 理想 K/V tensor payload；不能引用为“Llama 4K 显存”。

## 常见架构组件

### Pre-norm 与 RMSNorm

Llama 系公开模型常在子层前使用 RMSNorm：

\[
\operatorname{RMSNorm}(x)=g\odot
\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}
\]

它不减去均值，参数通常只有缩放向量。具体 epsilon、计算 dtype 和 norm 位置以 config/实现为准；它们会影响低精度稳定性和 checkpoint 兼容。

### SwiGLU

门控 MLP 常写为：

\[
\operatorname{MLP}(x)=W_d\left(\operatorname{SiLU}(W_gx)\odot W_ux\right)
\]

因此每层包含 gate/up/down 三个主要投影。LoRA 教程只改 `q_proj/v_proj` 与 all-linear 的容量、显存和质量不同，target modules 必须从实际模块名发现，不能从另一个模型机械复制。

### RoPE

RoPE 对 Q/K 的二维分量施加与位置相关的旋转，使点积编码相对位置信息。扩展长度可能使用插值、频率缩放或其他 rope 配置，但“模型能接受更多 token”不等于能在所有位置完成检索、计数和多跳综合。

上下文扩展评测至少覆盖：开头/中部/结尾 needle，多文档冲突，顺序判断，全局聚合和长输出一致性。

### GQA 与 KV Cache

若 query heads 为 \(H_q\)，KV heads 为 \(H_{kv}\)，每层 dense KV 的理想化字节约为：

\[
2\times H_{kv}\times d_{head}\times T\times bytes(dtype)
\]

再乘层数、batch/并发序列，并加入 block metadata、对齐与 runtime workspace。GQA 令多个 query heads 共享一组 K/V，减少 cache 与 decode 内存带宽；它不按同样比例减少 Q 投影或全部模型权重。

本仓库 config inspector 只在这些标准字段显式且自洽时计算；字段缺失或出现已知 MLA marker 就拒绝，而不是按家族名补默认值。即使计算成功，也只是 config-level deduction：未知的自定义代码、实际权重形状、量化 cache layout 或 runtime 转换仍可能改变实测结果。

## Base、Instruct 与聊天模板

Base 模型学习续写分布，适合继续预训练、研究和自定义后训练；Instruct 模型经过对话/偏好训练，依赖相应 chat template。常见错误包括：

- 给 Base 套 Instruct 模板并期待自动获得指令能力；
- 手写 `[INST]` 或 role token，而 checkpoint 使用不同代际模板；
- 训练和部署使用不同 BOS/EOS、system 或 generation prompt；
- tool schema 的序列化方式与训练模板不一致；
- tokenizer revision 漂移，但只固定 model revision。

上线前用 `tokenizer.apply_chat_template` 打印 token ids 与解码文本，检查单轮、多轮、system、tool result、空 assistant generation prompt 和停止 token。

## 参数量与内存账本

参数量不是显存。推理需要：权重 + KV Cache + 激活/临时张量 + kernel/runtime；训练还需要 adapter/梯度/优化器/保存激活。

4-bit 文件也不是每参数严格 0.5 byte 的完整运行时：group scale、zero/metadata、未量化层、反量化 workspace 和 allocator 都会增加占用。单卡选型应执行：

1. 读取 config 计算原始权重和 KV 理想值；
2. 选择目标 runtime 原生支持的量化格式；
3. 用最短序列和 micro-batch 1 dry-run；
4. 扫描真实输入/输出长度与并发；
5. 记录 allocated/reserved、峰值、tokens/s 和质量回归。

## LoRA/QLoRA 路线

实验至少固定 base revision、tokenizer/template、数据 manifest、target modules、rank/alpha、dtype/量化、sequence length 和 seed。比较：Base Prompt、Base+RAG、LoRA/QLoRA 与更高预算参考。

训练前必须检查实际 labels：system/user/padding/tool 是否按设计 mask，assistant 首尾 token 是否错位。合并 adapter 前后比较固定输入 logits；保存/重载后再比较，并重新跑领域、通用、安全和格式评测。

“可训练参数少”不代表激活少；长序列仍可能 OOM。降级顺序通常是 micro-batch 1、gradient checkpointing/高效 attention、按长度分布缩短序列、减少 target/rank，最后换小基座。每次改变都应创建新实验配置。

## Transformers 与 vLLM 交叉验证

Transformers 适合建立正确性基线、检查 logits/token/template；vLLM 适合连续批处理、PagedAttention 和 OpenAI-compatible 服务。比较时保持：

- 同一权重与 tokenizer revision；
- 同一 chat template 与 generation 参数；
- 同一量化语义，而不只看文件名；
- 同一输入/output budget；
- greedy token 或任务指标一致性容差。

若两边输出不同，先比较渲染后的 token ids、special tokens、generation defaults、dtype/量化和停止条件，再讨论 kernel 数值差异。

## 许可与供应链

“可下载权重”不自动等于 OSI 开源。不同 Llama 代际可能有独立社区许可、归因、可接受用途或规模条款。发布 adapter、合并权重、容器或托管服务前，核对具体版本许可和依赖模型条款。

权重、tokenizer 和 remote code 都是供应链输入。固定 commit/hash，优先使用 safetensors，默认 `trust_remote_code=False`；确需远程代码时先审查固定 revision，并在隔离环境运行。

## 可运行实验

选择一个许可证允许的小尺寸 Base 与 Instruct checkpoint：

1. 固定 revision，导出 config/tokenizer/template 摘要；
2. 比较同一文本在 Base/Instruct 模板下的 token 序列；
3. 计算权重与 1K/4K/8K token KV 理想容量；
4. 用 Transformers greedy 生成建立 token 基线；
5. 若有目标 GPU，用 4-bit/8-bit/BF16 比较任务质量、显存和速度；
6. 若有 vLLM，验证模板/token/停止条件并画并发—TTFT—吞吐 Pareto 曲线。

结果必须绑定 checkpoint 与硬件，不写“Llama 需要多少显存”这种无条件结论。

## 面试追问

1. RMSNorm 与 LayerNorm 在公式和工程上有何差异？
2. SwiGLU 为什么有三个主要投影，参数量如何估算？
3. GQA 减少哪部分 KV Cache，不能减少哪些计算？
4. RoPE 配置允许更长上下文为何不等于有效长上下文？
5. Base/Instruct 的训练目标、chat template 和部署接口怎样关联？
6. 如何证明量化或 adapter merge 没有造成不可接受回归？
7. “开放权重”和“开源软件”为什么不是同一个许可概念？

## 一手资料

- Meta，[Llama models repository](https://github.com/meta-llama/llama-models)，官方模型卡、prompt format 与许可入口。
- Touvron 等，[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)，早期 LLaMA 公开论文。
- Hugging Face，[Chat templates](https://huggingface.co/docs/transformers/en/chat_templating)，模板与 generation mask 机制。
- checkpoint 自带 `config.json`、tokenizer、model card 与 license；具体实验的最高优先级证据。
