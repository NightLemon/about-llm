# LoRA、QLoRA 与单卡工程：显存省在哪里，风险又留在哪里

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备在单张消费级 GPU 上做微调的开发者与算法工程师。
- **先修**：[微调总览](finetuning.md)、Transformer 线性层和基础训练循环。
- **首次阅读**：LoRA 更新 → Target modules → QLoRA → 显存账本 → Adapter 发布。
- **完成信号**：能解释一次 OOM 来自哪块内存，并让 Adapter 在全新 base load 上重现输出。
- **卡住时**：按[Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md#run)跑最小路径。

</div>

假设你想让一个 0.5B–1B 级 Instruct 模型更稳定地遵循团队的售后格式，但笔记本 GPU 容不下全参数训练。
LoRA 的思路是冻结原模型，只学习一组小更新；QLoRA 再把冻结基座压到低比特存储。

这两项技术能显著降低一部分显存，却不会自动解决数据、labels、activation、checkpoint 和评测问题。

## LoRA 实际给线性层加了什么

冻结基座权重 (W_0\in\mathbb R^{d_{out}\times d_{in}})，只训练低秩增量：

\[
y=xW_0^\top + \frac{\alpha}{r}xA^\top B^\top,
\quad A\in\mathbb R^{r\times d_{in}},
\quad B\in\mathbb R^{d_{out}\times r}.
\]

原矩阵有 (d_{out}d_{in}) 个参数，Adapter 只有 (r(d_{in}+d_{out})) 个。常见初始化让 (B=0)、(A) 随机，
于是训练开始时 LoRA 分支输出为 0，模型函数与基座相同。

`alpha/r` 控制增量尺度。Rank 提高会增加可表达方向，也增加参数、optimizer state 和过拟合机会；它不是“越大
越好”的质量旋钮。

推理时可以保留独立 Adapter，也可以合并：

\[
W'=W_0+\frac{\alpha}{r}BA.
\]

Merge 前后要在固定输入上比较 logits 或生成结果。数学公式相同，不代表 dtype、量化顺序和 tied weights 的具体实现
不会引入差异。

## Target modules 决定模型哪里能改变

不同 checkpoint 对同一功能使用不同模块名。先打印实际 model tree 和匹配结果，再决定 target list：

| 选择 | 训练容量与常见用途 | 代价 |
|---|---|---|
| `q_proj/v_proj` | 轻量行为或格式适配 | 可训练方向较少 |
| `q/k/v/o` | 更完整地调整 attention | 参数与 optimizer state 增加 |
| Attention + MLP linears | 更强领域/行为适配 | 显存、训练时间和过拟合风险更高 |
| Embedding/LM head | 新增 token 或输出头变化 | 保存、weight tying 与兼容更复杂 |

`trainable params = 0` 应立即失败，不能继续跑一个看似正常的训练循环。若新增 special token 却冻结 embedding 与
LM head 的相关行，模型也很难学会新 token。

Target modules 是实验变量。比较两种设置时固定 data、template、有效 token batch、更新步数与评测集。

## QLoRA 压缩的是冻结基座

QLoRA 把 base weights 以 4-bit 存储，矩阵计算时再反量化到 BF16/FP16；LoRA Adapter、gradients、optimizer states、
activations 和部分 kernels 仍使用更高精度。

经典组件包括：

- NF4：为近似正态权重分布设计的量化 levels；
- Double quantization：继续压缩量化 scales；
- Paged optimizer：在内存压力下管理 optimizer state；
- `prepare_model_for_kbit_training`：处理部分 norm、input gradient 与 checkpointing 配置。

这些名字不代表任意 GPU/runtime 都走到高效 kernel。Bitsandbytes、CUDA、driver、compute capability 与模型 shape
必须在目标设备验证。

支持 BF16 的硬件通常优先使用 BF16 compute；FP16 要额外关注 overflow 与 GradScaler。不要只看 `load_in_4bit=True`
就说“整个训练是 4-bit”。

## 显存账本要逐项写

训练峰值可按下面的结构理解：

\[
M=M_{base}+M_{adapter}+M_{grad}+M_{optimizer}
+M_{activation}+M_{workspace}+M_{runtime}.
\]

其中：

- Base：量化 codes、scales、未量化层与配置 buffers；
- Adapter/grad/optimizer：只随可训练参数增长，但 AdamW 通常有两组 moments；
- Activation：随 batch、sequence length、hidden size、层数和 checkpointing 改变；
- Workspace：attention、dequantization、temporary logits 与 fused kernels；
- Runtime：CUDA context、allocator reserve、compile/graph cache 和其他进程。

“7B × 4 bit ≈ 3.5 GB”只算了权重 codes 的理想 payload，不能直接当成 QLoRA 峰值。

仓库估算器适合先筛掉明显不可行的配置：

```powershell
python projects/single-gpu-finetuning/train_qlora.py `
  --model-id illustrative/model `
  --revision immutable-commit-placeholder `
  --num-parameters 7000000000 `
  --num-layers 32 `
  --hidden-size 4096 `
  --max-length 1024 `
  --micro-batch-size 1 `
  --gradient-accumulation 16 `
  --rank 16 `
  --alpha 32 `
  --target-modules q_proj,k_proj,v_proj,o_proj `
  --target-linears-per-layer 4 `
  --estimate-only
```

最终仍要在目标 GPU 记录 `torch.cuda.max_memory_allocated()`、`max_memory_reserved()`、step time 与实际 module tree。
Reserved − allocated 可能来自 allocator cache/碎片，不等同于内存泄漏。

## OOM 时按什么顺序降级

一次只改一项：

1. Micro-batch 降到 1；
2. 打开 gradient checkpointing，以额外计算换 activation；
3. 使用兼容的 SDPA/FlashAttention，并核对没有 fallback；
4. 按真实长度分布降低 max sequence 或做 length bucketing；
5. 减少 target modules 或 rank；
6. 换更小的 base model。

`empty_cache()` 释放的是 allocator 中未被 tensor 引用的缓存，无法消除仍存活的 tensor，也不会改变真实峰值。
每次降级都新建 run identity，因为它可能改变速度、容量甚至训练语义。

## Gradient accumulation 不只是“假装大 batch”

若两个 micro-batches 的有效监督 token 数不同，直接平均两个 batch mean 会让短 batch 的 token 权重更大。
Token-mean 目标应累计 loss sum 与有效 token count，再按整个 optimizer window 的分母缩放。

即使有效 token 总数相同，gradient accumulation 与一次大 batch 仍可能因 dropout、batch-dependent operation、
gradient clipping、scheduler step 和浮点归约顺序而不同。

记录 `tokens/update` 比 `samples/update` 更有意义，特别是变长对话和 packing 场景。最后一个不足完整 accumulation
window 的 batch 也要明确定义分母与是否执行 step。

公式和 DDP 对照见[分布式训练](../systems/distributed-training.md#global-batch-loss-normalization)。

## Gradient checkpointing 用时间换 activation

Checkpointing 不保存部分前向中间量，在 backward 时重算。它降低 activation memory，代价是额外计算。

训练时通常关闭 `use_cache`，因为 KV Cache 为自回归推理服务，与 activation recomputation 的目标不同。
验收时查看实际 peak VRAM 与 step time，不要只检查 config 中的布尔值。

## Labels 错了，PEFT 只会更便宜地学错

在 backward 前打印目标 tokenizer 的：

```text
rendered conversation
input_ids
assistant mask
final labels after collator
valid supervised token count
```

Chat template 会加入 system、tool 和 special tokens，不能按原始字符串位置猜 mask。System/user/padding 与不希望监督的
控制 token 应为 `-100`；目标 assistant tokens 的 label 才等于 input token ID。

仓库的 Qwen SFT 验证程序演示了一个真实故障：原生 template 没有 generation span，TRL assistant mask 全为 0；
审核 template 才把 assistant serialization 交给 collator。完整讲解见[SFT 数据闭环](sft-data-pipeline.md)。

## 目标 GPU 的最小 dry-run

进入长训练前，只跑一个 optimizer update：

1. 固定不可变 model/tokenizer/template revision；
2. 抽查 labels 与 truncation；
3. 确认只有 Adapter 参数 `requires_grad=True`；
4. Forward/backward 后检查所有训练梯度 finite；
5. 记录 peak allocated/reserved 与 token/s；
6. 保存 Adapter，并在新进程、新 base load 上重载。

然后逐步扩大到 10 steps、100 steps 和完整数据。不要在同一次实验中同时改 rank、target modules、LR、length 和
batch，否则首个退化没有可归因对象。

## Adapter 发布包应该包含什么

| 内容 | 作用 |
|---|---|
| Adapter weights + PEFT config | 定义低秩更新与 target modules |
| Immutable base identity | 防止加载到错误底座 |
| Tokenizer + chat template | 保持训练与推理序列化一致 |
| Training/data manifest identity | 能回溯生成来源 |
| Generation config | 固定 held-out 比较协议 |
| 文件清单、size/hash | 检查 bundle 完整性 |

动态 Adapter 便于一个 base 服务多个任务，但会增加 routing、batching 与版本管理；Merge 部署更简单，却失去动态
切换，且量化 base 的 merge 可能需要更多主存。

仓库 `smoke_peft.py` 用 tiny GPT 验证 freeze、save、fresh reload、safe merge、tokenizer/template 与 manifest：

```powershell
python projects/single-gpu-finetuning/smoke_peft.py `
  --steps 8 `
  --artifact-root artifacts/peft-export-control
```

在框架加载以前，检查程序会核对目录文件集、safetensors 结构、base/merged tensor signatures 和 LoRA A/B。
Unkeyed hash 只能发现意外漂移，不认证发布者；可写目录中的 verify-then-reopen 仍有 TOCTOU，需要不可变发布路径、
ACL/lease 或内容寻址句柄。

## 这次 Qwen LoRA 运行说明了什么

```powershell
python projects/single-gpu-finetuning/run_qwen_target_lora_control.py `
  --verify projects/single-gpu-finetuning/qwen2.5-0.5b-lora.recorded-report.json
```

这份运行记录来自指定版本的 Qwen2.5-0.5B-Instruct。它真实执行了 CPU FP32 assistant-only backward、
一次 AdamW step、base freeze、PEFT export 与 fresh reload。Reload logits 可以与导出前对上，但单样本 loss
反而上升。

因此我们可以确认这个 checkpoint 的 Adapter 路径确实执行过。训练超参数是否合理、held-out 质量是否改善，
以及 GPU/QLoRA 是否可用，需要新的实验。
精确参数量、fingerprints 与 artifact sizes 留在[项目控制台账](../evidence/project-controls.md)。

## 训练恢复比 Adapter 保存多得多

用于推理的 Adapter 目录通常没有：optimizer、scheduler、GradScaler、RNG、sampler/data cursor 和未完成 accumulation
window。要声称训练可恢复，先定义 checkpoint 落在哪个 commit boundary。

两种常见协议是：

- 只在 zero-grad/optimizer-committed boundary 保存，恢复后从下一未提交 sample 继续；
- 保存半窗口 gradients、position、divisor、crash-time RNG，并把 sidecar 与 base checkpoint 原子发布。

仓库用三个小型 CPU 故障实验，分别展示漏存 GradScaler、DataLoader prefetch cursor 和
consumed—optimizer-committed crash window 会造成什么后果。它们帮助理解恢复状态，但不能代替目标 PEFT
trainer 的 exact resume 验收。运行入口见
[Single-GPU Finetuning 深挖实验](../practice/projects/single-gpu-finetuning.md#controls)。

## 多 Adapter 什么时候值得

按租户或任务分别训练 Adapter 可以隔离发布节奏，却会增加版本、路由、缓存和 batching 碎片。
频繁变化的事实更适合 RAG，而不是不断增加 Adapter。

多个 LoRA 的加权、串联或融合也不保证行为是单个 Adapter 的线性组合。组合后要重新评测相互干扰、权限隔离和
敏感数据记忆风险。

## 常见故障从哪里查

| 现象 | 第一检查点 |
|---|---|
| Trainable params = 0 | Target module 名称与 model tree |
| Loss 不降 | Assistant mask、shift、LR、重复样本 |
| Train loss 很低但生成差 | 泄漏、teacher forcing、template 与 decode 配置 |
| NaN/Inf | FP16 overflow、异常样本、LR、量化/kernel |
| 训练正常，评测/保存 OOM | Generation KV、logits、merge 与 checkpoint 峰值 |
| Adapter 加载后行为异常 | Base/tokenizer/template revision、`modules_to_save` |
| Merge 偏差大 | Dtype、量化顺序、scale 与 tied weights |
| Resume 后漂移 | Optimizer/scheduler/scaler、RNG、data cursor、pending gradients |

最后用同一 base、data、template、有效 token batch、更新预算与 held-out cases 比较 LoRA、QLoRA 和其他 baseline。
报告 trainable/total params、peak memory、tokens/s、总时长、Adapter size、任务/通用/安全指标与失败样本。
