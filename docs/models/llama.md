# Llama：从开放权重到单卡可复现实验

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想理解、量化、微调或部署 Llama 系开放权重模型的开发者和算法工程师。
- **先修**：理解 decoder-only Transformer、tokenizer、KV Cache 和基础 GPU 内存。
- **首次阅读**：固定 checkpoint → 读懂 decoder block → 模板 → 内存 → 单卡实验 → 发布。
- **完成信号**：能从一个固定 checkpoint 推导结构和预算，并用真实运行校准公式。
- **卡住时**：先读 [Transformer](../core/transformer.md)，只选择一个小型 text-only Instruct checkpoint。

</div>

**模型导航**：[Transformer](../core/transformer.md) · [Transformers 项目](../practice/projects/transformers-basics.md) ·
[单卡微调](../practice/projects/single-gpu-finetuning.md) · [Llama 证据台账](../evidence/llama-controls.md)
{ .doc-nav }

Llama 是学习开放权重模型工程的好入口：你可以检查 config、tokenizer、weights，运行 forward，再把静态推导与真实显存、质量和延迟对账。

但 Llama 不是一个固定架构。不同代际、尺寸、Base/Instruct、text/multimodal 版本可能有不同词表、head 布局、RoPE、上下文、模板和许可。

本章用一条主线组织这些知识：选定一个具体 checkpoint，把它变成一个可复现、可评测、可回滚的单卡系统。

## 第一步不是下载权重

先写出实验对象：

~~~text
model repository + immutable revision
+ tokenizer revision and chat template
+ text-only Base or Instruct
+ Transformers/vLLM exact version
+ dtype or quantization
+ target GPU
+ task cases and generation budget
+ license review
~~~

如果只有“Llama 8B”这样的短名，你还不能安全推导结构、模板、显存或许可。

### 一个模型由哪些文件组成

可部署 checkpoint 通常需要：

~~~text
repository at immutable revision
├── config
├── tokenizer files and tokenizer config
├── chat template / prompt format
├── generation config
├── weight shards and index
├── model card
└── license / acceptable-use terms
~~~

实际文件名会变化。重点是记录 loader 真正读取的每个文件：relative path、size、SHA-256、resolved revision 和用途。

只固定 model revision，却让 tokenizer 或 template 漂移，仍然不能复现实验。Hash 能识别 bytes 是否变化，但无密钥 hash 不是发布者签名。

## 用证据阶梯约束结论

| 层级 | 证据 | 可以回答 |
|---|---|---|
| L1 | 固定 model card | 厂商对该发布声明了什么 |
| L2 | 固定 config/tokenizer bytes | 静态结构、token 协议候选 |
| L3 | weight inventory 与成功加载 | 指定权重是否完整进入 loader |
| L4 | 目标 runtime forward/generate | cache、logits 和执行路径 |
| L5 | 目标任务与硬件评测 | 质量、性能、容量和 SLO |

不要把不同层的结论拼起来。例如，model card 报告一个上下文长度，authored fixture 验证了 GQA 公式，另一个模型的 weight smoke 成功；三者不能合成“这个 Llama checkpoint 的长上下文已经验证”。

本仓库当前固定了一份 Llama 3.2 model-card 发布证据，但没有把它冒充真实 Llama 权重执行。精确 revision 和 control 范围见[证据台账](../evidence/llama-controls.md)。

## 从 config 画出一层 decoder

先读这些字段，而不是从家族名猜：

| 字段 | 你想知道什么 |
|---|---|
| hidden size / layers | residual width 与 block 数 |
| query heads / KV heads | MHA、GQA 或 MQA 候选 |
| head dimension | Q/K/V 每 head 维度 |
| intermediate size | gated MLP 宽度 |
| vocabulary / tied embeddings | embedding 与输出头形状 |
| norm epsilon / bias flags | 数值路径和参数量 |
| position / RoPE fields | 位置编码实现候选 |
| architecture / model type | loader 和代码路径候选 |

字段缺失、自定义 code 或实际 weight shape 不符时，应停止机械推导，转去核对同 revision 实现。

许多 Llama-style text checkpoint 使用 pre-norm decoder block：

\[
h' = h + \operatorname{Attention}(\operatorname{RMSNorm}(h)),
\]

\[
h'' = h' + \operatorname{MLP}(\operatorname{RMSNorm}(h')).
\]

下面四个机制决定了大量工程行为。

## RMSNorm：只按均方根缩放

\[
\operatorname{RMSNorm}(x)=g\odot
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}.
\]

与 LayerNorm 相比，它不先减去均值。直觉上，它控制向量整体尺度，让后续 attention/MLP 看到更稳定的输入。

实现和 checkpoint 兼容性仍取决于 epsilon、accumulation dtype、scale 参数 shape 和 norm 所在位置。RMSNorm 与 LayerNorm shape 相同，也不能互换权重语义。

## RoPE：旋转 Q/K 表达相对位置

Rotary Position Embedding（RoPE）把每对 Q/K 通道按位置旋转：

\[
R(p,\theta)
=
\begin{bmatrix}
\cos(p\theta)&-\sin(p\theta)\\
\sin(p\theta)&\cos(p\theta)
\end{bmatrix}.
\]

Q 与 K 使用对应旋转后，它们的点积带有相对位置结构。工程上必须固定 base frequency、使用维度、scaling 类型和实现版本。

只增加 max position 配置，不会自动得到可靠长上下文。协议接受、runtime 能跑和任务在远距离仍正确，是三件事。

## SwiGLU：为什么 MLP 有三组主权重

常见 gated MLP 写成：

\[
\operatorname{MLP}(x)=W_d
\left(\operatorname{SiLU}(W_gx)\odot W_ux\right).
\]

一个分支产生 gate，另一个分支产生 value，逐元素相乘后再投影回 residual width。

若 hidden width 为 \(d\)、intermediate width 为 \(m\)，忽略 bias 时主要参数约为：

\[
P_{\mathrm{MLP}}=3dm.
\]

这就是不能使用普通两层 MLP 的 \(2dm\) 公式的原因。精确值仍要从实际 state dict shape 重算。

## GQA：减少 K/V，不同比缩小整个模型

Grouped-Query Attention（GQA）让多个 query heads 共享较少的 K/V heads。

设 query heads 为 \(H_q\)、KV heads 为 \(H_{kv}\)、head dimension 为 \(d_h\)。标准 layout 下：

\[
P_Q=dH_qd_h,\qquad
P_K=P_V=dH_{kv}d_h.
\]

当 \(H_{kv}<H_q\) 时，K/V projection 与 KV Cache 会减少。但 Q/O projection、MLP、embedding 和 runtime workspace 不会按同一比例缩小。

所以 GQA 的价值主要体现在解码 cache 和部分投影，而不是“整个模型按 KV head 比例变小”。

## 从结构推导两本内存账

### 参数账

理论权重存储主项是：

\[
M_{\mathrm{weights}}
=\sum_i \operatorname{numel}(W_i)\times \operatorname{bytes}(dtype_i).
\]

参数量应从 actual name/shape 账本计算，并说明 tied embeddings、buffer、adapter 和量化 metadata 怎样处理。产品名中的 B 只是标签，不能替代重算。

### KV 账

标准 dense K/V layout 的理想 payload 为：

\[
M_{KV,\mathrm{ideal}}
=2LBTH_{kv}d_hs.
\]

其中 \(L\) 是层数，\(B\) 是 batch，\(T\) 是缓存 token 数，\(s\) 是每个元素的字节数。

这个公式不包含 page/block 对齐、allocator、prefix sharing、量化 scale、workspace、CUDA graph reserve、weights 或 activations。

因此它适合做预检和解释变量趋势，不能当作 GPU 峰值。若 checkpoint 使用自定义 attention，也要先确认公式前提成立。

### 运行峰值账

推理峰值可拆成：

\[
M_{\mathrm{peak}}\approx
M_{\mathrm{weights}}+M_{KV}+M_{\mathrm{activations}}
+M_{\mathrm{workspace}}+M_{\mathrm{allocator}}+M_{\mathrm{runtime}}.
\]

真实报告应说明 warm-up、采样时点和指标来源。PyTorch allocated、reserved、进程 RSS 与设备总占用并不是同一个数。

## Base、Instruct 与 template 是一个协议

Base checkpoint 主要学习 next-token continuation；Instruct checkpoint 还接受了指令或对话后训练。二者可能共享主干，却不是相同任务接口。

Chat template 决定：

- role 与 system 怎样序列化；
- BOS、EOS、header 和 turn-end token；
- assistant generation prompt；
- tool definition/call/result 的表示；
- 训练中 assistant-only labels 的边界。

在下载大权重前，就可以只用 tokenizer 检查：

~~~text
single user
system + user
multi-turn
empty assistant generation prompt
tool definition + call + result
Unicode and multilingual input
~~~

保存 rendered text、token IDs、special-token positions 与 template fingerprint。训练和部署必须使用同一协议。

成功 tokenize 不证明权重匹配或回答质量；Base 套上 Instruct template 也不会自动获得指令能力。

## Generation 也要固定最终配置

记录 decoding method、temperature、top-p/top-k、max new tokens、special token IDs、stop strings、penalties 和 runtime version。

Model defaults、generation config、显式 kwargs 与 serving 参数可能有不同 precedence。比较 Transformers 和 vLLM 前，应先打印最终 resolved config。

模型 EOS、应用 stop string、客户端取消和服务 finish reason 是不同终止层。截断用户可见字符串不能伪造模型真实 terminal，也不能改写 usage。

## 量化：4-bit 只是部分状态的描述

“4-bit 模型”通常表示部分 weights 的压缩存储。它没有说明：

- 哪些 layers 被量化；
- group size、axis、scale/zero dtype；
- packing、padding 和未量化模块；
- compute 与 accumulation dtype；
- KV Cache dtype；
- fused kernel 和 workspace；
- artifact、resident 与 peak memory。

因此 4-bit 不等于每个参数严格占 0.5 byte，也不保证速度更快。没有匹配 kernel 时，dequant 和数据搬运可能抵消收益。

量化验收要同时比较 reload 后的固定 logits/continuation、任务质量、峰值显存、TTFT、TPOT、吞吐和并发。CPU packing demo 不能替代目标 GPU 证据。

## LoRA/QLoRA：少训练参数不等于少一切内存

LoRA 对线性权重加入低秩更新：

\[
W'=W+\frac{\alpha}{r}BA.
\]

若输入/输出维度为 \(d_{in},d_{out}\)，该层可训练参数为：

\[
P_{\mathrm{LoRA}}=r(d_{in}+d_{out}).
\]

总数取决于实际 target modules。分离的 q/v projection、fused QKV 与 all-linear 会得到完全不同的参数账。

QLoRA 通常低比特存储 frozen base，但 adapter、gradients、optimizer、activations 和部分计算仍使用更高精度。Base 文件变小不代表训练峰值同比下降。

发布 adapter 时绑定 exact base、tokenizer、template、target modules、rank/scaling、数据、runtime 和评测。保存后在新的 base 实例重载，不能只验证内存中的 merge。

## 单张消费级 GPU 的渐进路线

### Phase 0：零权重预检

1. 固定 model card、revision 和许可。
2. 下载并 hash config、tokenizer 与 template。
3. 计算参数、理想 KV 和 weight shards 总字节。
4. 估算目标长度下的峰值组成。
5. 定义最小质量、安全和长上下文 cases。

若理论主项已经超过资源，就换更小 checkpoint 或经验证的量化方案，而不是等下载后 OOM。

### Phase 1：最小真实执行

1. batch 1、最短 prompt、明确 dtype/device。
2. 记录 loader 实际读取的 files 和 parameter inventory。
3. 执行一次 prefill 和带 cache 的单 token decode。
4. 与不使用 cache 的同序列 logits 对账。
5. 固定一次 greedy generation 和 stop reason。

这一步证明指定环境和输入的执行路径，不证明任务质量或长上下文。

### Phase 2：容量扫描

每次只改变一个变量：input length、max output、batch/concurrency、quantization 或 KV dtype。记录成功、OOM、timeout 和其他失败的完整分母。

推理通常先降低 batch/concurrency 和 token budget，再考虑经验证的量化或更小模型。每次降级都生成新 manifest，并重新跑质量回归。

### Phase 3：服务与微调

先用 Transformers 建立 token、logits、cache 和 generation baseline，再接入 vLLM 测 continuous batching、paged KV、prefix cache、取消和 offered-load。

微调则从 labels 检查、tiny-batch overfit、LoRA backward、adapter reload 开始，最后才扩大数据和序列长度。

## 长上下文要测“有效”，不只测“能放”

至少按位置和任务切片：

| Case | 想发现的问题 |
|---|---|
| 开头/中间/结尾 needle | 位置信息能否恢复 |
| 多 needle | 是否遗漏或混淆 |
| 冲突文档 | 是否遵循来源和时间 |
| 顺序/计数/聚合 | 是否维护全局状态 |
| 长输入 + 长输出 | 后段约束是否保持 |
| 多语言/代码 | tokenizer 与能力分布 |

报告 tokenizer token length，并把 OOM、timeout、truncation、refusal 与 wrong answer 放入分母。

Model card 报告的 context length、runtime 接受的长度和任务有效长度是三种不同结论。

## 开放权重仍有许可与供应链

可下载权重不自动等于 OSI open source。具体 Llama 发布可能有独立 Community License、acceptable-use、归因、再分发或规模条款。

发布前审查 exact version、adapter/derivative、tokenizer/code dependencies、训练数据、目标客户/地域/行业和 NOTICE。教材不能替代责任主体的法律判断。

生产 manifest 至少绑定：

~~~json
{
  "model_id": "<repository>",
  "revision": "<immutable-commit>",
  "files": [{"path": "...", "size": 0, "sha256": "..."}],
  "tokenizer_revision": "...",
  "template_sha256": "...",
  "runtime": "name + exact version",
  "dtype_or_quantization": "...",
  "adapter": null,
  "license_review_id": "...",
  "evaluation_artifact": "sha256:..."
}
~~~

升级 model、tokenizer、template、runtime 或 quantization 的任一项，都视为候选系统变化。使用 paired offline eval、shadow、canary，再保留完整旧 bundle 回滚。

## 一个最小学习项目

选择一个许可允许且资源可承受的 text-only Instruct checkpoint，交付四个 artifact：

1. **identity report**：revision、文件、config、tokenizer、template、license。
2. **execution report**：prefill/cache/greedy 对账和峰值显存。
3. **evaluation report**：逐 case 质量、安全、长上下文和失败分类。
4. **serving report**：Transformers/vLLM 输入对齐、TTFT、TPOT、吞吐、取消和回滚。

每个实验先写预测，再运行，再解释公式与实测的差异。这样你学到的是模型工程因果链，而不是一串命令。

可运行入口：

- [Transformers Basics](../practice/projects/transformers-basics.md)
- [Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md)
- [Inference Serving](../practice/projects/inference-serving.md)
- [Evaluation Gate](../practice/projects/evaluation-gate.md)

## 常见错误

- 把 Llama 当作固定 config，或只保存一个家族短名。
- 把 model-card 参数与上下文声明写成独立测量。
- 只固定 model，不固定 tokenizer、template 和 generation。
- 把 authored GQA 公式 fixture 写成目标 checkpoint 显存。
- 用文件位宽推断 GPU 峰值或端到端 speedup。
- 把 LoRA trainable parameters 少写成训练显存同比下降。
- 给 Base 套 Instruct template 并期待相同行为。
- 把 vLLM 能启动当成正确性、性能和取消证据。
- 把可下载权重写成 OSI 开源。
- 把 SHA-256 当作发布者签名。

## 面试时怎样回答

面对“怎样部署一个 Llama 模型”，按六步回答：

1. 固定 checkpoint、tokenizer、template、license 与 immutable revision。
2. 从 config/weight shapes 读取 RMSNorm、RoPE、SwiGLU 和 GQA。
3. 分开权重、KV、activation、workspace 与 allocator 预算。
4. 用 prefill/cache/greedy 建立真实正确性 baseline。
5. 对量化、adapter 和 serving runtime 做 paired evaluation。
6. 保存 release manifest，经过 canary 并能回滚完整 bundle。

继续追问时，应能推导 GQA 的 KV 变化、解释 QLoRA 哪些状态没有变成 4-bit，并说明 reported context 与 effective context 的差别。

## 自测

1. 为什么 model card、config、weights、runtime 和 task evaluation 不能互相借用结论？
2. SwiGLU 为什么有三组主要投影？GQA 又减少了哪些部分？
3. 理想 KV payload 与 GPU peak memory 之间还缺哪些项？
4. 为什么 model revision 固定而 tokenizer 漂移仍不可复现？
5. 单次量化 generation 看起来正常后，还缺哪些发布证据？

## 一手资料入口

- Meta，[Llama models repository](https://github.com/meta-llama/llama-models)。
- Touvron 等，[LLaMA](https://arxiv.org/abs/2302.13971)。
- Hugging Face，[Chat templates](https://huggingface.co/docs/transformers/en/chat_templating)。
- 本仓库固定的 model-card revision 与 controls 见[Llama 证据台账](../evidence/llama-controls.md)。
