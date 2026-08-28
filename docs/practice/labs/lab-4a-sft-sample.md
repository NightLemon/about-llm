# 实验 4A：追踪一个 SFT 样本

**实验导航**：[返回实验目录](../labs.md) ·
[微调总览](../../training/finetuning.md) ·
[SFT 数据管线](../../training/sft-data-pipeline.md) ·
[Single-GPU Finetuning](../projects/single-gpu-finetuning.md)
{ .doc-nav }

很多微调教程从 `Trainer(...)` 开始，于是最重要的事实被藏起来了：模型究竟看到了哪些 token，哪些位置产生 loss，梯度更新了谁，保存的 adapter 又依赖哪一个基座？

本实验只追踪一个样本。它不下载模型，也不追求生成质量；目标是先把因果链看清楚：

```text
messages
→ template bytes
→ input IDs / shifted targets
→ assistant-only labels
→ loss / backward
→ LoRA A、B
→ adapter-only artifact
→ fresh base reload
→ held-out comparison
```

## 你要回答的六个问题

运行前先写下预测：

1. `assistant:` 前缀本身是否应该产生监督？
2. next-token shift 后，答案第一个 token 对应 label 的哪个位置？
3. LoRA 的 B 为零时，包装前后的 logits 是否应完全一致？
4. 第一次 backward 时，A 和 B 是否都会有非零梯度？
5. adapter 能在当前对象里工作，是否证明它能被独立重载？
6. 训练样本 loss 下降，是否说明 held-out 质量提升？

## 运行

安装 Notebook 环境：

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[dev,torch,jax]"
python scripts/doctor.py --profile notebooks
~~~

打开：

```text
notebooks/04_sft_sample_lifecycle.ipynb
```

或在内存中从头执行，不把输出写回仓库：

~~~powershell
python scripts/execute_notebooks.py --pattern notebooks/04_sft_sample_lifecycle.ipynb
~~~

## 第一次阅读只盯四处

### 1. `answer_start`

Notebook 先序列化 system、user 与 `assistant:` 前缀，再追加答案。`answer_start` 是答案在原始 token 序列中的起点。

语言模型输入使用 `token_ids[:-1]`，target 使用 `token_ids[1:]`。因此判断某个 label 是否属于答案，要看它所预测的原始 token 位置，而不是直接把未移动的 mask 贴到 target 上。

### 2. 初始函数不变

`LoRALinear` 使用

\[
W' = W + \frac{\alpha}{r}BA.
\]

B 从零开始，所以刚包装时 \(BA=0\)。Notebook 要求包装前后 logits 在零容差下相同。若这一步失败，后续 loss 下降也无法区分是训练效果还是包装时已经改变了基座函数。

### 3. 冻结基座

优化器只接收 `requires_grad=True` 的 LoRA 参数。训练结束后，Notebook 再逐元素比较冻结的 base weight。

这比打印“trainable params 百分比”更有说服力：百分比是配置描述，权重未改变才是这次运行的观察。

### 4. Held-out 单独报告

Notebook 用另一个问题分别计算 base 和 adapter loss，但不要求 adapter 必须更好。一个随机初始化的小模型、一个训练样本和一个 held-out 样本没有资格支持质量结论。

这里真正要学的是：训练闭环和发布评测是两条相连但不同的证据链。

## 三个故意失败的实验

### 把所有 token 都设成 labels

把 `train_labels` 换成未 mask 的 target。训练 loss 可能更平滑，却同时监督 system、user 和模板文本。这不能叫 assistant-only SFT。

### 改 seed 后重载 adapter

保持 adapter 不变，只换一个随机种子重新初始化底座模型。张量形状完全对得上，加载也不报错，但输出就是变了。
所以一份 adapter 产物必须绑定它训练时用的**确切**底座模型与 revision；只存 A、B 两个矩阵是不够的。

### 用训练 loss 做发布判断

删掉留出（held-out）那一格，只展示最后一次训练 loss。这样你仍能证明优化器把这个样本拟合住了，
但没法把「原始模型」「Prompt / RAG 基线」和「微调后」三者放在一起比，也就发现不了退步。

## 从教学模型迁移到 Qwen/PEFT 时替换什么

| 教学 Notebook | 真实训练路径 |
|---|---|
| 手写 `system/user/assistant` 文本 | checkpoint 自带或审核后的 chat template |
| ByteTokenizer | 固定 revision 的目标 tokenizer |
| 输出头一个 LoRA module | 从真实 module tree 审核出的 target modules |
| `torch.save(adapter_state_dict)` | PEFT `save_pretrained` + manifest/base identity |
| 一个 held-out loss | 固定 cases、切片、任务指标与发布 gate |
| CPU 随机 MiniGPT | 目标 checkpoint、3070 Laptop 实测显存与吞吐 |

不要把教学代码换个模型名就当成生产训练。真实链路至少还要处理这几类问题：

- **数据形状**：padding、截断、多轮对话与 tool message；
- **训练机制**：梯度累积、混合精度（AMP）、断点续训；
- **数据治理**：许可、近重复、敏感信息。

## 交付物

完成实验后保存一页短报告：

```text
1. template 与 assistant span
2. input/label shape 和监督 token 数
3. trainable parameter names/count
4. 初始函数与冻结基座断言
5. training loss 首尾值
6. fresh-base reload 误差
7. base/adapter held-out 两列
8. 已证明与未证明
```

如果你能解释「为什么 all-token loss 更低反而可能是坏消息」，就可以进入真实 tokenizer 与模板的预检了。
如果还说不清标签左移（target shift）是怎么回事，先回到 [MiniGPT Notebook](../labs.md#lab-3)。
