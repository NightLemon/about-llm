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

保持 adapter 不变，但用不同 seed 构造 fresh base。即使 shape 完全匹配，输出也会改变。这就是为什么真实 adapter artifact 必须绑定精确 base model/revision，而不是只保存 A、B。

### 用训练 loss 做发布判断

删掉 held-out cell，只展示最后一个 training loss。你仍能证明优化器拟合了这个样本，却无法比较 base、Prompt/RAG baseline 与 adapter，更无法发现回归。

## 从教学模型迁移到 Qwen/PEFT 时替换什么

| 教学 Notebook | 真实训练路径 |
|---|---|
| 手写 `system/user/assistant` 文本 | checkpoint 自带或审核后的 chat template |
| ByteTokenizer | 固定 revision 的目标 tokenizer |
| 输出头一个 LoRA module | 从真实 module tree 审核出的 target modules |
| `torch.save(adapter_state_dict)` | PEFT `save_pretrained` + manifest/base identity |
| 一个 held-out loss | 固定 cases、切片、任务指标与发布 gate |
| CPU 随机 MiniGPT | 目标 checkpoint、3070 Laptop 实测显存与吞吐 |

不要把教学代码直接换一个模型名就称为生产训练。真实路径还要处理 padding、截断、multi-turn/tool messages、gradient accumulation、AMP、checkpoint resume、数据许可、近重复和敏感信息。

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

如果你能解释“为什么 all-token loss 更低也可能是坏消息”，就可以进入真实 tokenizer/template preflight；如果还说不清 target shift，先回到 [MiniGPT Notebook](../labs.md#lab-3)。
