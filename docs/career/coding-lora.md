# 编码题 5：LoRA Linear

**练习导航**：[返回编码轮索引](coding-round.md) · [PEFT/QLoRA](../training/peft-qlora-engineering.md) · [单卡微调](../practice/projects/single-gpu-finetuning.md)
{ .doc-nav }

本页是一道可以独立计时、实现和验收的练习。开始前先写清输入输出、错误契约和至少两个边界例。

**题面**：包装一个 `nn.Linear`，冻结其权重，只训练低秩增量。
有效权重为 \(W + \frac{\alpha}{r} BA\)，其中 \(A \in \mathbb{R}^{r \times d_{in}}\)、
\(B \in \mathbb{R}^{d_{out} \times r}\)。再实现把 adapter 合并回普通 `Linear` 的方法。

**必须先问清楚的三件事**：

- \(A\) 和 \(B\) 分别怎样初始化？（这决定了包装后模型行为是否改变。）
- 缩放是 \(\alpha/r\) 还是 \(\alpha\)？换 rank 做实验时这会影响可比性。
- 需要保存完整模型还是只保存 adapter？

**最小实现**：

```python
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype
        ))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype
        ))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def merged(self):
        """Return a separate Linear; do not modify the frozen base layer."""
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.copy_(self.base.weight + self.scaling * (self.lora_b @ self.lora_a))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged
```

\(B\) 初始化为**零**，\(A\) 用随机初始化。于是包装瞬间 \(BA = 0\)，模型输出与原来逐位相同，
训练是从原模型出发的连续微调，而不是一次随机扰动。两个都置零则梯度恒为零，永远学不动；
两个都随机则包装本身就破坏了预训练权重。

参数量从 \(d_{in} \times d_{out}\) 降到 \(r(d_{in} + d_{out})\)。
例如 \(d_{in}=d_{out}=4096\)、\(r=8\) 时，从约 1678 万降到约 6.6 万，约为原来的 0.4%。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 包装后行为不变 | \(B\) 初始化非零时训练起点已偏移 | 断言包装前后输出逐位相等 |
| base 确实冻结 | 忘记设 `requires_grad=False`，实际是全量微调 | 反向后断言 `base.weight.grad is None` |
| 合并等价 | 合并时漏乘 `scaling` | 断言 `merged(x)` 与 `forward(x)` 在容差内相等 |
| 保存重载 | 只存权重不存 rank/alpha，重载后形状或缩放错 | 存盘再读回，断言输出一致 |
| `rank` 非正 | 创建出空参数，静默变成恒等层 | 断言抛异常 |

第二条最值得主动提。"loss 在下降"并不能证明你在做 LoRA——全量微调的 loss 同样会降。
真正的证据是 base 梯度为 `None` 且可训练参数量符合预期。

**面试官的下一个追问**：

- *QLoRA 是不是"用 4-bit 训练"？* 不是，见[核心题第 9 题](interview-questions.md#qlora-scope)。
- *合并后还能切换回原模型吗？* 能，但要保留原始权重；`merged()` 应返回独立对象而不是原地修改。

**仓库参考实现**：`src/about_llm/finetuning/lora.py` 的 `LoRALinear`，
包含 `merged()` 与只导出 adapter 的 `adapter_state_dict()`。测试见 `tests/test_lora.py`。

```bash
python -m pytest tests/test_lora.py -q
```

## 完成信号

在不查看参考实现的情况下重新写一遍，并能解释复杂度、失败行为、一个反例以及测试为什么能推翻错误实现。
