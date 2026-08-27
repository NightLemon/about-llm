"""离线完成一次 Transformers 小模型的构建、训练和生成闭环。

模型使用随机初始化的微型 GPT-2 配置和两条合成 token 序列，不下载 checkpoint。
实验只验证 Transformers API、loss 反向传播和 generate 能连通，不代表真实语言质量。
"""

from __future__ import annotations

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from about_llm.integrations.transformers_tools import parameter_report


def run_smoke(steps: int = 12) -> dict[str, object]:
    """训练微型 GPT-2 若干步，并返回损失变化、参数量和生成张量形状。"""

    # 固定初始化，让初始损失、最终损失和贪心生成在同一软件版本下可重复。
    torch.manual_seed(23)

    # 词表、上下文和隐藏维度都刻意缩小，使实验能在 CPU 上几秒内完成。
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.train()
    # labels=input_ids 会由模型内部完成一位错位，学习根据前缀预测下一个 token。
    input_ids = torch.tensor([[1, 5, 7, 9, 2], [1, 4, 6, 8, 2]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    # 标准训练步：清梯度 → 前向计算 loss → 反向传播 → AdamW 更新。
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))

    # 切到 eval 后从每条序列的前两个 token 开始做确定性的贪心生成。
    model.eval()
    generated = model.generate(
        input_ids=input_ids[:, :2],
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=config.pad_token_id,
    )
    return {
        "parameter_report": parameter_report(model),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "generated_shape": list(generated.shape),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
