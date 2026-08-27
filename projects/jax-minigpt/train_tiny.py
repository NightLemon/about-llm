"""用 JAX/Optax 让微型 GPT 过拟合一个固定 batch。

实验从随机参数开始，JIT 编译一个真实训练步，并重复喂入两条相同序列。最终 loss 明显下降
说明前向、交叉熵、自动微分和优化器已经连通；这不是模型泛化或真实文本质量测试。
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax

from about_llm.from_scratch.gpt_jax import (
    JAXGPTConfig,
    adamw_optimizer,
    cross_entropy_loss,
    forward,
    init_params,
    make_train_step,
)


def build_parser() -> argparse.ArgumentParser:
    """定义训练步数、学习率和随机种子。"""

    parser = argparse.ArgumentParser(
        description="JIT a tiny JAX GPT train step and overfit one batch"
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """构造最小 GPT，执行 JIT 训练并汇总 loss 与耗时。"""

    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    # 模型和词表都刻意缩小，使 CPU 也能快速完成训练闭环。
    config = JAXGPTConfig(
        vocab_size=8,
        context_length=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
    )
    # targets 比 input_ids 左移一位，表示每个位置监督下一个 token。
    input_ids = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    targets = jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    # 参数初始化与优化器状态都从固定 seed 开始，便于复现实验。
    params = init_params(jax.random.key(args.seed), config)
    optimizer = adamw_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    optimizer_state = optimizer.init(params)
    train_step = make_train_step(config, optimizer)
    initial_loss = float(cross_entropy_loss(forward(params, input_ids, config), targets))
    step_seconds: list[float] = []
    gradient_norm = jnp.asarray(0.0)
    loss = jnp.asarray(initial_loss)

    # 第一步还包含 JIT 编译时间，所以单独保留，后续步才用于 steady-state 均值。
    for _ in range(args.steps):
        started = time.perf_counter()
        params, optimizer_state, loss, gradient_norm = train_step(
            params, optimizer_state, input_ids, targets
        )
        loss.block_until_ready()
        step_seconds.append(time.perf_counter() - started)

    final_loss = float(cross_entropy_loss(forward(params, input_ids, config), targets))
    parameter_count = sum(int(leaf.size) for leaf in jax.tree.leaves(params))
    steady = step_seconds[1:] or step_seconds
    report = {
        "versions": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
        },
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "verified_scope": "current JAX device; no multi-device or target-GPU claim",
        "network_performed": False,
        "seed": args.seed,
        "steps": args.steps,
        "parameter_count": parameter_count,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "final_preclip_gradient_norm": float(gradient_norm),
        "compile_plus_first_step_seconds": step_seconds[0],
        "mean_steady_step_seconds": sum(steady) / len(steady),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if final_loss < initial_loss else 1


if __name__ == "__main__":
    raise SystemExit(main())
