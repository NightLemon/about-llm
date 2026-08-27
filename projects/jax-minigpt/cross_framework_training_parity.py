"""比较 PyTorch 与 JAX 中带随机性的 AdamW 训练轨迹。

除了初始权重和输入，实验还对齐 dropout 随机掩码、优化器超参数和更新步数，
逐步比较 loss、梯度与参数。它比一次前向 parity 更能发现优化器或随机数语义差异。
"""

from __future__ import annotations

import json

from about_llm.from_scratch.gpt_cross_framework_training import (
    run_gpt_cross_framework_training_parity_control,
)


def main() -> None:
    """运行跨框架训练对照并输出每一步的误差。"""

    # 完整报告由底层同时收集两条轨迹，本入口只负责严格序列化。
    print(
        json.dumps(
            run_gpt_cross_framework_training_parity_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
