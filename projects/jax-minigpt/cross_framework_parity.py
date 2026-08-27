"""比较同一份 MiniGPT 参数在 PyTorch 与 JAX 中的前向数值。

实验把权重按明确映射复制到两个实现，输入同一批 token，并比较 logits 与 loss。
它验证的是本仓库两个实现的公式和张量布局一致，不代表两个框架在所有模型上等价。
"""

from __future__ import annotations

import json

from about_llm.from_scratch.gpt_cross_framework import (
    run_gpt_cross_framework_parity_control,
)


def main() -> None:
    """运行跨框架前向对照并打印误差与 shape。"""

    # allow_nan=False 防止某一框架产生 NaN 后仍得到可解析但无意义的报告。
    print(
        json.dumps(
            run_gpt_cross_framework_parity_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
