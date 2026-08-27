"""运行双进程 CPU/Gloo MoE All-to-All 训练实验。

它在前向路由和跨 rank 通信之后继续执行反向传播，用来检查专家参数是否收到梯度，
以及分布式结果是否与单进程参考计算一致。
"""

from __future__ import annotations

import json

from about_llm.from_scratch.moe_all_to_all_training import (
    run_moe_all_to_all_training_control,
)


def main() -> None:
    """运行训练对照并输出包含梯度证据的 JSON。"""

    # 禁止非有限值，确保通信或反向传播失败不会被序列化为看似正常的报告。
    print(
        json.dumps(
            run_moe_all_to_all_training_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
