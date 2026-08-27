"""运行带容量限制的双进程 MoE All-to-All 训练实验。

真正的路由、跨进程通信和反向传播在 ``about_llm.from_scratch`` 中实现；
本文件是读者入口，负责启动实验并把各 rank 的结果整理成可检查的 JSON。
"""

from __future__ import annotations

import json

from about_llm.from_scratch.moe_all_to_all_training import (
    run_moe_all_to_all_capacity_training_control,
)


def main() -> None:
    """执行容量受限训练，并禁止 JSON 中出现 NaN 或 Infinity。"""

    # allow_nan=False 会把数值异常变成显式错误，避免无效梯度混进实验报告。
    print(
        json.dumps(
            run_moe_all_to_all_capacity_training_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
