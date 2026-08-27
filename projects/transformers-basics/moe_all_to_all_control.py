"""运行双进程 CPU/Gloo MoE All-to-All 前向传播实验。

实验用两个本地进程模拟两个 data-parallel rank，观察 token 如何发往专家所在的 rank，
再把专家输出送回原 rank。这里不训练参数，只验证分发、通信和回收顺序。
"""

from __future__ import annotations

import json

from about_llm.from_scratch.moe_all_to_all import run_moe_all_to_all_control


def main() -> None:
    """执行双进程通信实验，并输出可重复检查的 JSON。"""

    # 底层实现会自行创建本地 Gloo 进程组，本入口只负责展示最终报告。
    print(
        json.dumps(
            run_moe_all_to_all_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
