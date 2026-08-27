"""运行双进程 MoE 容量分组实验。

实验关注专家容量有限时的路由决策：哪些 token 被接收、哪些溢出，以及两个 rank
是否对容量分组得到相同结论。底层使用 CPU/Gloo，因此无需 GPU 也能观察通信语义。
"""

from __future__ import annotations

import json

from about_llm.from_scratch.moe_distributed_capacity import (
    run_distributed_moe_capacity_control,
)


def main() -> None:
    """执行容量分组对照并输出严格 JSON。"""

    # sort_keys=True 让多次运行的输出顺序稳定，便于人工比较和测试。
    print(
        json.dumps(
            run_distributed_moe_capacity_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
