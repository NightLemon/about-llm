"""运行可训练 Top-k MoE 的确定性梯度实验。

这个入口调用最小 MoE 实现，依次完成路由、稀疏专家前向、损失计算和反向传播，
用固定输入回答“路由器和被选专家是否真的获得梯度”。
"""

from __future__ import annotations

import json

from about_llm.from_scratch.moe_training import run_trainable_moe_control


def main() -> None:
    """执行一次 MoE 训练步并打印路由与梯度报告。"""

    # 报告同时包含前向路由结果和反向梯度，便于把两阶段对应起来。
    report = run_trainable_moe_control()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
