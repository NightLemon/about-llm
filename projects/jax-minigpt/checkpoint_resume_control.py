"""验证 JAX/Optax 训练从 checkpoint 恢复后能延续同一条轨迹。

底层会分别运行“不间断训练”和“保存后在新进程恢复训练”，再比较参数、优化器状态、
随机数状态和 loss。使用新进程是为了排除内存中残留状态带来的假恢复。
"""

from __future__ import annotations

import json

from about_llm.from_scratch.jax_training_resume import (
    run_jax_training_resume_control,
)


def main() -> None:
    """执行跨进程恢复对照并输出逐项一致性报告。"""

    # 非有限值不允许进入 JSON，数值异常会直接令实验失败。
    print(
        json.dumps(
            run_jax_training_resume_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
