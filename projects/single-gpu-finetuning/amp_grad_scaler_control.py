"""在 CPU 上验证 AMP/GradScaler 的调用顺序、overflow 跳步与状态恢复。

底层真实执行 autocast、scale、backward、unscale、梯度裁剪和 scaler.update，并比较 checkpoint
恢复前后的 scale 轨迹。本入口只负责运行控制并输出严格 JSON。
"""

from __future__ import annotations

import json
import sys

from about_llm.finetuning.amp_scaler import run_cpu_amp_grad_scaler_control


def main() -> None:
    """执行 AMP 控制并以 UTF-8 输出全部 step 状态。"""

    # allow_nan=False 让梯度或 scale 异常直接成为序列化错误。
    payload = run_cpu_amp_grad_scaler_control().to_dict()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.buffer.write((encoded + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
