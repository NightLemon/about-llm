"""跨进程验证 CPU AMP 训练 checkpoint 是否能精确恢复完整训练状态。

正常控制比较不间断 baseline 与 phase1→新进程 resume；隐藏 worker 模式还会分别遗漏 scheduler、
scaler、RNG 或数据游标，展示“能加载权重”为什么不等于“延续同一训练轨迹”。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from about_llm.finetuning.training_resume import (
    WorkerMode,
    run_training_resume_process_control,
    run_training_resume_worker,
)


def parse_args() -> argparse.Namespace:
    """解析外部 control 模式与仅供子进程使用的 worker 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-mode",
        choices=(
            "baseline",
            "phase1",
            "resume",
            "omit-scheduler",
            "omit-scaler",
            "omit-rng",
            "omit-data",
            "wrong-scheduler",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--checkpoint-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """运行单个 worker 阶段，或协调多进程 baseline/resume 对照。"""

    args = parse_args()
    if args.worker_mode is None:
        # 默认父进程会依次启动独立 worker，确保恢复不依赖当前内存中的 Python 对象。
        if args.checkpoint_path is not None:
            raise SystemExit("--checkpoint-path is reserved for internal workers")
        payload = run_training_resume_process_control(Path(__file__))
    else:
        # worker 分支只由父控制调用，在给定 checkpoint 上执行一种正常或故障模式。
        if args.checkpoint_path is None:
            raise SystemExit("internal worker requires --checkpoint-path")
        payload = run_training_resume_worker(
            cast(WorkerMode, args.worker_mode), args.checkpoint_path
        )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.buffer.write((encoded + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
