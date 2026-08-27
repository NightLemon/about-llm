"""运行或复核 Transformers 工作线程的协作取消实验。

实时模式让小模型在线程中生成，并通过停止条件响应取消信号；``--verify`` 只核对记录报告。
重点是客户端断开后后台线程是否及时退出，而不只是 HTTP 层停止发送字节。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.inference.transformers_thread_cancellation_control import (
    load_and_verify_transformers_thread_cancellation_report,
    run_transformers_thread_cancellation_control,
)

PROJECT = Path(__file__).resolve().parent
DEFAULT_RECORDED_REPORT = (
    PROJECT / "transformers-thread-cancellation.recorded-report.json"
)


def main() -> None:
    """选择实时线程取消实验或无模型的离线报告验证。"""

    # 统一中文报告的终端编码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="RECORDED_REPORT",
        help="verify a recorded report without constructing a model or server",
    )
    args = parser.parse_args()
    report = (
        run_transformers_thread_cancellation_control()
        if args.verify is None
        else load_and_verify_transformers_thread_cancellation_report(args.verify)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
