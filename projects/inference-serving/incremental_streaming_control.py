"""运行或复核流式生成在客户端断连后的协作取消实验。

实际运行会启动本地 loopback 服务，逐块产生 token，并在客户端提前断开后传播取消信号；
``--verify`` 只检查仓库记录报告，不启动服务。实验关注资源是否停止继续工作。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.inference.incremental_streaming_control import (
    load_and_verify_incremental_streaming_report,
    run_incremental_streaming_control,
)

PROJECT = Path(__file__).resolve().parent
DEFAULT_RECORDED_REPORT = PROJECT / "incremental-streaming.recorded-report.json"


def main() -> None:
    """在实时控制和离线报告验证之间选择一条路径。"""

    # 报告含中文错误与状态说明，统一终端编码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="RECORDED_REPORT",
        help="verify a recorded report without starting a server",
    )
    args = parser.parse_args()
    # 无 --verify 时真正运行本地服务；有参数时不执行生成，仅复核证据。
    report = (
        run_incremental_streaming_control()
        if args.verify is None
        else load_and_verify_incremental_streaming_report(args.verify)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
