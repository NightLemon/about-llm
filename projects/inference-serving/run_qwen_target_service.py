"""运行或离线复核固定 Qwen2.5 checkpoint 的本地流式服务。

实时模式会启动 loopback HTTP 服务、发送请求并核对 SSE、停止原因、usage 和取消语义；
``--verify`` 不加载模型，只检查 recorded report 是否仍绑定同一 service manifest。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.inference.target_service_control import (
    load_and_verify_recorded_target_service_report,
    run_live_target_service_control,
)

PROJECT = Path(__file__).resolve().parent
DEFAULT_SERVICE_MANIFEST = PROJECT / "qwen2.5-0.5b-service.control.json"
DEFAULT_CHECKPOINT_MANIFEST = (
    PROJECT.parent
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    """根据 --verify 选择真实服务控制或离线证据验证。"""

    # 服务响应可能含中文，显式设置 UTF-8 后再打印 JSON。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-manifest", type=Path, default=DEFAULT_SERVICE_MANIFEST
    )
    parser.add_argument(
        "--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed checkpoint files to exist in the local cache",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="RECORDED_REPORT",
        help="verify a recorded report without loading the model or starting a server",
    )
    args = parser.parse_args()
    if args.verify is None:
        report = run_live_target_service_control(
            args.service_manifest,
            args.checkpoint_manifest,
            local_files_only=args.local_files_only,
        )
    else:
        report = load_and_verify_recorded_target_service_report(
            args.service_manifest,
            args.checkpoint_manifest,
            args.verify,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
