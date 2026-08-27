"""追踪一条训练样本从来源 revision 到数据集和 checkpoint 的血缘。

``trace`` 会根据规范重新构建 source→sample→dataset→run→checkpoint 图；``verify`` 会将
重算结果与仓库记录报告逐项比较。这样能回答某条训练数据来自哪里、影响过哪些产物。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from about_llm.training_data_lineage import (
    analyze_training_data_lineage,
    load_training_data_lineage_spec,
    verify_training_data_lineage_report,
)

PROJECT = Path(__file__).resolve().parent
DEFAULT_SPEC = PROJECT / "thread-8841.lineage.json"
DEFAULT_REPORT = PROJECT / "thread-8841.recorded-report.json"


def _build_parser() -> argparse.ArgumentParser:
    """定义重新追踪与验证已有报告两个子命令。"""

    parser = argparse.ArgumentParser(
        description=(
            "Trace one training item from source revisions to checkpoints, or verify "
            "the recorded teaching report."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="recompute the source-to-checkpoint trace")
    trace.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    trace.add_argument("--output", type=Path)
    trace.add_argument(
        "--overwrite",
        action="store_true",
        help="replace --output if it already exists",
    )

    verify = subparsers.add_parser(
        "verify", help="recompute and compare the recorded report"
    )
    verify.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    verify.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def _json_text(payload: object) -> str:
    """用稳定、严格且保留中文的格式序列化报告。"""

    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"


def _write_report(path: Path, payload: object, *, overwrite: bool) -> None:
    """写入新报告；默认拒绝覆盖已有证据文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    # x 模式保证目标已存在时原子失败，避免误覆盖一次已记录的实验结果。
    mode = "w" if overwrite else "x"
    try:
        with path.open(mode, encoding="utf-8", newline="\n") as output:
            output.write(_json_text(payload))
    except FileExistsError as error:
        raise ValueError(
            f"output already exists: {path}; pass --overwrite to replace it"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    """执行血缘重算或离线报告验证，并把输入错误交给 argparse 展示。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        # spec 定义节点、边和哈希约束，是两种命令共同的事实来源。
        spec = load_training_data_lineage_spec(args.spec)
        if args.command == "trace":
            # trace 分支可打印到 stdout，也可显式写入一个新 artifact。
            report = analyze_training_data_lineage(spec)
            if args.output is None:
                print(_json_text(report), end="")
            else:
                _write_report(args.output, report, overwrite=args.overwrite)
                print(
                    _json_text(
                        {
                            "case_id": report["case_id"],
                            "output": str(args.output),
                            "report_fingerprint": report["report_fingerprint"],
                            "written": True,
                        }
                    ),
                    end="",
                )
            return 0

        # verify 会重新计算，而不是只检查记录文件能否被 JSON 解析。
        report = verify_training_data_lineage_report(spec, args.report)
        print(
            _json_text(
                {
                    "case_id": report["case_id"],
                    "report_fingerprint": report["report_fingerprint"],
                    "verification_scope": "full_local_recomputation",
                    "verified": True,
                }
            ),
            end="",
        )
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
