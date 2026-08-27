"""在固定 Qwen checkpoint 上运行或复核单矩阵 INT4 量化实验。

默认路径加载目标矩阵、打包为 INT4、反量化并比较误差；``--verify`` 路径不加载模型，
只检查已有报告的 schema、数值关系和 manifest 指纹是否仍与当前实验一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.integrations.transformers_weight_quantization_control import (
    run_target_weight_quantization_control,
    verify_recorded_target_weight_quantization_report,
)

TARGET_DIRECTORY = Path(__file__).with_name("target-checkpoints")
DEFAULT_MANIFEST = TARGET_DIRECTORY / "qwen2.5-0.5b-instruct.control.json"


def main() -> int:
    """根据参数选择“实际运行量化”或“离线验证已有报告”。"""

    parser = argparse.ArgumentParser(
        description=(
            "Run one selected-matrix packed INT4 control on the reviewed Qwen "
            "CPU FP32 checkpoint, or verify a recorded closed report"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed snapshot to already exist in the local cache",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="verify a recorded report without loading the target checkpoint",
    )
    args = parser.parse_args()
    # 两条路径都先读取 manifest，确保离线报告也绑定到同一 checkpoint 身份。
    spec = load_checkpoint_control_spec(args.manifest)
    if args.verify is not None:
        # 离线验证适合 CI：无需下载权重，也能发现报告篡改或版本漂移。
        report = verify_recorded_target_weight_quantization_report(
            args.verify,
            expected_manifest_fingerprint=spec.manifest_fingerprint,
        )
    else:
        # 实际运行路径需要目标 checkpoint，并会重新测量量化误差。
        report = run_target_weight_quantization_control(
            spec,
            local_files_only=args.local_files_only,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
