"""核对 Llama、Qwen 和 DeepSeek 模型页面所引用的发布证据。

默认只验证仓库内清单、快照和哈希之间的关系；加入 ``--verify-upstream`` 后才会下载
固定 revision 的公开文件并逐字节校验。它验证资料来源，不加载模型权重或评测模型质量。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.model_release_evidence import verify_model_release_evidence

DEFAULT_MANIFEST = Path(__file__).with_name("release-evidence") / "manifest.json"


def main() -> None:
    """验证本地证据清单，并可选地与上游固定文件比对。"""

    parser = argparse.ArgumentParser(
        description=(
            "Verify hash-pinned Llama/Qwen/DeepSeek release evidence without "
            "loading model weights"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--verify-upstream",
        action="store_true",
        help="download immutable public artifacts and verify exact bytes",
    )
    args = parser.parse_args()
    # 网络验证是显式选项，默认运行保持离线、快速且可重复。
    report = verify_model_release_evidence(
        args.manifest,
        verify_upstream=args.verify_upstream,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
