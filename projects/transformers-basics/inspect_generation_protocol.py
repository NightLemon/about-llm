"""检查 tokenizer、模型配置和生成配置中的特殊 token 是否一致。

输入是仓库定义的 generation protocol JSON 快照。本实验只比较文件中明确记录的值，
不会猜测 Transformers 在特定版本运行时可能补上的默认值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.generation_contract import (
    inspect_generation_protocol_document,
    load_generation_protocol_json,
)


def main() -> None:
    """加载协议快照，执行跨组件一致性检查并打印结果。"""

    parser = argparse.ArgumentParser(
        description=(
            "Compare explicit tokenizer/model/generation special-token snapshots "
            "without inferring effective runtime defaults"
        )
    )
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    # 检查 BOS/EOS/PAD 等 ID 是否冲突，以及必需字段是否缺失。
    inspection = inspect_generation_protocol_document(
        load_generation_protocol_json(args.protocol)
    )
    print(json.dumps(inspection.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
