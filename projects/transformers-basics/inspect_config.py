"""读取本地模型 config.json，解释 Decoder 架构并估算标准 KV Cache 大小。

这里只读取配置元数据，不加载权重。估算公式只适用于能从配置确认的标准 Attention 布局；
遇到未知或特殊结构时程序会说明拒绝估算的原因，避免给出貌似精确的错误数字。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.model_config import (
    estimate_standard_kv_cache,
    inspect_decoder_config,
    load_model_config_json,
)


def main() -> None:
    """解析命令行参数，检查模型结构并输出不同上下文长度的 KV 估算。"""

    parser = argparse.ArgumentParser(
        description="Inspect a local decoder config without loading model weights"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--tokens", type=int, action="append")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--element-bytes",
        type=int,
        default=2,
        help="ideal bytes per K/V element; excludes scale and allocator metadata",
    )
    args = parser.parse_args()
    # 先把不同模型的字段归一化为统一 inspection，再判断能否套用标准 KV 公式。
    inspection = inspect_decoder_config(load_model_config_json(args.config))
    token_counts = (1024, 4096, 8192) if args.tokens is None else args.tokens
    estimates: list[dict[str, object]] = []
    if inspection.standard_kv_layout.applicable:
        # 每个估算都包含 batch、token 数、层数、KV head 数和元素字节数的共同影响。
        estimates = [
            estimate_standard_kv_cache(
                inspection,
                token_count=token_count,
                batch_size=args.batch_size,
                element_bytes=args.element_bytes,
            ).to_dict()
            for token_count in token_counts
        ]
    # 即使不能估算也返回结构检查结果和明确理由，供读者判断缺少哪个架构信息。
    payload = {
        "source_path": str(args.config),
        "inspection": inspection.to_dict(),
        "standard_kv_estimates": estimates,
        "estimate_refused": not inspection.standard_kv_layout.applicable,
        "estimate_refusal_reason": (
            None
            if inspection.standard_kv_layout.applicable
            else inspection.standard_kv_layout.reason
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
