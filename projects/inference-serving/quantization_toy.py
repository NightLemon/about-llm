"""在 CPU 上完整走一遍 groupwise weight-only 量化、打包、加载和矩阵乘。

实验不只估算低比特大小：它会实际生成 code 与 scale，按 bit stream 打包成二进制，重新加载，
再比较 FP32 与量化线性层输出。可选 artifact 路径使用新建模式，避免覆盖已有文件。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from about_llm.inference import (
    QUANTIZED_MATRIX_FORMAT_VERSION,
    PackedGroupwiseQuantizedMatrix,
    quantization_error,
    quantize_symmetric_groupwise,
    quantized_linear,
)


def run_experiment(
    *,
    seed: int,
    output_features: int,
    input_features: int,
    batch_size: int,
    bit_width: int,
    group_size: int,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """量化一张合成权重矩阵，验证字节往返并测量线性输出误差。"""

    # 固定 seed 让权重、输入和所有误差指标可重复。
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.25, size=(output_features, input_features)).astype(
        np.float32
    )
    # 专门放入一个全零 group，验证 absmax=0 不会产生除零或非法 scale。
    weights[0, : min(group_size, input_features)] = 0
    inputs = rng.normal(size=(batch_size, input_features)).astype(np.float32)
    # 每一行按连续 group 独立求 scale，随后把有符号 code 压入低比特字节流。
    quantized = quantize_symmetric_groupwise(
        weights,
        bit_width=bit_width,
        group_size=group_size,
    )
    packed = quantized.pack()
    unpacked = packed.unpack()
    # 内存 pack→unpack 必须逐 code、逐 scale 完全一致，误差只能来自量化本身。
    if not np.array_equal(unpacked.values, quantized.values) or not np.array_equal(
        unpacked.scales, quantized.scales
    ):
        raise RuntimeError("packed quantization round-trip changed codes or scales")
    # 再经过自描述二进制容器往返，覆盖 header、payload 和 SHA-256 完整性。
    serialized = packed.to_bytes()
    reloaded = PackedGroupwiseQuantizedMatrix.from_bytes(serialized)
    if not np.array_equal(reloaded.unpack().values, quantized.values) or not np.array_equal(
        reloaded.scales, quantized.scales
    ):
        raise RuntimeError("serialized quantization round-trip changed codes or scales")
    artifact_written = artifact_path is not None
    if artifact_path is not None:
        # xb 拒绝覆盖，防止命令误删已有量化 artifact。
        with artifact_path.open("xb") as handle:
            handle.write(serialized)
        disk_reloaded = PackedGroupwiseQuantizedMatrix.from_bytes(
            artifact_path.read_bytes()
        )
        if disk_reloaded.serialized_artifact_sha256 != packed.serialized_artifact_sha256:
            raise RuntimeError("written quantization artifact failed exact reload")
    # 最后真正执行量化线性层，并与原 FP32 权重的矩阵乘输出比较。
    reconstructed = packed.dequantize()
    baseline_output = inputs @ weights.T
    quantized_output = quantized_linear(inputs, unpacked)
    return {
        "schema_version": 2,
        "configuration": {
            "seed": seed,
            "output_features": output_features,
            "input_features": input_features,
            "batch_size": batch_size,
            "bit_width": bit_width,
            "group_size": group_size,
            "rounding": "numpy.rint on float32 normalized values",
            "code_range": [-quantized.maximum_code, quantized.maximum_code],
            "packing": "row-major offset-binary dense bitstream, LSB-first v1",
        },
        "storage": {
            "reference_fp32_weight_bytes": quantized.reference_fp32_weight_bytes,
            "ideal_packed_weight_bytes": quantized.ideal_packed_weight_bytes,
            "float32_scale_metadata_bytes": quantized.scale_metadata_bytes,
            "ideal_total_bytes": quantized.ideal_total_bytes,
            "ideal_compression_ratio": quantized.ideal_compression_ratio,
            "numpy_unpacked_reference_bytes": quantized.unpacked_reference_bytes,
            "actual_dense_packed_weight_bytes": packed.packed_weight_bytes,
            "actual_code_plus_fp32_scale_payload_bytes": packed.raw_payload_bytes,
            "final_byte_padding_bits": packed.padding_bits,
            "raw_payload_includes_container_header_or_alignment": False,
            "serialized_tensor_artifact_bytes": packed.serialized_artifact_bytes,
        },
        "packed_codes": {
            "hex": packed.packed_values.hex(),
            "sha256": packed.packed_values_sha256,
            "round_trip_codes_exact": True,
            "round_trip_scales_exact": True,
        },
        "tensor_artifact": {
            "format_version": QUANTIZED_MATRIX_FORMAT_VERSION,
            "byte_order": "little-endian header and float32 scales",
            "integrity": "trailing unkeyed SHA-256 over header and payload",
            "sha256": packed.serialized_artifact_sha256,
            "round_trip_exact": True,
            "written_to_disk": artifact_written,
            "path": str(artifact_path) if artifact_path is not None else None,
        },
        "weight_error": asdict(quantization_error(weights, reconstructed)),
        "linear_output_error": asdict(
            quantization_error(baseline_output, quantized_output)
        ),
        "scope": {
            "device": "CPU",
            "weights": "seeded synthetic matrix",
            "quantizer": "symmetric absmax per contiguous row group",
            "actual_low_bit_packing_executed": True,
            "self_contained_quantized_tensor_artifact_constructed": True,
            "self_contained_model_artifact_written": False,
            "fused_low_bit_kernel_executed": False,
            "calibration_or_gptq_awq_executed": False,
            "model_quality_or_latency_proved": False,
        },
    }


def positive_integer(value: str) -> int:
    """把命令行参数解析为严格正整数。"""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """定义矩阵 shape、位宽、group 大小和可选 artifact 路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-features", type=positive_integer, default=16)
    parser.add_argument("--input-features", type=positive_integer, default=33)
    parser.add_argument("--batch-size", type=positive_integer, default=8)
    parser.add_argument(
        "--bit-width", type=positive_integer, choices=range(2, 9), default=4
    )
    parser.add_argument("--group-size", type=positive_integer, default=8)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        help="optional new path for the strict binary tensor artifact; never overwritten",
    )
    return parser.parse_args()


def main() -> None:
    """执行量化闭环并打印真实字节数与误差。"""

    args = parse_args()
    artifact = run_experiment(
        seed=args.seed,
        output_features=args.output_features,
        input_features=args.input_features,
        batch_size=args.batch_size,
        bit_width=args.bit_width,
        group_size=args.group_size,
        artifact_path=args.artifact_path,
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
