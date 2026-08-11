"""Build, strictly reload, and execute a two-matrix quantized bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from about_llm.inference import (
    QUANTIZED_BUNDLE_FORMAT_VERSION,
    QUANTIZED_BUNDLE_SCHEMA_VERSION,
    NamedQuantizedMatrix,
    QuantizedBundleIdentity,
    QuantizedMatrixBundle,
    quantization_error,
    quantize_symmetric_groupwise,
    quantized_linear,
)


def run_toy(
    *,
    seed: int,
    bit_width: int,
    group_size: int,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0, 0.5, size=(3, 6)).astype(np.float32)
    first_weight = rng.normal(0, 0.4, size=(8, 6)).astype(np.float32)
    second_weight = rng.normal(0, 0.4, size=(3, 8)).astype(np.float32)
    first = quantize_symmetric_groupwise(
        first_weight, bit_width=bit_width, group_size=group_size
    ).pack()
    second = quantize_symmetric_groupwise(
        second_weight, bit_width=bit_width, group_size=group_size
    ).pack()
    bundle = QuantizedMatrixBundle(
        identity=QuantizedBundleIdentity(
            model_family="authored-two-layer-mlp",
            model_revision=f"fixture-seed-{seed}",
            tokenizer_id="integer-input-fixture",
            tokenizer_revision="not-a-text-tokenizer-v1",
            architecture_config={
                "input_features": 6,
                "hidden_features": 8,
                "output_features": 3,
                "activation": "tanh",
                "bias": False,
            },
        ),
        tensors=(
            NamedQuantizedMatrix("layer.0.weight", first),
            NamedQuantizedMatrix("layer.1.weight", second),
        ),
    )
    serialized = bundle.to_bytes()
    reloaded = QuantizedMatrixBundle.from_bytes(serialized)
    disk_round_trip = False
    if artifact_path is not None:
        bundle.write_new(artifact_path)
        reloaded = QuantizedMatrixBundle.read(artifact_path)
        disk_round_trip = True

    fp32_output = np.tanh(inputs @ first_weight.T) @ second_weight.T
    quantized_output = quantized_linear(
        np.tanh(
            quantized_linear(inputs, reloaded.get("layer.0.weight").unpack())
        ),
        reloaded.get("layer.1.weight").unpack(),
    )
    output_error = quantization_error(fp32_output, quantized_output)
    tensor_artifact_bytes = sum(
        len(item.matrix.to_bytes()) for item in bundle.tensors
    )
    return {
        "schema_version": 1,
        "bundle_format_version": QUANTIZED_BUNDLE_FORMAT_VERSION,
        "bundle_schema_version": QUANTIZED_BUNDLE_SCHEMA_VERSION,
        "identity": reloaded.identity.to_dict(),
        "tensor_names": list(reloaded.tensor_names),
        "tensor_count": len(reloaded.tensors),
        "reference_fp32_weight_bytes": first_weight.nbytes + second_weight.nbytes,
        "raw_quantized_payload_bytes": first.raw_payload_bytes + second.raw_payload_bytes,
        "individual_tensor_artifact_bytes": tensor_artifact_bytes,
        "bundle_artifact_bytes": len(serialized),
        "bundle_container_overhead_bytes": len(serialized) - tensor_artifact_bytes,
        "bundle_sha256": bundle.serialized_artifact_sha256,
        "exact_byte_round_trip": reloaded.to_bytes() == serialized,
        "exact_quantized_forward_round_trip": bool(
            np.array_equal(
                quantized_output,
                quantized_linear(
                    np.tanh(quantized_linear(inputs, first.unpack())),
                    second.unpack(),
                ),
            )
        ),
        "output_error_vs_fp32": {
            "mean_absolute_error": output_error.mean_absolute_error,
            "root_mean_squared_error": output_error.root_mean_squared_error,
            "maximum_absolute_error": output_error.maximum_absolute_error,
            "relative_l2_error": output_error.relative_l2_error,
        },
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "disk_round_trip": disk_round_trip,
        "scope": {
            "network_used": False,
            "multiple_named_quantized_matrices_embedded": True,
            "architecture_and_revision_identity_embedded": True,
            "tokenizer_identity_embedded": True,
            "tokenizer_payload_embedded": False,
            "unquantized_parameter_kinds_supported": False,
            "model_forward_implementation_embedded": False,
            "runtime_specific_layout_or_kernel": False,
            "fused_low_bit_execution": False,
            "cryptographic_origin_authenticated": False,
            "full_llm_checkpoint": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        help="optional new path; existing files are rejected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_toy(
                seed=args.seed,
                bit_width=args.bit_width,
                group_size=args.group_size,
                artifact_path=args.artifact_path,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
