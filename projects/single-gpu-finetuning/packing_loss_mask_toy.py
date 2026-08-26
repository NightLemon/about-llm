"""Print three ways to train on two documents packed into one causal sequence."""

from __future__ import annotations

import json
import sys
from typing import Any

from about_llm.finetuning.packing import PackedDocument, build_packed_causal_lm_example

DOCUMENTS = (
    PackedDocument(document_id="docA", token_ids=(11, 12)),
    PackedDocument(document_id="docB", token_ids=(21, 22)),
)


def run_toy() -> dict[str, Any]:
    profiles = {
        "concatenated": build_packed_causal_lm_example(
            DOCUMENTS,
            eos_token_id=2,
            pad_token_id=0,
            max_length=6,
            mask_cross_document_targets=False,
            isolate_document_attention=False,
            reset_position_ids=False,
        ),
        "masked_target_only": build_packed_causal_lm_example(
            DOCUMENTS,
            eos_token_id=2,
            pad_token_id=0,
            max_length=6,
            mask_cross_document_targets=True,
            isolate_document_attention=False,
            reset_position_ids=False,
        ),
        "document_isolated": build_packed_causal_lm_example(
            DOCUMENTS,
            eos_token_id=2,
            pad_token_id=0,
            max_length=6,
            mask_cross_document_targets=True,
            isolate_document_attention=True,
            reset_position_ids=True,
        ),
    }
    return {
        "schema_version": "about-llm.packing-loss-mask-toy.v1",
        "token_legend": {
            "0": "PAD",
            "2": "EOS",
            "11": "A1",
            "12": "A2",
            "21": "B1",
            "22": "B2",
        },
        "profiles": {name: profile.to_dict() for name, profile in profiles.items()},
        "comparison": {
            "cross_document_prediction": {
                "predictor_position": 2,
                "predictor_token": "EOS(docA)",
                "target_position": 3,
                "target_token": "B1(docB)",
                "concatenated_included_in_loss": profiles["concatenated"].loss_mask[3],
                "masked_target_included_in_loss": profiles[
                    "masked_target_only"
                ].loss_mask[3],
            },
            "doc_b_first_token_attention_keys": {
                name: list(profile.allowed_key_positions[3])
                for name, profile in profiles.items()
            },
        },
        "scope": {
            "authored_token_ids": True,
            "tokenizer_or_model_executed": False,
            "framework_collator_or_packed_kernel_executed": False,
            "one_fixed_decoder_only_sequence": True,
            "production_data_quality_or_throughput_proven": False,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run_toy(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
