from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.model_config import (
    MODEL_CONFIG_EVIDENCE_BOUNDARY,
    estimate_standard_kv_cache,
    inspect_decoder_config,
    load_model_config_json,
)

pytestmark = pytest.mark.formula

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "projects" / "transformers-basics" / "configs"


def test_standard_gqa_fixture_has_exact_ideal_kv_ledger() -> None:
    inspection = inspect_decoder_config(
        load_model_config_json(CONFIGS / "standard-gqa.example.json")
    )
    layout = inspection.standard_kv_layout
    estimate = estimate_standard_kv_cache(inspection, token_count=4096)

    assert inspection.config_fingerprint == (
        "sha256:16839fe12b7e1280d1a5fd60102387e1aed21d6dbf0a03c148c53da46b731e46"
    )
    assert layout.applicable is True
    assert layout.attention_kind == "gqa"
    assert layout.head_dim == 128
    assert layout.query_heads_per_kv_head == 4
    assert estimate.bytes_per_token_per_layer == 4096
    assert estimate.total_bytes == 536_870_912
    assert estimate.to_dict()["ideal_tensor_payload_only"] is True
    assert (
        estimate.to_dict()[
            "includes_allocator_metadata_alignment_workspace_or_scales"
        ]
        is False
    )


def test_moe_markers_do_not_change_standard_attention_kv_formula() -> None:
    inspection = inspect_decoder_config(
        load_model_config_json(CONFIGS / "moe-gqa.example.json")
    )
    estimate = estimate_standard_kv_cache(
        inspection,
        token_count=4096,
        batch_size=2,
        element_bytes=2,
    )

    assert inspection.standard_kv_layout.attention_kind == "gqa"
    assert set(inspection.moe_marker_fields) == {
        "moe_intermediate_size",
        "num_experts_per_tok",
        "num_local_experts",
    }
    assert inspection.to_dict()["known_moe_markers_present"] is True
    assert inspection.to_dict()["parameter_count_estimated"] is False
    assert estimate.bytes_per_token_per_layer == 2048
    assert estimate.total_bytes == 402_653_184


def test_known_mla_markers_fail_closed_for_standard_kv_estimate() -> None:
    inspection = inspect_decoder_config(
        load_model_config_json(CONFIGS / "mla-moe.example.json")
    )

    assert inspection.to_dict()["known_mla_markers_present"] is True
    assert set(inspection.mla_marker_fields) == {
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    }
    assert inspection.standard_kv_layout.applicable is False
    assert "standard dense K/V formula must not be applied" in (
        inspection.standard_kv_layout.reason
    )
    with pytest.raises(ValueError, match="standard KV estimate is not applicable"):
        estimate_standard_kv_cache(inspection, token_count=1024)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"num_key_value_heads": None}, "missing explicit standard attention fields"),
        ({"num_key_value_heads": 3}, "must be divisible"),
        ({"num_attention_heads": True}, "must be a positive integer"),
        ({"tie_word_embeddings": 1}, "must be a boolean"),
        ({"rope_scaling": "linear"}, "must be an object"),
        ({"kv_lora_rank": "512"}, "must be a positive integer"),
    ],
)
def test_inspector_rejects_or_refuses_ambiguous_config(
    mutation: dict[str, object], message: str
) -> None:
    config = {
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        **mutation,
    }

    if "missing" in message or "divisible" in message:
        inspection = inspect_decoder_config(config)
        assert inspection.standard_kv_layout.applicable is False
        assert message in inspection.standard_kv_layout.reason
    else:
        with pytest.raises(ValueError, match=message):
            inspect_decoder_config(config)


def test_explicit_head_dim_avoids_hidden_size_guess() -> None:
    inspection = inspect_decoder_config(
        {
            "hidden_size": 1000,
            "head_dim": 96,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        }
    )

    assert inspection.standard_kv_layout.applicable is True
    assert inspection.standard_kv_layout.head_dim == 96


@pytest.mark.parametrize(
    ("kv_heads", "expected_kind", "expected_group_size"),
    [(8, "mha", 1), (1, "mqa", 8)],
)
def test_standard_layout_classifies_mha_and_mqa(
    kv_heads: int, expected_kind: str, expected_group_size: int
) -> None:
    inspection = inspect_decoder_config(
        {
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": kv_heads,
        }
    )

    assert inspection.standard_kv_layout.attention_kind == expected_kind
    assert (
        inspection.standard_kv_layout.query_heads_per_kv_head
        == expected_group_size
    )


def test_inspection_snapshots_nested_config_and_states_evidence_boundary() -> None:
    config = {
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "rope_scaling": {"factor": 2.0, "type": "linear"},
    }
    inspection = inspect_decoder_config(config)
    config["rope_scaling"]["factor"] = 99.0  # type: ignore[index]

    assert inspection.core_fields["rope_scaling"]["factor"] == 2.0  # type: ignore[index]
    assert inspection.to_dict()["evidence_boundary"] == MODEL_CONFIG_EVIDENCE_BOUNDARY
    assert "does not inspect weights" in MODEL_CONFIG_EVIDENCE_BOUNDARY
    assert "establish effective context length or quality" in (
        MODEL_CONFIG_EVIDENCE_BOUNDARY
    )


@pytest.mark.parametrize(
    "bad_json",
    [
        '{"hidden_size":1024,"hidden_size":2048}',
        '{"hidden_size":NaN}',
        '{"hidden_size":Infinity}',
        "[]",
    ],
)
def test_local_config_loader_rejects_ambiguous_json(
    tmp_path: Path, bad_json: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(bad_json, encoding="utf-8")

    with pytest.raises(ValueError):
        load_model_config_json(path)


def test_local_config_loader_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(b'{"model_type":"bad-utf8-\xff"}')

    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_model_config_json(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_count", 0),
        ("batch_size", True),
        ("element_bytes", -1),
    ],
)
def test_kv_estimator_requires_positive_integer_scenario(
    field: str, value: object
) -> None:
    inspection = inspect_decoder_config(
        json.loads((CONFIGS / "standard-gqa.example.json").read_text(encoding="utf-8"))
    )
    kwargs = {"token_count": 1, "batch_size": 1, "element_bytes": 2}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        estimate_standard_kv_cache(inspection, **kwargs)  # type: ignore[arg-type]
