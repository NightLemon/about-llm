from __future__ import annotations

import pytest

from about_llm.finetuning.packing import (
    IGNORE_INDEX,
    PackedDocument,
    build_packed_causal_lm_example,
)

pytestmark = pytest.mark.formula

DOCUMENTS = (
    PackedDocument(document_id="docA", token_ids=(11, 12)),
    PackedDocument(document_id="docB", token_ids=(21, 22)),
)


def _build(
    *,
    eos_token_id: int = 2,
    pad_token_id: int = 0,
    max_length: int = 6,
    mask_cross_document_targets: bool = True,
    isolate_document_attention: bool = False,
    reset_position_ids: bool = False,
):
    return build_packed_causal_lm_example(
        DOCUMENTS,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_length=max_length,
        mask_cross_document_targets=mask_cross_document_targets,
        isolate_document_attention=isolate_document_attention,
        reset_position_ids=reset_position_ids,
    )


def test_masking_cross_document_target_changes_loss_not_attention() -> None:
    example = _build()

    assert example.input_ids == (11, 12, 2, 21, 22, 2)
    assert example.labels == (IGNORE_INDEX, 12, 2, IGNORE_INDEX, 22, 2)
    assert example.loss_mask == (False, True, True, False, True, True)
    assert example.objective_token_count == 4
    assert example.non_padding_occupancy == 1
    assert example.objective_token_utilization == pytest.approx(4 / 6)
    assert example.allowed_key_positions[3] == (0, 1, 2, 3)

    boundary_target = example.targets[2]
    assert boundary_target.predictor_token_id == 2
    assert boundary_target.target_token_id == 21
    assert boundary_target.crosses_document_boundary is True
    assert boundary_target.included_in_loss is False


def test_unmasked_concat_includes_eos_to_next_document_target() -> None:
    example = _build(mask_cross_document_targets=False)

    assert example.labels == (IGNORE_INDEX, 12, 2, 21, 22, 2)
    assert example.objective_token_count == 5
    assert example.targets[2].included_in_loss is True
    assert example.allowed_key_positions[3] == (0, 1, 2, 3)


def test_document_isolation_resets_positions_and_blocks_prior_document() -> None:
    example = _build(isolate_document_attention=True, reset_position_ids=True)

    assert example.position_ids == (0, 1, 2, 0, 1, 2)
    assert example.allowed_key_positions == (
        (0,),
        (0, 1),
        (0, 1, 2),
        (3,),
        (3, 4),
        (3, 4, 5),
    )
    assert example.labels == (IGNORE_INDEX, 12, 2, IGNORE_INDEX, 22, 2)


def test_padding_is_excluded_from_attention_and_both_utilization_counts() -> None:
    example = _build(max_length=8, isolate_document_attention=True)

    assert example.input_ids[-2:] == (0, 0)
    assert example.document_ids[-2:] == (None, None)
    assert example.labels[-2:] == (IGNORE_INDEX, IGNORE_INDEX)
    assert example.allowed_key_positions[-2:] == ((), ())
    assert example.non_padding_occupancy == pytest.approx(6 / 8)
    assert example.objective_token_utilization == pytest.approx(4 / 8)


def test_packer_rejects_ambiguous_or_oversized_inputs() -> None:
    with pytest.raises(ValueError, match="distinct"):
        _build(eos_token_id=2, pad_token_id=2)
    with pytest.raises(ValueError, match="exceed"):
        _build(max_length=5)
    with pytest.raises(ValueError, match="unique"):
        build_packed_causal_lm_example(
            (DOCUMENTS[0], DOCUMENTS[0]),
            eos_token_id=2,
            pad_token_id=0,
            max_length=6,
            mask_cross_document_targets=True,
            isolate_document_attention=True,
            reset_position_ids=True,
        )
