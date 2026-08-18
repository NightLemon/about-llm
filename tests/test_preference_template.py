from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from about_llm.finetuning import (
    audit_preference_tokenization,
    load_preference_records,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "projects" / "single-gpu-finetuning" / "preference.train.example.jsonl"
)
pytestmark = pytest.mark.contract


def _render(_: dict[str, Any]) -> dict[str, list[int]]:
    return {
        "prompt_ids": [1, 2],
        "prompt_chosen_ids": [1, 2, 3],
        "prompt_rejected_ids": [1, 2, 4, 5],
    }


def test_preference_tokenization_audit_binds_target_renderer_and_tokens() -> None:
    records = load_preference_records(FIXTURE)

    report = audit_preference_tokenization(
        records,
        render=_render,
        renderer_identity={"model": "local", "revision": "abc"},
        max_length=8,
    )

    assert report.record_count == 2
    assert report.samples[0].prompt_token_count == 2
    assert report.samples[0].chosen_completion_token_count == 1
    assert report.manifest_fingerprint.startswith("sha256:")
    assert report.to_dict()["scope"]["prompt_prefix_match_required"] is True


@pytest.mark.parametrize(
    ("rendered", "message"),
    (
        (
            {
                "prompt_ids": [1, 2],
                "prompt_chosen_ids": [1, 9, 3],
                "prompt_rejected_ids": [1, 2, 4],
            },
            "not an exact prefix",
        ),
        (
            {
                "prompt_ids": [1, 2],
                "prompt_chosen_ids": [1, 2, 3],
                "prompt_rejected_ids": [1, 2, 3],
            },
            "same completion ids",
        ),
        (
            {
                "prompt_ids": [1, 2],
                "prompt_chosen_ids": [1, 2, 3, 4, 5],
                "prompt_rejected_ids": [1, 2, 6],
            },
            "exceeds max_length",
        ),
    ),
)
def test_preference_tokenization_audit_fails_closed(
    rendered: dict[str, list[int]], message: str
) -> None:
    record = load_preference_records(FIXTURE)[0]

    with pytest.raises(ValueError, match=message):
        audit_preference_tokenization(
            (record,),
            render=lambda _: rendered,
            renderer_identity={"model": "local"},
            max_length=4,
        )
