from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "train_byte_bpe.py"


def test_byte_bpe_project_emits_reproducible_machine_readable_evidence() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--text",
            "abab",
            "--sample",
            "abab",
            "--vocab-size",
            "258",
            "--min-pair-frequency",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)

    assert payload["implementation"] == "about-llm.byte-bpe-reference.v1"
    assert payload["actual_vocab_size"] == 258
    assert payload["merges"] == [
        {
            "rank": 0,
            "left_id": ord("a"),
            "right_id": ord("b"),
            "new_id": 256,
            "bytes_hex": "6162",
            "utf8_preview": "ab",
        },
        {
            "rank": 1,
            "left_id": 256,
            "right_id": 256,
            "new_id": 257,
            "bytes_hex": "61626162",
            "utf8_preview": "abab",
        },
    ]
    assert payload["samples"][0]["token_ids"] == [257]
    assert payload["samples"][0]["round_trip"] is True
    assert "no normalization" in payload["evidence_boundary"]
