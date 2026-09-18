from __future__ import annotations

import pytest

pytestmark = pytest.mark.formula


def test_iteration_boundary_checkpoint_payload_is_fourteen_bytes_per_parameter() -> None:
    bf16_weight_bytes = 2
    fp32_master_weight_bytes = 4
    two_fp32_adam_moment_bytes = 2 * 4

    checkpoint_bytes = (
        bf16_weight_bytes + fp32_master_weight_bytes + two_fp32_adam_moment_bytes
    )

    assert checkpoint_bytes == 14


def test_live_training_ledger_cannot_reuse_checkpoint_payload_when_gradient_is_resident() -> None:
    checkpoint_bytes = 14
    fp32_gradient_bytes = 4

    assert checkpoint_bytes + fp32_gradient_bytes == 18
    assert checkpoint_bytes != 18
