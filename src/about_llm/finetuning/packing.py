"""A deterministic reference for packed causal-LM labels and attention."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

PACKED_CAUSAL_LM_VERSION = "about-llm.packed-causal-lm.v1"
IGNORE_INDEX = -100


def _validate_token_id(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class PackedDocument:
    """One already-tokenized document, before EOS is appended by the packer."""

    document_id: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise ValueError("token_ids must be a non-empty tuple")
        for index, token_id in enumerate(self.token_ids):
            _validate_token_id(token_id, f"token_ids[{index}]")


@dataclass(frozen=True)
class PackedNextTokenTarget:
    """The target at i + 1 that is predicted by the hidden state at i."""

    predictor_position: int
    target_position: int
    predictor_token_id: int
    target_token_id: int
    predictor_document_id: str
    target_document_id: str
    crosses_document_boundary: bool
    included_in_loss: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "predictor_position": self.predictor_position,
            "target_position": self.target_position,
            "predictor_token_id": self.predictor_token_id,
            "target_token_id": self.target_token_id,
            "predictor_document_id": self.predictor_document_id,
            "target_document_id": self.target_document_id,
            "crosses_document_boundary": self.crosses_document_boundary,
            "included_in_loss": self.included_in_loss,
        }


@dataclass(frozen=True)
class PackedCausalLMExample:
    """Materialized labels and attention for one packed fixed-length sequence."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    position_ids: tuple[int, ...]
    document_ids: tuple[str | None, ...]
    attention_mask: tuple[tuple[bool, ...], ...]
    targets: tuple[PackedNextTokenTarget, ...]
    mask_cross_document_targets: bool
    isolate_document_attention: bool
    reset_position_ids: bool
    non_padding_token_count: int
    objective_token_count: int

    @property
    def allocated_slots(self) -> int:
        return len(self.input_ids)

    @property
    def non_padding_occupancy(self) -> float:
        return self.non_padding_token_count / self.allocated_slots

    @property
    def objective_token_utilization(self) -> float:
        return self.objective_token_count / self.allocated_slots

    @property
    def allowed_key_positions(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(index for index, allowed in enumerate(row) if allowed)
            for row in self.attention_mask
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PACKED_CAUSAL_LM_VERSION,
            "config": {
                "mask_cross_document_targets": self.mask_cross_document_targets,
                "isolate_document_attention": self.isolate_document_attention,
                "reset_position_ids": self.reset_position_ids,
                "ignore_index": IGNORE_INDEX,
            },
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "loss_mask": [int(value) for value in self.loss_mask],
            "position_ids": list(self.position_ids),
            "document_ids": list(self.document_ids),
            "allowed_key_positions": [
                list(positions) for positions in self.allowed_key_positions
            ],
            "targets": [target.to_dict() for target in self.targets],
            "counts": {
                "allocated_slots": self.allocated_slots,
                "non_padding_tokens": self.non_padding_token_count,
                "objective_tokens": self.objective_token_count,
                "non_padding_occupancy": self.non_padding_occupancy,
                "objective_token_utilization": self.objective_token_utilization,
            },
        }


def build_packed_causal_lm_example(
    documents: Sequence[PackedDocument],
    *,
    eos_token_id: int,
    pad_token_id: int,
    max_length: int,
    mask_cross_document_targets: bool,
    isolate_document_attention: bool,
    reset_position_ids: bool,
) -> PackedCausalLMExample:
    """Pack documents and expose effective next-token targets.

    Labels are aligned with input positions as expected by common causal-LM
    implementations. Position zero is ignored because no hidden state inside the
    sequence predicts it. If requested, the first token of every later document is
    also ignored so an EOS token does not learn to predict the next document.
    """

    _validate_token_id(eos_token_id, "eos_token_id")
    _validate_token_id(pad_token_id, "pad_token_id")
    if eos_token_id == pad_token_id:
        raise ValueError("eos_token_id and pad_token_id must be distinct")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    options = (
        mask_cross_document_targets,
        isolate_document_attention,
        reset_position_ids,
    )
    if any(not isinstance(value, bool) for value in options):
        raise TypeError("packing options must be booleans")

    selected = tuple(documents)
    if not selected or any(not isinstance(item, PackedDocument) for item in selected):
        raise ValueError("documents must contain at least one PackedDocument")
    document_ids = tuple(item.document_id for item in selected)
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("document ids must be unique within one packed sequence")
    for document in selected:
        if eos_token_id in document.token_ids or pad_token_id in document.token_ids:
            raise ValueError("document token_ids must not contain EOS or PAD")

    required_slots = sum(len(document.token_ids) + 1 for document in selected)
    if required_slots > max_length:
        raise ValueError("documents plus EOS tokens exceed max_length")

    input_ids: list[int] = []
    owners: list[str | None] = []
    local_positions: list[int] = []
    for document in selected:
        input_ids.extend(document.token_ids)
        input_ids.append(eos_token_id)
        owners.extend((document.document_id,) * (len(document.token_ids) + 1))
        local_positions.extend(range(len(document.token_ids) + 1))

    non_padding_token_count = len(input_ids)
    padding = max_length - non_padding_token_count
    input_ids.extend((pad_token_id,) * padding)
    owners.extend((None,) * padding)
    local_positions.extend((0,) * padding)

    labels = [IGNORE_INDEX] * max_length
    for target_position in range(1, non_padding_token_count):
        crosses_boundary = owners[target_position - 1] != owners[target_position]
        if not (mask_cross_document_targets and crosses_boundary):
            labels[target_position] = input_ids[target_position]
    loss_mask = tuple(label != IGNORE_INDEX for label in labels)

    position_ids = tuple(
        local_positions[position] if reset_position_ids else position
        for position in range(non_padding_token_count)
    ) + (0,) * padding

    attention_mask = tuple(
        tuple(
            owners[query] is not None
            and owners[key] is not None
            and key <= query
            and (
                not isolate_document_attention
                or owners[key] == owners[query]
            )
            for key in range(max_length)
        )
        for query in range(max_length)
    )

    targets = tuple(
        PackedNextTokenTarget(
            predictor_position=predictor_position,
            target_position=predictor_position + 1,
            predictor_token_id=input_ids[predictor_position],
            target_token_id=input_ids[predictor_position + 1],
            predictor_document_id=owners[predictor_position] or "",
            target_document_id=owners[predictor_position + 1] or "",
            crosses_document_boundary=(
                owners[predictor_position] != owners[predictor_position + 1]
            ),
            included_in_loss=loss_mask[predictor_position + 1],
        )
        for predictor_position in range(non_padding_token_count - 1)
    )

    return PackedCausalLMExample(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        loss_mask=loss_mask,
        position_ids=position_ids,
        document_ids=tuple(owners),
        attention_mask=attention_mask,
        targets=targets,
        mask_cross_document_targets=mask_cross_document_targets,
        isolate_document_attention=isolate_document_attention,
        reset_position_ids=reset_position_ids,
        non_padding_token_count=non_padding_token_count,
        objective_token_count=sum(loss_mask),
    )
