# ruff: noqa: RUF001 -- Full-width punctuation is intentional in Chinese learner output.
"""Walk through split leakage, NLL, one logit update, and sliced metrics."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final

LABELS: Final = ("refund", "logistics", "fraud_review")


@dataclass(frozen=True, slots=True)
class TicketRow:
    row_id: str
    thread_id: str
    label: str


ROWS: Final = (
    TicketRow("row-001", "thread-100", "refund"),
    TicketRow("row-002", "thread-100", "refund"),
    TicketRow("row-003", "thread-200", "logistics"),
    TicketRow("row-004", "thread-200", "logistics"),
    TicketRow("row-005", "thread-300", "fraud_review"),
    TicketRow("row-006", "thread-400", "fraud_review"),
)

ROW_LEVEL_SPLIT: Final = {
    "row-001": "train",
    "row-002": "test",
    "row-003": "train",
    "row-004": "test",
    "row-005": "train",
    "row-006": "validation",
}

THREAD_LEVEL_SPLIT: Final = {
    "row-001": "train",
    "row-002": "train",
    "row-003": "train",
    "row-004": "train",
    "row-005": "validation",
    "row-006": "test",
}


def _split_threads(assignments: dict[str, str]) -> dict[str, list[str]]:
    result = {"train": [], "validation": [], "test": []}
    for split in result:
        result[split] = sorted({row.thread_id for row in ROWS if assignments[row.row_id] == split})
    return result


def _overlapping_threads(split_threads: dict[str, list[str]]) -> list[str]:
    owners: dict[str, set[str]] = {}
    for split, thread_ids in split_threads.items():
        for thread_id in thread_ids:
            owners.setdefault(thread_id, set()).add(split)
    return sorted(thread_id for thread_id, splits in owners.items() if len(splits) > 1)


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def _classification_step(
    logits: tuple[float, ...], *, target_index: int, learning_rate: float
) -> dict[str, object]:
    if len(logits) != len(LABELS):
        raise ValueError("logits must contain one value per label")
    if not 0 <= target_index < len(logits):
        raise ValueError("target_index is outside the label set")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    probabilities = _softmax(logits)
    loss = -math.log(probabilities[target_index])
    gradient = tuple(
        probability - (1.0 if index == target_index else 0.0)
        for index, probability in enumerate(probabilities)
    )
    updated_logits = tuple(
        value - learning_rate * derivative
        for value, derivative in zip(logits, gradient, strict=True)
    )
    updated_probabilities = _softmax(updated_logits)
    updated_loss = -math.log(updated_probabilities[target_index])
    return {
        "logits": logits,
        "probabilities": probabilities,
        "target_index": target_index,
        "target_label": LABELS[target_index],
        "negative_log_likelihood": loss,
        "logit_gradient": gradient,
        "learning_rate": learning_rate,
        "updated_logits": updated_logits,
        "updated_probabilities": updated_probabilities,
        "updated_negative_log_likelihood": updated_loss,
    }


def build_walkthrough() -> dict[str, object]:
    row_split_threads = _split_threads(ROW_LEVEL_SPLIT)
    thread_split_threads = _split_threads(THREAD_LEVEL_SPLIT)
    prediction = _classification_step(
        (2.0, 0.5, -1.0),
        target_index=2,
        learning_rate=2.0,
    )
    return {
        "task": {
            "prediction_time": "first_route_request",
            "sample_unit": "thread_id",
            "labels": LABELS,
            "high_cost_error": "missed_fraud_review",
        },
        "split_audit": {
            "row_level": {
                "threads": row_split_threads,
                "overlap_threads": _overlapping_threads(row_split_threads),
            },
            "thread_level": {
                "threads": thread_split_threads,
                "overlap_threads": _overlapping_threads(thread_split_threads),
            },
        },
        "prediction": prediction,
        "metrics": {
            "total_tickets": 100,
            "correct_tickets": 99,
            "fraud_review_tickets": 1,
            "fraud_review_true_positives": 0,
            "accuracy": 0.99,
            "fraud_review_recall": 0.0,
        },
        "scope": {
            "real_ticket_text_or_trained_model_used": False,
            "logits_treated_as_direct_parameters_for_one_local_step": True,
            "generalization_or_business_value_measured": False,
        },
    }


def _vector(values: object) -> str:
    if not isinstance(values, tuple):
        raise TypeError("expected a tuple")
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def render_walkthrough(walkthrough: dict[str, object]) -> str:
    split_audit = walkthrough["split_audit"]
    prediction = walkthrough["prediction"]
    metrics = walkthrough["metrics"]
    if not isinstance(split_audit, dict) or not isinstance(prediction, dict):
        raise TypeError("walkthrough sections must be objects")
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be an object")
    row_level = split_audit["row_level"]
    thread_level = split_audit["thread_level"]
    if not isinstance(row_level, dict) or not isinstance(thread_level, dict):
        raise TypeError("split audit entries must be objects")

    return "\n".join(
        [
            "工单分类最小学习闭环",
            "",
            "1. 先检查切分单位",
            f"逐行切分的跨集合 thread: {row_level['overlap_threads']}",
            f"按 thread 切分的跨集合 thread: {thread_level['overlap_threads']}",
            "结论: 同一 thread 出现在训练和测试中时，测试集不再代表新会话。",
            "",
            "2. 再手算一条 fraud_review 样本",
            f"logits: {_vector(prediction['logits'])}",
            f"probabilities: {_vector(prediction['probabilities'])}",
            f"target: {prediction['target_label']}",
            f"NLL: {float(prediction['negative_log_likelihood']):.4f}",
            f"dL/dlogits = p-y: {_vector(prediction['logit_gradient'])}",
            "",
            "3. 只把 logits 当作参数，观察一次局部梯度步",
            f"updated logits: {_vector(prediction['updated_logits'])}",
            f"updated probabilities: {_vector(prediction['updated_probabilities'])}",
            f"updated NLL: {float(prediction['updated_negative_log_likelihood']):.4f}",
            "结论: 目标类别概率上升、NLL 下降，只说明这一步的训练方向符合目标。",
            "",
            "4. 最后把总体指标拆开",
            f"accuracy: {int(metrics['correct_tickets'])}/{int(metrics['total_tickets'])} "
            f"= {float(metrics['accuracy']):.2f}",
            "fraud_review recall: "
            f"{int(metrics['fraud_review_true_positives'])}/"
            f"{int(metrics['fraud_review_tickets'])} "
            f"= {float(metrics['fraud_review_recall']):.2f}",
            "结论: 99% accuracy 可以与风险类 recall=0 同时出现。",
            "",
            "本实验没有训练真实模型，也没有测量泛化或业务收益。",
        ]
    )


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    print(render_walkthrough(build_walkthrough()))


if __name__ == "__main__":
    main()
