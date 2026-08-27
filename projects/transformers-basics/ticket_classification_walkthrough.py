# ruff: noqa: RUF001 -- Full-width punctuation is intentional in Chinese learner output.
"""用一个最小工单分类任务串起数据切分、训练目标和分类指标。

这个实验没有训练神经网络，而是把真实训练流程中最容易混淆的四件事压缩到一个文件里：

1. 先确定一条样本究竟是一行数据，还是一段完整会话；
2. 比较错误的逐行切分和正确的按会话切分；
3. 对一条样本手算 softmax、负对数似然（NLL）和一次梯度更新；
4. 对比总体 accuracy 与高风险类别 recall，观察总体分数如何掩盖关键失败。

运行 ``python projects/transformers-basics/ticket_classification_walkthrough.py`` 可以看到完整过程。
文件中的小数据和 logits 都是为了方便手算而设置的，不代表真实模型效果。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final

# 三个标签依次对应“退款、物流、欺诈复核”。元组顺序也是 logits 的类别顺序。
LABELS: Final = ("refund", "logistics", "fraud_review")


@dataclass(frozen=True, slots=True)
class TicketRow:
    """一行工单数据；同一个 thread_id 可以包含多行连续对话。"""

    row_id: str
    thread_id: str
    label: str


# 两段会话各有两行，故意用来展示逐行随机切分造成的会话泄漏。
ROWS: Final = (
    TicketRow("row-001", "thread-100", "refund"),
    TicketRow("row-002", "thread-100", "refund"),
    TicketRow("row-003", "thread-200", "logistics"),
    TicketRow("row-004", "thread-200", "logistics"),
    TicketRow("row-005", "thread-300", "fraud_review"),
    TicketRow("row-006", "thread-400", "fraud_review"),
)

# 错误示范：把每一行独立切分后，thread-100 和 thread-200 同时出现在训练集与测试集。
ROW_LEVEL_SPLIT: Final = {
    "row-001": "train",
    "row-002": "test",
    "row-003": "train",
    "row-004": "test",
    "row-005": "train",
    "row-006": "validation",
}

# 正确示范：同一段会话的所有行属于同一个集合，测试时才是在评估“没见过的新会话”。
THREAD_LEVEL_SPLIT: Final = {
    "row-001": "train",
    "row-002": "train",
    "row-003": "train",
    "row-004": "train",
    "row-005": "validation",
    "row-006": "test",
}


def _split_threads(assignments: dict[str, str]) -> dict[str, list[str]]:
    """把“行属于哪个集合”的映射转换成“每个集合包含哪些会话”。"""

    result = {"train": [], "validation": [], "test": []}
    for split in result:
        result[split] = sorted({row.thread_id for row in ROWS if assignments[row.row_id] == split})
    return result


def _overlapping_threads(split_threads: dict[str, list[str]]) -> list[str]:
    """找出跨集合出现的会话；非空结果意味着发生了数据泄漏。"""

    owners: dict[str, set[str]] = {}
    for split, thread_ids in split_threads.items():
        for thread_id in thread_ids:
            owners.setdefault(thread_id, set()).add(split)
    return sorted(thread_id for thread_id, splits in owners.items() if len(splits) > 1)


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    """将任意实数分数转换为和为 1 的类别概率。"""

    # 先减去最大值，避免较大的 logit 经过 exp 后溢出；该变换不会改变 softmax 结果。
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def _classification_step(
    logits: tuple[float, ...], *, target_index: int, learning_rate: float
) -> dict[str, object]:
    """对一条分类样本手算 NLL，并直接在 logits 上走一步梯度下降。

    真实训练会通过反向传播更新模型权重。本函数故意把 logits 暂时当作可训练参数，
    只观察损失函数希望三个类别分数分别向哪个方向移动。
    """

    # 先检查输入契约，防止“类别数错位”等问题悄悄污染后面的教学结论。
    if len(logits) != len(LABELS):
        raise ValueError("logits must contain one value per label")
    if not 0 <= target_index < len(logits):
        raise ValueError("target_index is outside the label set")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    probabilities = _softmax(logits)

    # 单标签分类的交叉熵就是正确类别概率的负对数：p 越接近 1，损失越接近 0。
    loss = -math.log(probabilities[target_index])

    # softmax 与交叉熵合并后的 logit 梯度为 p-y。
    # 正确类别处 y=1，因此梯度为负；其余类别 y=0，因此梯度为正。
    gradient = tuple(
        probability - (1.0 if index == target_index else 0.0)
        for index, probability in enumerate(probabilities)
    )

    # 梯度下降执行“新值 = 旧值 - 学习率 × 梯度”：抬高正确类别，压低错误类别。
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
    """计算实验的四组结果，返回便于测试和渲染的结构化数据。"""

    # 阶段一：分别审计两种切分方法，看是否有会话同时落入多个集合。
    row_split_threads = _split_threads(ROW_LEVEL_SPLIT)
    thread_split_threads = _split_threads(THREAD_LEVEL_SPLIT)

    # 阶段二：模型最初偏向 refund（logit=2.0），但正确答案是 fraud_review（索引 2）。
    prediction = _classification_step(
        (2.0, 0.5, -1.0),
        target_index=2,
        learning_rate=2.0,
    )
    return {
        # 先写清预测发生的时间点和样本单位，才能判断特征、标签与切分是否合法。
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
        # 阶段三：构造“总体只错一例，但恰好漏掉全部欺诈复核”的指标反例。
        "metrics": {
            "total_tickets": 100,
            "correct_tickets": 99,
            "fraud_review_tickets": 1,
            "fraud_review_true_positives": 0,
            "accuracy": 0.99,
            "fraud_review_recall": 0.0,
        },
        # 明确证据边界，避免把一次局部手算误解成真实模型或业务效果验证。
        "scope": {
            "real_ticket_text_or_trained_model_used": False,
            "logits_treated_as_direct_parameters_for_one_local_step": True,
            "generalization_or_business_value_measured": False,
        },
    }


def _vector(values: object) -> str:
    """把数值元组格式化为适合命令行阅读的四位小数向量。"""

    if not isinstance(values, tuple):
        raise TypeError("expected a tuple")
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def render_walkthrough(walkthrough: dict[str, object]) -> str:
    """把结构化结果按“切分→损失→更新→指标”的学习顺序渲染为中文。"""

    # 渲染前保留运行时类型检查，使格式错误在靠近来源的位置失败。
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
    """配置 Windows 终端编码并打印实验结果。"""

    # Windows 旧终端的默认编码不一定是 UTF-8，显式设置后中文输出更稳定。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    print(render_walkthrough(build_walkthrough()))


if __name__ == "__main__":
    main()
