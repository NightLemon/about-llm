# ruff: noqa: RUF001 -- Full-width punctuation is intentional in learner output.
"""从矩阵乘法开始，手算一次三候选 next-token 预测和参数更新。

模型读到“天空通常是”后，要在“红 / 蓝 / 。”中选择下一个 token。实验依次展示
hidden state 如何变成 logits、softmax 如何得到概率、NLL 如何衡量错误，以及梯度下降
如何让“蓝”的概率上升。所有数字都很小，读者可以跟着命令行输出逐项复算。
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Final

# 候选 token 的顺序决定 logits、概率和输出矩阵列的共同含义。
TOKENS: Final = ("红", "蓝", "。")

# 把上游 Transformer 的输出压缩为两个可手算的特征；真实模型通常有数千维。
HIDDEN_STATE: Final = (1.0, 1.0)

# shape=[2, 3]：两维 hidden state 分别连接到三个候选 token。
OUTPUT_WEIGHT: Final = (
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
)
# 正确答案“蓝”位于候选列表的索引 1。
TARGET_INDEX: Final = 1
LEARNING_RATE: Final = 0.5
FINITE_DIFFERENCE_EPSILON: Final = 1e-5


def _matrix_multiply(
    row: Sequence[float], matrix: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    """计算一个行向量与二维矩阵的乘积，不依赖 NumPy。"""

    # [1, input_dim] @ [input_dim, output_dim] 的内侧维度必须相等。
    if not row or len(row) != len(matrix):
        raise ValueError("row width must equal the number of matrix rows")
    output_width = len(matrix[0])
    if output_width == 0 or any(len(matrix_row) != output_width for matrix_row in matrix):
        raise ValueError("matrix must be non-empty and rectangular")
    # 每个输出元素都是输入行与矩阵对应列的逐项乘积之和。
    return tuple(
        sum(row[input_index] * matrix[input_index][output_index] for input_index in range(len(row)))
        for output_index in range(output_width)
    )


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """用数值稳定的 softmax 将 logits 转成概率。"""

    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("logits must be a non-empty sequence of finite numbers")
    # 整体减去最大 logit 不改变概率，却能避免 exp(很大的数) 溢出。
    maximum = max(logits)
    shifted_exponentials = tuple(math.exp(value - maximum) for value in logits)
    total = sum(shifted_exponentials)
    return tuple(value / total for value in shifted_exponentials)


def _negative_log_likelihood(probabilities: Sequence[float], target_index: int) -> float:
    """返回正确 token 的负对数似然；正确概率越高，损失越低。"""

    if not 0 <= target_index < len(probabilities):
        raise ValueError("target_index is outside the probability vector")
    target_probability = probabilities[target_index]
    if not 0.0 < target_probability <= 1.0:
        raise ValueError("target probability must be in (0, 1]")
    return -math.log(target_probability)


def _loss_for_weight(
    hidden_state: Sequence[float],
    weight: Sequence[Sequence[float]],
    target_index: int,
) -> float:
    """从一份输出权重重新计算 loss，供有限差分梯度检查使用。"""

    logits = _matrix_multiply(hidden_state, weight)
    return _negative_log_likelihood(_softmax(logits), target_index)


def build_walkthrough() -> dict[str, object]:
    """计算从 logits 到一次权重更新的全部中间结果。"""

    # 阶段一：hidden state 乘输出矩阵，得到三个候选 token 的未归一化分数。
    logits = _matrix_multiply(HIDDEN_STATE, OUTPUT_WEIGHT)

    # 阶段二：把 softmax 的“减最大值 → exp → 归一化”拆开保存，方便逐步展示。
    maximum = max(logits)
    shifted_logits = tuple(value - maximum for value in logits)
    shifted_exponentials = tuple(math.exp(value) for value in shifted_logits)
    probabilities = _softmax(logits)
    loss = _negative_log_likelihood(probabilities, TARGET_INDEX)

    # 阶段三：softmax + NLL 对 logits 的导数可化简为 p-y。
    logit_gradient = tuple(
        probability - (1.0 if index == TARGET_INDEX else 0.0)
        for index, probability in enumerate(probabilities)
    )
    # 输出层梯度是 hidden_state 的列向量与 logit_gradient 的外积，shape 仍为 [2, 3]。
    weight_gradient = tuple(
        tuple(hidden_value * derivative for derivative in logit_gradient)
        for hidden_value in HIDDEN_STATE
    )

    # 用有限差分独立检查权重[0,0]的解析梯度：分别加减 epsilon 后观察 loss 斜率。
    row_index = 0
    column_index = 0
    plus_weight = [list(row) for row in OUTPUT_WEIGHT]
    minus_weight = [list(row) for row in OUTPUT_WEIGHT]
    plus_weight[row_index][column_index] += FINITE_DIFFERENCE_EPSILON
    minus_weight[row_index][column_index] -= FINITE_DIFFERENCE_EPSILON
    finite_difference = (
        _loss_for_weight(HIDDEN_STATE, plus_weight, TARGET_INDEX)
        - _loss_for_weight(HIDDEN_STATE, minus_weight, TARGET_INDEX)
    ) / (2.0 * FINITE_DIFFERENCE_EPSILON)

    # 阶段四：所有权重沿负梯度移动半步，再完整重算 logits、概率和 loss。
    updated_weight = tuple(
        tuple(
            value - LEARNING_RATE * weight_gradient[row_index][column_index]
            for column_index, value in enumerate(row)
        )
        for row_index, row in enumerate(OUTPUT_WEIGHT)
    )
    updated_logits = _matrix_multiply(HIDDEN_STATE, updated_weight)
    updated_probabilities = _softmax(updated_logits)
    updated_loss = _negative_log_likelihood(updated_probabilities, TARGET_INDEX)

    return {
        "prompt": "天空通常是",
        "tokens": TOKENS,
        "hidden_state": HIDDEN_STATE,
        "output_weight": OUTPUT_WEIGHT,
        "matrix_shape": [2, 3],
        "logits": logits,
        "maximum_logit": maximum,
        "shifted_logits": shifted_logits,
        "shifted_exponentials": shifted_exponentials,
        "probabilities": probabilities,
        "target_index": TARGET_INDEX,
        "target_token": TOKENS[TARGET_INDEX],
        "negative_log_likelihood": loss,
        "logit_gradient": logit_gradient,
        "weight_gradient": weight_gradient,
        "finite_difference": {
            "weight_index": [row_index, column_index],
            "epsilon": FINITE_DIFFERENCE_EPSILON,
            "numerical_derivative": finite_difference,
            "analytic_derivative": weight_gradient[row_index][column_index],
        },
        "learning_rate": LEARNING_RATE,
        "updated_weight": updated_weight,
        "updated_logits": updated_logits,
        "updated_probabilities": updated_probabilities,
        "updated_negative_log_likelihood": updated_loss,
    }


def _vector(values: object) -> str:
    """把数值元组渲染为保留四位小数的向量。"""

    if not isinstance(values, tuple):
        raise TypeError("expected a tuple")
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def render_walkthrough(walkthrough: dict[str, object]) -> str:
    """按“分数→概率→梯度→更新”的顺序生成中文讲解。"""

    # 这些检查让结构化计算与显示层解耦后仍能尽早发现类型错位。
    finite_difference = walkthrough["finite_difference"]
    if not isinstance(finite_difference, dict):
        raise TypeError("finite_difference must be an object")
    probabilities = walkthrough["probabilities"]
    updated_probabilities = walkthrough["updated_probabilities"]
    if not isinstance(probabilities, tuple) or not isinstance(updated_probabilities, tuple):
        raise TypeError("probabilities must be tuples")
    target_index = int(walkthrough["target_index"])

    return "\n".join(
        [
            "从两个数字到一次模型更新",
            "",
            f"问题：模型读到“{walkthrough['prompt']}”，"
            f"下一个 token 应该是“{walkthrough['target_token']}”。",
            f"候选 token：{walkthrough['tokens']}",
            "",
            "1. hidden state 乘输出矩阵，得到三个分数",
            f"hidden state：{_vector(walkthrough['hidden_state'])}，shape=[1,2]",
            f"output weight shape：{walkthrough['matrix_shape']}",
            "第一个 logit = 1×1 + 1×1 = 2",
            "第二个 logit = 1×0 + 1×1 = 1",
            "第三个 logit = 1×0 + 1×0 = 0",
            f"logits：{_vector(walkthrough['logits'])}",
            "",
            "2. softmax 把分数变成总和为 1 的概率",
            f"先减最大值：{_vector(walkthrough['shifted_logits'])}",
            f"再取指数：{_vector(walkthrough['shifted_exponentials'])}",
            f"最后除以总和：{_vector(probabilities)}",
            f"正确 token“{walkthrough['target_token']}”的概率：{probabilities[target_index]:.4f}",
            f"NLL = -ln(正确 token 概率) = {float(walkthrough['negative_log_likelihood']):.4f}",
            "",
            "3. 梯度告诉我们：哪个分数该升，哪个分数该降",
            f"dL/dlogits = p-y：{_vector(walkthrough['logit_gradient'])}",
            "“蓝”的梯度为负，所以沿负梯度更新时，它的 logit 会升高。",
            f"权重[0,0]的公式梯度：{float(finite_difference['analytic_derivative']):.6f}",
            f"把该权重上下扰动后算出的斜率：{float(finite_difference['numerical_derivative']):.6f}",
            "",
            "4. 沿负梯度更新一次，再算一遍",
            f"更新后的 logits：{_vector(walkthrough['updated_logits'])}",
            f"更新后的概率：{_vector(updated_probabilities)}",
            f"正确 token 概率：{probabilities[target_index]:.4f} → "
            f"{updated_probabilities[target_index]:.4f}",
            f"NLL：{float(walkthrough['negative_log_likelihood']):.4f} → "
            f"{float(walkthrough['updated_negative_log_likelihood']):.4f}",
            "",
            "结论：这次更新让当前样本的正确 token 概率上升、损失下降。",
            "真实 LLM 会对完整词表和大量参数重复这条计算链；这个小例子只负责把数学关系摊开。",
        ]
    )


def main() -> int:
    """配置中文终端输出并运行完整手算过程。"""

    # Windows 终端默认编码可能不是 UTF-8，显式配置可避免中文乱码。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    print(render_walkthrough(build_walkthrough()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
