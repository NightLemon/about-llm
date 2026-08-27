"""手算一次 temperature、top-k、top-p 与 repetition penalty 采样。

第一组从已知概率开始，按固定顺序裁剪并用固定均匀数做 inverse-CDF 抽样；第二组展示
repetition penalty 对正、负 logits 的方向不同，以及重复 token 只应惩罚一次。
"""

from __future__ import annotations

import json

import numpy as np

from about_llm.inference import SamplingConfig, sample_next_token


def main() -> None:
    """运行核采样与有符号重复惩罚两组固定输入。"""

    # top-k 先保留三个候选，top-p 再按累计概率裁剪；uniform=0.6 固定最终抽样。
    step = sample_next_token(
        np.log(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)),
        config=SamplingConfig(temperature=1, top_k=3, top_p=0.7),
        uniform=0.6,
    )
    # 正 logit 要除以 penalty，负 logit 要乘以 penalty，二者都会降低已出现 token 的偏好。
    signed_penalty = sample_next_token(
        [2.0, -2.0, 0.5],
        config=SamplingConfig(repetition_penalty=2),
        prior_token_ids=(0, 1, 1),
        uniform=0,
    )
    artifact = {
        "exact_top_k_then_top_p_fixture": step.to_dict(),
        "signed_repetition_penalty_fixture": {
            "input_logits": signed_penalty.input_logits.tolist(),
            "prior_token_ids": signed_penalty.prior_token_ids,
            "adjusted_logits": signed_penalty.repetition_adjusted_logits.tolist(),
        },
        "scope": {
            "authored_finite_logits_processed": True,
            "fixed_uniform_inverse_cdf_executed": True,
            "processor_order_and_tie_break_fixed": True,
            "model_forward_or_tokenizer_executed": False,
            "multi_token_eos_stop_or_kv_modeled": False,
            "runtime_default_equivalence_proved": False,
            "generation_quality_latency_or_throughput_proved": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
