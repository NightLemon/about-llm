from __future__ import annotations

import json

from about_llm.evaluation import holm_bonferroni_correction


def main() -> None:
    p_values = [0.04, 0.01, 0.03, 0.20]
    result = holm_bonferroni_correction(p_values, alpha=0.05)
    artifact = {
        "input_order": {
            "interpretation": "four prespecified hypotheses in authored input order",
            "p_values": p_values,
        },
        "holm": result.to_dict(),
        "scope": {
            "arbitrary_dependence_fwer_control_requires_valid_input_p_values": True,
            "effect_size_or_practical_importance_estimated": False,
            "family_prespecified_or_selection_bias_repaired": False,
            "holm_rank_and_running_maximum_executed": True,
            "repeated_peeking_or_optional_stopping_repaired": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
