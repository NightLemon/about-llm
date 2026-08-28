"""演示 MinHash/LSH 如何召回近重复候选，并用精确 Jaccard 二次确认。

文本先做 Unicode/空白归一化并切字符 n-gram；LSH 只负责低成本产生 candidate pairs，
随后精确重算相似度。实验还枚举全部 pair 审计 LSH 漏召回，提醒候选生成不是保证。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from about_llm.finetuning.minhash_lsh import (
    MinHashLSHConfig,
    audit_minhash_lsh_recall,
    exact_recheck_lsh_candidates,
    generate_minhash_lsh_candidates,
    lsh_candidate_probability,
)
from about_llm.finetuning.near_duplicate import (
    NearDuplicateProfile,
    character_ngrams,
    normalize_near_duplicate_text,
)

# 样例同时含空白复制、近似改写、无关文本与不同 split。
AUTHORED_TEXTS = {
    "train-1": "A transformer predicts the next token from its prefix.",
    "validation-copy": "A  transformer predicts the next token from its prefix.",
    "test-near": "A transformer predicts each next token from its prefix.",
    "test-unrelated": "SQLite transactions serialize local writers.",
    "validation-other": "Retrieval metrics need explicit relevance labels.",
}


def run_toy(
    *,
    ngram_size: int,
    threshold: float,
    num_hashes: int,
    bands: int,
    seed: int,
) -> dict[str, Any]:
    """构建 n-gram 集，运行 LSH、精确复核和穷举 recall 审计。"""

    # normalization profile 是数据契约的一部分，改变它会改变 n-gram 与相似度。
    items = {
        item_id: character_ngrams(
            normalize_near_duplicate_text(
                text, profile=NearDuplicateProfile.NFC_WHITESPACE
            ),
            size=ngram_size,
        )
        for item_id, text in AUTHORED_TEXTS.items()
    }
    config = MinHashLSHConfig(num_hashes=num_hashes, bands=bands, seed=seed)
    # LSH 只缩小 O(n²) 比较集合，任何自动动作前仍要 exact recheck。
    candidates = generate_minhash_lsh_candidates(items, config=config)
    exact_rechecks = exact_recheck_lsh_candidates(
        items, candidates, threshold=threshold
    )
    recall = audit_minhash_lsh_recall(items, candidates, threshold=threshold)
    return {
        "schema_version": 1,
        "authored_fixture": True,
        "authored_texts": AUTHORED_TEXTS,
        "authored_relationships": {
            "validation-copy": "whitespace-only copy of train-1 after normalization",
            "test-near": "near rewrite of train-1",
            "test-unrelated": "different topic from train-1",
            "validation-other": "different topic from train-1",
        },
        "network_used": False,
        "normalization_profile": NearDuplicateProfile.NFC_WHITESPACE.value,
        "ngram_size": ngram_size,
        "threshold": threshold,
        "candidate_report": candidates.to_dict(),
        "exact_candidate_rechecks": [item.to_dict() for item in exact_rechecks],
        "exhaustive_recall_audit": recall.to_dict(),
        "ideal_independent_minhash_collision_probability": {
            "at_threshold": lsh_candidate_probability(
                threshold,
                bands=config.bands,
                rows_per_band=config.rows_per_band,
            ),
            "at_0_5": lsh_candidate_probability(
                0.5,
                bands=config.bands,
                rows_per_band=config.rows_per_band,
            ),
            "at_0_9": lsh_candidate_probability(
                0.9,
                bands=config.bands,
                rows_per_band=config.rows_per_band,
            ),
        },
        "conclusion": (
            "LSH produces candidates only; exact Jaccard confirms candidates, and the "
            "exhaustive pair audit measures misses for this authored fixture"
        ),
        "scope": {
            "candidate_source": "Unicode-codepoint character n-gram sets",
            "exact_recheck_required_before_action": True,
            "exhaustive_recall_audit_is_quadratic": True,
            "candidate_recall_guaranteed": False,
            "semantic_or_translation_duplicate_detection": False,
            "threshold_calibrated_for_real_domain": False,
            "hash_or_manifest_authenticates_data": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """定义 n-gram、阈值、hash 数、band 数和随机种子。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num-hashes", type=int, default=64)
    parser.add_argument("--bands", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """运行候选召回实验并打印概率与漏召回审计。"""

    args = parse_args()
    print(
        json.dumps(
            run_toy(
                ngram_size=args.ngram_size,
                threshold=args.threshold,
                num_hashes=args.num_hashes,
                bands=args.bands,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
