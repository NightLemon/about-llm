"""构建并验证一个带 HMAC 链的模型评测发布台账。

三条记录依次绑定 baseline manifest、candidate manifest 和比较结论；每条记录都包含前一条
摘要和当前 artifact 哈希。实验展示如何发现删除、换序或篡改，但公开测试密钥不提供生产认证。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.evaluation import (
    EvaluationReleaseLedger,
    append_evaluation_release_artifact,
    load_evaluation_release_ledger,
    verify_evaluation_release_ledger,
    write_evaluation_release_ledger,
)

PROJECT = Path(__file__).resolve().parent

# 这些密钥公开写在仓库中，只用于学习签名协议，绝不能作为生产密钥。
FIXTURE_KEYS = {
    "fixture-hmac-2026-a": bytes.fromhex("11" * 32),
    "fixture-hmac-2026-b": bytes.fromhex("22" * 32),
}
ARTIFACT_PATHS = {
    "baseline-run-manifest": PROJECT / "run.baseline.manifest.example.json",
    "candidate-run-manifest": PROJECT / "run.candidate.manifest.example.json",
    "release-comparison": PROJECT / "comparison.example.json",
}


def build_fixture() -> EvaluationReleaseLedger:
    """按时间顺序重建三条发布记录，并在最后一条轮换签名密钥。"""

    # 第一条记录 baseline，append 会计算 artifact 哈希并建立链头。
    ledger = append_evaluation_release_artifact(
        None,
        release_id="authored-eval-release-001",
        artifact_id="baseline-run-manifest",
        artifact_kind="evaluation_run_manifest",
        artifact_path=ARTIFACT_PATHS["baseline-run-manifest"],
        decision="recorded",
        recorded_at="2026-08-07T09:00:00+08:00",
        key_id="fixture-hmac-2026-a",
        secret_key=FIXTURE_KEYS["fixture-hmac-2026-a"],
    )
    # 第二条记录 candidate manifest，并绑定第一条记录的摘要。
    ledger = append_evaluation_release_artifact(
        ledger,
        release_id="authored-eval-release-002",
        artifact_id="candidate-run-manifest",
        artifact_kind="evaluation_run_manifest",
        artifact_path=ARTIFACT_PATHS["candidate-run-manifest"],
        decision="recorded",
        recorded_at="2026-08-07T09:01:00+08:00",
        key_id="fixture-hmac-2026-a",
        secret_key=FIXTURE_KEYS["fixture-hmac-2026-a"],
    )
    # 最后一条批准比较结果，同时演示 key_id 轮换后链仍可验证。
    return append_evaluation_release_artifact(
        ledger,
        release_id="authored-eval-release-003",
        artifact_id="release-comparison",
        artifact_kind="evaluation_comparison",
        artifact_path=ARTIFACT_PATHS["release-comparison"],
        decision="approved",
        recorded_at="2026-08-07T09:02:00+08:00",
        key_id="fixture-hmac-2026-b",
        secret_key=FIXTURE_KEYS["fixture-hmac-2026-b"],
    )


def main() -> None:
    """重建或读取台账，再核对签名链、artifact 字节和可信链头。"""

    parser = argparse.ArgumentParser(
        description="Build or verify the public authenticated release-ledger fixture"
    )
    parser.add_argument(
        "--write",
        type=Path,
        help="exclusive-create a canonical snapshot; refuses to overwrite",
    )
    args = parser.parse_args()
    # 每次都重建 expected，保证 checked-in 台账确实来自当前 artifact 内容。
    expected = build_fixture()
    if args.write is not None:
        write_evaluation_release_ledger(args.write, expected)
        ledger = expected
    else:
        ledger = load_evaluation_release_ledger(
            PROJECT / "release-ledger.example.json"
        )
        if ledger != expected:
            raise ValueError("checked-in ledger does not match rebuilt artifact bytes")
    # trusted_head 防止攻击者连同全部记录一起替换成另一条自洽的链。
    verification = verify_evaluation_release_ledger(
        ledger,
        key_resolver=FIXTURE_KEYS,
        artifact_paths=ARTIFACT_PATHS,
        trusted_head=expected.head,
    )
    payload = verification.to_dict()
    payload["fixture_keys_are_public_test_values"] = True
    payload["production_key_custody_proven"] = False
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
