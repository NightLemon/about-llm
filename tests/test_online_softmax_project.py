from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "online_softmax_demo.py"


def test_online_softmax_project_entry_reports_bounded_evidence() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)

    assert completed.stderr == ""
    assert report["implementation"] == "about-llm.online-softmax-oracle.v1"
    assert report["observations"] == {
        "key_block_count": 3,
        "logical_peak_score_elements": 15,
        "full_score_elements": 35,
        "max_abs_error_vs_dense": report["observations"][
            "max_abs_error_vs_dense"
        ],
        "all_outputs_finite": True,
    }
    assert report["observations"]["max_abs_error_vs_dense"] <= 1e-12
    assert report["scope"] == {
        "online_path_materialized_complete_score_or_probability": False,
        "dense_reference_materialized_for_comparison": True,
        "float64_online_accumulation": True,
        "real_arithmetic_equivalence_claimed": True,
        "bitwise_equivalence_claimed": False,
        "cuda_or_gpu_kernel_executed": False,
        "flashattention_backend_executed": False,
        "hbm_traffic_peak_memory_or_performance_measured": False,
    }
