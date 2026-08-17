"""Install the built wheel in a clean venv and exercise every console script."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path

CONSOLE_SCRIPTS = (
    "about-llm-agent",
    "about-llm-cloud-contract",
    "about-llm-eval",
    "about-llm-inference-analyze",
    "about-llm-preference-data",
    "about-llm-rag",
    "about-llm-sft-data",
    "about-llm-synthetic-audit",
)

EXPECTED_TARGET_SFT_LABEL_VERSION_LINES = (
    "about-llm.target-sft-label-control.v2",
    "about-llm.target-sft-label-control-report.v2",
)


def _environment_paths(environment: Path) -> tuple[Path, Path]:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe", environment / "Scripts"
    return environment / "bin" / "python", environment / "bin"


def smoke_wheel(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="about-llm-wheel-") as temp_directory:
        environment = Path(temp_directory) / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python, scripts_directory = _environment_paths(environment)
        install_target = f"{wheel.resolve()}[agents,api,evaluation]"
        subprocess.run(
            [str(python), "-m", "pip", "install", install_target],
            check=True,
            timeout=600,
        )
        for script_name in CONSOLE_SCRIPTS:
            suffix = ".exe" if os.name == "nt" else ""
            executable = scripts_directory / f"{script_name}{suffix}"
            if not executable.is_file():
                raise FileNotFoundError(f"missing installed console script: {executable}")
            completed = subprocess.run(
                [str(executable), "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{script_name} --help failed with exit code {completed.returncode}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            print(f"OK: {script_name}")

        online_attention_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import numpy as np; "
                    "from about_llm.from_scratch import "
                    "blockwise_online_attention, causal_mask; "
                    "x=np.eye(2, dtype=np.float64); "
                    "r=blockwise_online_attention(x,x,x,block_size=1,"
                    "mask=causal_mask(2)); "
                    "assert r.key_block_count == 2; "
                    "assert r.logical_peak_score_elements == 2; "
                    "assert r.full_score_elements == 4; "
                    "np.testing.assert_allclose(r.output[0], [1.0,0.0]); "
                    "assert not hasattr(r,'probabilities'); print('OK')"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if online_attention_import.stdout.strip() != "OK":
            raise RuntimeError("installed blockwise online-softmax oracle failed")
        print("OK: about_llm.from_scratch.blockwise_online_attention")

        verifier_selection_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from fractions import Fraction; "
                    "from about_llm.inference import VerifierCandidate, "
                    "analyze_verifier_guided_best_of_n; "
                    "c=(VerifierCandidate('wrong',5,20,False),"
                    "VerifierCandidate('correct',4,80,True),"
                    "VerifierCandidate('verifier_hack',1,99,False)); "
                    "r=analyze_verifier_guided_best_of_n(c,sample_count=16); "
                    "assert r.oracle_success_probability=="
                    "Fraction(152544843904,152587890625); "
                    "assert r.selected_success_probability=="
                    "Fraction(28951056265019,156250000000000); "
                    "assert r.expected_selected_verifier_score=="
                    "Fraction(954783461138377521,10000000000000000); "
                    "assert r.logical_candidate_sequences==3**16; print('OK')"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if verifier_selection_import.stdout.strip() != "OK":
            raise RuntimeError("installed verifier best-of-N oracle failed")
        print("OK: about_llm.inference.verifier_selection")

        self_consistency_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from fractions import Fraction; "
                    "from about_llm.inference import BinaryVoteRegime, "
                    "analyze_latent_regime_binary_majority; "
                    "iid=(BinaryVoteRegime('iid',1,3,2),); "
                    "mix=(BinaryVoteRegime('easy',1,9,1),"
                    "BinaryVoteRegime('hard',1,3,7)); "
                    "a=analyze_latent_regime_binary_majority(iid,sample_count=11); "
                    "b=analyze_latent_regime_binary_majority(mix,sample_count=11); "
                    "assert a.majority_success_probability=="
                    "Fraction(36791901,48828125); "
                    "assert b.majority_success_probability=="
                    "Fraction(13474113561,25000000000); "
                    "assert b.pairwise_success_correlation==Fraction(3,8); "
                    "assert a.logical_binary_vote_sequences==2**11; print('OK')"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if self_consistency_import.stdout.strip() != "OK":
            raise RuntimeError("installed binary self-consistency oracle failed")
        print("OK: about_llm.inference.self_consistency")

        sequential_peeking_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from fractions import Fraction; "
                    "from about_llm.evaluation import "
                    "analyze_repeated_two_sided_sign_tests; "
                    "a=analyze_repeated_two_sided_sign_tests("
                    "(10,20,30,40,50),per_look_alpha=Fraction(1,20)); "
                    "b=analyze_repeated_two_sided_sign_tests("
                    "(10,20,30,40,50),per_look_alpha=Fraction(1,100)); "
                    "assert a.familywise_null_rejection_probability=="
                    "Fraction(7109832616777,70368744177664); "
                    "assert b.familywise_null_rejection_probability=="
                    "Fraction(2142139082367,140737488355328); "
                    "assert a.logical_binary_sign_sequences==2**50; print('OK')"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if sequential_peeking_import.stdout.strip() != "OK":
            raise RuntimeError("installed sequential peeking oracle failed")
        print("OK: about_llm.evaluation.sequential")

        gradient_accumulation_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from fractions import Fraction; "
                    "from about_llm.finetuning import CategoricalMicrobatch, "
                    "CategoricalTokenRecord, DDPGradientAccumulationAnalysis, "
                    "DDPTokenMeanAnalysis, "
                    "analyze_default_ddp_gradient_accumulation, "
                    "analyze_default_ddp_token_mean, "
                    "analyze_masked_token_gradient_accumulation; "
                    "m=(CategoricalMicrobatch('short',"
                    "(CategoricalTokenRecord('s',(9,1),0),)),"
                    "CategoricalMicrobatch('long',"
                    "(CategoricalTokenRecord('l1',(4,1),1),"
                    "CategoricalTokenRecord('l2',(4,1),1),"
                    "CategoricalTokenRecord('l3',(4,1),1)))); "
                    "r=analyze_masked_token_gradient_accumulation(m); "
                    "assert r.full_batch_class_aggregate_logit_gradient=="
                    "(Fraction(23,40),Fraction(-23,40)); "
                    "assert r.count_scaled_accumulated_class_aggregate_logit_gradient=="
                    "r.full_batch_class_aggregate_logit_gradient; "
                    "assert r.naive_equal_microbatch_class_aggregate_logit_gradient=="
                    "(Fraction(7,20),Fraction(-7,20)); "
                    "d=analyze_default_ddp_token_mean("
                    "m,data_parallel_world_size=2); "
                    "assert isinstance(d,DDPTokenMeanAnalysis); "
                    "assert d.correct_local_loss_sum_scale==Fraction(1,2); "
                    "assert d.missing_world_size_local_loss_sum_scale==Fraction(1,4); "
                    "assert d.correctly_scaled_default_ddp_class_aggregate_logit_gradient=="
                    "(Fraction(23,40),Fraction(-23,40)); "
                    "assert d.missing_world_size_default_ddp_class_aggregate_logit_gradient=="
                    "(Fraction(23,80),Fraction(-23,80)); "
                    "assert d.equal_rank_local_mean_class_aggregate_logit_gradient=="
                    "(Fraction(7,20),Fraction(-7,20)); "
                    "a=analyze_default_ddp_gradient_accumulation("
                    "((m[0],),(m[1],)),data_parallel_world_size=2,"
                    "unclipped_sgd_learning_rate=Fraction(7,20)); "
                    "assert isinstance(a,DDPGradientAccumulationAnalysis); "
                    "assert a.accumulation_steps==1; "
                    "assert a.correct_local_loss_sum_scale==Fraction(1,2); "
                    "assert a.one_sync_after_accumulation_class_aggregate_logit_gradient=="
                    "(Fraction(23,40),Fraction(-23,40)); "
                    "assert a.unclipped_sgd_parameter_delta=="
                    "(Fraction(-161,800),Fraction(161,800)); print('OK')"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if gradient_accumulation_import.stdout.strip() != "OK":
            raise RuntimeError("installed gradient-accumulation oracle failed")
        print("OK: about_llm.finetuning.gradient_accumulation")

        model_evidence_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.model_release_evidence import "
                    "MODEL_RELEASE_EVIDENCE_VERSION; print(MODEL_RELEASE_EVIDENCE_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if model_evidence_import.stdout.strip() != (
            "about-llm.model-release-evidence.v1"
        ):
            raise RuntimeError("installed model release evidence module failed")
        print("OK: about_llm.model_release_evidence")

        checkpoint_control_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.integrations.transformers_checkpoint_control import "
                    "TRANSFORMERS_CHECKPOINT_CONTROL_VERSION; "
                    "print(TRANSFORMERS_CHECKPOINT_CONTROL_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if checkpoint_control_import.stdout.strip() != (
            "about-llm.transformers-checkpoint-control.v1"
        ):
            raise RuntimeError("installed Transformers checkpoint control module failed")
        print("OK: about_llm.integrations.transformers_checkpoint_control")

        activation_patching_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.integrations."
                    "transformers_activation_patching_control import "
                    "TARGET_ACTIVATION_PATCHING_CONTROL_VERSION, "
                    "TARGET_ACTIVATION_PATCHING_REPORT_VERSION; "
                    "print(TARGET_ACTIVATION_PATCHING_CONTROL_VERSION); "
                    "print(TARGET_ACTIVATION_PATCHING_REPORT_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if activation_patching_import.stdout.splitlines() != [
            "about-llm.target-activation-patching-control.v1",
            "about-llm.target-activation-patching-control-report.v1",
        ]:
            raise RuntimeError("installed target activation-patching module failed")
        print(
            "OK: about_llm.integrations."
            "transformers_activation_patching_control"
        )

        rag_transformers_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.rag.transformers_control import "
                    "RAG_TRANSFORMERS_CONTROL_VERSION; "
                    "print(RAG_TRANSFORMERS_CONTROL_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if rag_transformers_import.stdout.strip() != (
            "about-llm.rag-transformers-control.v1"
        ):
            raise RuntimeError("installed RAG Transformers control module failed")
        print("OK: about_llm.rag.transformers_control")

        rag_publication_policy_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.rag.citations import CitationContext; "
                    "from about_llm.rag.models import Document; "
                    "from about_llm.rag.generation_policy import "
                    "DEFAULT_RAG_PUBLICATION_POLICY, RAG_PUBLICATION_POLICY_VERSION, "
                    "evaluate_post_generation; "
                    "print(RAG_PUBLICATION_POLICY_VERSION); "
                    "print(DEFAULT_RAG_PUBLICATION_POLICY.fingerprint); "
                    "context=CitationContext('<source/>', "
                    "{'S1': Document('doc', 'evidence', 'tenant')}); "
                    "decision=evaluate_post_generation(context, 'rejected raw'); "
                    "print('raw_output' in decision.to_public_dict())"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if rag_publication_policy_import.stdout.splitlines() != [
            "about-llm.rag-publication-policy.v1",
            "sha256:4e59d11cefc5ed9e6cc55a4c36a572e0ed698a8583527bcfbb4eb78b99722449",
            "False",
        ]:
            raise RuntimeError("installed RAG publication policy module failed")
        print("OK: about_llm.rag.generation_policy")

        guarded_rag_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.rag import "
                    "RAG_GUARDED_TRANSFORMERS_CONTROL_VERSION; "
                    "from about_llm.rag.guarded_transformers_control import "
                    "RAG_GUARDED_TRANSFORMERS_REPORT_VERSION; "
                    "print(RAG_GUARDED_TRANSFORMERS_CONTROL_VERSION); "
                    "print(RAG_GUARDED_TRANSFORMERS_REPORT_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if guarded_rag_import.stdout.splitlines() != [
            "about-llm.rag-guarded-transformers-control.v1",
            "about-llm.rag-guarded-transformers-control-report.v1",
        ]:
            raise RuntimeError("installed guarded RAG Transformers control module failed")
        print("OK: about_llm.rag.guarded_transformers_control")

        target_lora_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.finetuning import "
                    "TARGET_LORA_CONTROL_VERSION, TARGET_LORA_REPORT_VERSION; "
                    "print(TARGET_LORA_CONTROL_VERSION); "
                    "print(TARGET_LORA_REPORT_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if target_lora_import.stdout.splitlines() != [
            "about-llm.target-lora-control.v1",
            "about-llm.target-lora-control-report.v1",
        ]:
            raise RuntimeError("installed target LoRA control module failed")
        print("OK: about_llm.finetuning.target_lora_control")

        target_dpo_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.finetuning import "
                    "TARGET_DPO_CONTROL_VERSION, TARGET_DPO_REPORT_VERSION; "
                    "print(TARGET_DPO_CONTROL_VERSION); "
                    "print(TARGET_DPO_REPORT_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if target_dpo_import.stdout.splitlines() != [
            "about-llm.target-dpo-control.v1",
            "about-llm.target-dpo-control-report.v1",
        ]:
            raise RuntimeError("installed target DPO control module failed")
        print("OK: about_llm.finetuning.target_dpo_control")

        target_sft_label_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.finetuning import "
                    "TARGET_SFT_LABEL_CONTROL_VERSION, TARGET_SFT_LABEL_REPORT_VERSION; "
                    "print(TARGET_SFT_LABEL_CONTROL_VERSION); "
                    "print(TARGET_SFT_LABEL_REPORT_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if tuple(target_sft_label_import.stdout.splitlines()) != (
            EXPECTED_TARGET_SFT_LABEL_VERSION_LINES
        ):
            raise RuntimeError("installed target SFT label control module failed")
        print("OK: about_llm.finetuning.target_sft_label_control")

        target_service_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.inference.target_service_control import "
                    "TARGET_SERVICE_CONTROL_VERSION, TARGET_SERVICE_REPORT_VERSION; "
                    "print(TARGET_SERVICE_CONTROL_VERSION); "
                    "print(TARGET_SERVICE_REPORT_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if target_service_import.stdout.splitlines() != [
            "about-llm.target-service-control.v1",
            "about-llm.target-service-control-report.v1",
        ]:
            raise RuntimeError("installed target service control module failed")
        print("OK: about_llm.inference.target_service_control")

        incremental_control = subprocess.run(
            [
                str(python),
                "-m",
                "about_llm.inference.incremental_streaming_control",
                "run",
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        incremental_report = json.loads(incremental_control.stdout)
        if not (
            incremental_report.get("implementation")
            == "about-llm.incremental-streaming-control.v1"
            and incremental_report.get("complete_stream", {}).get(
                "client_sse_done_observed"
            )
            is True
            and incremental_report.get("disconnect_stream", {}).get(
                "postclose_backend_asyncio_cancelled_error_observed"
            )
            is True
            and incremental_report.get("disconnect_stream", {}).get(
                "postclose_backend_emitted_token_ids"
            )
            == [201]
            and incremental_report.get("scope", {}).get(
                "transformers_generation_thread_cancellation_proven"
            )
            is False
            and incremental_control.stderr == ""
        ):
            raise RuntimeError("installed incremental-streaming control failed")
        print("OK: about_llm.inference.incremental_streaming_control")

        thread_cancellation_import = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from about_llm.inference."
                    "transformers_thread_cancellation_control import "
                    "TRANSFORMERS_THREAD_CANCELLATION_CONTROL_VERSION, "
                    "TRANSFORMERS_THREAD_CANCELLATION_REPORT_VERSION; "
                    "print(TRANSFORMERS_THREAD_CANCELLATION_CONTROL_VERSION); "
                    "print(TRANSFORMERS_THREAD_CANCELLATION_REPORT_VERSION)"
                ),
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if thread_cancellation_import.stdout.splitlines() != [
            "about-llm.transformers-thread-cancellation-control.v1",
            "about-llm.transformers-thread-cancellation-control-report.v1",
        ]:
            raise RuntimeError("installed Transformers thread-cancellation module failed")
        print(
            "OK: about_llm.inference.transformers_thread_cancellation_control import"
        )

        mcp_sdk_memory_control = subprocess.run(
            [str(python), "-m", "about_llm.agents.mcp_sdk_memory"],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        mcp_sdk_memory_report = json.loads(mcp_sdk_memory_control.stdout)
        if not (
            mcp_sdk_memory_report.get("control_version")
            == "about-llm.mcp-sdk-memory-control.v1"
            and mcp_sdk_memory_report.get("runtime", {}).get("sdk_version")
            == "1.29.0"
            and mcp_sdk_memory_report.get("transport", {}).get(
                "official_sdk_memory_stream"
            )
            is True
            and mcp_sdk_memory_report.get("transport", {}).get("tcp_http") is False
            and mcp_sdk_memory_report.get("calls", {}).get(
                "invalid_schema_handler_delta"
            )
            == 0
            and mcp_sdk_memory_report.get("calls", {}).get(
                "unknown_tool_handler_delta"
            )
            == 1
            and mcp_sdk_memory_report.get("scope", {}).get(
                "authentication_or_authorization_proven"
            )
            is False
            and mcp_sdk_memory_control.stderr == ""
        ):
            raise RuntimeError("installed official MCP SDK memory control failed")
        print("OK: about_llm.agents.mcp_sdk_memory control")

        mcp_sdk_stdio_control = subprocess.run(
            [str(python), "-m", "about_llm.agents.mcp_sdk_stdio"],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        mcp_sdk_stdio_report = json.loads(mcp_sdk_stdio_control.stdout)
        if not (
            mcp_sdk_stdio_report.get("control_version")
            == "about-llm.mcp-sdk-stdio-control.v1"
            and mcp_sdk_stdio_report.get("runtime", {}).get("sdk_version")
            == "1.29.0"
            and mcp_sdk_stdio_report.get("transport", {}).get(
                "client_launched_server_subprocess"
            )
            is True
            and mcp_sdk_stdio_report.get("transport", {}).get(
                "os_stdin_stdout_pipes"
            )
            is True
            and mcp_sdk_stdio_report.get("server_receipt", {}).get(
                "handler_events"
            )
            == ["fixture.add", "fixture.missing"]
            and mcp_sdk_stdio_report.get("scope", {}).get(
                "malformed_raw_framing_controls_executed"
            )
            is False
            and mcp_sdk_stdio_report.get("scope", {}).get(
                "authentication_or_authorization_proven"
            )
            is False
            and mcp_sdk_stdio_control.stderr == ""
        ):
            raise RuntimeError("installed official MCP SDK stdio control failed")
        print("OK: about_llm.agents.mcp_sdk_stdio control")

        mcp_sdk_http_control = subprocess.run(
            [
                str(python),
                "-m",
                "about_llm.agents.mcp_sdk_streamable_http",
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        mcp_sdk_http_report = json.loads(mcp_sdk_http_control.stdout)
        if not (
            mcp_sdk_http_report.get("control_version")
            == "about-llm.mcp-sdk-streamable-http-control.v1"
            and mcp_sdk_http_report.get("runtime", {}).get("sdk_version")
            == "1.29.0"
            and mcp_sdk_http_report.get("transport", {}).get(
                "real_ipv4_loopback_tcp_http"
            )
            is True
            and mcp_sdk_http_report.get("transport", {}).get(
                "mcp_session_termination_delete_observed"
            )
            is True
            and mcp_sdk_http_report.get("http_observations", {}).get(
                "mcp_response_count"
            )
            == 9
            and mcp_sdk_http_report.get("server_receipt", {}).get(
                "handler_events"
            )
            == ["fixture.add", "fixture.missing"]
            and mcp_sdk_http_report.get("scope", {}).get(
                "authentication_or_authorization_proven"
            )
            is False
            and mcp_sdk_http_control.stderr == ""
        ):
            raise RuntimeError(
                "installed official MCP SDK Streamable HTTP control failed"
            )
        print("OK: about_llm.agents.mcp_sdk_streamable_http control")

        mcp_control = subprocess.run(
            [str(python), "-m", "about_llm.agents.mcp_stdio", "control"],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        report = json.loads(mcp_control.stdout)
        if not (
            report.get("implementation") == "about-llm.mcp-stdio-control.v1"
            and report.get("transport", {}).get(
                "client_launched_server_subprocess"
            )
            is True
            and report.get("scope", {}).get("external_network_or_remote_server_called")
            is False
            and mcp_control.stderr == ""
        ):
            raise RuntimeError("installed MCP stdio subprocess control failed")
        print("OK: about_llm.agents.mcp_stdio control")

        mcp_http_control = subprocess.run(
            [
                str(python),
                "-m",
                "about_llm.agents.mcp_streamable_http",
                "control",
            ],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        mcp_http_report = json.loads(mcp_http_control.stdout)
        if not (
            mcp_http_report.get("implementation")
            == "about-llm.mcp-streamable-http-control.v1"
            and mcp_http_report.get("network", {}).get("real_tcp_http") is True
            and mcp_http_report.get("cancellation", {}).get(
                "jsonrpc_response_after_cancellation_count"
            )
            == 0
            and mcp_http_report.get("evidence_limits", {}).get(
                "remote_or_cross_vendor_interoperability_proven"
            )
            is False
            and mcp_http_control.stderr == ""
        ):
            raise RuntimeError("installed MCP Streamable HTTP loopback control failed")
        print("OK: about_llm.agents.mcp_streamable_http control")

        a2a_control = subprocess.run(
            [str(python), "-m", "about_llm.agents.a2a_loopback", "control"],
            cwd=temp_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        a2a_report = json.loads(a2a_control.stdout)
        if not (
            a2a_report.get("implementation") == "about-llm.a2a-loopback-control.v1"
            and a2a_report.get("network", {}).get("real_tcp_http") is True
            and a2a_report.get("task", {}).get("local_verifier_passed") is True
            and a2a_report.get("evidence_limits", {}).get(
                "remote_or_cross_vendor_interoperability_proven"
            )
            is False
            and a2a_control.stderr == ""
        ):
            raise RuntimeError("installed A2A loopback subprocess control failed")
        print("OK: about_llm.agents.a2a_loopback control")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels = sorted(args.wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected exactly one wheel in {args.wheel_dir}, found {len(wheels)}")
    smoke_wheel(wheels[0])


if __name__ == "__main__":
    main()
