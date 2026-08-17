from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from about_llm.integrations.transformers_activation_patching_control import (  # noqa: E402
    QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL,
    ActivationPatchingProtocol,
    execute_loaded_activation_patching_control,
    normalized_patch_recovery,
    verify_recorded_activation_patching_report,
)
from about_llm.integrations.transformers_checkpoint_control import (  # noqa: E402
    load_checkpoint_control_spec,
)
from about_llm.llmops import artifact_fingerprint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_MANIFEST = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)
SCRIPT = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "run_qwen_activation_patching_control.py"
)
RECORDED_REPORT = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.activation-patching.recorded-report.json"
)


class TinyCausalLayer(torch.nn.Module):
    def forward(self, hidden: Any) -> Any:
        positions = torch.arange(
            1,
            hidden.shape[1] + 1,
            dtype=hidden.dtype,
            device=hidden.device,
        ).view(1, -1, 1)
        return hidden + torch.cumsum(hidden, dim=1) / positions


class TinyBackbone(torch.nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int, layer_count: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            TinyCausalLayer() for _ in range(layer_count)
        )

    def forward(self, input_ids: Any) -> Any:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TinyHookedCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(91)
        self.config = SimpleNamespace(
            model_type="tiny-hooked",
            hidden_size=8,
            vocab_size=16,
        )
        self.model = TinyBackbone(16, 8, 3)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)

    def forward(
        self, *, input_ids: Any, use_cache: bool, return_dict: bool
    ) -> Any:
        assert use_cache is False
        assert return_dict is True
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids)))


class TinyProtocolTokenizer:
    def __init__(self, protocol: ActivationPatchingProtocol) -> None:
        self.protocol = protocol

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert messages[0] == {
            "role": "system",
            "content": self.protocol.system_message,
        }
        content = messages[1]["content"]
        if content == self.protocol.clean_user_message:
            return list(self.protocol.clean_input_ids)
        if content == self.protocol.corrupt_user_message:
            return list(self.protocol.corrupt_input_ids)
        raise AssertionError("unexpected authored fixture message")

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert len(token_ids) == 1
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return f"<{token_ids[0]}>"


def _tiny_fixture() -> tuple[
    ActivationPatchingProtocol, TinyHookedCausalLM, TinyProtocolTokenizer
]:
    model = TinyHookedCausalLM().eval()
    clean = (1, 2, 3, 4)
    corrupt = (1, 5, 3, 4)
    with torch.inference_mode():
        clean_logits = model(
            input_ids=torch.tensor([clean]), use_cache=False, return_dict=True
        ).logits[0, -1]
        corrupt_logits = model(
            input_ids=torch.tensor([corrupt]), use_cache=False, return_dict=True
        ).logits[0, -1]
    delta = clean_logits - corrupt_logits
    positive = int(torch.argmax(delta).item())
    negative = int(torch.argmin(delta).item())
    protocol = ActivationPatchingProtocol(
        model_id="fixture/tiny-hooked",
        revision="a" * 40,
        expected_model_class="TinyHookedCausalLM",
        expected_model_type="tiny-hooked",
        expected_hidden_size=8,
        expected_layer_count=3,
        system_message="fixture system",
        clean_user_message="clean fixture",
        corrupt_user_message="corrupt fixture",
        clean_input_ids=clean,
        corrupt_input_ids=corrupt,
        source_position=1,
        metric_position=3,
        positive_token_text=f"<{positive}>",
        positive_token_id=positive,
        negative_token_text=f"<{negative}>",
        negative_token_id=negative,
        source_layer_indices=(0, 1, 2),
        future_token_text="<8>",
        future_token_id=8,
        expected_clean_top_token_id=None,
        expected_corrupt_top_token_id=None,
        minimum_clean_corrupt_gap=1e-6,
        control_tolerance=1e-6,
    )
    return protocol, model, TinyProtocolTokenizer(protocol)


def test_reviewed_qwen_protocol_binds_template_tokens_metric_and_sites() -> None:
    protocol = QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL

    assert protocol.fingerprint == (
        "sha256:e34b2bfe2999fe52acb18e8f1908d89db286db042be67ad4f2343d7b83ed6702"
    )
    assert protocol.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert len(protocol.clean_input_ids) == 26
    assert [
        index
        for index, pair in enumerate(
            zip(protocol.clean_input_ids, protocol.corrupt_input_ids, strict=True)
        )
        if pair[0] != pair[1]
    ] == [19]
    assert protocol.clean_input_ids[19] == 9625
    assert protocol.corrupt_input_ids[19] == 9856
    assert (protocol.positive_token_id, protocol.negative_token_id) == (59604, 94409)
    assert protocol.source_layer_indices == (0, 11, 23)
    assert protocol.metric_position == 25


def test_tiny_real_hooks_pass_constructive_and_causal_structure_controls() -> None:
    protocol, model, tokenizer = _tiny_fixture()

    report = execute_loaded_activation_patching_control(
        protocol, model=model, tokenizer=tokenizer
    )

    conditions = {item["name"]: item for item in report["conditions"]}
    assert len(conditions) == 6
    assert conditions["full_prefix_first_layer_positive_control"][
        "normalized_recovery"
    ] == pytest.approx(1.0, abs=1e-6)
    assert conditions["readout_position_final_layer_positive_control"][
        "normalized_recovery"
    ] == pytest.approx(1.0, abs=1e-6)
    assert conditions["future_position_first_layer_negative_control"][
        "normalized_recovery"
    ] == pytest.approx(0.0, abs=1e-6)
    assert conditions["source_position_layer_2"]["normalized_recovery"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert report["structural_controls"]["all_passed"] is True
    assert report["execution"]["total_forward_count"] == 10
    assert report["execution"]["hook_count_after_control"] == 0


def test_protocol_drift_fails_before_intervention() -> None:
    protocol, model, tokenizer = _tiny_fixture()
    tokenizer.protocol = ActivationPatchingProtocol(
        **{
            **protocol.__dict__,
            "clean_input_ids": (1, 2, 3, 9),
        }
    )

    with pytest.raises(ValueError, match="clean chat-template token ids drifted"):
        execute_loaded_activation_patching_control(
            protocol, model=model, tokenizer=tokenizer
        )


def test_recovery_is_unclipped_and_rejects_small_denominator() -> None:
    assert normalized_patch_recovery(
        clean_metric=1.0, corrupt_metric=0.0, patched_metric=2.0
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="denominator is too small"):
        normalized_patch_recovery(
            clean_metric=1.0,
            corrupt_metric=1.0 + 1e-10,
            patched_metric=1.2,
        )


def test_target_script_help_does_not_load_optional_model() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert "Qwen2.5-0.5B-Instruct" in completed.stdout
    assert completed.stderr == ""


def test_recorded_target_report_is_bound_and_scope_limited() -> None:
    if not RECORDED_REPORT.exists():
        pytest.skip("recorded target report is created by the reviewed slow control")
    spec = load_checkpoint_control_spec(CHECKPOINT_MANIFEST)

    report = verify_recorded_activation_patching_report(
        RECORDED_REPORT,
        expected_checkpoint_manifest_fingerprint=spec.manifest_fingerprint,
    )

    assert report["scope"]["target_checkpoint_weights_loaded"] is True
    assert report["scope"]["unique_natural_circuit_proven"] is False
    assert report["scope"]["model_quality_or_factual_reliability_proven"] is False
    assert report["result"]["baseline"]["clean_top_token_text"] == "Paris"
    assert report["result"]["baseline"]["corrupt_top_token_text"] == "Berlin"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [("scope", "scope drifted"), ("result", "result drifted")],
)
def test_recorded_target_report_cooperative_rehash_tampering_is_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    if not RECORDED_REPORT.exists():
        pytest.skip("recorded target report is created by the reviewed slow control")
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    if mutation == "scope":
        report["scope"]["unique_natural_circuit_proven"] = True
    else:
        report["result"]["baseline"]["clean_metric"] = 10.0
    projection = {
        key: value for key, value in report.items() if key != "report_fingerprint"
    }
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(projection)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    spec = load_checkpoint_control_spec(CHECKPOINT_MANIFEST)

    with pytest.raises(ValueError, match=message):
        verify_recorded_activation_patching_report(
            path,
            expected_checkpoint_manifest_fingerprint=spec.manifest_fingerprint,
        )
