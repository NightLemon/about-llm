from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from about_llm.finetuning.ppo_text import (  # noqa: E402
    batch_prompt_completions,
    build_text_contexts,
    build_text_control_tokenizer,
    enumerate_stopped_responses,
    render_text_control_prompt,
)

pytestmark = pytest.mark.contract


def test_text_control_tokenizer_prompt_and_padding_are_explicit() -> None:
    tokenizer = build_text_control_tokenizer()
    rendered, prompt_ids = render_text_control_prompt(tokenizer)
    assert rendered == (
        "<|system|> Return one word. </s> "
        "<|user|> Say good. </s> <|assistant|> "
    )
    assert len(tokenizer) == 13
    assert len(prompt_ids) == 10
    prefixes = torch.tensor([[11], [12]], dtype=torch.long)
    contexts, masks = build_text_contexts(
        prompt_ids,
        prefixes,
        max_context_length=12,
        pad_token_id=1,
    )
    assert contexts.shape == (2, 12)
    assert torch.equal(contexts[:, 10], prefixes[:, 0])
    assert torch.equal(contexts[:, 11], torch.tensor([1, 1]))
    assert torch.equal(masks.sum(dim=1), torch.tensor([11, 11]))


def test_stopped_response_enumeration_distinguishes_eos_from_cap() -> None:
    responses = enumerate_stopped_responses(vocab_size=3, eos_token_id=2, horizon=2)
    assert len(responses) == 7
    assert len(set(responses)) == 7
    assert (2,) in responses
    assert all(not (len(response) == 2 and response[0] == 2) for response in responses)
    assert sum(len(response) == 1 for response in responses) == 1
    assert sum(len(response) == 2 for response in responses) == 6
    restricted = enumerate_stopped_responses(
        vocab_size=13,
        eos_token_id=2,
        horizon=2,
        allowed_token_ids=(2, 6, 7, 8, 9, 10, 11, 12),
    )
    assert len(restricted) == 57
    assert all(token_id in {2, 6, 7, 8, 9, 10, 11, 12} for row in restricted for token_id in row)


def test_completion_batch_rejects_silent_truncation() -> None:
    ids, mask = batch_prompt_completions(
        (3, 4), ((5,), (6, 2)), max_context_length=4, pad_token_id=1
    )
    assert ids.tolist() == [[3, 4, 5, 1], [3, 4, 6, 2]]
    assert mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    with pytest.raises(ValueError, match="exceed"):
        batch_prompt_completions(
            (3, 4), ((5, 6, 7),), max_context_length=4, pad_token_id=1
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vocab_size": 0, "eos_token_id": 0, "horizon": 2}, "vocab_size"),
        ({"vocab_size": 3, "eos_token_id": 3, "horizon": 2}, "eos_token_id"),
        ({"vocab_size": 3, "eos_token_id": 2, "horizon": 0}, "horizon"),
    ],
)
def test_stopped_response_enumeration_rejects_invalid_configuration(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        enumerate_stopped_responses(**kwargs)
