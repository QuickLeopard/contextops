"""Tests for the `curator` CLI subcommand's judge-selection logic in
`contextops_bench.__main__`.

`_select_curator_judge()` is a pure function extracted from `curator(args)`
specifically so this precedence logic (--echo-judge > --litellm-judge >
provider=="echo" fallback > default self-judging) is testable without a
real network call or constructing a full `argparse.Namespace`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextops.clients import EchoJudge, LiteLLMJudge
from contextops_bench.curator_bench import _ClientAsJudge, CurationBenchItem
from contextops_bench.__main__ import (
    _check_dataset_mutual_exclusivity,
    _load_or_generate_curator_dataset,
    _select_curator_judge,
    _SYNTHETIC_ONLY_DEFAULTS,
)


class _FakeClient:
    PROVIDER = "fake"


def test_echo_judge_flag_always_wins():
    judge, label = _select_curator_judge(
        client=_FakeClient(), model="claude-haiku-4-5", provider="direct_zen",
        echo_judge=True, litellm_judge=False, judge_model=None,
    )
    assert isinstance(judge, EchoJudge)
    assert label == "echo"


def test_echo_provider_without_flags_falls_back_to_echo_judge():
    """Self-judging against EchoClient's fixed output would be meaningless."""
    judge, label = _select_curator_judge(
        client=_FakeClient(), model="fake-model", provider="echo",
        echo_judge=False, litellm_judge=False, judge_model=None,
    )
    assert isinstance(judge, EchoJudge)
    assert label == "echo"


def test_default_real_provider_self_judges_with_same_model():
    judge, label = _select_curator_judge(
        client=_FakeClient(), model="claude-haiku-4-5", provider="direct_zen",
        echo_judge=False, litellm_judge=False, judge_model=None,
    )
    assert isinstance(judge, _ClientAsJudge)
    assert label == "self"


def test_default_real_provider_self_judges_with_different_judge_model():
    """--judge-model pointing at a sibling model (same provider/key) should
    be labeled distinctly from same-model self-judging.
    """
    judge, label = _select_curator_judge(
        client=_FakeClient(), model="claude-haiku-4-5", provider="direct_zen",
        echo_judge=False, litellm_judge=False, judge_model="claude-sonnet-4-6",
    )
    assert isinstance(judge, _ClientAsJudge)
    assert label == "self:claude-sonnet-4-6"


def test_litellm_judge_flag_used_when_requested():
    judge, label = _select_curator_judge(
        client=_FakeClient(), model="claude-haiku-4-5", provider="direct_zen",
        echo_judge=False, litellm_judge=True, judge_model="gpt-4o-mini",
    )
    assert isinstance(judge, LiteLLMJudge)
    assert label == "litellm:gpt-4o-mini"


def test_litellm_judge_wins_over_default_self_judge():
    judge, label = _select_curator_judge(
        client=_FakeClient(), model="claude-haiku-4-5", provider="direct_zen",
        echo_judge=False, litellm_judge=True, judge_model=None,
    )
    assert isinstance(judge, LiteLLMJudge)
    assert label == "litellm:claude-haiku-4-5"


def test_echo_judge_wins_over_echo_provider_fallback():
    """Both echo_judge=True and provider=="echo" -> still just "echo", no
    contradiction, first branch wins deterministically.
    """
    judge, label = _select_curator_judge(
        client=_FakeClient(), model="fake-model", provider="echo",
        echo_judge=True, litellm_judge=False, judge_model=None,
    )
    assert isinstance(judge, EchoJudge)
    assert label == "echo"


# --- _check_dataset_mutual_exclusivity ------------------------------------

def test_mutual_exclusivity_noop_when_no_dataset():
    """No dataset => never raises, regardless of other flag values."""
    _check_dataset_mutual_exclusivity(
        dataset=None, noise_ratio=0.99, chunks_per_item=99, dup_rate=0.99,
        seed=1, prompt_style="code", embedder="openai",
    )  # must not raise


def test_mutual_exclusivity_noop_when_dataset_and_all_defaults():
    _check_dataset_mutual_exclusivity(
        dataset=Path("data.json"), **_SYNTHETIC_ONLY_DEFAULTS,
    )  # must not raise


def test_mutual_exclusivity_raises_on_embedder_override():
    overrides = dict(_SYNTHETIC_ONLY_DEFAULTS)
    overrides["embedder"] = "openai"
    with pytest.raises(ValueError, match="--embedder"):
        _check_dataset_mutual_exclusivity(dataset=Path("data.json"), **overrides)


def test_mutual_exclusivity_raises_on_prompt_style_override():
    overrides = dict(_SYNTHETIC_ONLY_DEFAULTS)
    overrides["prompt_style"] = "code"
    with pytest.raises(ValueError, match="--prompt-style"):
        _check_dataset_mutual_exclusivity(dataset=Path("data.json"), **overrides)


def test_mutual_exclusivity_raises_on_multiple_overrides_lists_all():
    overrides = dict(_SYNTHETIC_ONLY_DEFAULTS)
    overrides["noise_ratio"] = 0.5
    overrides["seed"] = 7
    with pytest.raises(ValueError) as exc_info:
        _check_dataset_mutual_exclusivity(dataset=Path("data.json"), **overrides)
    assert "--noise-ratio" in str(exc_info.value)
    assert "--seed" in str(exc_info.value)


# --- _load_or_generate_curator_dataset -------------------------------------

def test_load_or_generate_uses_synthetic_generator_by_default():
    items = _load_or_generate_curator_dataset(
        dataset=None, n=5, **_SYNTHETIC_ONLY_DEFAULTS,
    )
    assert len(items) == 5
    assert all(isinstance(it, CurationBenchItem) for it in items)


def test_load_or_generate_uses_dataset_file_when_given(tmp_path):
    data = [{
        "query": "What is the capital of France?",
        "expected": "Paris",
        "chunks": [{"text": "Paris is the capital of France.", "similarity": 0.9}],
    }]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))

    items = _load_or_generate_curator_dataset(
        dataset=path, n=999, **_SYNTHETIC_ONLY_DEFAULTS,
    )
    assert len(items) == 1
    assert items[0].query == "What is the capital of France?"


def test_load_or_generate_raises_on_conflicting_flags(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([]))
    overrides = dict(_SYNTHETIC_ONLY_DEFAULTS)
    overrides["embedder"] = "tfidf"
    with pytest.raises(ValueError, match="--embedder"):
        _load_or_generate_curator_dataset(dataset=path, n=5, **overrides)


def test_load_or_generate_applies_prompt_style():
    overrides = dict(_SYNTHETIC_ONLY_DEFAULTS)
    overrides["prompt_style"] = "code"
    items = _load_or_generate_curator_dataset(dataset=None, n=3, **overrides)
    assert len(items) == 3


def test_load_or_generate_applies_embedder():
    overrides = dict(_SYNTHETIC_ONLY_DEFAULTS)
    overrides["embedder"] = "tfidf"
    items = _load_or_generate_curator_dataset(dataset=None, n=3, **overrides)
    for it in items:
        for c in it.chunks:
            assert 0.0 <= c.similarity <= 1.0
