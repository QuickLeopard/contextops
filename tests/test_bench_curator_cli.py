"""Tests for the `curator` CLI subcommand's judge-selection logic in
`contextops_bench.__main__`.

`_select_curator_judge()` is a pure function extracted from `curator(args)`
specifically so this precedence logic (--echo-judge > --litellm-judge >
provider=="echo" fallback > default self-judging) is testable without a
real network call or constructing a full `argparse.Namespace`.
"""

from __future__ import annotations

from contextops.clients import EchoJudge, LiteLLMJudge
from contextops_bench.curator_bench import _ClientAsJudge
from contextops_bench.__main__ import _select_curator_judge


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
