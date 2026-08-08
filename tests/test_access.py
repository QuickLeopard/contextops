import hashlib


from contextops.access import (
    Principal,
    RoleBasedPolicyEngine,
    TaggedContent,
    apply_access_policy,
)
from contextops.models import HistoryMessage, Prompt


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_public_content_always_allowed():
    p = Principal(id="alice", roles={"support"})
    contents = [TaggedContent(text="hello", section="query")]
    result = apply_access_policy(contents, p)
    assert result.prompt.query == "hello"
    assert result.redacted == []
    assert len(result.decisions) == 1
    assert result.decisions[0].action == "included"


def test_redacts_missing_role():
    p = Principal(id="bob", roles={"support"})
    contents = [
        TaggedContent(text="public context", section="context"),
        TaggedContent(
            text="secret docs",
            section="documents",
            required_roles={"executive"},
        ),
    ]
    result = apply_access_policy(contents, p)
    assert result.prompt.context == "public context"
    assert result.prompt.documents == ""
    assert len(result.redacted) == 1
    assert result.redacted[0][1] == "missing required roles: executive"


def test_allows_matching_role():
    p = Principal(id="carol", roles={"support", "executive"})
    contents = [
        TaggedContent(
            text="secret docs",
            section="documents",
            required_roles={"executive"},
        ),
    ]
    result = apply_access_policy(contents, p)
    assert result.prompt.documents == "secret docs"
    assert result.redacted == []


def test_documents_joined():
    p = Principal(id="dave", roles={"support"})
    contents = [
        TaggedContent(text="chunk one", section="documents"),
        TaggedContent(text="chunk two", section="documents"),
    ]
    result = apply_access_policy(contents, p)
    assert result.prompt.documents == "chunk one\n\nchunk two"


def test_history_converted_to_messages():
    p = Principal(id="eve", roles={"support"})
    contents = [
        TaggedContent(text="msg 1", section="history"),
        TaggedContent(text="msg 2", section="history"),
    ]
    result = apply_access_policy(contents, p)
    assert result.prompt.history == [
        HistoryMessage(role="user", content="msg 1"),
        HistoryMessage(role="user", content="msg 2"),
    ]


def test_redaction_decision_hash_matches_content():
    p = Principal(id="frank", roles=set())
    contents = [
        TaggedContent(
            text="top secret",
            section="documents",
            required_roles={"classified"},
        ),
    ]
    result = apply_access_policy(contents, p)
    decision = result.decisions[0]
    assert decision.action == "redacted"
    assert decision.content_hash == _hash("top secret")
    assert decision.principal_id == "frank"
    assert decision.section == "documents"


def test_multiple_roles_any_match():
    engine = RoleBasedPolicyEngine()
    p = Principal(id="grace", roles={"a", "b"})
    content = TaggedContent(text="x", section="query", required_roles={"c", "b"})
    assert engine.is_allowed(p, content) is True


def test_custom_engine_can_override():
    class DenyAll:
        def is_allowed(self, principal, content):
            return False

    p = Principal(id="henry", roles={"admin"})
    contents = [TaggedContent(text="anything", section="query")]
    result = apply_access_policy(contents, p, engine=DenyAll())
    assert result.prompt.query == ""
    assert len(result.redacted) == 1


def test_empty_content_list():
    p = Principal(id="iris", roles={"support"})
    result = apply_access_policy([], p)
    assert result.prompt == Prompt()
    assert result.included == []
    assert result.redacted == []
    assert result.decisions == []
