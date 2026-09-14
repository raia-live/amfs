"""A briefing books reuse of what it hands over — but only when asked to.

The gap this closes: briefing *reported* recall_count and never incremented it,
so the agent that follows the documented workflow (brief first, then work) was
the one penalised. Entries it was briefed on stayed at zero for ever, no value
line was shown for a memory that had just done its job, and the write-only ratio
counted briefed knowledge as never read.

The reason it is opt-in rather than automatic is the other half of the story.
The same endpoint serves an agent about to act and a dashboard panel rendering a
briefing for a human, and crediting a page view as reuse is a bug this codebase
has already shipped once — a nine-entry topic reached 395 recalls that way. Only
the caller can tell the two apart, so the caller has to say.
"""

from __future__ import annotations

import asyncio
import types
from datetime import datetime, timezone

import pytest

server = pytest.importorskip("amfs_http.server")

from amfs_core.models import Digest, DigestType  # noqa: E402


def _digest(scope: str, hot: list[dict]) -> Digest:
    return Digest(
        digest_type=DigestType.ENTITY,
        scope=scope,
        summary={"narrative": f"about {scope}", "hot_context": hot},
        entry_count=len(hot),
        source_agents=["a"],
        compiled_at=datetime.now(timezone.utc),
        namespace="default",
        branch="main",
    )


class _RecordingAdapter:
    """Counts increments without needing a database."""

    def __init__(self) -> None:
        self.credited: list[tuple[str, str]] = []

    def increment_recall_count(self, entity_path: str, key: str, **_kw) -> None:
        self.credited.append((entity_path, key))


class _FakeMemory:
    def __init__(self, digests, adapter):
        self._digests = digests
        self._adapter = adapter

    def briefing(self, **_kw):
        return list(self._digests)


def _request():
    return types.SimpleNamespace(state=types.SimpleNamespace(visibility_filter=None))


@pytest.fixture
def wired(monkeypatch):
    adapter = _RecordingAdapter()
    digests = [_digest("looptest/deploy", [
        {"entity_path": "looptest/deploy", "key": "top-priority", "value": "v"},
        {"entity_path": "looptest/deploy", "key": "second", "value": "v"},
    ])]
    monkeypatch.setattr(server, "_async_adapter", None, raising=False)
    monkeypatch.setattr(server, "_get_memory", lambda: _FakeMemory(digests, adapter))
    return adapter, digests


def _call(credit_reuse):
    return asyncio.run(server.get_briefing(
        _request(), entity_path="looptest/deploy", agent_id=None, limit=10,
        credit_reuse=credit_reuse, _auth=None,
    ))


class TestBriefingCreditsReuse:
    def test_credits_the_top_surfaced_entry_when_asked(self, wired):
        adapter, _ = wired
        out = _call(True)
        assert out["total"] == 1
        assert adapter.credited == [("looptest/deploy", "top-priority")], (
            "the briefing must book reuse of the knowledge it handed over"
        )

    def test_credits_nothing_by_default(self, wired):
        """The dashboard renders briefings on page view and must stay free."""
        adapter, _ = wired
        _call(False)
        assert adapter.credited == []

    def test_caps_credit_rather_than_crediting_everything_shown(self, wired):
        """Two entries surfaced, one credited — the 395x inflation lesson."""
        adapter, _ = wired
        _call(True)
        assert len(adapter.credited) == server.REUSE_CREDIT_K

    def test_ignores_digests_without_hot_context(self, monkeypatch):
        """A compiled narrative is a synthesis about entries, not the entries."""
        adapter = _RecordingAdapter()
        narrative_only = Digest(
            digest_type=DigestType.ENTITY,
            scope="looptest/deploy",
            summary={"narrative": "no hot context here", "key_facts": ["a fact"]},
            entry_count=3,
            source_agents=["a"],
            compiled_at=datetime.now(timezone.utc),
            namespace="default",
            branch="main",
        )
        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(server, "_get_memory",
                            lambda: _FakeMemory([narrative_only], adapter))
        _call(True)
        assert adapter.credited == []

    def test_credit_never_precedes_visibility_filtering(self, monkeypatch):
        """An entry the caller may not see must not be credited to them."""
        adapter = _RecordingAdapter()
        hidden = _digest("other/secret", [
            {"entity_path": "other/secret", "key": "not-yours", "value": "v"},
        ])
        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(server, "_get_memory", lambda: _FakeMemory([hidden], adapter))

        class _Vis:
            def should_filter(self):
                return True

        monkeypatch.setattr(server, "_get_visibility_filter", lambda _r: _Vis())
        monkeypatch.setattr(server, "_filter_briefing_digests", lambda _v, _d: [])

        asyncio.run(server.get_briefing(
            _request(), entity_path="other/secret", agent_id=None, limit=10,
            credit_reuse=True, _auth=None,
        ))
        assert adapter.credited == []

    def test_accounting_failure_does_not_break_the_briefing(self, monkeypatch):
        """Reuse accounting is best-effort; a briefing must still be returned."""
        class _Exploding(_RecordingAdapter):
            def increment_recall_count(self, entity_path, key, **_kw):
                raise RuntimeError("db down")

        digests = [_digest("looptest/deploy", [
            {"entity_path": "looptest/deploy", "key": "top", "value": "v"},
        ])]
        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(server, "_get_memory",
                            lambda: _FakeMemory(digests, _Exploding()))
        out = _call(True)
        assert out["total"] == 1

    def test_deduplicates_an_entry_surfaced_by_two_digests(self, monkeypatch):
        adapter = _RecordingAdapter()
        same = {"entity_path": "looptest/deploy", "key": "shared", "value": "v"}
        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(server, "_get_memory", lambda: _FakeMemory(
            [_digest("looptest/deploy", [same]), _digest("looptest", [same])], adapter))
        _call(True)
        assert adapter.credited == [("looptest/deploy", "shared")]
