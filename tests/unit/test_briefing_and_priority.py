"""Unit tests for BriefingService (WS2/WS4) and priority sort."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

pytest.importorskip("amfs_cortex", reason="amfs_cortex not installed")

from amfs_core.models import (
    Digest,
    DigestType,
    MemoryEntry,
    Provenance,
    SearchQuery,
)
from amfs_cortex.briefing import BriefingService


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _entry(
    key: str,
    *,
    entity_path: str = "svc",
    confidence: float = 1.0,
    recall_count: int = 0,
    outcome_count: int = 0,
    agent_id: str = "a",
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=f"value-{key}",
        provenance=Provenance(agent_id=agent_id, session_id="s", written_at=_now()),
        confidence=confidence,
        recall_count=recall_count,
        outcome_count=outcome_count,
    )


def _entity_digest(
    scope: str,
    *,
    entry_count: int = 5,
    age_hours: float = 1.0,
) -> Digest:
    return Digest(
        digest_type=DigestType.ENTITY,
        scope=scope,
        summary={"narrative": f"Summary of {scope}", "agents": ["a1"]},
        entry_count=entry_count,
        source_agents=["a1"],
        compiled_at=_now() - timedelta(hours=age_hours),
        namespace="default",
        branch="main",
    )


def _mock_adapter(
    digests: list[Digest] | None = None,
    search_results: list[MemoryEntry] | None = None,
    branches: list | None = None,
):
    adapter = MagicMock()
    adapter.list_digests.return_value = digests or []
    adapter.search.return_value = search_results or []
    adapter.list_branches.return_value = branches or []
    return adapter


# ──────────────────────────────────────────────────────────────
# Hot-context injection
# ──────────────────────────────────────────────────────────────


class TestHotContextInjection:
    def test_entity_digest_gets_hot_context(self) -> None:
        digest = _entity_digest("svc")
        entries = [_entry("k1"), _entry("k2"), _entry("k3")]
        adapter = _mock_adapter(digests=[digest], search_results=entries)

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert len(result) == 1
        assert "hot_context" in result[0].summary
        assert len(result[0].summary["hot_context"]) == 3

    def test_hot_context_entry_fields(self) -> None:
        digest = _entity_digest("svc")
        entries = [_entry("k1", confidence=0.85, recall_count=5, outcome_count=2)]
        adapter = _mock_adapter(digests=[digest], search_results=entries)

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        hot = result[0].summary["hot_context"][0]
        assert hot["key"] == "k1"
        assert hot["confidence"] == 0.85
        assert hot["recall_count"] == 5
        assert hot["outcome_count"] == 2

    def test_non_entity_digest_not_modified(self) -> None:
        digest = Digest(
            digest_type=DigestType.AGENT_BRIEF,
            scope="agent-a",
            summary={"narrative": "Agent brief"},
            entry_count=1,
            compiled_at=_now(),
        )
        adapter = _mock_adapter(digests=[digest], search_results=[_entry("k1")])

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc", agent_id="agent-a")

        for d in result:
            assert "hot_context" not in d.summary

    def test_empty_search_no_hot_context(self) -> None:
        digest = _entity_digest("svc")
        adapter = _mock_adapter(digests=[digest], search_results=[])

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert "hot_context" not in result[0].summary

    def test_search_exception_handled_gracefully(self) -> None:
        digest = _entity_digest("svc")
        adapter = _mock_adapter(digests=[digest])
        adapter.search.side_effect = Exception("DB timeout")

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert len(result) == 1
        assert "hot_context" not in result[0].summary

    def test_search_uses_priority_sort(self) -> None:
        digest = _entity_digest("svc")
        adapter = _mock_adapter(digests=[digest], search_results=[_entry("k1")])

        svc = BriefingService(adapter)
        svc.briefing(entity_path="svc")

        search_call = adapter.search.call_args
        query = search_call[0][0]
        assert isinstance(query, SearchQuery)
        assert query.sort_by == "priority"
        assert query.limit == 3


# ──────────────────────────────────────────────────────────────
# Standalone hot-context (no compiled digest)
# ──────────────────────────────────────────────────────────────


class TestStandaloneHotContext:
    def test_creates_synthetic_digest_when_no_digests(self) -> None:
        entries = [_entry("k1"), _entry("k2")]
        adapter = _mock_adapter(digests=[], search_results=entries)

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert len(result) == 1
        assert result[0].digest_type == DigestType.ENTITY
        assert result[0].scope == "svc"
        assert "hot_context" in result[0].summary

    def test_no_synthetic_if_no_entries(self) -> None:
        adapter = _mock_adapter(digests=[], search_results=[])

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert len(result) == 0

    def test_synthetic_digest_has_correct_fields(self) -> None:
        entries = [_entry("k1")]
        adapter = _mock_adapter(digests=[], search_results=entries)

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        d = result[0]
        assert d.digest_type == DigestType.ENTITY
        assert d.scope == "svc"
        assert d.entry_count == 1
        assert "narrative" in d.summary
        assert "hot_context" in d.summary


# ──────────────────────────────────────────────────────────────
# Consolidation notice injection
# ──────────────────────────────────────────────────────────────


class TestConsolidationNotice:
    def _mock_branch(self, name: str, status: str = "active"):
        b = MagicMock()
        b.name = name
        b.status = status
        return b

    def test_pending_branches_add_notice(self) -> None:
        digest = _entity_digest("svc")
        branches = [
            self._mock_branch("cortex/consolidation/svc/20260101-120000"),
            self._mock_branch("cortex/consolidation/svc/20260102-120000"),
        ]
        adapter = _mock_adapter(
            digests=[digest], search_results=[_entry("k1")], branches=branches,
        )

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert "consolidation_notice" in result[0].summary
        notice = result[0].summary["consolidation_notice"]
        assert notice["pending_proposals"] == 2

    def test_branches_for_other_entity_ignored(self) -> None:
        digest = _entity_digest("svc")
        branches = [
            self._mock_branch("cortex/consolidation/other/20260101-120000"),
        ]
        adapter = _mock_adapter(
            digests=[digest], search_results=[_entry("k1")], branches=branches,
        )

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert "consolidation_notice" not in result[0].summary

    def test_no_pending_branches_no_notice(self) -> None:
        digest = _entity_digest("svc")
        adapter = _mock_adapter(
            digests=[digest], search_results=[_entry("k1")], branches=[],
        )

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert "consolidation_notice" not in result[0].summary

    def test_branches_exception_handled(self) -> None:
        digest = _entity_digest("svc")
        adapter = _mock_adapter(digests=[digest], search_results=[_entry("k1")])
        adapter.list_branches.side_effect = Exception("Not available")

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert "consolidation_notice" not in result[0].summary


# ──────────────────────────────────────────────────────────────
# Full briefing flow
# ──────────────────────────────────────────────────────────────


class TestBriefingFlow:
    def test_entity_path_with_digests_gets_injection(self) -> None:
        digest = _entity_digest("svc")
        adapter = _mock_adapter(digests=[digest], search_results=[_entry("k1")])

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert len(result) == 1
        assert "hot_context" in result[0].summary

    def test_no_entity_path_no_injection(self) -> None:
        digest = _entity_digest("svc")
        adapter = _mock_adapter(digests=[digest])

        svc = BriefingService(adapter)
        result = svc.briefing()

        for d in result:
            assert "hot_context" not in d.summary

    def test_multiple_digests_sorted_by_score(self) -> None:
        d1 = _entity_digest("svc", entry_count=10, age_hours=0.5)
        d2 = Digest(
            digest_type=DigestType.AGENT_BRIEF,
            scope="agent-x",
            summary={"narrative": "Brief", "entities_written": ["svc"]},
            entry_count=3,
            compiled_at=_now() - timedelta(hours=2),
        )
        adapter = _mock_adapter(digests=[d1, d2], search_results=[_entry("k1")])

        svc = BriefingService(adapter)
        result = svc.briefing(entity_path="svc")

        assert result[0].digest_type == DigestType.ENTITY


# ──────────────────────────────────────────────────────────────
# Priority sort (with real adapter)
# ──────────────────────────────────────────────────────────────


class TestPrioritySort:
    def test_priority_sort_reranks_by_composite_score(self) -> None:
        """Priority sort uses PriorityScorer which weighs recall_count, not just confidence."""
        from amfs_core.tiering import PriorityScorer

        high_conf = _entry("high-conf", confidence=1.0, recall_count=0)
        high_recall = _entry("high-recall", confidence=0.7, recall_count=50)

        scorer = PriorityScorer()
        score_conf = scorer.score(high_conf)
        score_recall = scorer.score(high_recall)

        assert score_recall > score_conf, (
            "High recall_count should boost priority above raw confidence"
        )

        entries = [high_conf, high_recall]
        scores = scorer.score_batch(entries)
        entries.sort(key=lambda e: scores.get(e.entry_key, 0.0), reverse=True)

        assert entries[0].key == "high-recall"

    def test_priority_sort_respects_limit(self, tmp_path: Path) -> None:
        from amfs.memory import AgentMemory
        from amfs_filesystem.adapter import FilesystemAdapter

        adapter = FilesystemAdapter(root=tmp_path / ".amfs")
        mem = AgentMemory(adapter=adapter, agent_id="test", session_id="s1")

        for i in range(10):
            mem.write("svc", f"k{i}", f"val{i}")

        results = mem.search(entity_path="svc", sort_by="priority", limit=3)
        assert len(results) == 3

    def test_priority_sort_query_model(self) -> None:
        sq = SearchQuery(entity_path="svc", sort_by="priority", limit=5)
        assert sq.sort_by == "priority"

    def test_confidence_sort_still_works(self, tmp_path: Path) -> None:
        from amfs.memory import AgentMemory
        from amfs_filesystem.adapter import FilesystemAdapter

        adapter = FilesystemAdapter(root=tmp_path / ".amfs")
        mem = AgentMemory(adapter=adapter, agent_id="test", session_id="s1")

        mem.write("svc", "low", "val", confidence=0.3)
        mem.write("svc", "high", "val", confidence=0.9)

        results = mem.search(entity_path="svc", sort_by="confidence")
        assert results[0].key == "high"
