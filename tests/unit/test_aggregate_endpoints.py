"""Tests for the dashboard aggregate endpoints and shared aggregate helpers.

Covers:
- amfs_core.aggregates pure functions (entity summaries, extended stats,
  cross-agent share stats)
- AdapterABC default implementations of entity_summaries / stats_extended /
  share_stats
- HTTP handlers: /api/v1/entries pagination + meta fields, /api/v1/entities,
  extended /api/v1/stats, and /api/v1/traces/share-stats (including route
  precedence over /traces/{trace_id}).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from amfs_core.aggregates import (
    entity_summaries_from_entries,
    extended_stats_from_entries,
    share_stats_from_traces,
)
from amfs_core.models import (
    DecisionTrace,
    MemoryEntry,
    Provenance,
    TraceEntry,
)

NOW = datetime.now(timezone.utc)


def _entry(
    entity_path: str = "repo/module",
    key: str = "k",
    agent_id: str = "agent-a",
    confidence: float = 0.8,
    written_at: datetime | None = None,
    recall_count: int = 0,
    outcome_count: int = 0,
    content_hash: str | None = None,
    memory_type: str = "fact",
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value={"v": 1},
        provenance=Provenance(
            agent_id=agent_id,
            session_id="s1",
            written_at=written_at or NOW,
        ),
        confidence=confidence,
        recall_count=recall_count,
        outcome_count=outcome_count,
        content_hash=content_hash,
        memory_type=memory_type,
    )


def _trace(
    reader: str,
    authors: list[str | None],
    created_at: datetime | None = None,
) -> DecisionTrace:
    return DecisionTrace(
        id=f"tr-{reader}-{len(authors)}",
        agent_id=reader,
        session_id="s1",
        causal_entries=[
            TraceEntry(
                entity_path="repo/module",
                key=f"k{i}",
                version=1,
                confidence=0.9,
                written_by=a,
            )
            for i, a in enumerate(authors)
        ],
        created_at=created_at or NOW,
    )


# ---------------------------------------------------------------------------
# Pure aggregate helpers
# ---------------------------------------------------------------------------


class TestEntitySummaries:
    def test_groups_and_sorts_by_last_updated(self) -> None:
        old = NOW - timedelta(days=3)
        entries = [
            _entry("repo/a", "k1", "agent-a", 0.6, old),
            _entry("repo/a", "k2", "agent-b", 1.0, NOW, content_hash="abc"),
            _entry("repo/b", "k1", "agent-a", 0.9, old),
        ]
        summaries = entity_summaries_from_entries(entries)

        assert [s["entity_path"] for s in summaries] == ["repo/a", "repo/b"]
        a = summaries[0]
        assert a["entry_count"] == 2
        assert a["avg_confidence"] == pytest.approx(0.8)
        assert a["last_agent"] == "agent-b"
        assert a["agents"] == ["agent-a", "agent-b"]
        assert a["hashed_count"] == 1

    def test_empty(self) -> None:
        assert entity_summaries_from_entries([]) == []

    def test_sums_recall_counts_per_entity(self) -> None:
        entries = [
            _entry("repo/a", "k1", "agent-a", recall_count=5, written_at=NOW),
            _entry("repo/a", "k2", "agent-b", recall_count=2, written_at=NOW),
            _entry("repo/b", "k1", "agent-a", recall_count=9, written_at=NOW),
        ]
        by_path = {s["entity_path"]: s for s in entity_summaries_from_entries(entries)}
        assert by_path["repo/a"]["total_recalls"] == 7
        assert by_path["repo/b"]["total_recalls"] == 9


class TestExtendedStats:
    def test_counts_recalls_and_weekly_delta(self) -> None:
        entries = [
            _entry("repo/a", "k1", "agent-a", recall_count=5, written_at=NOW),
            _entry(
                "repo/a",
                "k2",
                "agent-b",
                recall_count=2,
                written_at=NOW - timedelta(days=10),
                memory_type="belief",
            ),
            _entry(
                "repo/b",
                "k1",
                "agent-a",
                outcome_count=1,
                written_at=NOW - timedelta(days=30),
            ),
        ]
        stats = extended_stats_from_entries(entries)

        assert stats["total_entries"] == 3
        assert stats["total_entities"] == 2
        assert stats["total_agents"] == 2
        assert stats["total_recalls"] == 7
        assert stats["entries_this_week"] == 1
        assert stats["entries_last_week"] == 1
        assert stats["outcome_linked_count"] == 1
        assert stats["memory_type_counts"] == {"fact": 2, "belief": 1}
        assert stats["oldest_entry_at"] == NOW - timedelta(days=30)
        assert stats["newest_entry_at"] == NOW

    def test_recall_deltas_unknown_from_entries(self) -> None:
        # Entries carry only a recall counter (no timestamps), so the pure
        # path reports None — clients hide the trend instead of showing 0.
        stats = extended_stats_from_entries([_entry("repo/a", "k", "agent-a")])
        assert stats["recalls_this_week"] is None
        assert stats["recalls_last_week"] is None

    def test_empty(self) -> None:
        stats = extended_stats_from_entries([])
        assert stats["total_entries"] == 0
        assert stats["total_recalls"] == 0
        assert stats["confidence_avg"] == 0.0
        assert stats["oldest_entry_at"] is None


class TestShareStats:
    def test_counts_cross_agent_pairs_only(self) -> None:
        traces = [
            # reader-a used writer-b's knowledge twice, own knowledge once
            _trace("reader-a", ["writer-b", "writer-b", "reader-a"]),
            _trace("reader-c", ["writer-b", None]),
        ]
        stats = share_stats_from_traces(traces)

        assert stats["total"] == 3
        assert stats["pairs"][0] == {
            "reader": "reader-a",
            "author": "writer-b",
            "count": 2,
        }
        assert {"reader": "reader-c", "author": "writer-b", "count": 1} in stats["pairs"]

    def test_since_filter(self) -> None:
        old = NOW - timedelta(days=60)
        traces = [
            _trace("reader-a", ["writer-b"], created_at=old),
            _trace("reader-a", ["writer-b"], created_at=NOW),
        ]
        stats = share_stats_from_traces(traces, since=NOW - timedelta(days=30))
        assert stats["total"] == 1

    def test_agent_ids_restricts_both_sides(self) -> None:
        traces = [
            _trace("reader-a", ["writer-b"]),
            _trace("reader-a", ["hidden-agent"]),
            _trace("hidden-agent", ["writer-b"]),
        ]
        stats = share_stats_from_traces(traces, agent_ids=["reader-a", "writer-b"])
        assert stats["total"] == 1
        assert stats["pairs"] == [
            {"reader": "reader-a", "author": "writer-b", "count": 1}
        ]

    def test_pair_limit_caps_pairs_not_total(self) -> None:
        traces = [
            _trace("r1", ["w1", "w1"]),
            _trace("r2", ["w2"]),
            _trace("r3", ["w3"]),
        ]
        stats = share_stats_from_traces(traces, pair_limit=1)
        assert stats["total"] == 4
        assert len(stats["pairs"]) == 1
        assert stats["pairs"][0]["count"] == 2


# ---------------------------------------------------------------------------
# AdapterABC default implementations
# ---------------------------------------------------------------------------


def _fake_adapter(entries: list[MemoryEntry], traces: list[DecisionTrace]):
    from amfs_core.abc import AdapterABC

    class FakeAdapter(AdapterABC):
        def read(self, entity_path, key, *, min_confidence=0.0):
            return None

        def write(self, entry):
            return entry

        def list(self, entity_path=None, *, include_superseded=False):
            return entries

        def watch(self, entity_path, callback):
            raise NotImplementedError

        def commit_outcome(self, record):
            return []

        def list_traces(self, **kwargs):
            return traces

    return FakeAdapter()


class TestAdapterDefaults:
    def test_entity_summaries_with_agent_filter(self) -> None:
        adapter = _fake_adapter(
            [
                _entry("repo/a", "k1", "agent-a"),
                _entry("repo/a", "k2", "agent-b"),
            ],
            [],
        )
        summaries = adapter.entity_summaries(agent_ids=["agent-a"])
        assert summaries[0]["entry_count"] == 1
        assert summaries[0]["agents"] == ["agent-a"]

    def test_stats_extended_matches_helper(self) -> None:
        entries = [_entry("repo/a", "k1", recall_count=3)]
        adapter = _fake_adapter(entries, [])
        assert adapter.stats_extended() == extended_stats_from_entries(entries)

    def test_share_stats_delegates(self) -> None:
        traces = [_trace("reader-a", ["writer-b"])]
        adapter = _fake_adapter([], traces)
        assert adapter.share_stats()["total"] == 1


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

import amfs_http.server as server  # noqa: E402


@pytest.fixture()
def entries() -> list[MemoryEntry]:
    return [
        _entry("repo/a", "k1", "agent-a", 0.9, NOW, recall_count=4),
        _entry("repo/a", "k2", "agent-b", 0.7, NOW - timedelta(days=1), recall_count=9),
        _entry("repo/b", "k1", "agent-a", 0.5, NOW - timedelta(days=2), recall_count=1),
    ]


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch, entries: list[MemoryEntry]
) -> TestClient:
    traces = [_trace("reader-a", ["writer-b", "writer-b"])]
    adapter = _fake_adapter(entries, traces)

    mem = MagicMock()
    mem.namespace = "test-ns"
    mem._adapter = adapter
    mem.list.side_effect = lambda *a, **kw: adapter.list()

    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda request: None)
    return TestClient(server.app)


class TestEntriesPagination:
    def test_default_returns_all_with_total(self, client: TestClient) -> None:
        res = client.get("/api/v1/entries")
        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 3
        assert len(body["entries"]) == 3
        assert "value" in body["entries"][0]

    def test_sort_limit_offset(self, client: TestClient) -> None:
        res = client.get(
            "/api/v1/entries", params={"sort": "recall_count", "limit": 1, "offset": 1}
        )
        body = res.json()
        assert body["total"] == 3
        assert len(body["entries"]) == 1
        assert body["entries"][0]["recall_count"] == 4

    def test_fields_meta_strips_values(self, client: TestClient) -> None:
        res = client.get("/api/v1/entries", params={"fields": "meta"})
        body = res.json()
        assert all("value" not in e for e in body["entries"])
        assert all("recall_count" in e for e in body["entries"])

    def test_invalid_sort_rejected(self, client: TestClient) -> None:
        res = client.get("/api/v1/entries", params={"sort": "nope"})
        assert res.status_code == 422


class TestEntitiesEndpoint:
    def test_returns_summaries_without_values(self, client: TestClient) -> None:
        res = client.get("/api/v1/entities")
        assert res.status_code == 200
        body = res.json()
        assert len(body["entities"]) == 2
        first = body["entities"][0]
        assert first["entity_path"] == "repo/a"
        assert first["entry_count"] == 2
        assert "value" not in first
        assert "entries" not in first

    def test_visibility_filter_applied(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vis = MagicMock()
        vis.should_filter.return_value = True
        vis.filter_entries.side_effect = lambda entries: [
            e for e in entries if e.provenance.agent_id == "agent-b"
        ]
        monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)

        res = client.get("/api/v1/entities")
        body = res.json()
        assert len(body["entities"]) == 1
        assert body["entities"][0]["entry_count"] == 1


class TestStatsExtended:
    def test_includes_extended_fields(self, client: TestClient) -> None:
        res = client.get("/api/v1/stats")
        assert res.status_code == 200
        body = res.json()
        # Backward-compatible MemoryStats fields
        assert body["total_entries"] == 3
        assert body["total_entities"] == 2
        assert body["confidence_avg"] == pytest.approx(0.7)
        # New extended fields
        assert body["total_recalls"] == 14
        assert "entries_this_week" in body
        assert body["memory_type_counts"] == {"fact": 3}

    def test_visibility_branch_has_same_shape(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vis = MagicMock()
        vis.should_filter.return_value = True
        vis.filter_entries.side_effect = lambda entries: entries[:1]
        monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)

        res = client.get("/api/v1/stats")
        body = res.json()
        assert body["total_entries"] == 1
        assert body["total_recalls"] == 4
        assert "confidence_avg" in body
        assert "oldest_entry_at" in body


class TestShareStatsEndpoint:
    def test_share_stats_route_wins_over_trace_id(self, client: TestClient) -> None:
        res = client.get("/api/v1/traces/share-stats")
        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 2
        assert body["pairs"] == [
            {"reader": "reader-a", "author": "writer-b", "count": 2}
        ]

    def test_visibility_restricts_agents(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vis = MagicMock()
        vis.should_filter.return_value = True
        vis.get_visible_agent_ids.return_value = {"someone-else"}
        monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)

        res = client.get("/api/v1/traces/share-stats")
        assert res.json()["total"] == 0
