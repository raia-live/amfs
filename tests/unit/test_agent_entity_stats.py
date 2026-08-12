"""Unit tests for the per-(agent, entity) aggregate behind authority ranking.

The Postgres adapter overrides this with a GROUP BY, so the contract pinned
here is the one both implementations have to satisfy — most importantly the
prefix scoping, where 'a/b' must cover 'a/b/c' without ever matching 'a/bc'.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("amfs_core", reason="amfs_core not installed")

from amfs_core.abc import AdapterABC
from amfs_core.aggregates import agent_entity_stats_from_entries
from amfs_core.models import MemoryEntry, Provenance

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


def _entry(
    agent_id: str,
    entity_path: str,
    key: str,
    *,
    confidence: float = 0.8,
    recalls: int = 0,
    outcomes: int = 0,
    days_ago: float = 1.0,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value="v",
        confidence=confidence,
        recall_count=recalls,
        outcome_count=outcomes,
        provenance=Provenance(
            agent_id=agent_id,
            session_id=f"session-{agent_id}",
            written_at=NOW - timedelta(days=days_ago),
        ),
    )


class _ListAdapter(AdapterABC):
    """Minimal adapter exercising the ABC's list()-based default."""

    def __init__(self, entries: list[MemoryEntry]) -> None:
        self._entries = entries

    def list(self, entity_path=None, *, include_superseded=False, branch="main"):
        if entity_path is None:
            return list(self._entries)
        return [e for e in self._entries if e.entity_path == entity_path]

    # Unused abstract surface.
    def write(self, *a, **k): raise NotImplementedError
    def read(self, *a, **k): raise NotImplementedError
    def delete(self, *a, **k): raise NotImplementedError
    def history(self, *a, **k): raise NotImplementedError
    def commit_outcome(self, *a, **k): raise NotImplementedError
    def watch(self, *a, **k): raise NotImplementedError


class TestGrouping:

    def test_one_row_per_agent_entity_pair(self):
        rows = agent_entity_stats_from_entries([
            _entry("a", "myapp/auth", "k1"),
            _entry("a", "myapp/auth", "k2"),
            _entry("b", "myapp/auth", "k3"),
            _entry("a", "myapp/billing", "k4"),
        ])
        pairs = {(r["agent_id"], r["entity_path"]): r for r in rows}
        assert len(pairs) == 3
        assert pairs[("a", "myapp/auth")]["entry_count"] == 2
        assert pairs[("b", "myapp/auth")]["entry_count"] == 1

    def test_counts_recalls_and_validated_outcomes(self):
        rows = agent_entity_stats_from_entries([
            _entry("a", "myapp/auth", "k1", recalls=3, outcomes=1),
            _entry("a", "myapp/auth", "k2", recalls=4, outcomes=0),
        ])
        assert rows[0]["total_recalls"] == 7
        # Entries *with* an outcome, not the sum of outcome counts: the
        # question is how many memories were validated, not how often.
        assert rows[0]["outcome_linked_count"] == 1

    def test_tracks_first_and_last_write(self):
        rows = agent_entity_stats_from_entries([
            _entry("a", "myapp/auth", "k1", days_ago=10),
            _entry("a", "myapp/auth", "k2", days_ago=2),
        ])
        assert rows[0]["last_written"] == NOW - timedelta(days=2)
        assert rows[0]["first_written"] == NOW - timedelta(days=10)

    def test_averages_confidence(self):
        rows = agent_entity_stats_from_entries([
            _entry("a", "myapp/auth", "k1", confidence=0.6),
            _entry("a", "myapp/auth", "k2", confidence=1.0),
        ])
        assert rows[0]["avg_confidence"] == pytest.approx(0.8)

    def test_empty_input_yields_no_rows(self):
        assert agent_entity_stats_from_entries([]) == []


class TestPrefixScoping:

    def _adapter(self) -> _ListAdapter:
        return _ListAdapter([
            _entry("a", "myapp/auth", "k1"),
            _entry("a", "myapp/auth/tokens", "k2"),
            _entry("a", "myapp/authorisation", "k3"),
            _entry("a", "other/thing", "k4"),
        ])

    def test_scope_includes_descendants(self):
        rows = self._adapter().agent_entity_stats(entity_path="myapp/auth")
        assert {r["entity_path"] for r in rows} == {"myapp/auth", "myapp/auth/tokens"}

    def test_scope_excludes_sibling_with_shared_prefix(self):
        """'myapp/auth' must not swallow 'myapp/authorisation'."""
        rows = self._adapter().agent_entity_stats(entity_path="myapp/auth")
        assert "myapp/authorisation" not in {r["entity_path"] for r in rows}

    def test_trailing_slash_is_tolerated(self):
        rows = self._adapter().agent_entity_stats(entity_path="myapp/auth/")
        assert {r["entity_path"] for r in rows} == {"myapp/auth", "myapp/auth/tokens"}

    def test_unscoped_returns_every_pair(self):
        rows = self._adapter().agent_entity_stats()
        assert len(rows) == 4


class TestAgentFiltering:

    def test_restricts_to_named_writers(self):
        adapter = _ListAdapter([
            _entry("mine", "myapp/auth", "k1"),
            _entry("theirs", "myapp/auth", "k2"),
        ])
        rows = adapter.agent_entity_stats(agent_ids=["mine"])
        assert [r["agent_id"] for r in rows] == ["mine"]

    def test_empty_allow_list_hides_everything(self):
        """A user who owns no agents must not see an unfiltered account."""
        adapter = _ListAdapter([_entry("theirs", "myapp/auth", "k1")])
        assert adapter.agent_entity_stats(agent_ids=[]) == []


class TestSlashedKeys:

    def test_a_key_containing_a_slash_does_not_move_the_entity(self):
        """6.6% of production keys contain a slash; grouping is by column."""
        rows = agent_entity_stats_from_entries([
            _entry("a", "sweep/abc", "active-negotiation/xyz"),
        ])
        assert rows[0]["entity_path"] == "sweep/abc"
