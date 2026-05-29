"""Unit tests for consolidation HTTP endpoint logic (WS2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("amfs_cortex", reason="amfs_cortex not installed")

from amfs_core.models import (
    ConsolidationReport,
    MemoryEntry,
    MemoryType,
    Provenance,
    SearchQuery,
)
from amfs_cortex.consolidator import ConsolidationStrategy


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _entry(
    key: str,
    *,
    entity_path: str = "svc",
    confidence: float = 1.0,
    memory_type: MemoryType = MemoryType.FACT,
    recall_count: int = 5,
    outcome_count: int = 1,
    age_days: float = 0.0,
    agent_id: str = "a",
) -> MemoryEntry:
    written_at = _now() - timedelta(days=age_days)
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=f"value-{key}",
        provenance=Provenance(agent_id=agent_id, session_id="s", written_at=written_at),
        confidence=confidence,
        memory_type=memory_type,
        recall_count=recall_count,
        outcome_count=outcome_count,
    )


class TestConsolidationStrategyIntegration:
    """Integration-style tests that verify the strategy produces correct report types."""

    def test_run_returns_consolidation_report(self) -> None:
        adapter = MagicMock()
        adapter.list.return_value = [_entry("k1"), _entry("k2")]
        adapter.list_branches.return_value = []
        strategy = ConsolidationStrategy(adapter)

        report = strategy.run()

        assert isinstance(report, ConsolidationReport)
        assert report.entity_path == "*"
        assert isinstance(report.auto_archived, int)
        assert isinstance(report.proposals_created, int)
        assert report.consolidated_at is not None

    def test_run_entity_returns_scoped_report(self) -> None:
        entries = [_entry("k1", entity_path="myapp/auth")]
        adapter = MagicMock()
        adapter.search.return_value = entries
        adapter.list_branches.return_value = []
        strategy = ConsolidationStrategy(adapter)

        report = strategy.run_entity("myapp/auth")

        assert report.entity_path == "myapp/auth"

    def test_find_candidates_with_real_entries(self) -> None:
        entries = [
            _entry("k1", agent_id="a1", confidence=0.9),
            _entry("k1", agent_id="a2", confidence=0.85),
            _entry("k1", agent_id="a3", confidence=0.8),
        ]
        adapter = MagicMock()
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")

        convergent = [p for p in proposals if p.strategy == "convergent_knowledge"]
        assert len(convergent) >= 1
        assert all(p.entity_path == "svc" for p in proposals)

    def test_report_serialization(self) -> None:
        adapter = MagicMock()
        adapter.list.return_value = []
        strategy = ConsolidationStrategy(adapter)

        report = strategy.run()
        data = report.model_dump(mode="json")

        assert "auto_archived" in data
        assert "proposals_created" in data
        assert "compression_ratio" in data
        assert "consolidated_at" in data


class TestConsolidationStatusMetrics:
    """Test the metrics that the /consolidation/status endpoint would return."""

    def test_count_auto_archived_from_activity_log(self) -> None:
        activity_log = [
            {"type": "consolidation_run", "auto_archived": 5, "timestamp": "2026-01-01"},
            {"type": "consolidation_run", "auto_archived": 3, "timestamp": "2026-01-02"},
            {"type": "consolidation_error", "timestamp": "2026-01-03"},
        ]

        total = sum(
            entry.get("auto_archived", 0)
            for entry in activity_log
            if entry.get("type") == "consolidation_run"
        )

        assert total == 8

    def test_count_pending_consolidation_branches(self) -> None:
        branches = [
            MagicMock(name="cortex/consolidation/svc/20260101"),
            MagicMock(name="cortex/consolidation/auth/20260102"),
            MagicMock(name="feature/unrelated"),
            MagicMock(name="cortex/consolidation/api/20260103"),
        ]
        for b in branches:
            b.name = b._mock_name

        pending = [b for b in branches if b.name.startswith("cortex/consolidation/")]

        assert len(pending) == 3

    def test_entities_ready_threshold(self) -> None:
        entries = [
            _entry(f"k{i}", entity_path="big-entity") for i in range(15)
        ] + [
            _entry(f"k{i}", entity_path="small-entity") for i in range(3)
        ]

        entity_counts: dict[str, int] = {}
        for e in entries:
            entity_counts[e.entity_path] = entity_counts.get(e.entity_path, 0) + 1

        ready = [ep for ep, count in entity_counts.items() if count >= 10]

        assert "big-entity" in ready
        assert "small-entity" not in ready
