"""Unit tests for new continual-learning models (WS2 + WS4)."""

from __future__ import annotations

from datetime import datetime, timezone

from amfs_core.models import (
    ConsolidationProposal,
    ConsolidationReport,
    DigestType,
    EventType,
    SearchQuery,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestConsolidationProposal:
    def test_creation_all_fields(self) -> None:
        p = ConsolidationProposal(
            id="p1",
            entity_path="myapp/auth",
            branch_name="cortex/consolidation/myapp/auth/20260101-120000",
            strategy="convergent_knowledge",
            risk_tier="review_required",
            source_entry_keys=["myapp/auth/k1", "myapp/auth/k2", "myapp/auth/k3"],
            proposed_value={"merged": True},
            proposed_confidence=0.85,
            compression_ratio=3.0,
            rationale="3 agents converged on same value",
            created_at=_now(),
        )
        assert p.id == "p1"
        assert p.entity_path == "myapp/auth"
        assert p.strategy == "convergent_knowledge"
        assert len(p.source_entry_keys) == 3

    def test_default_status_pending(self) -> None:
        p = ConsolidationProposal(
            id="p2",
            entity_path="svc",
            branch_name="cortex/consolidation/svc/ts",
            strategy="outcome_rollup",
            risk_tier="review_required",
            source_entry_keys=["svc/k1"],
            proposed_value="summary",
            proposed_confidence=0.9,
            compression_ratio=5.0,
            rationale="rollup",
            created_at=_now(),
        )
        assert p.status == "pending"
        assert p.reviewed_by is None
        assert p.reviewed_at is None

    def test_serialization_roundtrip(self) -> None:
        now = _now()
        p = ConsolidationProposal(
            id="p3",
            entity_path="svc",
            branch_name="cortex/consolidation/svc/ts",
            strategy="convergent_knowledge",
            risk_tier="review_required",
            source_entry_keys=["svc/a", "svc/b"],
            proposed_value={"nested": [1, 2]},
            proposed_confidence=0.75,
            compression_ratio=2.0,
            rationale="test",
            status="approved",
            created_at=now,
            reviewed_by="admin",
            reviewed_at=now,
        )
        data = p.model_dump(mode="json")
        restored = ConsolidationProposal.model_validate(data)
        assert restored.id == p.id
        assert restored.proposed_value == {"nested": [1, 2]}
        assert restored.status == "approved"
        assert restored.reviewed_by == "admin"


class TestConsolidationReport:
    def test_creation(self) -> None:
        r = ConsolidationReport(
            entity_path="myapp/auth",
            auto_archived=5,
            proposals_created=2,
            proposals_auto_merged=0,
            compression_ratio=1.5,
            consolidated_at=_now(),
        )
        assert r.auto_archived == 5
        assert r.proposals_created == 2

    def test_serialization_roundtrip(self) -> None:
        r = ConsolidationReport(
            entity_path="*",
            auto_archived=10,
            proposals_created=3,
            proposals_auto_merged=1,
            compression_ratio=2.0,
            consolidated_at=_now(),
        )
        data = r.model_dump(mode="json")
        restored = ConsolidationReport.model_validate(data)
        assert restored.auto_archived == 10
        assert restored.compression_ratio == 2.0
        assert restored.entity_path == "*"


class TestDigestTypeTracePattern:
    def test_trace_pattern_exists(self) -> None:
        assert DigestType.TRACE_PATTERN == "trace_pattern"

    def test_trace_pattern_in_values(self) -> None:
        assert "trace_pattern" in [dt.value for dt in DigestType]


class TestConsolidationEventTypes:
    def test_consolidation_proposed(self) -> None:
        assert EventType.CONSOLIDATION_PROPOSED == "consolidation_proposed"

    def test_consolidation_approved(self) -> None:
        assert EventType.CONSOLIDATION_APPROVED == "consolidation_approved"

    def test_consolidation_rejected(self) -> None:
        assert EventType.CONSOLIDATION_REJECTED == "consolidation_rejected"

    def test_consolidation_auto_merged(self) -> None:
        assert EventType.CONSOLIDATION_AUTO_MERGED == "consolidation_auto_merged"

    def test_all_four_distinct(self) -> None:
        vals = {
            EventType.CONSOLIDATION_PROPOSED,
            EventType.CONSOLIDATION_APPROVED,
            EventType.CONSOLIDATION_REJECTED,
            EventType.CONSOLIDATION_AUTO_MERGED,
        }
        assert len(vals) == 4


class TestSearchQueryPriority:
    def test_default_sort_by_confidence(self) -> None:
        sq = SearchQuery()
        assert sq.sort_by == "confidence"

    def test_priority_sort_option(self) -> None:
        sq = SearchQuery(sort_by="priority")
        assert sq.sort_by == "priority"

    def test_roundtrip(self) -> None:
        sq = SearchQuery(entity_path="svc", sort_by="priority", limit=5)
        data = sq.model_dump()
        restored = SearchQuery(**data)
        assert restored.sort_by == "priority"
        assert restored.limit == 5
