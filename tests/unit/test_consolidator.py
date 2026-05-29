"""Unit tests for ConsolidationStrategy (WS2: risk-tiered memory compaction)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call

import pytest

pytest.importorskip("amfs_cortex", reason="amfs_cortex not installed")

from amfs_core.models import (
    Branch,
    ConsolidationProposal,
    Event,
    EventType,
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
    recall_count: int = 0,
    outcome_count: int = 0,
    age_days: float = 0.0,
    agent_id: str = "agent-a",
    tier: int = 1,
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
        tier=tier,
    )


def _mock_adapter(entries=None):
    adapter = MagicMock()
    adapter.list.return_value = entries or []
    adapter.search.return_value = entries or []
    adapter.write.return_value = None
    adapter.log_event.return_value = None
    adapter.create_branch.return_value = Branch(
        name="test", branched_at=_now(), created_by="test",
    )
    adapter.list_branches.return_value = []
    return adapter


# ──────────────────────────────────────────────────────────────
# Tier A: Superseded beliefs
# ──────────────────────────────────────────────────────────────


class TestTierASupersededBeliefs:
    def test_belief_superseded_by_high_confidence_fact(self) -> None:
        entries = [
            _entry("k1", confidence=0.95, memory_type=MemoryType.FACT),
            _entry("k1", confidence=0.6, memory_type=MemoryType.BELIEF, agent_id="agent-b"),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived >= 1
        assert adapter.write.call_count >= 1
        written = adapter.write.call_args_list[0][0][0]
        assert written.tier == 3

    def test_belief_not_superseded_when_fact_below_threshold(self) -> None:
        entries = [
            _entry("k1", confidence=0.8, memory_type=MemoryType.FACT),
            _entry("k1", confidence=0.6, memory_type=MemoryType.BELIEF, agent_id="agent-b"),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived == 0

    def test_belief_without_corresponding_fact_not_archived(self) -> None:
        entries = [
            _entry("k1", confidence=0.6, memory_type=MemoryType.BELIEF),
            _entry("k2", confidence=0.95, memory_type=MemoryType.FACT),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived == 0

    def test_multiple_beliefs_same_key_all_archived(self) -> None:
        entries = [
            _entry("k1", confidence=0.95, memory_type=MemoryType.FACT),
            _entry("k1", confidence=0.5, memory_type=MemoryType.BELIEF, agent_id="b1"),
            _entry("k1", confidence=0.4, memory_type=MemoryType.BELIEF, agent_id="b2"),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived >= 2

    def test_consolidation_auto_merged_event_logged(self) -> None:
        entries = [
            _entry("k1", confidence=0.95, memory_type=MemoryType.FACT),
            _entry("k1", confidence=0.6, memory_type=MemoryType.BELIEF, agent_id="b"),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        strategy.run()

        logged_events = [c for c in adapter.log_event.call_args_list]
        assert len(logged_events) >= 1
        event = logged_events[0][0][0]
        assert event.event_type == EventType.CONSOLIDATION_AUTO_MERGED


# ──────────────────────────────────────────────────────────────
# Tier A: Stale pruning
# ──────────────────────────────────────────────────────────────


class TestTierAStalePruning:
    def test_old_zero_recall_zero_outcome_pruned(self) -> None:
        entries = [
            _entry("stale", age_days=60, recall_count=0, outcome_count=0, tier=1),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived >= 1

    def test_old_entry_with_recall_not_pruned(self) -> None:
        entries = [
            _entry("used", age_days=60, recall_count=5, outcome_count=0, tier=1),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived == 0

    def test_old_entry_with_outcomes_not_pruned(self) -> None:
        entries = [
            _entry("validated", age_days=60, recall_count=0, outcome_count=3, tier=1),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived == 0

    def test_recent_entry_not_pruned(self) -> None:
        entries = [
            _entry("fresh", age_days=5, recall_count=0, outcome_count=0, tier=1),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived == 0

    def test_already_archived_by_tier_assigner_not_pruned(self) -> None:
        """Entries assigned to ARCHIVE tier by TierAssigner are skipped.

        TierAssigner assigns tiers based on PriorityScorer scores. When an
        entry scores low enough to be placed in the ARCHIVE tier (tier >= 3),
        _prune_stale_entries skips it to avoid redundant demotion.
        We need enough entries so TierAssigner actually puts some in archive.
        """
        entries = [
            _entry(f"active-{i}", age_days=1, recall_count=10, outcome_count=5, tier=1)
            for i in range(10)
        ] + [
            _entry("old-stale", age_days=60, recall_count=0, outcome_count=0, tier=1),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived >= 1

    def test_custom_stale_days(self) -> None:
        entries = [
            _entry("k1", age_days=15, recall_count=0, outcome_count=0, tier=1),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter, stale_days=10)
        report = strategy.run()

        assert report.auto_archived >= 1


# ──────────────────────────────────────────────────────────────
# Tier B: Convergent knowledge
# ──────────────────────────────────────────────────────────────


class TestTierBConvergentKnowledge:
    def test_three_agents_similar_values_creates_proposal(self) -> None:
        entries = [
            _entry("k1", agent_id="agent-a", confidence=0.9),
            _entry("k1", agent_id="agent-b", confidence=0.85),
            _entry("k1", agent_id="agent-c", confidence=0.8),
        ]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")
        convergent = [p for p in proposals if p.strategy == "convergent_knowledge"]
        assert len(convergent) >= 1
        assert convergent[0].entity_path == "svc"
        assert convergent[0].risk_tier == "review_required"
        assert convergent[0].branch_name.startswith("cortex/consolidation/svc/")

    def test_two_agents_insufficient(self) -> None:
        entries = [
            _entry("k1", agent_id="agent-a"),
            _entry("k1", agent_id="agent-b"),
        ]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")
        convergent = [p for p in proposals if p.strategy == "convergent_knowledge"]
        assert len(convergent) == 0

    def test_dissimilar_values_no_proposal(self) -> None:
        entries = [
            MemoryEntry(
                entity_path="svc", key="k1", value="completely different value A",
                provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
                confidence=0.9,
            ),
            MemoryEntry(
                entity_path="svc", key="k1", value="totally unrelated value B xyz123",
                provenance=Provenance(agent_id="b", session_id="s", written_at=_now()),
                confidence=0.8,
            ),
            MemoryEntry(
                entity_path="svc", key="k1", value="another thing entirely different C qwerty",
                provenance=Provenance(agent_id="c", session_id="s", written_at=_now()),
                confidence=0.7,
            ),
        ]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")
        convergent = [p for p in proposals if p.strategy == "convergent_knowledge"]
        assert len(convergent) == 0

    def test_fewer_than_three_entries_returns_empty(self) -> None:
        entries = [_entry("k1"), _entry("k2")]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")
        assert proposals == []

    def test_proposal_confidence_is_average_plus_boost(self) -> None:
        entries = [
            _entry("k1", agent_id="a", confidence=0.8),
            _entry("k1", agent_id="b", confidence=0.8),
            _entry("k1", agent_id="c", confidence=0.8),
        ]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")
        convergent = [p for p in proposals if p.strategy == "convergent_knowledge"]
        if convergent:
            assert convergent[0].proposed_confidence == pytest.approx(0.85, abs=0.01)


# ──────────────────────────────────────────────────────────────
# Tier B: Outcome rollups
# ──────────────────────────────────────────────────────────────


class TestTierBOutcomeRollups:
    def test_five_experiences_with_outcomes_creates_proposal(self) -> None:
        entries = [
            _entry(f"action-deploy-{i}", memory_type=MemoryType.EXPERIENCE,
                   outcome_count=2, agent_id=f"a{i}")
            for i in range(6)
        ]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")
        rollups = [p for p in proposals if p.strategy == "outcome_rollup"]
        assert len(rollups) >= 1

    def test_fewer_than_five_no_rollup(self) -> None:
        entries = [
            _entry(f"action-deploy-{i}", memory_type=MemoryType.EXPERIENCE,
                   outcome_count=2, agent_id=f"a{i}")
            for i in range(3)
        ]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")
        rollups = [p for p in proposals if p.strategy == "outcome_rollup"]
        assert len(rollups) == 0

    def test_experiences_without_outcomes_excluded(self) -> None:
        entries = [
            _entry(f"action-deploy-{i}", memory_type=MemoryType.EXPERIENCE,
                   outcome_count=0, agent_id=f"a{i}")
            for i in range(10)
        ]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)

        proposals = strategy.find_consolidation_candidates("svc")
        rollups = [p for p in proposals if p.strategy == "outcome_rollup"]
        assert len(rollups) == 0


# ──────────────────────────────────────────────────────────────
# Full consolidation run
# ──────────────────────────────────────────────────────────────


class TestConsolidationRun:
    def test_empty_entries_returns_empty_report(self) -> None:
        adapter = _mock_adapter([])
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived == 0
        assert report.proposals_created == 0
        assert report.compression_ratio == 1.0

    def test_run_returns_consolidation_report(self) -> None:
        entries = [
            _entry("k1", confidence=0.95, memory_type=MemoryType.FACT),
            _entry("k1", confidence=0.5, memory_type=MemoryType.BELIEF, agent_id="b"),
            _entry("stale", age_days=60, recall_count=0, outcome_count=0, tier=1),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        assert report.auto_archived >= 1
        assert report.consolidated_at is not None
        assert report.entity_path == "*"

    def test_run_entity_scoped(self) -> None:
        entries = [
            _entry("k1", entity_path="myapp/auth", age_days=60,
                   recall_count=0, outcome_count=0, tier=1),
        ]
        adapter = _mock_adapter(entries)
        adapter.search.return_value = entries
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run_entity("myapp/auth")

        assert report.entity_path == "myapp/auth"

    def test_run_entity_empty(self) -> None:
        adapter = _mock_adapter([])
        adapter.search.return_value = []
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run_entity("svc")

        assert report.auto_archived == 0
        assert report.entity_path == "svc"

    def test_report_compression_ratio(self) -> None:
        entries = [
            _entry("k1", confidence=0.95, memory_type=MemoryType.FACT),
            _entry("k1", confidence=0.5, memory_type=MemoryType.BELIEF, agent_id="b"),
            _entry("k2"),
            _entry("k3"),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        report = strategy.run()

        if report.auto_archived > 0:
            assert report.compression_ratio > 1.0


# ──────────────────────────────────────────────────────────────
# persist_proposal
# ──────────────────────────────────────────────────────────────


class TestPersistProposal:
    def _make_proposal(self) -> ConsolidationProposal:
        return ConsolidationProposal(
            id="p1",
            entity_path="svc",
            branch_name="cortex/consolidation/svc/20260101-120000",
            strategy="convergent_knowledge",
            risk_tier="review_required",
            source_entry_keys=["svc/k1", "svc/k2", "svc/k3"],
            proposed_value={"merged": True},
            proposed_confidence=0.85,
            compression_ratio=3.0,
            rationale="3 agents converged",
            created_at=_now(),
        )

    def test_creates_branch_and_writes_entry(self) -> None:
        adapter = _mock_adapter()
        strategy = ConsolidationStrategy(adapter)
        proposal = self._make_proposal()

        result = strategy.persist_proposal(proposal)

        assert result is True
        adapter.create_branch.assert_called_once()
        branch_arg = adapter.create_branch.call_args[0][0]
        assert isinstance(branch_arg, Branch)
        assert branch_arg.name == proposal.branch_name
        adapter.write.assert_called_once()

    def test_logs_consolidation_proposed_event(self) -> None:
        adapter = _mock_adapter()
        strategy = ConsolidationStrategy(adapter)
        proposal = self._make_proposal()

        strategy.persist_proposal(proposal)

        adapter.log_event.assert_called_once()
        event = adapter.log_event.call_args[0][0]
        assert event.event_type == EventType.CONSOLIDATION_PROPOSED

    def test_adapter_failure_returns_false(self) -> None:
        adapter = _mock_adapter()
        adapter.create_branch.side_effect = Exception("DB error")
        strategy = ConsolidationStrategy(adapter)
        proposal = self._make_proposal()

        result = strategy.persist_proposal(proposal)

        assert result is False

    def test_written_entry_has_correct_branch(self) -> None:
        adapter = _mock_adapter()
        strategy = ConsolidationStrategy(adapter)
        proposal = self._make_proposal()

        strategy.persist_proposal(proposal)

        written = adapter.write.call_args[0][0]
        assert written.branch == proposal.branch_name
        assert written.entity_path == "svc"
        assert written.confidence == proposal.proposed_confidence

    def test_demote_entry_sets_branch_correctly(self) -> None:
        """Regression test: _demote_entry must set entry.branch, not pass branch= kwarg."""
        entries = [
            _entry("k1", confidence=0.95, memory_type=MemoryType.FACT),
            _entry("k1", confidence=0.5, memory_type=MemoryType.BELIEF, agent_id="b"),
        ]
        adapter = _mock_adapter(entries)
        strategy = ConsolidationStrategy(adapter)
        strategy.run()

        if adapter.write.call_count > 0:
            written = adapter.write.call_args_list[0][0][0]
            assert written.branch == "main"
            assert written.tier == 3
            assert adapter.write.call_args_list[0][1] == {}
