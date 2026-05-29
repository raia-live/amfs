"""ConsolidationStrategy — risk-tiered memory compaction.

Implements safe, staged memory consolidation with two risk tiers:

Tier A (auto-safe): Non-semantic changes applied directly to main.
  - Superseded beliefs: belief superseded by a fact on the same key.
  - Stale pruning: zero-recall, zero-outcome entries below archive threshold.

Tier B (proposal-required): Semantic merges that create branch-based
proposals for human or agent review.
  - Convergent knowledge: 3+ agents wrote converging values (similarity > 0.9).
  - Outcome-validated rollup: 5+ experience entries all with outcomes.

Tier B is implemented in amfs-internal's MemoryDistiller to keep
semantic-risk operations in Pro.  This module provides Tier A and the
shared infrastructure for both tiers.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from amfs_core.models import (
    Branch,
    ConsolidationProposal,
    ConsolidationReport,
    Event,
    EventType,
    MemoryEntry,
    MemoryType,
    Provenance,
    SearchQuery,
)
from amfs_core.tiering import PriorityScorer, TierAssigner

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter

logger = logging.getLogger(__name__)

_SYSTEM_AGENT = "cortex/consolidator"
_STALE_DAYS = 30
_STALE_RECALL_THRESHOLD = 0


class ConsolidationStrategy:
    """Tier A consolidation: auto-safe operations on main.

    These rules only *demote* or *archive* existing entries; they never
    create new merged content.  No semantic risk.
    """

    def __init__(
        self,
        adapter: PostgresAdapter,
        namespace: str = "default",
        stale_days: int = _STALE_DAYS,
    ) -> None:
        self._adapter = adapter
        self._namespace = namespace
        self._stale_days = stale_days
        self._scorer = PriorityScorer()
        self._assigner = TierAssigner()

    def run(self, *, branch: str = "main") -> ConsolidationReport:
        """Execute Tier A consolidation and detect Tier B proposals.

        Tier A: auto-archives superseded beliefs and stale entries.
        Tier B: detects candidates and persists them as branch-based proposals.
        """
        entries = self._adapter.list(branch=branch)
        if not entries:
            return self._empty_report()

        auto_archived = 0
        auto_archived += self._archive_superseded_beliefs(entries, branch)
        auto_archived += self._prune_stale_entries(entries, branch)

        entity_paths = {e.entity_path for e in entries}
        proposals_created = 0
        for ep in list(entity_paths)[:50]:
            try:
                candidates = self.find_consolidation_candidates(ep, branch=branch)
                for proposal in candidates:
                    if self.persist_proposal(proposal, source_branch=branch):
                        proposals_created += 1
            except Exception:
                logger.debug("Tier B detection failed for %s", ep, exc_info=True)

        return ConsolidationReport(
            entity_path="*",
            auto_archived=auto_archived,
            proposals_created=proposals_created,
            proposals_auto_merged=0,
            compression_ratio=self._compute_compression(len(entries), auto_archived),
            consolidated_at=datetime.now(timezone.utc),
        )

    def run_entity(
        self, entity_path: str, *, branch: str = "main",
    ) -> ConsolidationReport:
        """Execute Tier A consolidation and detect Tier B proposals for a single entity."""
        entries = self._adapter.search(
            SearchQuery(entity_path=entity_path, limit=1000),
            branch=branch,
        )
        if not entries:
            return self._empty_report(entity_path)

        auto_archived = 0
        auto_archived += self._archive_superseded_beliefs(entries, branch)
        auto_archived += self._prune_stale_entries(entries, branch)

        proposals_created = 0
        try:
            candidates = self.find_consolidation_candidates(entity_path, branch=branch)
            for proposal in candidates:
                if self.persist_proposal(proposal, source_branch=branch):
                    proposals_created += 1
        except Exception:
            logger.debug("Tier B detection failed for %s", entity_path, exc_info=True)

        return ConsolidationReport(
            entity_path=entity_path,
            auto_archived=auto_archived,
            proposals_created=proposals_created,
            proposals_auto_merged=0,
            compression_ratio=self._compute_compression(len(entries), auto_archived),
            consolidated_at=datetime.now(timezone.utc),
        )

    def find_consolidation_candidates(
        self, entity_path: str, *, branch: str = "main",
    ) -> list[ConsolidationProposal]:
        """Identify Tier B candidates (convergent knowledge, outcome rollup).

        Returns proposals but does NOT create branches — the caller
        (Pro distiller or HTTP endpoint) decides whether to persist them.
        """
        entries = self._adapter.search(
            SearchQuery(entity_path=entity_path, limit=1000, sort_by="confidence"),
            branch=branch,
        )
        if len(entries) < 3:
            return []

        proposals: list[ConsolidationProposal] = []
        proposals.extend(self._find_convergent_knowledge(entity_path, entries))
        proposals.extend(self._find_outcome_rollups(entity_path, entries))

        return proposals

    def persist_proposal(
        self, proposal: ConsolidationProposal, *, source_branch: str = "main",
    ) -> bool:
        """Create a branch for a Tier B proposal and write the proposed entry.

        Returns True if the branch and entry were created successfully.
        The branch follows the naming convention ``cortex/consolidation/{entity}/{timestamp}``
        so BriefingService and the proposals UI can discover it.
        """
        try:
            branch_obj = Branch(
                namespace=self._namespace,
                name=proposal.branch_name,
                parent_branch=source_branch,
                branched_at=datetime.now(timezone.utc),
                created_by=_SYSTEM_AGENT,
                description=proposal.rationale,
            )
            self._adapter.create_branch(branch_obj)

            consolidated_entry = MemoryEntry(
                entity_path=proposal.entity_path,
                key=f"consolidated-{proposal.strategy}-{proposal.id[:8]}",
                value=proposal.proposed_value,
                confidence=proposal.proposed_confidence,
                version=1,
                memory_type=MemoryType.FACT,
                branch=proposal.branch_name,
                provenance=Provenance(
                    agent_id=_SYSTEM_AGENT,
                    session_id=f"consolidation-{proposal.id}",
                    written_at=datetime.now(timezone.utc),
                    pattern_refs=proposal.source_entry_keys[:10],
                ),
            )
            self._adapter.write(consolidated_entry)

            self._adapter.log_event(Event(
                namespace=self._namespace,
                agent_id=_SYSTEM_AGENT,
                branch=proposal.branch_name,
                event_type=EventType.CONSOLIDATION_PROPOSED,
                summary=f"Tier B proposal for {proposal.entity_path}: {proposal.strategy}",
                details={
                    "proposal_id": proposal.id,
                    "entity_path": proposal.entity_path,
                    "strategy": proposal.strategy,
                    "source_entry_count": len(proposal.source_entry_keys),
                    "proposed_confidence": proposal.proposed_confidence,
                },
            ))

            return True
        except Exception:
            logger.warning(
                "Failed to persist consolidation proposal %s for %s",
                proposal.id, proposal.entity_path, exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Tier A: Superseded beliefs
    # ------------------------------------------------------------------

    def _archive_superseded_beliefs(
        self, entries: list[MemoryEntry], branch: str,
    ) -> int:
        """Archive beliefs that have been superseded by high-confidence facts.

        A belief is superseded when a fact exists on the same entity/key
        with confidence >= 0.9.
        """
        by_key: dict[tuple[str, str], list[MemoryEntry]] = {}
        for e in entries:
            by_key.setdefault((e.entity_path, e.key), []).append(e)

        archived = 0
        for (ep, key), group in by_key.items():
            facts = [
                e for e in group
                if e.memory_type == MemoryType.FACT and e.confidence >= 0.9
            ]
            beliefs = [
                e for e in group
                if e.memory_type == MemoryType.BELIEF
            ]
            if not facts or not beliefs:
                continue

            for belief in beliefs:
                try:
                    self._demote_entry(belief, branch, reason="superseded_by_fact")
                    archived += 1
                except Exception:
                    logger.debug(
                        "Failed to archive superseded belief %s/%s",
                        ep, key, exc_info=True,
                    )

        return archived

    # ------------------------------------------------------------------
    # Tier A: Stale pruning
    # ------------------------------------------------------------------

    def _prune_stale_entries(
        self, entries: list[MemoryEntry], branch: str,
    ) -> int:
        """Demote tier for entries with zero recall, zero outcomes, and old age.

        Does not delete — only adjusts tier so they're deprioritised in retrieval.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._stale_days)
        scores = self._scorer.score_batch(entries)
        tiers = self._assigner.assign(entries, scores)

        archived = 0
        for entry in entries:
            ek = entry.entry_key
            current_tier = tiers.get(ek, 1)
            if current_tier >= 3:
                continue

            is_stale = (
                entry.provenance.written_at < cutoff
                and entry.recall_count <= _STALE_RECALL_THRESHOLD
                and entry.outcome_count == 0
            )
            if is_stale:
                try:
                    self._demote_entry(entry, branch, reason="stale_pruning")
                    archived += 1
                except Exception:
                    logger.debug(
                        "Failed to demote stale entry %s",
                        ek, exc_info=True,
                    )

        return archived

    # ------------------------------------------------------------------
    # Tier B candidate detection (proposal generation, not execution)
    # ------------------------------------------------------------------

    def _find_convergent_knowledge(
        self,
        entity_path: str,
        entries: list[MemoryEntry],
    ) -> list[ConsolidationProposal]:
        """Find keys where 3+ agents wrote converging values (similarity > 0.9)."""
        by_key: dict[str, list[MemoryEntry]] = {}
        for e in entries:
            by_key.setdefault(e.key, []).append(e)

        proposals: list[ConsolidationProposal] = []
        for key, group in by_key.items():
            agents = {e.provenance.agent_id for e in group}
            if len(agents) < 3:
                continue

            representative = max(group, key=lambda e: e.confidence)
            similar_count = 0
            for e in group:
                if e is representative:
                    continue
                sim = self._value_similarity(representative.value, e.value)
                if sim >= 0.9:
                    similar_count += 1

            if similar_count < 2:
                continue

            avg_confidence = sum(e.confidence for e in group) / len(group)
            proposals.append(ConsolidationProposal(
                id=str(uuid.uuid4()),
                entity_path=entity_path,
                branch_name=f"cortex/consolidation/{entity_path}/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                strategy="convergent_knowledge",
                risk_tier="review_required",
                source_entry_keys=[e.entry_key for e in group],
                proposed_value=representative.value,
                proposed_confidence=round(min(avg_confidence + 0.05, 1.0), 3),
                compression_ratio=round(len(group) / 1.0, 1),
                rationale=(
                    f"{len(agents)} agents wrote converging values for '{key}' "
                    f"(similarity > 0.9). Proposing merge to single authoritative entry."
                ),
                created_at=datetime.now(timezone.utc),
            ))

        return proposals

    def _find_outcome_rollups(
        self,
        entity_path: str,
        entries: list[MemoryEntry],
    ) -> list[ConsolidationProposal]:
        """Find experience entries with outcomes that could be rolled up."""
        experiences = [
            e for e in entries
            if e.memory_type == MemoryType.EXPERIENCE and e.outcome_count > 0
        ]
        if len(experiences) < 5:
            return []

        by_key_prefix: dict[str, list[MemoryEntry]] = {}
        for e in experiences:
            prefix = e.key.rsplit("-", 1)[0] if "-" in e.key else e.key
            by_key_prefix.setdefault(prefix, []).append(e)

        proposals: list[ConsolidationProposal] = []
        for prefix, group in by_key_prefix.items():
            if len(group) < 5:
                continue

            avg_conf = sum(e.confidence for e in group) / len(group)
            total_outcomes = sum(e.outcome_count for e in group)

            proposals.append(ConsolidationProposal(
                id=str(uuid.uuid4()),
                entity_path=entity_path,
                branch_name=f"cortex/consolidation/{entity_path}/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                strategy="outcome_rollup",
                risk_tier="review_required",
                source_entry_keys=[e.entry_key for e in group],
                proposed_value={
                    "type": "outcome_validated_summary",
                    "source_count": len(group),
                    "total_outcomes": total_outcomes,
                    "key_prefix": prefix,
                },
                proposed_confidence=round(min(avg_conf + 0.1, 0.95), 3),
                compression_ratio=round(len(group) / 1.0, 1),
                rationale=(
                    f"{len(group)} outcome-validated experience entries with prefix '{prefix}' "
                    f"({total_outcomes} total outcomes). Proposing summary rollup."
                ),
                created_at=datetime.now(timezone.utc),
            ))

        return proposals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _demote_entry(
        self, entry: MemoryEntry, branch: str, *, reason: str,
    ) -> None:
        """Demote an entry by setting tier to ARCHIVE and logging the event."""
        entry.tier = 3
        entry.branch = branch
        self._adapter.write(entry)
        try:
            self._adapter.log_event(Event(
                namespace=self._namespace,
                agent_id=_SYSTEM_AGENT,
                branch=branch,
                event_type=EventType.CONSOLIDATION_AUTO_MERGED,
                summary=f"Auto-archived {entry.entity_path}/{entry.key}: {reason}",
                details={
                    "entity_path": entry.entity_path,
                    "key": entry.key,
                    "reason": reason,
                    "previous_confidence": entry.confidence,
                },
            ))
        except Exception:
            logger.debug("Failed to log consolidation event", exc_info=True)

    @staticmethod
    def _value_similarity(a: Any, b: Any) -> float:
        """Compute string similarity between two values."""
        sa, sb = str(a), str(b)
        if sa == sb:
            return 1.0
        return SequenceMatcher(None, sa, sb).ratio()

    @staticmethod
    def _compute_compression(total: int, archived: int) -> float:
        if total == 0:
            return 1.0
        remaining = total - archived
        return round(total / max(remaining, 1), 2)

    def _empty_report(self, entity_path: str = "*") -> ConsolidationReport:
        return ConsolidationReport(
            entity_path=entity_path,
            auto_archived=0,
            proposals_created=0,
            proposals_auto_merged=0,
            compression_ratio=1.0,
            consolidated_at=datetime.now(timezone.utc),
        )
