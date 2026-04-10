"""PatternDetector — scans memory for collaboration-aware patterns.

Detects eight pattern types aligned with the "GitHub for agent memory" model:

Tier 1 — Collaboration Health:
  - knowledge_conflict: Different agents wrote different values for the same key
  - stale_knowledge: Old entries still being read but never updated
  - orphaned_branch: Branches with no activity that haven't been merged
  - redundant_writes: Multiple agents storing similar information

Tier 2 — Collaboration Insights:
  - single_point_of_knowledge: Only one agent knows about a critical entity
  - passive_consumer: An agent reads knowledge it never contributes to
  - unreviewed_changes: Open PRs that have been waiting too long for review

Legacy (kept):
  - recurring_failure: Entries repeatedly in incident causal chains
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field

from amfs_core.models import (
    Branch,
    BranchStatus,
    MemoryEntry,
    OutcomeRecord,
    PullRequest,
    PullRequestStatus,
)

PATTERN_CATEGORIES = {
    "collaboration": {
        "label": "Collaboration Health",
        "description": "Issues affecting how agents work together on shared knowledge",
        "types": [
            "knowledge_conflict",
            "stale_knowledge",
            "orphaned_branch",
            "redundant_writes",
        ],
    },
    "insights": {
        "label": "Collaboration Insights",
        "description": "Opportunities to improve agent teamwork",
        "types": [
            "single_point_of_knowledge",
            "passive_consumer",
            "unreviewed_changes",
        ],
    },
    "reliability": {
        "label": "Reliability",
        "description": "Recurring problems that need attention",
        "types": ["recurring_failure"],
    },
}

PATTERN_METADATA: dict[str, dict[str, str]] = {
    "knowledge_conflict": {
        "label": "Knowledge Conflict",
        "icon": "git-merge",
        "hint": "Review conflicting entries and merge or overwrite with the correct value.",
    },
    "stale_knowledge": {
        "label": "Stale Knowledge",
        "icon": "clock",
        "hint": "Update or archive these entries — other agents may be relying on outdated information.",
    },
    "orphaned_branch": {
        "label": "Orphaned Branch",
        "icon": "git-branch",
        "hint": "Merge or close this branch to keep the memory workspace clean.",
    },
    "redundant_writes": {
        "label": "Redundant Writes",
        "icon": "copy",
        "hint": "Consolidate into a single authoritative entry to avoid confusion.",
    },
    "single_point_of_knowledge": {
        "label": "Single Point of Knowledge",
        "icon": "user",
        "hint": "Have another agent review or replicate this knowledge for resilience.",
    },
    "passive_consumer": {
        "label": "Passive Consumer",
        "icon": "eye",
        "hint": "This agent depends on knowledge it never contributes to — consider sharing back.",
    },
    "unreviewed_changes": {
        "label": "Unreviewed Changes",
        "icon": "git-pull-request",
        "hint": "Review and merge or close this PR to unblock knowledge flow.",
    },
    "recurring_failure": {
        "label": "Recurring Failure",
        "icon": "alert-triangle",
        "hint": "Investigate the linked entries — they keep appearing in incident chains.",
    },
}

ALL_PATTERN_TYPES = list(PATTERN_METADATA.keys())


class DetectedPattern(BaseModel):
    """A single detected pattern with metadata."""

    pattern_type: str
    severity: str  # "info", "warning", "critical"
    category: str  # "collaboration", "insights", "reliability"
    entity_path: str
    description: str
    details: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PatternReport(BaseModel):
    """Aggregated report from a full pattern detection scan."""

    patterns: list[DetectedPattern] = Field(default_factory=list)
    scanned_entries: int = 0
    scanned_outcomes: int = 0
    scan_duration_ms: float = 0.0
    scanned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def by_type(self) -> dict[str, list[DetectedPattern]]:
        result: dict[str, list[DetectedPattern]] = defaultdict(list)
        for p in self.patterns:
            result[p.pattern_type].append(p)
        return dict(result)

    @property
    def by_severity(self) -> dict[str, list[DetectedPattern]]:
        result: dict[str, list[DetectedPattern]] = defaultdict(list)
        for p in self.patterns:
            result[p.severity].append(p)
        return dict(result)

    @property
    def by_category(self) -> dict[str, list[DetectedPattern]]:
        result: dict[str, list[DetectedPattern]] = defaultdict(list)
        for p in self.patterns:
            result[p.category].append(p)
        return dict(result)


class PatternDetector:
    """Scans memory entries, branches, and PRs for collaboration patterns.

    Parameters
    ----------
    stale_days:
        Entries older than this with active readers are flagged as stale.
    orphan_days:
        Branches inactive for this many days are flagged as orphaned.
    pr_stale_days:
        Open PRs older than this are flagged as unreviewed.
    similarity_threshold:
        Minimum string similarity (0–1) to flag redundant writes.
    incident_threshold:
        Minimum incident appearances for recurring failure detection.
    """

    def __init__(
        self,
        *,
        stale_days: int = 14,
        orphan_days: int = 7,
        pr_stale_days: int = 3,
        similarity_threshold: float = 0.75,
        incident_threshold: int = 2,
    ) -> None:
        self.stale_days = stale_days
        self.orphan_days = orphan_days
        self.pr_stale_days = pr_stale_days
        self.similarity_threshold = similarity_threshold
        self.incident_threshold = incident_threshold

    def analyze(
        self,
        entries: list[MemoryEntry],
        *,
        outcome_data: list[OutcomeRecord] | None = None,
        branches: list[Branch] | None = None,
        pull_requests: list[PullRequest] | None = None,
    ) -> PatternReport:
        """Run all detectors and return a consolidated report."""
        start = datetime.now(timezone.utc)
        outcomes = outcome_data or []

        patterns: list[DetectedPattern] = []

        # Tier 1: Collaboration Health
        patterns.extend(self._detect_knowledge_conflicts(entries))
        patterns.extend(self._detect_stale_knowledge(entries))
        patterns.extend(self._detect_orphaned_branches(branches or []))
        patterns.extend(self._detect_redundant_writes(entries))

        # Tier 2: Collaboration Insights
        patterns.extend(self._detect_single_point_of_knowledge(entries))
        patterns.extend(self._detect_passive_consumers(entries))
        patterns.extend(self._detect_unreviewed_changes(pull_requests or []))

        # Legacy: Reliability
        patterns.extend(self._detect_recurring_failures(entries, outcomes))

        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return PatternReport(
            patterns=patterns,
            scanned_entries=len(entries),
            scanned_outcomes=len(outcomes),
            scan_duration_ms=elapsed,
        )

    # ── Tier 1: Collaboration Health ──────────────────────────────────

    def _detect_knowledge_conflicts(
        self, entries: list[MemoryEntry]
    ) -> list[DetectedPattern]:
        """Find keys where different agents wrote different values."""
        key_agents: dict[tuple[str, str], dict[str, MemoryEntry]] = defaultdict(dict)
        for e in entries:
            compound = (e.entity_path, e.key)
            key_agents[compound][e.provenance.agent_id] = e

        patterns: list[DetectedPattern] = []
        for (ep, key), agent_entries in key_agents.items():
            if len(agent_entries) < 2:
                continue

            values = {}
            for aid, entry in agent_entries.items():
                val_str = str(entry.value) if entry.value is not None else ""
                values[aid] = val_str

            distinct_values = set(values.values())
            if len(distinct_values) < 2:
                continue

            agents_list = list(agent_entries.keys())
            severity = "critical" if len(agents_list) > 2 else "warning"

            patterns.append(DetectedPattern(
                pattern_type="knowledge_conflict",
                severity=severity,
                category="collaboration",
                entity_path=ep,
                description=(
                    f"{len(agents_list)} agents wrote different values for '{key}' "
                    f"in {ep}"
                ),
                details={
                    "key": key,
                    "agents": agents_list,
                    "value_count": len(distinct_values),
                    "latest_writes": {
                        aid: e.provenance.written_at.isoformat()
                        for aid, e in agent_entries.items()
                    },
                },
            ))

        return patterns

    def _detect_stale_knowledge(
        self, entries: list[MemoryEntry]
    ) -> list[DetectedPattern]:
        """Find old entries that are still being read by other agents."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.stale_days)
        patterns: list[DetectedPattern] = []

        entity_stale: dict[str, list[MemoryEntry]] = defaultdict(list)
        for e in entries:
            is_old = e.provenance.written_at < cutoff
            is_read = (e.recall_count or 0) > 0
            if is_old and is_read:
                entity_stale[e.entity_path].append(e)
            elif is_old and e.outcome_count == 0:
                entity_stale[e.entity_path].append(e)

        for ep, stale in entity_stale.items():
            if len(stale) < 2:
                continue

            actively_read = [e for e in stale if (e.recall_count or 0) > 0]
            oldest = min(e.provenance.written_at for e in stale)
            age_days = (datetime.now(timezone.utc) - oldest).days

            severity = "warning"
            if actively_read:
                severity = "critical" if len(actively_read) > 3 else "warning"

            desc = f"{len(stale)} entries in '{ep}' are older than {self.stale_days} days"
            if actively_read:
                desc += f" and {len(actively_read)} are still being read by other agents"

            patterns.append(DetectedPattern(
                pattern_type="stale_knowledge",
                severity=severity,
                category="collaboration",
                entity_path=ep,
                description=desc,
                details={
                    "stale_count": len(stale),
                    "actively_read_count": len(actively_read),
                    "oldest_age_days": age_days,
                    "keys": [e.key for e in stale[:10]],
                },
            ))

        return patterns

    def _detect_orphaned_branches(
        self, branches: list[Branch]
    ) -> list[DetectedPattern]:
        """Find active branches with no recent activity."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.orphan_days)
        patterns: list[DetectedPattern] = []

        for b in branches:
            if b.status != BranchStatus.ACTIVE:
                continue
            if b.name == "main":
                continue

            created = b.created_at or b.branched_at
            if created > cutoff:
                continue

            age_days = (datetime.now(timezone.utc) - created).days
            severity = "warning" if age_days < 30 else "critical"

            patterns.append(DetectedPattern(
                pattern_type="orphaned_branch",
                severity=severity,
                category="collaboration",
                entity_path=f"_branches/{b.name}",
                description=(
                    f"Branch '{b.name}' has been active for {age_days} days "
                    f"without being merged (created by {b.created_by})"
                ),
                details={
                    "branch_name": b.name,
                    "created_by": b.created_by,
                    "created_at": created.isoformat(),
                    "age_days": age_days,
                    "parent": b.parent_branch,
                },
            ))

        return patterns

    def _detect_redundant_writes(
        self, entries: list[MemoryEntry]
    ) -> list[DetectedPattern]:
        """Find different agents writing similar values to similar keys."""
        entity_entries: dict[str, list[MemoryEntry]] = defaultdict(list)
        for e in entries:
            entity_entries[e.entity_path].append(e)

        patterns: list[DetectedPattern] = []
        seen_pairs: set[tuple[str, str]] = set()

        for ep, group in entity_entries.items():
            if len(group) < 2:
                continue

            by_agent: dict[str, list[MemoryEntry]] = defaultdict(list)
            for e in group:
                by_agent[e.provenance.agent_id].append(e)

            if len(by_agent) < 2:
                continue

            agents = list(by_agent.keys())
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    a1, a2 = agents[i], agents[j]
                    for e1 in by_agent[a1]:
                        for e2 in by_agent[a2]:
                            pair_key = tuple(sorted([e1.key, e2.key]))
                            if pair_key in seen_pairs:
                                continue

                            if e1.key == e2.key:
                                continue

                            v1 = str(e1.value or "")[:500]
                            v2 = str(e2.value or "")[:500]
                            if not v1 or not v2:
                                continue

                            sim = SequenceMatcher(None, v1, v2).ratio()
                            if sim >= self.similarity_threshold:
                                seen_pairs.add(pair_key)
                                patterns.append(DetectedPattern(
                                    pattern_type="redundant_writes",
                                    severity="info",
                                    category="collaboration",
                                    entity_path=ep,
                                    description=(
                                        f"'{e1.key}' (by {a1}) and '{e2.key}' (by {a2}) "
                                        f"have {sim:.0%} similar content in {ep}"
                                    ),
                                    details={
                                        "key_a": e1.key,
                                        "agent_a": a1,
                                        "key_b": e2.key,
                                        "agent_b": a2,
                                        "similarity": round(sim, 3),
                                    },
                                ))
                            if len(patterns) > 50:
                                return patterns

        return patterns

    # ── Tier 2: Collaboration Insights ────────────────────────────────

    def _detect_single_point_of_knowledge(
        self, entries: list[MemoryEntry]
    ) -> list[DetectedPattern]:
        """Find entities where only one agent has written knowledge."""
        entity_agents: dict[str, set[str]] = defaultdict(set)
        entity_counts: dict[str, int] = defaultdict(int)

        for e in entries:
            if e.entity_path.startswith("_system/"):
                continue
            entity_agents[e.entity_path].add(e.provenance.agent_id)
            entity_counts[e.entity_path] += 1

        patterns: list[DetectedPattern] = []
        for ep, agents in entity_agents.items():
            if len(agents) != 1:
                continue
            count = entity_counts[ep]
            if count < 3:
                continue

            sole_agent = next(iter(agents))
            severity = "warning" if count >= 5 else "info"

            patterns.append(DetectedPattern(
                pattern_type="single_point_of_knowledge",
                severity=severity,
                category="insights",
                entity_path=ep,
                description=(
                    f"Only '{sole_agent}' has written to '{ep}' "
                    f"({count} entries) — bus factor is 1"
                ),
                details={
                    "agent": sole_agent,
                    "entry_count": count,
                },
            ))

        return patterns

    def _detect_passive_consumers(
        self, entries: list[MemoryEntry]
    ) -> list[DetectedPattern]:
        """Find agents that only read from entities they never write to.

        Uses recall_count > 0 and cross-references with write authorship.
        """
        entity_writers: dict[str, set[str]] = defaultdict(set)
        entity_readers: dict[str, set[str]] = defaultdict(set)

        for e in entries:
            if e.entity_path.startswith("_system/"):
                continue
            entity_writers[e.entity_path].add(e.provenance.agent_id)

        all_agents: set[str] = set()
        for e in entries:
            all_agents.add(e.provenance.agent_id)

        agent_write_entities: dict[str, set[str]] = defaultdict(set)
        for e in entries:
            agent_write_entities[e.provenance.agent_id].add(e.entity_path)

        agent_reads: dict[str, set[str]] = defaultdict(set)
        for e in entries:
            if (e.recall_count or 0) > 0:
                for agent in all_agents:
                    if agent != e.provenance.agent_id:
                        agent_reads[agent].add(e.entity_path)

        patterns: list[DetectedPattern] = []
        for agent, read_entities in agent_reads.items():
            write_entities = agent_write_entities.get(agent, set())
            read_only = read_entities - write_entities
            if len(read_only) < 2:
                continue

            patterns.append(DetectedPattern(
                pattern_type="passive_consumer",
                severity="info",
                category="insights",
                entity_path=f"_agents/{agent}",
                description=(
                    f"Agent '{agent}' reads from {len(read_only)} entities "
                    f"it never writes to"
                ),
                details={
                    "agent": agent,
                    "read_only_entities": sorted(read_only)[:10],
                    "read_only_count": len(read_only),
                    "write_count": len(write_entities),
                },
            ))

        return patterns

    def _detect_unreviewed_changes(
        self, pull_requests: list[PullRequest]
    ) -> list[DetectedPattern]:
        """Find open PRs that have been waiting too long."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.pr_stale_days)
        patterns: list[DetectedPattern] = []

        for pr in pull_requests:
            if pr.status != PullRequestStatus.OPEN:
                continue

            created = pr.created_at
            if created is None or created > cutoff:
                continue

            age_days = (datetime.now(timezone.utc) - created).days
            severity = "critical" if age_days > 14 else "warning"

            patterns.append(DetectedPattern(
                pattern_type="unreviewed_changes",
                severity=severity,
                category="collaboration",
                entity_path=f"_branches/{pr.source_branch}",
                description=(
                    f"PR '{pr.title}' has been open for {age_days} days "
                    f"without review ({pr.source_branch} → {pr.target_branch})"
                ),
                details={
                    "pr_title": pr.title,
                    "source_branch": pr.source_branch,
                    "target_branch": pr.target_branch,
                    "created_by": pr.created_by,
                    "created_at": created.isoformat(),
                    "age_days": age_days,
                },
            ))

        return patterns

    # ── Legacy: Reliability ───────────────────────────────────────────

    def _detect_recurring_failures(
        self,
        entries: list[MemoryEntry],
        outcomes: list[OutcomeRecord],
    ) -> list[DetectedPattern]:
        """Find entries that repeatedly appear in incident causal chains."""
        incident_types = {
            "critical_failure", "failure", "minor_failure",
            "p1_incident", "p2_incident", "regression",
        }
        incident_outcomes = [
            o for o in outcomes if o.outcome_type.value in incident_types
        ]

        entry_incident_count: dict[str, int] = defaultdict(int)
        for outcome in incident_outcomes:
            for ek in outcome.causal_entry_keys:
                entry_incident_count[ek] += 1

        entry_map = {e.entry_key: e for e in entries}
        patterns: list[DetectedPattern] = []

        for entry_key, count in entry_incident_count.items():
            if count < self.incident_threshold:
                continue
            entry = entry_map.get(entry_key)
            ep = entry.entity_path if entry else entry_key.rsplit("/", 1)[0]
            severity = "critical" if count >= self.incident_threshold * 2 else "warning"

            patterns.append(DetectedPattern(
                pattern_type="recurring_failure",
                severity=severity,
                category="reliability",
                entity_path=ep,
                description=(
                    f"Entry '{entry_key}' appeared in {count} incident causal chains"
                ),
                details={
                    "entry_key": entry_key,
                    "incident_count": count,
                    "confidence": entry.confidence if entry else None,
                },
            ))

        return patterns
