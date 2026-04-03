"""PatternDetector — scans memory entries and outcome data for recurring patterns.

Detects four pattern types:
  - Recurring failures: entries that repeatedly appear in incident causal chains
  - Hot entities: entity paths with disproportionate write/outcome activity
  - Stale clusters: groups of old entries with no outcome links
  - Confidence drift: entries whose confidence diverges from their entity average
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from amfs_core.models import MemoryEntry, OutcomeRecord


class DetectedPattern(BaseModel):
    """A single detected pattern with metadata."""

    pattern_type: str  # "recurring_failure", "hot_entity", "stale_cluster", "confidence_drift"
    severity: str  # "info", "warning", "critical"
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


class PatternDetector:
    """Scans memory entries and outcome data for four pattern types.

    Parameters
    ----------
    incident_threshold:
        Minimum number of incident appearances before flagging as recurring.
    stale_days:
        Entries older than this without outcome links are considered stale.
    hot_entity_stddev:
        Entity activity must exceed mean + this many stddevs to be "hot".
    drift_stddev:
        Confidence must diverge by this many stddevs from entity mean.
    """

    def __init__(
        self,
        *,
        incident_threshold: int = 2,
        stale_days: int = 30,
        hot_entity_stddev: float = 2.0,
        drift_stddev: float = 2.0,
    ) -> None:
        self.incident_threshold = incident_threshold
        self.stale_days = stale_days
        self.hot_entity_stddev = hot_entity_stddev
        self.drift_stddev = drift_stddev

    def analyze(
        self,
        entries: list[MemoryEntry],
        *,
        outcome_data: list[OutcomeRecord] | None = None,
    ) -> PatternReport:
        """Run all detectors and return a consolidated report."""
        start = datetime.now(timezone.utc)
        outcomes = outcome_data or []

        patterns: list[DetectedPattern] = []
        patterns.extend(self._detect_recurring_failures(entries, outcomes))
        patterns.extend(self._detect_hot_entities(entries, outcomes))
        patterns.extend(self._detect_stale_clusters(entries))
        patterns.extend(self._detect_confidence_drift(entries))

        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return PatternReport(
            patterns=patterns,
            scanned_entries=len(entries),
            scanned_outcomes=len(outcomes),
            scan_duration_ms=elapsed,
        )

    def _detect_recurring_failures(
        self,
        entries: list[MemoryEntry],
        outcomes: list[OutcomeRecord],
    ) -> list[DetectedPattern]:
        """Find entries that repeatedly appear in incident causal chains."""
        incident_types = {"p1_incident", "p2_incident", "regression"}
        incident_outcomes = [o for o in outcomes if o.outcome_type.value in incident_types]

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
                entity_path=ep,
                description=f"Entry '{entry_key}' appeared in {count} incident causal chains",
                details={
                    "entry_key": entry_key,
                    "incident_count": count,
                    "confidence": entry.confidence if entry else None,
                },
            ))

        return patterns

    def _detect_hot_entities(
        self,
        entries: list[MemoryEntry],
        outcomes: list[OutcomeRecord],
    ) -> list[DetectedPattern]:
        """Find entity paths with disproportionate activity."""
        entity_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            entity_counts[e.entity_path] += 1

        for o in outcomes:
            for ek in o.causal_entry_keys:
                parts = ek.rsplit("/", 1)
                if len(parts) == 2:
                    entity_counts[parts[0]] += 1

        if len(entity_counts) < 2:
            return []

        counts = list(entity_counts.values())
        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts) if len(counts) > 1 else 0.0
        threshold = mean + self.hot_entity_stddev * stdev

        if stdev == 0:
            return []

        patterns: list[DetectedPattern] = []
        for ep, count in entity_counts.items():
            if count > threshold:
                patterns.append(DetectedPattern(
                    pattern_type="hot_entity",
                    severity="info",
                    entity_path=ep,
                    description=(
                        f"Entity '{ep}' has {count} interactions "
                        f"(mean={mean:.1f}, threshold={threshold:.1f})"
                    ),
                    details={
                        "activity_count": count,
                        "mean": round(mean, 2),
                        "stddev": round(stdev, 2),
                        "threshold": round(threshold, 2),
                    },
                ))

        return patterns

    def _detect_stale_clusters(
        self,
        entries: list[MemoryEntry],
    ) -> list[DetectedPattern]:
        """Find groups of old entries with no outcome links."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.stale_days)
        entity_stale: dict[str, list[MemoryEntry]] = defaultdict(list)

        for e in entries:
            if e.outcome_count == 0 and e.provenance.written_at < cutoff:
                entity_stale[e.entity_path].append(e)

        patterns: list[DetectedPattern] = []
        for ep, stale_entries in entity_stale.items():
            if len(stale_entries) < 2:
                continue
            oldest = min(e.provenance.written_at for e in stale_entries)
            age_days = (datetime.now(timezone.utc) - oldest).days

            patterns.append(DetectedPattern(
                pattern_type="stale_cluster",
                severity="warning",
                entity_path=ep,
                description=(
                    f"{len(stale_entries)} entries in '{ep}' have no outcome links "
                    f"and are older than {self.stale_days} days"
                ),
                details={
                    "stale_count": len(stale_entries),
                    "oldest_age_days": age_days,
                    "keys": [e.key for e in stale_entries[:10]],
                },
            ))

        return patterns

    def _detect_confidence_drift(
        self,
        entries: list[MemoryEntry],
    ) -> list[DetectedPattern]:
        """Find entries whose confidence diverges from their entity average."""
        entity_entries: dict[str, list[MemoryEntry]] = defaultdict(list)
        for e in entries:
            entity_entries[e.entity_path].append(e)

        patterns: list[DetectedPattern] = []
        for ep, group in entity_entries.items():
            if len(group) < 3:
                continue
            confidences = [e.confidence for e in group]
            mean = statistics.mean(confidences)
            stdev = statistics.stdev(confidences)
            if stdev == 0:
                continue

            for e in group:
                z_score = abs(e.confidence - mean) / stdev
                if z_score >= self.drift_stddev:
                    direction = "above" if e.confidence > mean else "below"
                    patterns.append(DetectedPattern(
                        pattern_type="confidence_drift",
                        severity="info",
                        entity_path=ep,
                        description=(
                            f"Entry '{e.key}' confidence ({e.confidence:.3f}) is "
                            f"{z_score:.1f}σ {direction} entity mean ({mean:.3f})"
                        ),
                        details={
                            "key": e.key,
                            "confidence": round(e.confidence, 4),
                            "entity_mean": round(mean, 4),
                            "entity_stddev": round(stdev, 4),
                            "z_score": round(z_score, 2),
                        },
                    ))

        return patterns
