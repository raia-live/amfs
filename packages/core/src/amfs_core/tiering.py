"""Tiered memory scoring and assignment.

Implements a priority scoring formula inspired by the HMO paper's
adaptive scoring mechanism, adapted for AMFS's four-signal confidence
model (time + memory_type + outcomes + access frequency).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from amfs_core.models import MemoryEntry, MemoryTier, TierConfig


class PriorityScorer:
    """Compute composite priority scores for memory entries.

    The formula mirrors HMO's S_m with AMFS-specific signals:

        S_m = (alpha * importance + beta * recency) * freq_boost * time_decay

    where:
        importance = entry.importance_score or entry.confidence
        recency    = exp(-lambda * age_days)
        freq_boost = ln(1 + recall_count)   (at least 1.0)
        time_decay = exp(-lambda * age_days / max(1, ln(1 + recall_count)))
    """

    def __init__(self, config: TierConfig | None = None) -> None:
        self._config = config or TierConfig()

    def score(self, entry: MemoryEntry, *, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        age = now - entry.provenance.written_at
        age_days = max(0.0, age.total_seconds() / 86400.0)

        importance = getattr(entry, "importance_score", None) or entry.confidence
        importance = max(0.0, min(1.0, float(importance)))

        cfg = self._config
        recency = math.exp(-cfg.decay_lambda * age_days)

        freq_boost = max(1.0, math.log1p(entry.recall_count))

        denominator = max(1.0, math.log1p(entry.recall_count))
        time_decay = math.exp(-cfg.decay_lambda * age_days / denominator)

        return (cfg.alpha * importance + cfg.beta * recency) * freq_boost * time_decay

    def score_batch(
        self, entries: list[MemoryEntry], *, now: datetime | None = None
    ) -> dict[str, float]:
        now = now or datetime.now(timezone.utc)
        return {entry.entry_key: self.score(entry, now=now) for entry in entries}


class TierAssigner:
    """Assign memory tiers based on priority scores."""

    def __init__(self, config: TierConfig | None = None) -> None:
        self._config = config or TierConfig()

    def assign(
        self, entries: list[MemoryEntry], scores: dict[str, float]
    ) -> dict[str, int]:
        """Assign tiers to entries based on their scores.

        Returns a mapping of entry_key -> tier (1=HOT, 2=WARM, 3=ARCHIVE).
        """
        ranked = sorted(entries, key=lambda e: scores.get(e.entry_key, 0.0), reverse=True)

        result: dict[str, int] = {}
        hot_cap = self._config.hot_capacity
        warm_cap = self._config.warm_capacity

        for i, entry in enumerate(ranked):
            if i < hot_cap:
                result[entry.entry_key] = MemoryTier.HOT
            elif i < hot_cap + warm_cap:
                result[entry.entry_key] = MemoryTier.WARM
            else:
                result[entry.entry_key] = MemoryTier.ARCHIVE

        return result

    def assign_with_scores(
        self,
        entries: list[MemoryEntry],
        scorer: PriorityScorer,
        *,
        now: datetime | None = None,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Score and assign tiers in one pass."""
        scores = scorer.score_batch(entries, now=now)
        tiers = self.assign(entries, scores)
        return tiers, scores
