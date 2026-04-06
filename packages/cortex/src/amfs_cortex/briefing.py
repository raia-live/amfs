"""BriefingService — reads pre-compiled digests and ranks by relevance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from amfs_core.models import Digest, DigestType

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter


class BriefingService:
    """Serves pre-compiled digests ranked by relevance for a given context."""

    def __init__(self, adapter: PostgresAdapter, namespace: str = "default") -> None:
        self._adapter = adapter
        self._namespace = namespace

    def briefing(
        self,
        entity_path: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        branch: str = "main",
    ) -> list[Digest]:
        """Get a ranked list of relevant digests for the given context.

        Ranking (OSS, rule-based):
        1. Direct entity match (highest)
        2. Source digests for connectors with events on the same entity
        3. Agent briefs for agents that wrote to the same entity
        4. Recency-weighted
        5. Entry count weighted
        """
        all_digests = self._adapter.list_digests(namespace=self._namespace, branch=branch)
        if not all_digests:
            return []

        scored: list[tuple[float, Digest]] = []
        now = datetime.now(timezone.utc)

        for d in all_digests:
            score = self._score(d, entity_path, agent_id, now)
            if score > 0:
                d.staleness_ms = int((now - d.compiled_at).total_seconds() * 1000)
                scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def _score(
        self,
        digest: Digest,
        entity_path: str | None,
        agent_id: str | None,
        now: datetime,
    ) -> float:
        score = 0.0

        if entity_path:
            if digest.digest_type == DigestType.ENTITY and digest.scope == entity_path:
                score += 100.0
            elif digest.digest_type == DigestType.SOURCE:
                touched = digest.summary.get("entities_touched", [])
                if entity_path in touched:
                    score += 60.0
            elif digest.digest_type == DigestType.AGENT_BRIEF:
                entities = digest.summary.get("entities_written", [])
                if entity_path in entities:
                    score += 40.0

        if agent_id:
            if digest.digest_type == DigestType.AGENT_BRIEF and digest.scope == agent_id:
                score += 100.0
            elif digest.digest_type == DigestType.ENTITY:
                agents = digest.summary.get("agents", [])
                if agent_id in agents:
                    score += 50.0

        if score == 0:
            return 0.0

        age_hours = max((now - digest.compiled_at).total_seconds() / 3600, 0.01)
        recency_boost = min(10.0 / age_hours, 20.0)
        score += recency_boost

        entry_boost = min(digest.entry_count * 0.5, 15.0)
        score += entry_boost

        score += digest.anticipation_score * 30.0

        return score
