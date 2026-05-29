"""BriefingService — reads pre-compiled digests and ranks by relevance.

Includes hot-context injection: the top priority-scored entries are
surfaced alongside digests so agents always see the most important
memories, not just the most recently written.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from amfs_core.models import Digest, DigestType, MemoryEntry, SearchQuery

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter

logger = logging.getLogger(__name__)

_HOT_CONTEXT_LIMIT = 3


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
        digests = [d for _, d in scored[:limit]]

        if entity_path:
            if digests:
                self._inject_hot_context(digests, entity_path, branch)
                self._inject_consolidation_notice(digests, entity_path, branch)
            else:
                self._inject_standalone_hot_context(digests, entity_path, branch)

        return digests

    def _inject_hot_context(
        self,
        digests: list[Digest],
        entity_path: str,
        branch: str,
    ) -> None:
        """Attach top priority-scored entries to the lead entity digest.

        This gives agents the equivalent of soft attention over the memory
        store: the most important entries are always surfaced, not just those
        mentioned in the compiled digest summary.
        """
        try:
            entries = self._adapter.search(
                SearchQuery(
                    entity_path=entity_path,
                    sort_by="priority",
                    limit=_HOT_CONTEXT_LIMIT,
                ),
                branch=branch,
            )
        except Exception:
            logger.debug("Hot-context injection failed for %s", entity_path, exc_info=True)
            return

        if not entries:
            return

        hot_entries = [
            {
                "key": e.key,
                "value": e.value,
                "confidence": round(e.confidence, 3),
                "agent": e.provenance.agent_id,
                "outcome_count": e.outcome_count,
                "recall_count": e.recall_count,
            }
            for e in entries
        ]

        for d in digests:
            if d.digest_type == DigestType.ENTITY and d.scope == entity_path:
                d.summary["hot_context"] = hot_entries
                break

    def _inject_standalone_hot_context(
        self,
        digests: list[Digest],
        entity_path: str,
        branch: str,
    ) -> None:
        """Create a minimal digest with hot-context when no compiled digest exists.

        Ensures agents always get priority-scored entries even for entities
        that haven't been compiled yet.
        """
        try:
            entries = self._adapter.search(
                SearchQuery(
                    entity_path=entity_path,
                    sort_by="priority",
                    limit=_HOT_CONTEXT_LIMIT,
                ),
                branch=branch,
            )
        except Exception:
            return

        if not entries:
            return

        hot_entries = [
            {
                "key": e.key,
                "value": e.value,
                "confidence": round(e.confidence, 3),
                "agent": e.provenance.agent_id,
                "outcome_count": e.outcome_count,
                "recall_count": e.recall_count,
            }
            for e in entries
        ]

        digests.append(Digest(
            digest_type=DigestType.ENTITY,
            scope=entity_path,
            summary={
                "narrative": f"No compiled digest yet for {entity_path}. Showing top priority entries.",
                "hot_context": hot_entries,
            },
            entry_count=len(entries),
            source_agents=[],
            compiled_at=datetime.now(timezone.utc),
            namespace=self._namespace,
            branch=branch,
        ))

    def _inject_consolidation_notice(
        self,
        digests: list[Digest],
        entity_path: str,
        branch: str,
    ) -> None:
        """Surface pending consolidation branches for this entity."""
        try:
            branches = self._adapter.list_branches(
                namespace=self._namespace,
                status="active",
            )
            pending = [
                b for b in branches
                if b.name.startswith(f"cortex/consolidation/{entity_path}/")
            ]
        except Exception:
            logger.debug("Consolidation notice injection failed", exc_info=True)
            return

        if not pending:
            return

        notice = {
            "pending_proposals": len(pending),
            "branch_names": [b.name for b in pending[:5]],
            "message": (
                f"There {'is' if len(pending) == 1 else 'are'} {len(pending)} pending "
                f"consolidation proposal{'s' if len(pending) != 1 else ''} for this entity. "
                f"Review at /cortex/proposals."
            ),
        }

        for d in digests:
            if d.digest_type == DigestType.ENTITY and d.scope == entity_path:
                d.summary["consolidation_notice"] = notice
                break

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
            elif digest.digest_type == DigestType.CONNECTION_MAP and digest.scope == entity_path:
                score += 80.0
            elif digest.digest_type == DigestType.SOURCE:
                touched = digest.summary.get("entities_touched", [])
                if entity_path in touched:
                    score += 60.0
            elif digest.digest_type == DigestType.AGENT_BRIEF:
                entities = digest.summary.get("entities_written", [])
                if entity_path in entities:
                    score += 40.0
            elif digest.digest_type == DigestType.CONNECTION_MAP:
                connected = digest.summary.get("connected_entities", [])
                if entity_path in connected:
                    score += 30.0

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
