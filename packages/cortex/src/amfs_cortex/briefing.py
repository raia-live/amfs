"""BriefingService — reads pre-compiled digests and ranks by relevance.

Includes hot-context injection: the top priority-scored entries are
surfaced alongside digests so agents always see the most important
memories, not just the most recently written.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from amfs_core.authority import rank_authors
from amfs_core.models import Digest, DigestType, MemoryEntry, SearchQuery

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter

logger = logging.getLogger(__name__)

_HOT_CONTEXT_LIMIT = 3

# Three authors, three keys each. This rides along on the one call every agent
# is told to make first, so it is paying for itself in context window on every
# task — enough to route, not enough to be worth skimming past.
_WHO_TO_ASK_LIMIT = 3
_WHO_TO_ASK_KEYS = 3


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
            self._inject_who_to_ask(digests, entity_path, agent_id)

        return digests

    def _inject_who_to_ask(
        self,
        digests: list[Digest],
        entity_path: str,
        agent_id: str | None,
    ) -> None:
        """Name the agents worth asking about this entity, and what to read.

        Injected at serve time rather than compiled into the digest, for two
        reasons that both rule compilation out. Authority depends on recency,
        and the worker recompiles on a debounce, so a compiled ranking is stale
        by construction. And this names *other* agents, so it has to be
        filtered per caller — a compiled digest is shared by everyone who asks
        for it, which is the wrong granularity for a disclosure surface.

        The caller-side visibility filter is the control that enforces the
        second point; see the http-server's briefing handler.
        """
        try:
            stats = self._adapter.agent_entity_stats(entity_path=entity_path)
        except Exception:
            logger.debug("who_to_ask injection failed for %s", entity_path, exc_info=True)
            return

        ranked = rank_authors(entity_path, stats=stats, limit=_WHO_TO_ASK_LIMIT)

        # Asking an agent to read from itself is noise, not routing. Drop the
        # caller only after ranking, so its own writes still set the share
        # denominator and a colleague's contribution is not overstated.
        if agent_id:
            ranked = [a for a in ranked if a.agent_id != agent_id]
        if not ranked:
            return

        top_keys = self._top_keys_by_agent(entity_path, [a.agent_id for a in ranked])

        block = []
        for author in ranked:
            keys = top_keys.get(author.agent_id, [])
            item: dict[str, Any] = {
                "agent_id": author.agent_id,
                "reason": author.reason,
                "entry_count": author.entry_count,
                "share": round(author.share, 3),
                "validated_outcomes": author.validated_outcomes,
                "top_keys": keys,
            }
            # The literal call closes the "I know who, but not what to read"
            # gap that makes a bare recommendation useless.
            if keys:
                item["call"] = (
                    f'amfs_read_from("{author.agent_id}", "{entity_path}", "{keys[0]}")'
                )
            block.append(item)

        for d in digests:
            if d.digest_type == DigestType.ENTITY and d.scope == entity_path:
                d.summary["who_to_ask"] = block
                return

    def _top_keys_by_agent(
        self,
        entity_path: str,
        agent_ids: list[str],
    ) -> dict[str, list[str]]:
        """Highest-priority keys each named agent wrote on this entity."""
        result: dict[str, list[str]] = {}
        for aid in agent_ids:
            try:
                entries = self._adapter.search(
                    SearchQuery(
                        entity_path=entity_path,
                        agent_id=aid,
                        sort_by="priority",
                        limit=_WHO_TO_ASK_KEYS,
                        include_artifacts=False,
                    ),
                )
            except Exception:
                logger.debug("who_to_ask key lookup failed for %s", aid, exc_info=True)
                continue
            result[aid] = [e.key for e in entries]
        return result

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
                    # Working files shouldn't dominate the "what you know" context
                    # an agent reads at task start.
                    include_artifacts=False,
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
                    # Working files shouldn't dominate the "what you know" context
                    # an agent reads at task start.
                    include_artifacts=False,
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
