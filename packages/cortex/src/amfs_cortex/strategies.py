"""Rule-based compilation strategy — OSS default.

Pure structured aggregation with no LLM calls or external dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from amfs_core.models import Digest, DigestType

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter


class RuleBasedStrategy:
    """Compiles digests from structured aggregation of memory entries."""

    def compile_entity(
        self, entity_path: str, adapter: PostgresAdapter, namespace: str
    ) -> Digest | None:
        from amfs_core.models import SearchQuery

        entries = adapter.search(SearchQuery(
            entity_path=entity_path,
            limit=1000,
            sort_by="confidence",
        ))
        if not entries:
            return None

        agents: set[str] = set()
        external_sources: set[str] = set()
        total_confidence = 0.0
        outcome_linked = 0
        top_keys: list[dict[str, Any]] = []

        for e in entries:
            aid = e.provenance.agent_id
            if aid.startswith(("webhook/", "external/")):
                external_sources.add(aid.split("/", 1)[1])
            else:
                agents.add(aid)
            total_confidence += e.confidence
            if e.outcome_count > 0:
                outcome_linked += 1

        sorted_entries = sorted(entries, key=lambda e: e.confidence, reverse=True)
        for e in sorted_entries[:5]:
            top_keys.append({
                "key": e.key,
                "confidence": round(e.confidence, 3),
                "agent": e.provenance.agent_id,
                "updated": e.provenance.written_at.isoformat(),
            })

        last_write = max(e.provenance.written_at for e in entries)

        return Digest(
            digest_type=DigestType.ENTITY,
            scope=entity_path,
            summary={
                "total_keys": len(entries),
                "top_keys": top_keys,
                "agents": sorted(agents),
                "external_sources": sorted(external_sources),
                "avg_confidence": round(total_confidence / len(entries), 3) if entries else 0,
                "last_write": last_write.isoformat(),
                "outcome_linked": outcome_linked,
                "external_events": sum(
                    1 for e in entries
                    if e.provenance.agent_id.startswith(("webhook/", "external/"))
                ),
            },
            entry_count=len(entries),
            source_agents=sorted(agents | {f"webhook/{s}" for s in external_sources} | {f"external/{s}" for s in external_sources}),
            compiled_at=datetime.now(timezone.utc),
            namespace=namespace,
        )

    def compile_agent_brief(
        self, agent_id: str, adapter: PostgresAdapter, namespace: str
    ) -> Digest | None:
        from amfs_core.models import SearchQuery

        entries = adapter.search(SearchQuery(
            agent_id=agent_id,
            limit=1000,
            sort_by="recency",
        ))
        if not entries:
            return None

        entities_written: set[str] = set()
        total_entries = len(entries)
        top_knowledge: list[dict[str, Any]] = []

        for e in entries:
            entities_written.add(e.entity_path)

        by_confidence = sorted(entries, key=lambda e: e.confidence, reverse=True)
        for e in by_confidence[:5]:
            top_knowledge.append({
                "entity": e.entity_path,
                "key": e.key,
                "confidence": round(e.confidence, 3),
            })

        recent_entities = []
        seen: set[str] = set()
        for e in entries:
            if e.entity_path not in seen:
                recent_entities.append(e.entity_path)
                seen.add(e.entity_path)
            if len(recent_entities) >= 5:
                break

        outcomes_committed = sum(1 for e in entries if e.outcome_count > 0)
        last_active = max(e.provenance.written_at for e in entries)

        return Digest(
            digest_type=DigestType.AGENT_BRIEF,
            scope=agent_id,
            summary={
                "entities_written": sorted(entities_written),
                "total_entries": total_entries,
                "top_knowledge": top_knowledge,
                "recent_activity": recent_entities,
                "outcomes_committed": outcomes_committed,
                "last_active": last_active.isoformat(),
            },
            entry_count=total_entries,
            source_agents=[agent_id],
            compiled_at=datetime.now(timezone.utc),
            namespace=namespace,
        )

    def compile_source(
        self, source_id: str, adapter: PostgresAdapter, namespace: str
    ) -> Digest | None:
        from amfs_core.models import SearchQuery

        webhook_entries = adapter.search(SearchQuery(
            agent_id=f"webhook/{source_id}",
            limit=1000,
            sort_by="recency",
        ))
        external_entries = adapter.search(SearchQuery(
            agent_id=f"external/{source_id}",
            limit=1000,
            sort_by="recency",
        ))
        entries = webhook_entries + external_entries
        if not entries:
            return None

        entities_touched: set[str] = set()
        event_types: dict[str, int] = {}
        recent_events: list[dict[str, Any]] = []

        for e in entries:
            entities_touched.add(e.entity_path)
            key_parts = e.key.split("-", 1)
            etype = key_parts[0] if key_parts else "unknown"
            event_types[etype] = event_types.get(etype, 0) + 1

        for e in entries[:10]:
            recent_events.append({
                "key": e.key,
                "entity": e.entity_path,
                "written_at": e.provenance.written_at.isoformat(),
            })

        last_event = max(e.provenance.written_at for e in entries)

        return Digest(
            digest_type=DigestType.SOURCE,
            scope=source_id,
            summary={
                "total_events": len(entries),
                "recent_events": recent_events,
                "entities_touched": sorted(entities_touched),
                "event_types": event_types,
                "last_event": last_event.isoformat(),
            },
            entry_count=len(entries),
            source_agents=[f"webhook/{source_id}", f"external/{source_id}"],
            compiled_at=datetime.now(timezone.utc),
            namespace=namespace,
        )
