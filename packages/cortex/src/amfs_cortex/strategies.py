"""Rule-based compilation strategy — OSS default.

Pure structured aggregation with no LLM calls or external dependencies.
Generates both structured data and a human-readable narrative.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from amfs_core.models import Digest, DigestType

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter


def _confidence_label(c: float) -> str:
    if c >= 0.95:
        return "very high"
    if c >= 0.8:
        return "high"
    if c >= 0.6:
        return "moderate"
    if c >= 0.4:
        return "low"
    return "very low"


def _pluralize(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


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
        risk_keys: list[str] = []

        for e in entries:
            aid = e.provenance.agent_id
            if aid.startswith(("webhook/", "external/")):
                external_sources.add(aid.split("/", 1)[1])
            else:
                agents.add(aid)
            total_confidence += e.confidence
            if e.outcome_count > 0:
                outcome_linked += 1
            if e.key.startswith("risk-"):
                risk_keys.append(e.key)

        sorted_entries = sorted(entries, key=lambda e: e.confidence, reverse=True)
        for e in sorted_entries[:5]:
            top_keys.append({
                "key": e.key,
                "confidence": round(e.confidence, 3),
                "agent": e.provenance.agent_id,
                "updated": e.provenance.written_at.isoformat(),
            })

        last_write = max(e.provenance.written_at for e in entries)
        avg_conf = round(total_confidence / len(entries), 3) if entries else 0
        ext_event_count = sum(
            1 for e in entries
            if e.provenance.agent_id.startswith(("webhook/", "external/"))
        )

        narrative = self._narrate_entity(
            entity_path, entries, agents, external_sources,
            avg_conf, outcome_linked, risk_keys, ext_event_count,
        )

        return Digest(
            digest_type=DigestType.ENTITY,
            scope=entity_path,
            summary={
                "narrative": narrative,
                "total_keys": len(entries),
                "top_keys": top_keys,
                "agents": sorted(agents),
                "external_sources": sorted(external_sources),
                "avg_confidence": avg_conf,
                "last_write": last_write.isoformat(),
                "outcome_linked": outcome_linked,
                "external_events": ext_event_count,
            },
            entry_count=len(entries),
            source_agents=sorted(agents | {f"webhook/{s}" for s in external_sources} | {f"external/{s}" for s in external_sources}),
            compiled_at=datetime.now(timezone.utc),
            namespace=namespace,
        )

    @staticmethod
    def _narrate_entity(
        entity_path: str,
        entries: list,
        agents: set[str],
        external_sources: set[str],
        avg_conf: float,
        outcome_linked: int,
        risk_keys: list[str],
        ext_event_count: int,
    ) -> str:
        parts: list[str] = []

        parts.append(
            f"{entity_path} has {_pluralize(len(entries), 'knowledge entry')} "
            f"from {_pluralize(len(agents), 'agent')} "
            f"with {_confidence_label(avg_conf)} average confidence ({avg_conf:.0%})."
        )

        if outcome_linked > 0:
            ratio = outcome_linked / len(entries)
            parts.append(
                f"{_pluralize(outcome_linked, 'entry')} "
                f"({ratio:.0%}) validated by production outcomes."
            )

        if risk_keys:
            parts.append(
                f"Active risks: {', '.join(risk_keys)}."
            )

        if ext_event_count > 0:
            sources = ", ".join(sorted(external_sources))
            parts.append(
                f"{_pluralize(ext_event_count, 'external event')} ingested from {sources}."
            )

        return " ".join(parts)

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

        narrative = self._narrate_agent(
            agent_id, total_entries, entities_written,
            top_knowledge, outcomes_committed,
        )

        return Digest(
            digest_type=DigestType.AGENT_BRIEF,
            scope=agent_id,
            summary={
                "narrative": narrative,
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

    @staticmethod
    def _narrate_agent(
        agent_id: str,
        total_entries: int,
        entities_written: set[str],
        top_knowledge: list[dict[str, Any]],
        outcomes_committed: int,
    ) -> str:
        parts: list[str] = []

        parts.append(
            f"{agent_id} has written {_pluralize(total_entries, 'entry')} "
            f"across {_pluralize(len(entities_written), 'entity')}."
        )

        if top_knowledge:
            top = top_knowledge[0]
            parts.append(
                f"Highest-confidence knowledge: {top['key']} in {top['entity']} "
                f"({_confidence_label(top['confidence'])}, {top['confidence']:.0%})."
            )

        if outcomes_committed > 0:
            parts.append(
                f"{_pluralize(outcomes_committed, 'entry')} validated by outcomes."
            )

        return " ".join(parts)

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

        narrative = (
            f"{source_id} has sent {_pluralize(len(entries), 'event')} "
            f"touching {_pluralize(len(entities_touched), 'entity')} "
            f"({', '.join(sorted(entities_touched))})."
        )

        return Digest(
            digest_type=DigestType.SOURCE,
            scope=source_id,
            summary={
                "narrative": narrative,
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
