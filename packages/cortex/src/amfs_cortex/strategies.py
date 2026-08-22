"""Rule-based compilation strategy — OSS default.

Pure structured aggregation with no LLM calls or external dependencies.
Generates both structured data and a human-readable narrative.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from amfs_core.models import Digest, DigestType

logger = logging.getLogger(__name__)

#: How many entries a single entity digest compiles from. The search is ordered
#: by confidence, so a digest for an entity with more keys than this is built
#: from the top slice — which the schema profile / materialized aggregates must
#: disclose, since presenting a partial rollup as the entity's totals would
#: disagree with a full amfs_aggregate.
_ENTITY_DIGEST_LIMIT = 1000

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter


def _structured_profile(entries: list) -> tuple[dict | None, dict | None]:
    """Compute a schema profile + materialized aggregates for tabular entities.

    This is what makes heavy (e.g. room) usage fast: an agent reading the digest
    learns the shape of the data and gets common numeric rollups precomputed,
    without listing every record. Returns (None, None) when the entries are not
    record-shaped (plain-text memories), so ordinary entities are unaffected.
    Never raises — profiling must not break digest compilation.
    """
    try:
        from amfs_core.aggregates import aggregate_entries, room_schema_profile

        base = room_schema_profile(entries)
        candidates = base.get("row_path_candidates") or []
        row_path = candidates[0]["field"] if candidates else None
        profile = room_schema_profile(entries, row_path=row_path) if row_path else base

        field_names = [f["field"] for f in profile["fields"] if f["field"] != "_value"]
        if not field_names or profile["total_rows"] == 0:
            return None, None

        materialized: dict = {"total_rows": profile["total_rows"]}
        if row_path:
            materialized["row_path"] = row_path
        numeric_fields = [
            f["field"] for f in profile["fields"]
            if "numeric_range" in f and f["field"] != "_value"
        ]
        field_stats: dict = {}
        for fld in numeric_fields[:10]:
            res = aggregate_entries(entries, op="stats", field=fld, row_path=row_path)
            field_stats[fld] = {k: res.get(k) for k in ("n", "sum", "mean", "min", "max")}
        if field_stats:
            materialized["numeric_fields"] = field_stats
        return profile, materialized
    except Exception:  # noqa: BLE001 - profiling is best-effort
        logger.debug("schema profiling failed for entity digest", exc_info=True)
        return None, None


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
    if n == 1:
        return f"1 {word}"
    if word.endswith("y") and not word.endswith(("ay", "ey", "oy", "uy")):
        return f"{n} {word[:-1]}ies"
    return f"{n} {word}s"


class RuleBasedStrategy:
    """Compiles digests from structured aggregation of memory entries."""

    def compile_entity(
        self, entity_path: str, adapter: PostgresAdapter, namespace: str, branch: str = "main",
    ) -> Digest | None:
        from amfs_core.models import SearchQuery

        entries = adapter.search(SearchQuery(
            entity_path=entity_path,
            limit=_ENTITY_DIGEST_LIMIT,
            sort_by="confidence",
            # Digest narrative/top_keys summarise knowledge, not stored code files.
            include_artifacts=False,
        ), branch=branch)
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

        summary: dict[str, Any] = {
            "narrative": narrative,
            "total_keys": len(entries),
            "top_keys": top_keys,
            "agents": sorted(agents),
            "external_sources": sorted(external_sources),
            "avg_confidence": avg_conf,
            "last_write": last_write.isoformat(),
            "outcome_linked": outcome_linked,
            "external_events": ext_event_count,
        }

        # Room/tabular entities: carry an always-fresh schema profile and
        # precomputed numeric rollups so an agent can orient and get common
        # aggregates in this one briefing call instead of reading every record.
        schema_profile, materialized = _structured_profile(entries)
        if schema_profile is not None:
            # The digest is built from at most _ENTITY_DIGEST_LIMIT entries. When
            # the entity has more, these rollups are a top-confidence sample, not
            # the entity's totals — say so, and point at amfs_aggregate (which
            # runs over the full set) for exact numbers.
            if len(entries) >= _ENTITY_DIGEST_LIMIT:
                note = (
                    f"Computed from the {_ENTITY_DIGEST_LIMIT} highest-confidence "
                    "entries; this entity has at least that many, so treat these "
                    "as a sample and use amfs_aggregate for exact totals."
                )
                schema_profile["sampled"] = True
                schema_profile["sample_size"] = len(entries)
                schema_profile["note"] = note
                if materialized is not None:
                    materialized["sampled"] = True
                    materialized["note"] = note
            summary["schema_profile"] = schema_profile
            summary["materialized_aggregates"] = materialized

        return Digest(
            digest_type=DigestType.ENTITY,
            scope=entity_path,
            summary=summary,
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
        self, agent_id: str, adapter: PostgresAdapter, namespace: str, branch: str = "main",
    ) -> Digest | None:
        from amfs_core.models import SearchQuery

        entries = adapter.search(SearchQuery(
            agent_id=agent_id,
            limit=1000,
            sort_by="recency",
        ), branch=branch)
        if not entries:
            return None

        entities_written: set[str] = set()
        total_entries = len(entries)
        top_knowledge: list[dict[str, Any]] = []
        memory_types: dict[str, int] = {}
        risk_entries: list[str] = []
        pattern_refs: set[str] = set()

        for e in entries:
            entities_written.add(e.entity_path)
            mt = e.memory_type.value if hasattr(e.memory_type, "value") else str(e.memory_type)
            memory_types[mt] = memory_types.get(mt, 0) + 1
            if e.key.startswith("risk-"):
                risk_entries.append(e.key)
            for pr in e.provenance.pattern_refs:
                pattern_refs.add(pr)

        by_confidence = sorted(entries, key=lambda e: e.confidence, reverse=True)
        for e in by_confidence[:5]:
            top_knowledge.append({
                "entity": e.entity_path,
                "key": e.key,
                "confidence": round(e.confidence, 3),
            })

        recent_entities: list[str] = []
        seen: set[str] = set()
        for e in entries:
            if e.entity_path not in seen:
                recent_entities.append(e.entity_path)
                seen.add(e.entity_path)
            if len(recent_entities) >= 5:
                break

        outcomes_committed = sum(1 for e in entries if e.outcome_count > 0)
        last_active = max(e.provenance.written_at for e in entries)

        expertise_areas = self._derive_expertise(entries)

        all_entries = adapter.search(SearchQuery(limit=5000, sort_by="recency"), branch=branch)
        other_agents_on_entities: dict[str, set[str]] = {}
        for e in all_entries:
            if e.provenance.agent_id != agent_id and e.entity_path in entities_written:
                other_agents_on_entities.setdefault(e.entity_path, set()).add(e.provenance.agent_id)

        collaboration_score = len([ep for ep in entities_written if ep in other_agents_on_entities])
        solo_entities = [ep for ep in entities_written if ep not in other_agents_on_entities]

        anticipation = self._compute_anticipation(
            entries, outcomes_committed, total_entries, len(entities_written),
        )

        narrative = self._narrate_agent(
            agent_id, total_entries, entities_written,
            top_knowledge, outcomes_committed, expertise_areas,
            collaboration_score, solo_entities, risk_entries,
            memory_types, recent_entities,
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
                "expertise_areas": expertise_areas,
                "memory_types": memory_types,
                "collaboration_score": collaboration_score,
                "solo_entities": sorted(solo_entities),
                "risk_entries": risk_entries,
                "pattern_refs": sorted(pattern_refs),
            },
            entry_count=total_entries,
            source_agents=[agent_id],
            compiled_at=datetime.now(timezone.utc),
            namespace=namespace,
            anticipation_score=anticipation,
        )

    @staticmethod
    def _derive_expertise(entries: list) -> list[str]:
        """Derive expertise areas from entity paths the agent writes to most."""
        entity_counts: dict[str, int] = {}
        for e in entries:
            entity_counts[e.entity_path] = entity_counts.get(e.entity_path, 0) + 1

        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        areas: list[str] = []
        for ep, count in sorted_entities[:5]:
            short = ep.split("/")[-1] if "/" in ep else ep
            areas.append(short)
        return areas

    @staticmethod
    def _compute_anticipation(
        entries: list, outcomes_committed: int, total_entries: int, entity_count: int,
    ) -> float:
        """Compute an anticipation score (0–1) based on activity signals."""
        if total_entries == 0:
            return 0.0

        score = 0.0

        outcome_ratio = outcomes_committed / total_entries if total_entries else 0
        score += outcome_ratio * 0.3

        now = datetime.now(timezone.utc)
        recent_count = sum(
            1 for e in entries
            if (now - e.provenance.written_at).total_seconds() < 86400
        )
        recency_ratio = min(recent_count / max(total_entries, 1), 1.0)
        score += recency_ratio * 0.3

        breadth = min(entity_count / 5, 1.0)
        score += breadth * 0.2

        depth = min(total_entries / 20, 1.0)
        score += depth * 0.2

        return round(min(score, 1.0), 3)

    @staticmethod
    def _narrate_agent(
        agent_id: str,
        total_entries: int,
        entities_written: set[str],
        top_knowledge: list[dict[str, Any]],
        outcomes_committed: int,
        expertise_areas: list[str],
        collaboration_score: int,
        solo_entities: list[str],
        risk_entries: list[str],
        memory_types: dict[str, int],
        recent_entities: list[str],
    ) -> str:
        parts: list[str] = []

        if expertise_areas:
            areas_str = ", ".join(expertise_areas[:3])
            parts.append(f"{agent_id} specializes in {areas_str}.")
        else:
            parts.append(f"{agent_id} is active across {_pluralize(len(entities_written), 'entity')}.")

        if collaboration_score > 0:
            parts.append(
                f"Collaborates with other agents on "
                f"{_pluralize(collaboration_score, 'shared entity')}."
            )

        if solo_entities and len(solo_entities) <= 3:
            solo_str = ", ".join(e.split("/")[-1] for e in solo_entities)
            parts.append(f"Sole contributor to {solo_str}.")
        elif solo_entities:
            parts.append(
                f"Sole contributor to {_pluralize(len(solo_entities), 'entity')} — "
                f"knowledge bus factor is 1."
            )

        if outcomes_committed > 0:
            ratio = outcomes_committed / total_entries if total_entries else 0
            parts.append(
                f"{ratio:.0%} of knowledge is validated by outcomes."
            )
        elif total_entries > 5:
            parts.append("No outcomes recorded yet — consider committing outcomes to validate knowledge.")

        if risk_entries:
            parts.append(f"Tracking {_pluralize(len(risk_entries), 'active risk')}.")

        beliefs = memory_types.get("belief", 0)
        if beliefs > 0:
            parts.append(
                f"{_pluralize(beliefs, 'entry')} marked as beliefs (unvalidated hypotheses)."
            )

        if recent_entities:
            recent_str = ", ".join(e.split("/")[-1] for e in recent_entities[:3])
            parts.append(f"Recently focused on: {recent_str}.")

        return " ".join(parts)

    def compile_source(
        self, source_id: str, adapter: PostgresAdapter, namespace: str, branch: str = "main",
    ) -> Digest | None:
        from amfs_core.models import SearchQuery

        webhook_entries = adapter.search(SearchQuery(
            agent_id=f"webhook/{source_id}",
            limit=1000,
            sort_by="recency",
        ), branch=branch)
        external_entries = adapter.search(SearchQuery(
            agent_id=f"external/{source_id}",
            limit=1000,
            sort_by="recency",
        ), branch=branch)
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

    def compile_trace_patterns(
        self, entity_path: str, adapter: "PostgresAdapter", namespace: str, branch: str = "main",
    ) -> Digest | None:
        """Delegate to TracePatternExtractor for rule-based pattern mining.

        Also persists extracted pattern entries as first-class memories so
        they appear in search results and briefings.
        """
        from amfs_cortex.trace_miner import TracePatternExtractor
        extractor = TracePatternExtractor(adapter, namespace=namespace)

        pattern_entries = extractor.extract_patterns(entity_path, branch=branch)
        for entry in pattern_entries:
            try:
                entry.branch = branch
                adapter.write(entry)
            except Exception:
                logger.debug("Failed to persist pattern entry %s", entry.key, exc_info=True)

        return extractor.compile_trace_digest(entity_path, branch=branch)

    def compile_connection_map(
        self, entity_path: str, adapter: "PostgresAdapter", namespace: str, branch: str = "main",
    ) -> Digest | None:
        """Compile a CONNECTION_MAP digest from the knowledge graph for an entity."""
        edges = adapter.list_graph_edges(
            entity=entity_path,
            namespace=namespace,
            branch=branch,
            limit=500,
        )
        if not edges:
            return None

        by_relation: dict[str, list] = {}
        connected_entities: set[str] = set()
        for edge in edges:
            by_relation.setdefault(edge.relation, []).append(edge)
            if edge.source_entity != entity_path:
                connected_entities.add(edge.source_entity)
            if edge.target_entity != entity_path:
                connected_entities.add(edge.target_entity)

        relation_summary: dict[str, Any] = {}
        for rel, rel_edges in by_relation.items():
            targets = set()
            for e in rel_edges:
                targets.add(e.target_entity if e.source_entity == entity_path else e.source_entity)
            relation_summary[rel] = {
                "count": len(rel_edges),
                "entities": sorted(targets)[:10],
                "avg_confidence": round(sum(e.confidence for e in rel_edges) / len(rel_edges), 3),
            }

        parts: list[str] = []
        parts.append(
            f"{entity_path} has {_pluralize(len(edges), 'graph edge')} "
            f"connecting to {_pluralize(len(connected_entities), 'entity')}."
        )
        for rel, info in relation_summary.items():
            parts.append(
                f"  {rel}: {_pluralize(info['count'], 'edge')} "
                f"to {', '.join(info['entities'][:5])}."
            )

        return Digest(
            digest_type=DigestType.CONNECTION_MAP,
            scope=entity_path,
            summary={
                "narrative": " ".join(parts),
                "total_edges": len(edges),
                "connected_entities": sorted(connected_entities),
                "relations": relation_summary,
            },
            entry_count=len(edges),
            source_agents=[],
            compiled_at=datetime.now(timezone.utc),
            namespace=namespace,
            branch=branch,
        )
