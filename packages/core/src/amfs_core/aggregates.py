"""Pure aggregate computations over in-memory entries/traces.

Shared by AdapterABC default implementations and by HTTP handlers that must
aggregate over an already visibility-filtered list (where SQL can't express
the per-user room semantics). Keeping one implementation guarantees the
Python fallback and the filtered path always agree.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amfs_core.models import DecisionTrace, MemoryEntry


def entity_summaries_from_entries(entries: "list[MemoryEntry]") -> list[dict]:
    """Group entries per entity path: count, avg confidence, last write, agents."""
    grouped: dict[str, list] = {}
    for entry in entries:
        grouped.setdefault(entry.entity_path, []).append(entry)

    summaries: list[dict] = []
    for entity_path, group in grouped.items():
        group_sorted = sorted(
            group, key=lambda e: e.provenance.written_at, reverse=True
        )
        summaries.append(
            {
                "entity_path": entity_path,
                "entry_count": len(group),
                "avg_confidence": sum(e.confidence for e in group) / len(group),
                "last_updated": group_sorted[0].provenance.written_at,
                "last_agent": group_sorted[0].provenance.agent_id,
                "agents": sorted({e.provenance.agent_id for e in group}),
                "hashed_count": sum(1 for e in group if e.content_hash),
            }
        )
    summaries.sort(key=lambda s: s["last_updated"], reverse=True)
    return summaries


def extended_stats_from_entries(entries: "list[MemoryEntry]") -> dict:
    """MemoryStats fields plus recall totals, weekly deltas, and type counts."""
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)

    agents: dict[str, int] = {}
    entities: dict[str, int] = {}
    memory_type_counts: dict[str, int] = {}
    confidences: list[float] = []
    outcome_linked = 0
    total_recalls = 0
    this_week = 0
    last_week = 0
    oldest: datetime | None = None
    newest: datetime | None = None

    for entry in entries:
        aid = entry.provenance.agent_id
        agents[aid] = agents.get(aid, 0) + 1
        entities[entry.entity_path] = entities.get(entry.entity_path, 0) + 1
        mt = str(getattr(entry.memory_type, "value", entry.memory_type))
        memory_type_counts[mt] = memory_type_counts.get(mt, 0) + 1
        confidences.append(entry.confidence)
        if entry.outcome_count > 0:
            outcome_linked += 1
        total_recalls += entry.recall_count
        written = entry.provenance.written_at
        if written >= week_ago:
            this_week += 1
        elif written >= two_weeks_ago:
            last_week += 1
        if oldest is None or written < oldest:
            oldest = written
        if newest is None or written > newest:
            newest = written

    return {
        "total_entries": len(entries),
        "total_entities": len(entities),
        "total_agents": len(agents),
        "agents": agents,
        "entities": entities,
        "confidence_avg": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "confidence_min": min(confidences) if confidences else 0.0,
        "confidence_max": max(confidences) if confidences else 0.0,
        "outcome_linked_count": outcome_linked,
        "oldest_entry_at": oldest,
        "newest_entry_at": newest,
        "total_recalls": total_recalls,
        "entries_this_week": this_week,
        "entries_last_week": last_week,
        "memory_type_counts": memory_type_counts,
    }


def share_stats_from_traces(
    traces: "list[DecisionTrace]",
    *,
    since: datetime | None = None,
    pair_limit: int = 20,
    agent_ids: list[str] | None = None,
) -> dict:
    """Cross-agent share counts: causal entries authored by a different agent."""
    allowed = set(agent_ids) if agent_ids is not None else None

    pair_counts: dict[tuple[str, str], int] = {}
    total = 0
    for trace in traces:
        if since is not None and trace.created_at < since:
            continue
        if allowed is not None and trace.agent_id not in allowed:
            continue
        for ce in trace.causal_entries:
            author = ce.written_by
            if not author or author == trace.agent_id:
                continue
            if allowed is not None and author not in allowed:
                continue
            total += 1
            key = (trace.agent_id, author)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    pairs = [
        {"reader": reader, "author": author, "count": count}
        for (reader, author), count in sorted(
            pair_counts.items(), key=lambda kv: kv[1], reverse=True
        )
    ]
    return {"total": total, "pairs": pairs[:pair_limit]}
