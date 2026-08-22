"""Pure aggregate computations over in-memory entries/traces.

Shared by AdapterABC default implementations and by HTTP handlers that must
aggregate over an already visibility-filtered list (where SQL can't express
the per-user room semantics). Keeping one implementation guarantees the
Python fallback and the filtered path always agree.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amfs_core.models import DecisionTrace, MemoryEntry


# ── Recall-weighted "avoided re-research" token credit ─────────────────
# Per reuse we credit the measured content size (~4 chars/token), clamped:
# - floor: even a tiny memory replaces *some* lookup work, but a one-line
#   preference must not be credited like a re-derived investigation;
# - ceil: a single huge payload can't inflate the estimate — re-deriving one
#   memory realistically costs a few file reads + searches, no more.
# Dashboards and the MCP value ledger surface these numbers to users, so the
# same constants are mirrored in the Postgres adapter SQL, the dashboard's
# value-metrics.ts, and the MCP server's value_ledger.py. Keep them in sync.
RECALL_TOKENS_FLOOR = 200
RECALL_TOKENS_CEIL = 4000
CHARS_PER_TOKEN = 4

# How many hits of a query-driven lookup (retrieve/search) bump recall_count.
# One: per-entry reuse should count the memory an agent took, not the candidate
# list it was shown. Crediting the top three inflated the number roughly
# threefold — one real session made 4 lookups, was credited 16 reuses, and
# exactly 1 of those memories changed what the agent did.
#
# The Pro MCP server's chat recap (value_ledger.py) deliberately no longer
# mirrors this: it estimates what a single CALL delivered, so it credits every
# entry returned. The two answer different questions — this one is the
# dashboard's per-entry source of truth — and are not expected to match.
REUSE_CREDIT_K = 1


def recall_tokens_saved(entry: "MemoryEntry") -> int:
    """Estimated tokens of re-research avoided by this entry's recalls."""
    recalls = entry.recall_count or 0
    if recalls <= 0:
        return 0
    try:
        chars = len(json.dumps(entry.value, default=str))
    except (TypeError, ValueError):
        chars = len(str(entry.value))
    per_recall = min(max(chars // CHARS_PER_TOKEN, RECALL_TOKENS_FLOOR), RECALL_TOKENS_CEIL)
    return recalls * per_recall


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
                "total_recalls": sum(e.recall_count for e in group),
                "recalled_tokens_saved": sum(recall_tokens_saved(e) for e in group),
            }
        )
    summaries.sort(key=lambda s: s["last_updated"], reverse=True)
    return summaries


def agent_entity_stats_from_entries(entries: "list[MemoryEntry]") -> list[dict]:
    """Group entries per (agent, entity) pair — who has written where.

    One row per pair, carrying the evidence an authority ranking needs:
    volume (``entry_count``), reuse (``total_recalls``), validation
    (``outcome_linked_count``), recency (``last_written``) and stated certainty
    (``avg_confidence``). Scoring lives in ``authority.py``; this only counts.
    """
    grouped: dict[tuple[str, str], list] = {}
    for entry in entries:
        key = (entry.provenance.agent_id, entry.entity_path)
        grouped.setdefault(key, []).append(entry)

    rows: list[dict] = []
    for (agent_id, entity_path), group in grouped.items():
        rows.append(
            {
                "agent_id": agent_id,
                "entity_path": entity_path,
                "entry_count": len(group),
                "avg_confidence": sum(e.confidence for e in group) / len(group),
                "last_written": max(e.provenance.written_at for e in group),
                "first_written": min(e.provenance.written_at for e in group),
                "total_recalls": sum(e.recall_count or 0 for e in group),
                "outcome_linked_count": sum(
                    1 for e in group if (e.outcome_count or 0) > 0
                ),
            }
        )
    rows.sort(key=lambda r: (r["entity_path"], -r["entry_count"], r["agent_id"]))
    return rows


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
    recalled_tokens = 0
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
        recalled_tokens += recall_tokens_saved(entry)
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
        "recalled_tokens_saved": recalled_tokens,
        "entries_this_week": this_week,
        "entries_last_week": last_week,
        # Entries carry only a recall counter, not recall timestamps, so the
        # pure-Python path can't compute weekly reuse deltas. None (vs 0) lets
        # clients distinguish "unknown" from "no reuse" and hide the trend.
        "recalls_this_week": None,
        "recalls_last_week": None,
        "memory_type_counts": memory_type_counts,
    }


# ── Bulk aggregation + schema profiling over structured entry values ───
# These power three things that share one implementation so they can never
# disagree: the server-side /api/v1/aggregate endpoint, the MCP amfs_aggregate
# local fallback, and the room-index schema profile carried on entity digests.
# The input is a list of MemoryEntry whose values are records (dicts) or JSON
# arrays of records; the caller is responsible for having already scoped and
# visibility-filtered the entries.

AGGREGATE_OPS = ("count", "sum", "mean", "min", "max", "stats")
_DEFAULT_MAX_ENUM = 25


def coerce_value(value: object) -> object:
    """Return a structured value, parsing JSON strings when possible.

    Entry values arrive either already parsed (Postgres JSONB / the HTTP
    response shape) or as JSON strings; callers must handle both, so this is
    the single coercion point.
    """
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _get_path(row: object, path: str) -> object:
    """Dotted-path getter over dict rows; returns None if any segment misses."""
    cur = row
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def iter_rows(entries: "list[MemoryEntry]", row_path: str | None = None) -> list:
    """Flatten entries into the records to aggregate over.

    Without ``row_path`` each entry's coerced value is one row (or, if it is a
    list, each element is a row). With ``row_path`` the value at that dotted
    path is expected to be a list — one row per element — which is how a batch
    entry like ``{"listings": [...]}`` fans out into per-listing rows.
    """
    rows: list = []
    for entry in entries:
        val = coerce_value(getattr(entry, "value", entry))
        if row_path:
            target = _get_path(val, row_path) if isinstance(val, dict) else None
            if isinstance(target, list):
                rows.extend(target)
            elif target is not None:
                rows.append(target)
        elif isinstance(val, list):
            rows.extend(val)
        else:
            rows.append(val)
    return rows


def _to_number(x: object) -> float | None:
    """Coerce to a number, treating bools and non-numeric strings as absent."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _numeric_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "sum": 0, "mean": None, "min": None, "max": None}
    total = sum(values)
    return {
        "n": len(values),
        "sum": total,
        "mean": total / len(values),
        "min": min(values),
        "max": max(values),
    }


def _group_key(g: object) -> str:
    if g is None:
        return "null"
    if isinstance(g, (str, int, float, bool)):
        return str(g)
    return json.dumps(g, default=str, sort_keys=True)


def aggregate_entries(
    entries: "list[MemoryEntry]",
    *,
    op: str,
    field: str | None = None,
    group_by: str | None = None,
    row_path: str | None = None,
) -> dict:
    """Compute one aggregate over the records inside ``entries``.

    ``op`` is one of AGGREGATE_OPS. Numeric ops (sum/mean/min/max/stats) require
    ``field``; ``count`` does not. ``group_by`` produces one result per distinct
    value of that field. ``row_path`` flattens a list-valued field into rows
    first. Non-numeric field values are ignored for numeric ops (never coerced
    to zero), so ``n`` reports how many rows actually contributed.
    """
    if op not in AGGREGATE_OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {list(AGGREGATE_OPS)}")
    if op != "count" and not field:
        raise ValueError(f"op {op!r} requires a field")

    rows = iter_rows(entries, row_path)

    def _compute(group_rows: list) -> dict:
        if op == "count":
            return {"count": len(group_rows)}
        values = [
            n for r in group_rows
            if (n := _to_number(_get_path(r, field) if isinstance(r, dict) else None)) is not None
        ]
        summary = _numeric_summary(values)
        if op == "stats":
            return summary
        return {op: summary[op], "n": summary["n"]}

    result: dict = {
        "op": op,
        "field": field,
        "row_path": row_path,
        "total_rows": len(rows),
    }
    if group_by:
        groups: dict[str, list] = {}
        for r in rows:
            g = _get_path(r, group_by) if isinstance(r, dict) else None
            groups.setdefault(_group_key(g), []).append(r)
        result["group_by"] = group_by
        result["groups"] = [
            {"key": key, **_compute(group_rows)}
            for key, group_rows in sorted(groups.items(), key=lambda kv: kv[0])
        ]
    else:
        result.update(_compute(rows))
    return result


def _type_name(v: object) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _row_path_candidates(entries: "list[MemoryEntry]") -> list[dict]:
    """Detect list-of-object fields at the entry-value level.

    These are the ``row_path`` values an agent should aggregate over (e.g. a
    batch entry ``{"listings": [ {...}, ... ]}`` yields candidate ``listings``).
    """
    seen: dict[str, dict] = {}
    for entry in entries:
        val = coerce_value(getattr(entry, "value", entry))
        if not isinstance(val, dict):
            continue
        for k, v in val.items():
            if isinstance(v, list) and v and any(isinstance(el, dict) for el in v):
                info = seen.setdefault(k, {"field": k, "entries": 0, "total_rows": 0})
                info["entries"] += 1
                info["total_rows"] += len(v)
    return sorted(seen.values(), key=lambda c: c["total_rows"], reverse=True)


def room_schema_profile(
    entries: "list[MemoryEntry]",
    *,
    row_path: str | None = None,
    max_enum: int = _DEFAULT_MAX_ENUM,
) -> dict:
    """A compact, always-cheap map of what a room/entity contains.

    Returns the fields present across records, their inferred types, non-null
    counts, numeric ranges, low-cardinality value sets (enums), per-key record
    counts, and the ``row_path`` candidates an agent should aggregate over. The
    output is a few KB regardless of how many records the room holds, so an
    agent reads it once and knows exactly which amfs_aggregate / amfs_export
    calls to make instead of pulling everything to find out.
    """
    rows = iter_rows(entries, row_path)

    fields: dict[str, dict] = {}

    def _observe(name: str, value: object) -> None:
        info = fields.setdefault(
            name,
            {"types": set(), "non_null": 0, "numbers": [], "values": set()},
        )
        if value is None:
            info["types"].add("null")
            return
        info["non_null"] += 1
        info["types"].add(_type_name(value))
        num = _to_number(value)
        if num is not None:
            info["numbers"].append(num)
        if info["values"] is None:
            return
        if isinstance(value, (str, int, float, bool)):
            info["values"].add(value)
            if len(info["values"]) > max_enum:
                info["values"] = None
        else:
            info["values"] = None

    for r in rows:
        if isinstance(r, dict):
            for k, v in r.items():
                _observe(k, v)
        else:
            _observe("_value", r)

    field_out: list[dict] = []
    for name, info in sorted(fields.items()):
        item: dict = {
            "field": name,
            "types": sorted(info["types"]),
            "non_null": info["non_null"],
        }
        if info["numbers"]:
            item["numeric_range"] = {
                "min": min(info["numbers"]),
                "max": max(info["numbers"]),
            }
        if info["values"] is None:
            item["cardinality"] = f">{max_enum}"
        else:
            item["cardinality"] = len(info["values"])
            item["values"] = sorted(info["values"], key=lambda x: str(x))
        field_out.append(item)

    per_key: dict[str, int] = {}
    for entry in entries:
        key = getattr(entry, "key", None)
        if key is not None:
            per_key[key] = per_key.get(key, 0) + 1

    return {
        "total_entries": len(entries),
        "total_rows": len(rows),
        "row_path": row_path,
        "fields": field_out,
        "record_counts_by_key": per_key,
        "row_path_candidates": _row_path_candidates(entries),
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
