"""Who is the authority on a topic, and why — one scorer, four consumers.

The dashboard's "Who Knows What" page, the briefing's ``who_to_ask`` block,
retrieval's author weighting, and agent discovery all need to answer the same
question: of the agents that have written here, whose memory should be trusted
first? Answering it in four places guarantees four different answers, and the
one users notice is the explanation — a dashboard that says an agent is the
primary author while the briefing recommends someone else reads as a bug.

So the score and the human sentence explaining it are produced together, here,
and every surface reuses both verbatim.

This lives in ``amfs_core`` rather than in Cortex because ``amfs-pro-api``
depends on core but not on cortex, and the Pro expertise endpoint is one of the
four consumers. Nothing here touches storage: it is a pure function of the rows
``AdapterABC.agent_entity_stats()`` returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# How the four signals trade off. Validated outcomes lead deliberately: an
# agent whose memories demonstrably shaped a successful outcome is a better
# person to ask than one who simply wrote more, and volume alone is the signal
# most easily gamed by a chatty agent.
_W_SHARE = 0.30
_W_VALIDATED = 0.30
_W_REUSE = 0.20
_W_VOLUME = 0.20

# Recency scales the blend rather than adding to it, because staleness
# discounts every other signal at once: an agent that was authoritative a year
# ago and has not written since is less useful on all four counts, not just on
# recency. Floored so a dormant author is demoted, never erased — on a quiet
# topic the only author may not have written in months and is still the answer.
_RECENCY_HALF_LIFE_DAYS = 30.0
_RECENCY_FLOOR = 0.5

# Share thresholds for the coarse label the UI colours cells by.
_PRIMARY_SHARE = 0.6
_CONTRIBUTOR_SHARE = 0.2


@dataclass
class AuthorRank:
    """One agent's standing on one entity, with the sentence that explains it."""

    agent_id: str
    entity_path: str
    entry_count: int
    share: float
    validated_outcomes: int
    total_recalls: int
    last_written: datetime | None
    score: float
    strength: str
    reason: str
    top_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "entity_path": self.entity_path,
            "entry_count": self.entry_count,
            "share": round(self.share, 4),
            "validated_outcomes": self.validated_outcomes,
            "total_recalls": self.total_recalls,
            "last_written": self.last_written,
            "score": round(self.score, 4),
            "strength": self.strength,
            "reason": self.reason,
            "top_keys": list(self.top_keys),
        }


def _as_aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _recency_multiplier(last_written: datetime | None, now: datetime) -> float:
    if last_written is None:
        return _RECENCY_FLOOR
    days = max((now - last_written).total_seconds() / 86400.0, 0.0)
    decayed = 0.5 ** (days / _RECENCY_HALF_LIFE_DAYS)
    return _RECENCY_FLOOR + (1.0 - _RECENCY_FLOOR) * decayed


def _normalise(value: float, peak: float) -> float:
    """Scale against the strongest author on this topic, not a global maximum.

    Authority is a comparison between the agents that wrote here. Normalising
    against an account-wide maximum would make every author on a quiet topic
    look weak, which is the flaw in the old page's global-max colour scale.
    """
    if peak <= 0:
        return 0.0
    return min(value / peak, 1.0)


def humanise_age(last_written: datetime | None, now: datetime | None = None) -> str:
    """"2 days ago" — the recency half of an authority explanation."""
    if last_written is None:
        return "at an unknown time"
    now = now or datetime.now(timezone.utc)
    seconds = max((now - last_written).total_seconds(), 0.0)
    if seconds < 3600:
        return "in the last hour"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(seconds // 86400)
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def _build_reason(
    *,
    entry_count: int,
    validated: int,
    recalls: int,
    last_written: datetime | None,
    now: datetime,
) -> str:
    """The one sentence every surface shows. Only claims what was measured."""
    parts = [f"wrote {entry_count} memor{'ies' if entry_count != 1 else 'y'} here"]
    if validated:
        parts.append(
            f"{validated} with validated outcome{'s' if validated != 1 else ''}"
        )
    if recalls:
        parts.append(
            f"recalled {recalls} time{'s' if recalls != 1 else ''} by other sessions"
        )
    parts.append(f"active {humanise_age(last_written, now)}")
    return ", ".join(parts)


def _merge_by_agent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse an agent's rows into one, summing its evidence.

    Rolling a subtree up gives one row per (agent, child path); left as they
    are, an agent that wrote across three child paths would be ranked three
    times and its evidence split between the copies.
    """
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        agent_id = str(row.get("agent_id") or "")
        acc = merged.get(agent_id)
        if acc is None:
            merged[agent_id] = {
                "agent_id": agent_id,
                "entry_count": int(row.get("entry_count") or 0),
                "outcome_linked_count": int(row.get("outcome_linked_count") or 0),
                "total_recalls": int(row.get("total_recalls") or 0),
                "last_written": row.get("last_written"),
            }
            continue
        acc["entry_count"] += int(row.get("entry_count") or 0)
        acc["outcome_linked_count"] += int(row.get("outcome_linked_count") or 0)
        acc["total_recalls"] += int(row.get("total_recalls") or 0)
        newer = _as_aware(row.get("last_written"))
        current = _as_aware(acc.get("last_written"))
        if newer and (current is None or newer > current):
            acc["last_written"] = row.get("last_written")
    return list(merged.values())


def rank_authors(
    entity_path: str,
    *,
    stats: list[dict[str, Any]],
    limit: int | None = 3,
    now: datetime | None = None,
    top_keys: dict[str, list[str]] | None = None,
    include_descendants: bool = False,
) -> list[AuthorRank]:
    """Rank the agents that have written to *entity_path*, strongest first.

    *stats* are rows from ``AdapterABC.agent_entity_stats()``; rows for other
    entities are ignored, so the caller can pass the whole account's stats once
    and rank many topics from it without a query per topic.

    *include_descendants* folds rows beneath *entity_path* into it, so an agent
    that has only written to ``a/b/tokens`` still ranks as an authority on
    ``a/b``. Callers that treat each path as its own column — the expertise
    grid — want the default, where a child path is a topic in its own right.
    Callers asking "who should I ask about this area" want it on, and it
    matches the prefix scoping ``agent_entity_stats`` already applies to the
    query, so subtree rows are ranked rather than fetched and thrown away.

    *top_keys* optionally maps an agent id to its most relevant keys on this
    entity, so a recommendation can name what to read rather than only whom to
    ask. *limit* of ``None`` returns every author.
    """
    now = now or datetime.now(timezone.utc)
    prefix = entity_path.rstrip("/") + "/"
    matched = [
        r for r in stats
        if r.get("entity_path") == entity_path
        or (
            include_descendants
            and str(r.get("entity_path") or "").startswith(prefix)
        )
    ]
    if not matched:
        return []

    rows = _merge_by_agent(matched)

    total_entries = sum(int(r.get("entry_count") or 0) for r in rows)
    peak_validated = max(float(r.get("outcome_linked_count") or 0) for r in rows)
    peak_recalls = max(float(r.get("total_recalls") or 0) for r in rows)
    peak_volume = max(float(r.get("entry_count") or 0) for r in rows)

    ranked: list[AuthorRank] = []
    for row in rows:
        entry_count = int(row.get("entry_count") or 0)
        validated = int(row.get("outcome_linked_count") or 0)
        recalls = int(row.get("total_recalls") or 0)
        last_written = _as_aware(row.get("last_written"))

        share = entry_count / total_entries if total_entries else 0.0
        # Log-scaled so the gap between 1 and 10 memories counts for more than
        # the gap between 100 and 110 — past a point, more of the same tells
        # you nothing new about who to ask.
        volume = math.log1p(entry_count) / math.log1p(peak_volume) if peak_volume else 0.0

        blended = (
            _W_SHARE * share
            + _W_VALIDATED * _normalise(validated, peak_validated)
            + _W_REUSE * _normalise(recalls, peak_recalls)
            + _W_VOLUME * min(volume, 1.0)
        )
        score = blended * _recency_multiplier(last_written, now)

        if share >= _PRIMARY_SHARE:
            strength = "primary"
        elif share >= _CONTRIBUTOR_SHARE:
            strength = "contributor"
        else:
            strength = "touched"

        ranked.append(
            AuthorRank(
                agent_id=str(row.get("agent_id") or ""),
                entity_path=entity_path,
                entry_count=entry_count,
                share=share,
                validated_outcomes=validated,
                total_recalls=recalls,
                last_written=last_written,
                score=score,
                strength=strength,
                reason=_build_reason(
                    entry_count=entry_count,
                    validated=validated,
                    recalls=recalls,
                    last_written=last_written,
                    now=now,
                ),
                top_keys=list((top_keys or {}).get(str(row.get("agent_id") or ""), [])),
            )
        )

    # Ties broken by entry count then agent id so the order is stable across
    # calls — an ordering that reshuffles between renders reads as churn.
    ranked.sort(key=lambda a: (-a.score, -a.entry_count, a.agent_id))
    return ranked if limit is None else ranked[:limit]


def is_siloed(ranked: list[AuthorRank]) -> bool:
    """True when exactly one agent has written here.

    Callers showing this to a user whose view is filtered must word it as "one
    of your agents" — the caller cannot see agents it has no access to, so an
    account-wide claim would be one it cannot support.
    """
    return len(ranked) == 1
