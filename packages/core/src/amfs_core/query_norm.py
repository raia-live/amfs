"""Temporal query normalization for retrieval.

Natural-language recall queries frequently carry temporal intent ("what did we
ship *yesterday*", "the plugin delivered *last week*"). Left in the query text,
those words are embedded as topical noise — they drift the query vector away
from the actual subject and never become the recency signal the user meant.

``normalize_temporal`` splits that intent out: it returns the topical remainder
(safe to embed / feed to full-text search) plus a recency signal the caller can
turn into a stronger recency weight and/or a ``written_after`` lower bound.

Pure/deterministic and dependency-free so it is trivially unit-testable and can
be shared by the OSS ``/api/v1/retrieve`` path and the Pro retriever.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class TemporalNorm:
    """Result of stripping temporal intent from a query.

    - ``topical``: query text with temporal phrases removed (falls back to the
      original text when stripping would leave it empty, e.g. query was only
      "yesterday").
    - ``matched``: whether any temporal expression was found.
    - ``written_after``: optional lower bound on ``written_at`` implied by the
      phrase (timezone-aware UTC). Callers should treat this as a soft signal.
    - ``recency_weight_boost``: multiplier the caller can apply to its recency
      weight (1.0 = no change). Only > 1.0 when a temporal phrase matched.
    """

    topical: str
    matched: bool
    written_after: datetime | None
    recency_weight_boost: float


# Ordered longest/most-specific first so "last week" wins over "week", etc.
# Each entry maps a regex to the number of days back the window should cover.
_PHRASES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bjust now\b", re.I), 1),
    (re.compile(r"\bearlier today\b", re.I), 1),
    (re.compile(r"\bthis morning\b", re.I), 1),
    (re.compile(r"\bthis afternoon\b", re.I), 1),
    (re.compile(r"\btonight\b", re.I), 1),
    (re.compile(r"\btoday\b", re.I), 1),
    (re.compile(r"\byesterday\b", re.I), 2),
    (re.compile(r"\bthe day before yesterday\b", re.I), 3),
    (re.compile(r"\blast week\b", re.I), 14),
    (re.compile(r"\bthis week\b", re.I), 7),
    (re.compile(r"\bpast week\b", re.I), 7),
    (re.compile(r"\blast month\b", re.I), 60),
    (re.compile(r"\bthis month\b", re.I), 30),
    (re.compile(r"\bpast month\b", re.I), 30),
    (re.compile(r"\b(?:lately|recently|recent|latest|newest)\b", re.I), 7),
]

# Numeric phrases: "3 days ago", "past 2 weeks", "last 5 months".
_NUMERIC = re.compile(
    r"\b(?:in the )?(?:past|last)?\s*(\d{1,3})\s*(day|days|week|weeks|month|months)\s*(?:ago)?\b",
    re.I,
)
_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}


def normalize_temporal(query: str, *, now: datetime | None = None) -> TemporalNorm:
    """Extract temporal intent from ``query``.

    ``now`` is injectable for deterministic tests; defaults to current UTC.
    """
    if not query or not query.strip():
        return TemporalNorm(topical=query or "", matched=False, written_after=None, recency_weight_boost=1.0)

    now = now or datetime.now(timezone.utc)
    remaining = query
    max_days: float | None = None

    # Numeric phrases first (most specific), then keyword phrases.
    def _consume_numeric(m: re.Match[str]) -> str:
        nonlocal max_days
        n = int(m.group(1))
        days = n * _UNIT_DAYS[m.group(2).lower()]
        # +1 day buffer so an entry written "3 days ago" isn't excluded by a
        # sub-day boundary difference.
        days += 1
        max_days = days if max_days is None else max(max_days, days)
        return " "

    remaining = _NUMERIC.sub(_consume_numeric, remaining)

    for pattern, days in _PHRASES:
        if pattern.search(remaining):
            max_days = float(days) if max_days is None else max(max_days, float(days))
            remaining = pattern.sub(" ", remaining)

    matched = max_days is not None

    # Clean up leftover connective words / whitespace left by stripping.
    topical = re.sub(r"\s+", " ", remaining).strip()
    topical = re.sub(r"^(?:the|from|in|on|since|of|for|about)\s+", "", topical, flags=re.I).strip()
    topical = re.sub(r"\s+(?:the|from|in|on|since|of|for|about)$", "", topical, flags=re.I).strip()
    if not topical:
        # Query was purely temporal — keep original so we never embed "".
        topical = query.strip()

    if not matched:
        return TemporalNorm(topical=topical, matched=False, written_after=None, recency_weight_boost=1.0)

    written_after = now - timedelta(days=float(max_days))
    # Tighter windows (more explicit recency intent) get a bigger recency boost,
    # capped so recency never fully dominates semantic relevance.
    boost = 2.0 if max_days <= 14 else 1.5
    return TemporalNorm(
        topical=topical,
        matched=True,
        written_after=written_after,
        recency_weight_boost=boost,
    )
