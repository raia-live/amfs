"""The number: is the agent getting better, and does it recover when the
world changes.

Everything else in the evidence model is about one entry. This is about the
agent — its sessions in order, each ending in an outcome, and the success
rate along that sequence. An engine that calls itself self-improving owes
this curve to whoever runs it: "episode 1: 40%, episode 12: 85%" is a claim
that can be checked, and "three sessions to recover after the rule changed"
is what it is worth when it is true.

Two readings, both pure functions of the outcome sequence:

*The curve.* The sessions are cut into ``episodes`` consecutive bins of equal
size and each bin's success rate is reported, with the first and latest
``window`` sessions for the headline. Bins rather than a rolling line because
a dashboard wants a dozen points, not two thousand, and equal-size bins keep
every point an honest average over the same number of sessions.

*Recovery.* A regime shift is the moment an entry the agent relied on was
discredited (``MemoryEntry.discredited_at``; ``amfs_core.evidence``). The
rate over the ``window`` sessions before that moment is the bar; the agent has
recovered when its rolling rate after the shift reaches the bar again, and
the number of sessions it took is the cost of the change. A shift with fewer
than ``MIN_SESSIONS`` sessions on either side is not measured — one success
is not a recovery and one session is not a baseline.

Verification is reported alongside (``verified_share``): what fraction of the
outcomes behind the curve were decided outside the agent. A curve built from
the agent's own word is a weaker claim than one CI graded, and the reader
should see which it is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from amfs_core.evidence import is_known, is_success

#: Sessions a rolling rate needs before it is a rate: the first and latest
#: readings, the pre-shift bar and the post-shift recovery all require it.
MIN_SESSIONS = 3
DEFAULT_WINDOW = 10
DEFAULT_EPISODES = 12
MAX_EPISODES = 60


def _session(row: Any) -> tuple[datetime, bool, str | None] | None:
    """``(at, success, verified_by)`` from a row the adapter returned — a
    mapping or an object with ``committed_at`` / ``outcome_type``."""
    if isinstance(row, Mapping):
        at = row.get("at") or row.get("committed_at")
        outcome_type = row.get("outcome_type")
        verified_by = row.get("verified_by")
    else:
        at = getattr(row, "committed_at", None) or getattr(row, "at", None)
        outcome_type = getattr(row, "outcome_type", None)
        verified_by = getattr(row, "verified_by", None)
    outcome_type = getattr(outcome_type, "value", outcome_type)
    if at is None or not is_known(outcome_type):
        return None
    verified = str(verified_by).strip() if verified_by else None
    return at, is_success(outcome_type), verified or None


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def _shift_at(shift: Any) -> datetime | None:
    if isinstance(shift, Mapping):
        return shift.get("at") or shift.get("discredited_at")
    return getattr(shift, "at", None) or getattr(shift, "discredited_at", None)


def _shift_label(shift: Any) -> dict[str, Any]:
    if isinstance(shift, Mapping):
        ep, key = shift.get("entity_path"), shift.get("key")
    else:
        ep, key = getattr(shift, "entity_path", None), getattr(shift, "key", None)
    return {"entity_path": ep, "key": key}


def recovery_after(
    sessions: list[tuple[datetime, bool, str | None]],
    at: datetime,
    *,
    window: int = DEFAULT_WINDOW,
    min_sessions: int = MIN_SESSIONS,
) -> dict[str, Any]:
    """How the agent fared after one shift at ``at``.

    ``bar`` is the rate over the up to ``window`` sessions before ``at``;
    ``sessions_to_recover`` is how many sessions after it the rolling rate
    (over the last ``window`` of them, once there are ``min_sessions``) took
    to reach the bar, or ``None`` while it has not. ``measured`` is False when
    there is nothing to measure, and ``reason`` says why: ``short_before`` or
    ``short_after`` when a side has fewer than ``min_sessions``, ``no_bar``
    when the agent was failing every session before the shift — a rolling
    rate is always at or above zero, so "back to 0 %" would be a recovery
    claimed for free. What follows a no-bar shift is improvement, read off
    the curve, not recovery.
    """
    before = [ok for (t, ok, _v) in sessions if t < at][-window:]
    after = [ok for (t, ok, _v) in sessions if t >= at]
    out: dict[str, Any] = {
        "at": at.isoformat(),
        "before": len(before),
        "after": len(after),
        "bar": _rate(before),
        "measured": False,
        "reason": None,
        "recovered": False,
        "sessions_to_recover": None,
        "rate_after": _rate(after[-window:]),
    }
    if len(before) < min_sessions:
        out["reason"] = "short_before"
        return out
    if len(after) < min_sessions:
        out["reason"] = "short_after"
        return out
    if not any(before):
        out["reason"] = "no_bar"
        return out
    out["measured"] = True
    bar = out["bar"]
    for k in range(min_sessions, len(after) + 1):
        rolling = after[max(0, k - window) : k]
        if _rate(rolling) >= bar:
            out["recovered"] = True
            out["sessions_to_recover"] = k
            break
    return out


def learning_curve(
    rows: Iterable[Any],
    *,
    shifts: Iterable[Any] = (),
    window: int = DEFAULT_WINDOW,
    episodes: int = DEFAULT_EPISODES,
    min_sessions: int = MIN_SESSIONS,
) -> dict[str, Any]:
    """The agent's success over its sessions, and its recovery after each
    shift. ``rows`` are outcomes (any order; sorted here by time). ``shifts``
    carry ``at`` (or ``discredited_at``) and, for the label, ``entity_path``
    and ``key``.
    """
    window = max(min_sessions, int(window or DEFAULT_WINDOW))
    episodes = max(1, min(int(episodes or DEFAULT_EPISODES), MAX_EPISODES))
    sessions = sorted(
        (s for s in (_session(r) for r in rows) if s is not None), key=lambda s: s[0]
    )
    n = len(sessions)
    flags = [ok for (_t, ok, _v) in sessions]
    out: dict[str, Any] = {
        "sessions": n,
        "successes": sum(flags),
        "success_rate": _rate(flags),
        "window": window,
        "first": None,
        "latest": None,
        "change": None,
        "curve": [],
        "recovery": {
            "shifts": 0,
            "measured": 0,
            "recovered": 0,
            "median_sessions_to_recover": None,
            "open": [],
            "events": [],
        },
        "verified_share": _rate([v is not None for (_t, _ok, v) in sessions]),
        "span": {
            "from": sessions[0][0].isoformat() if sessions else None,
            "to": sessions[-1][0].isoformat() if sessions else None,
        },
    }
    if n == 0:
        return out

    # The headline compares two windows that never overlap: the first and the
    # latest ``window`` sessions when there are at least two windows' worth,
    # shorter equal halves until then, and nothing while a half would be
    # under ``min_sessions`` — there is no "then" and "now" in five sessions.
    half = min(window, n // 2)
    if half >= min_sessions:
        first = flags[:half]
        latest = flags[-half:]
        out["first"] = {"sessions": half, "success_rate": _rate(first)}
        out["latest"] = {"sessions": half, "success_rate": _rate(latest)}
        out["change"] = round(out["latest"]["success_rate"] - out["first"]["success_rate"], 4)

    bins = min(episodes, n)
    size, rem = divmod(n, bins)
    start = 0
    for i in range(bins):
        stop = start + size + (1 if i < rem else 0)
        chunk = sessions[start:stop]
        out["curve"].append(
            {
                "episode": i + 1,
                "sessions": len(chunk),
                "success_rate": _rate([ok for (_t, ok, _v) in chunk]),
                "from": chunk[0][0].isoformat(),
                "to": chunk[-1][0].isoformat(),
            }
        )
        start = stop

    events: list[dict[str, Any]] = []
    for shift in shifts:
        at = _shift_at(shift)
        if at is None:
            continue
        event = {
            **_shift_label(shift),
            **recovery_after(sessions, at, window=window, min_sessions=min_sessions),
        }
        events.append(event)
    events.sort(key=lambda e: e["at"])
    measured = [e for e in events if e["measured"]]
    recovered = [e for e in measured if e["recovered"]]
    costs = sorted(e["sessions_to_recover"] for e in recovered)
    median = None
    if costs:
        mid = len(costs) // 2
        median = costs[mid] if len(costs) % 2 else (costs[mid - 1] + costs[mid]) / 2
    out["recovery"] = {
        "shifts": len(events),
        "measured": len(measured),
        "recovered": len(recovered),
        "median_sessions_to_recover": median,
        "open": [
            {"entity_path": e["entity_path"], "key": e["key"], "at": e["at"], "after": e["after"]}
            for e in measured
            if not e["recovered"]
        ],
        "events": events,
    }
    return out


__all__ = [
    "DEFAULT_EPISODES",
    "DEFAULT_WINDOW",
    "MAX_EPISODES",
    "MIN_SESSIONS",
    "learning_curve",
    "recovery_after",
]
