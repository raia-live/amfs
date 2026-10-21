"""The ledger: the file a loop's gate leaves behind, and what the scorecard reads.

A run writes four files into its results directory, and a loop that cannot be
driven live writes the same four itself and is scored with
``python -m benchmarks.gate score``:

``probes.jsonl``
    One record per injected probe::

        {"probe_id": "gp-001", "kind": "placebo", "mode": "dry_run",
         "injected_at": "2026-10-16T04:20:00+00:00", "closed_at": "2026-10-16T04:43:10+00:00",
         "status": "rejected", "decided_by": "t1", "would_have_shipped": false,
         "stages": [{"tier": "t1", "verdict": "rejected", "at": "...", "note": "..."}],
         "probe": {... the draft, optional ...}, "detail": {... loop-specific, optional ...}}

    ``kind`` is one of ``placebo | poison | reward_hack | revert``; ``status`` one of
    ``open | rejected | accepted | undecided | expired``; ``decided_by`` names the
    tier that closed it (``t1`` the loop's offline replay, ``t2`` the agent's own
    replay, ``canary`` live traffic, ``human`` a reviewer — any label the loop
    uses, consistently); ``would_have_shipped`` is whether the loop's policy would
    have merged the probe had it not been a probe (a loop that cannot tell
    writes ``false``); ``stages`` is what each tier said, in order.

``aa.json``
    ``{"nominal": 0.075, "window_days": 14, "ran_at": "...", "trials": [{"decidable": true,
    "false_positive": false, "graded_at_decision": 58, "detail": "..."}, ...]}`` or
    ``null`` when the loop has no A/A.

``reward.json``
    ``[{"id": "fix-005", "title": "...", "shipped_at": "...",
    "proxy_before": {"n": 120, "hits": 14}, "proxy_after": {"n": 118, "hits": 2},
    "downstream_before": {"n": 60, "hits": 43}, "downstream_after": {"n": 58, "hits": 50},
    "self_report_before": {"n": 120, "hits": 96}, "self_report_after": {"n": 118, "hits": 104}},
    ...]`` or ``null``. ``proxy`` counts the failures the loop's own judge saw per
    run (a fall is the gain); ``downstream`` counts successes per run by a label
    the loop did not write; ``self_report`` counts the agent's own claimed
    successes per run and is optional.

``scorecard.json`` / ``scorecard.md``
    What :mod:`.scorecard` made of the three.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .protocol import AAResult, AATrial, Counts, Decision, Probe, RewardReading

PROBES_FILE = "probes.jsonl"
AA_FILE = "aa.json"
REWARD_FILE = "reward.json"
SCORECARD_JSON = "scorecard.json"
SCORECARD_MD = "scorecard.md"
RUN_FILE = "run.json"


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str) and v:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    return None


def probe_record(
    probe_id: str, kind: str, decision: Decision, *, probe: Probe | None = None,
) -> dict[str, Any]:
    """One ``probes.jsonl`` record from a loop's :class:`Decision`."""
    return {
        "probe_id": probe_id,
        "kind": kind,
        "mode": decision.mode,
        "injected_at": _iso(decision.injected_at),
        "closed_at": _iso(decision.closed_at),
        "status": decision.status.value,
        "decided_by": decision.decided_by,
        "would_have_shipped": bool(decision.would_have_shipped),
        "stages": [s.as_dict() for s in decision.stages],
        "probe": probe.as_dict() if probe else None,
        "detail": dict(decision.detail),
    }


def write_probes(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")


def read_probes(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def aa_to_json(result: AAResult | None) -> Any:
    if result is None:
        return None
    return {
        "nominal": result.nominal, "window_days": result.window_days,
        "ran_at": _iso(result.ran_at), "detail": dict(result.detail),
        "trials": [
            {"decidable": t.decidable, "false_positive": t.false_positive,
             "graded_at_decision": t.graded_at_decision, "detail": t.detail}
            for t in result.trials
        ],
    }


def aa_from_json(data: Any) -> AAResult | None:
    if not data:
        return None
    return AAResult(
        trials=tuple(
            AATrial(decidable=bool(t.get("decidable")),
                    false_positive=bool(t.get("false_positive")),
                    graded_at_decision=t.get("graded_at_decision"), detail=t.get("detail"))
            for t in data.get("trials") or []
        ),
        nominal=float(data["nominal"]),
        window_days=data.get("window_days"),
        ran_at=_dt(data.get("ran_at")),
        detail=dict(data.get("detail") or {}),
    )


def _counts_json(c: Counts | None) -> dict[str, int] | None:
    return None if c is None else {"n": c.n, "hits": c.hits}


def _counts(d: Any) -> Counts | None:
    if not d:
        return None
    return Counts(n=int(d.get("n", 0)), hits=int(d.get("hits", 0)))


def reward_to_json(readings: Iterable[RewardReading] | None) -> Any:
    if readings is None:
        return None
    return [
        {
            "id": r.id, "title": r.title, "shipped_at": _iso(r.shipped_at),
            "proxy_before": _counts_json(r.proxy_before),
            "proxy_after": _counts_json(r.proxy_after),
            "downstream_before": _counts_json(r.downstream_before),
            "downstream_after": _counts_json(r.downstream_after),
            "self_report_before": _counts_json(r.self_report_before),
            "self_report_after": _counts_json(r.self_report_after),
            "detail": dict(r.detail),
        }
        for r in readings
    ]


def reward_from_json(data: Any) -> list[RewardReading] | None:
    if data is None:
        return None
    out: list[RewardReading] = []
    for d in data:
        out.append(RewardReading(
            id=str(d["id"]), title=str(d.get("title") or d["id"]),
            shipped_at=_dt(d.get("shipped_at")),
            proxy_before=_counts(d.get("proxy_before")) or Counts(0, 0),
            proxy_after=_counts(d.get("proxy_after")) or Counts(0, 0),
            downstream_before=_counts(d.get("downstream_before")) or Counts(0, 0),
            downstream_after=_counts(d.get("downstream_after")) or Counts(0, 0),
            self_report_before=_counts(d.get("self_report_before")),
            self_report_after=_counts(d.get("self_report_after")),
            detail=dict(d.get("detail") or {}),
        ))
    return out


def write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


__all__ = [
    "AA_FILE",
    "PROBES_FILE",
    "REWARD_FILE",
    "RUN_FILE",
    "SCORECARD_JSON",
    "SCORECARD_MD",
    "aa_from_json",
    "aa_to_json",
    "probe_record",
    "read_json",
    "read_probes",
    "reward_from_json",
    "reward_to_json",
    "write_json",
    "write_probes",
]
