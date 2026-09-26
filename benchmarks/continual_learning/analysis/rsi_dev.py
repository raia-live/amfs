"""Development-seed evidence for prereg amendment (h): did the situation contract and the
engine fixes do what they predicted?

Reads one or more runs under ``results/`` and reports, per (run, family, arm, seed) cell
and pooled per (run, family, arm), the six quantities amendment (h) predicts:

  (i)   recommendation present on searched attempts (``rec_present``; predicted >= 98 %)
  (ii)  re-try of an action that already failed on the same task (``retry_rate``; < 5 %)
  (iii) changed-class recovery: episodes of a changed class, after the change, until its
        first success; share of changed classes recovered within 3 (``recovered_le3``; >= 80 %)
  (iv)  changed-class post-change success (``changed_post``; support >= 80 %, ci-fix >= 85 %)
  (v)   cell success over episodes 20-59 (``late``; support >= 0.86 / 0.90, ci-fix >= 0.90)
  (vi)  RSI arm only: episodes from the change to the first proposal (``first_fix_lag``;
        <= 3), share of fixes with ``source = repair_agent`` (>= 80 %), lever distribution

and, as the mechanism behind them, the recommendation mode mix, the ``act`` follow rate
(``taken == suggested_action``), the explore/sweep follow rate, and the label count per
cell. Windows are matched: only episodes ``< --episodes`` (default 60) are read, so a
120-episode grid cell compares with a 60-episode pilot cell.

    python -m continual_learning.analysis.rsi_dev --runs gridv5c-core-sl rsi-pilot-4 \
        rsi-pilot-5a rsi-pilot-5b --seeds 11 23 --fleet 1

Nothing here is a confirmatory statistic; it is the development readout the amendment
promised to report.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parents[1] / "results"
LATE_FROM = 20


def _episodes(run: str, *, max_episode: int, seeds: set[int] | None, fleet: int | None,
              scenarios: set[str] | None) -> list[dict[str, Any]]:
    path = RESULTS / run / "episodes.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("episode", 0) >= max_episode:
                continue
            if seeds and e.get("seed") not in seeds:
                continue
            if fleet is not None and int(e.get("fleet") or 1) != fleet:
                continue
            if scenarios and e.get("scenario") not in scenarios:
                continue
            e["_run"] = run
            out.append(e)
    return out


def _reports(run: str) -> list[dict[str, Any]]:
    path = RESULTS / run / "arm_reports.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "    -"


#: The tag each family puts the task's class under (support ``issue``, ci-fix ``failure``,
#: analytics/concierge ``class``; the rest fall back to whichever of these they carry).
CLASS_TAGS = ("issue", "failure", "class", "segment", "kind")


def task_class(e: dict[str, Any]) -> str:
    tags = e.get("tags") or {}
    for k in CLASS_TAGS:
        if tags.get(k) is not None:
            return str(tags[k])
    return "?"


def cell_metrics(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """The (h) quantities for one cell's episodes (one run, family, arm, seed)."""
    eps = sorted(eps, key=lambda e: e["episode"])
    n = len(eps)
    ok = sum(1 for e in eps if e.get("success"))
    late = [e for e in eps if e["episode"] >= LATE_FROM]
    late_ok = sum(1 for e in late if e.get("success"))
    post = [e for e in eps if (e.get("tags") or {}).get("post_change")]
    changed_post = [e for e in post if (e.get("tags") or {}).get("change_class")]
    changed_post_ok = sum(1 for e in changed_post if e.get("success"))
    unchanged_post = [e for e in post if not (e.get("tags") or {}).get("change_class")]
    unchanged_post_ok = sum(1 for e in unchanged_post if e.get("success"))

    # (iii) recovery per changed class: how many of its post-change episodes fail before
    # its first post-change success, and how many fail in total.
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in changed_post:
        by_class[task_class(e)].append(e)
    recovery: dict[str, dict[str, int]] = {}
    for cls, items in by_class.items():
        failed_before = 0
        first_ok: int | None = None
        for i, e in enumerate(items):
            if e.get("success"):
                first_ok = i
                break
            failed_before += 1
        recovery[cls] = {"episodes": len(items), "failed_before_first_success": failed_before,
                         "first_success_index": first_ok if first_ok is not None else -1,
                         "failed_total": sum(1 for e in items if not e.get("success"))}
    recovered_le3 = sum(1 for r in recovery.values() if 0 <= r["first_success_index"] <= 2)

    # (i), (ii), mechanism.
    searched = 0
    rec_present = 0
    modes: Counter[str] = Counter()
    act_n = act_followed = 0
    explore_n = explore_followed = 0
    labels: set[str] = set()
    for e in eps:
        for s in e.get("searches_log") or []:
            searched += 1
            rec = s.get("recommendation") or {}
            mode = rec.get("mode")
            if mode:
                rec_present += 1
                modes[mode] += 1
            if s.get("situation"):
                labels.add(str(s["situation"]))
            taken = s.get("taken")
            if mode == "act" and rec.get("suggested_action") and taken:
                act_n += 1
                act_followed += int(taken == rec["suggested_action"])
            if mode == "explore" and taken:
                plan = rec.get("plan") or ([rec["suggested_action"]] if rec.get("suggested_action") else [])
                if plan:
                    explore_n += 1
                    explore_followed += int(taken == plan[0])
    attempts = sum(int(e.get("attempts") or 0) for e in eps)
    retries = 0
    repeated = 0
    for e in eps:
        answers = [a for a in (e.get("answers") or []) if a]
        retries += max(0, len(answers) - 1)
        repeated += len(answers) - len(dict.fromkeys(answers))
    return {
        "n": n, "ok": ok, "late_n": len(late), "late_ok": late_ok,
        "changed_post_n": len(changed_post), "changed_post_ok": changed_post_ok,
        "unchanged_post_n": len(unchanged_post), "unchanged_post_ok": unchanged_post_ok,
        "classes_changed": len(recovery), "recovered_le3": recovered_le3,
        "failed_per_changed_class": (statistics.mean(r["failed_total"] for r in recovery.values())
                                     if recovery else None),
        "recovery": recovery,
        "searched": searched, "rec_present": rec_present, "modes": dict(modes),
        "act_n": act_n, "act_followed": act_followed,
        "explore_n": explore_n, "explore_followed": explore_followed,
        "attempts": attempts, "retries": retries, "repeated": repeated,
        "labels": len(labels),
        "first_attempt_ok": sum(1 for e in eps if e.get("first_attempt_success")),
        "cost_usd": sum(float((e.get("usage") or {}).get("cost_usd") or 0.0) for e in eps),
    }


def rsi_metrics(report: dict[str, Any], change_at: int) -> dict[str, Any]:
    """(vi) from an RSI arm report: first proposal's lag in episodes after the change, fix
    sources and levers, shipped count."""
    rep = report.get("report") or {}
    ledger = sorted((rep.get("ledger") or []), key=lambda r: float(r.get("sealed_at") or 0))
    fixes = rep.get("fixes") or []

    def _episode_at(iso: str | None) -> int | None:
        if not iso:
            return None
        try:
            t = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
        ep = None
        for r in ledger:
            if float(r.get("sealed_at") or 0) <= t:
                ep = int(r.get("episode", -1))
        return ep

    proposals = sorted((_episode_at(f.get("created_at")) for f in fixes if f.get("created_at")),
                       key=lambda x: (x is None, x))
    first = next((p for p in proposals if p is not None), None)
    sources = Counter(str(f.get("source") or "?") for f in fixes)
    levers = Counter(str(f.get("lever") or "?") for f in fixes)
    statuses = Counter(str(f.get("status") or "?") for f in fixes)
    shipped = [f for f in fixes if f.get("shipped_at")]
    ship_lag = [
        (_episode_at(f.get("shipped_at")) or 0) - change_at for f in shipped
        if _episode_at(f.get("shipped_at")) is not None
    ]
    return {
        "fixes": len(fixes), "first_fix_lag": (first - change_at) if first is not None else None,
        "sources": dict(sources), "levers": dict(levers), "statuses": dict(statuses),
        "shipped": len(shipped), "ship_lag": ship_lag,
        "incidents": len((rep.get("incidents") or {}).get("items") or []),
        "sealed": rep.get("sessions_sealed"),
    }


def summarise(runs: list[str], *, seeds: set[int] | None, fleet: int | None,
              scenarios: set[str] | None, max_episode: int) -> dict[str, Any]:
    cells: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for run in runs:
        eps = _episodes(run, max_episode=max_episode, seeds=seeds, fleet=fleet, scenarios=scenarios)
        by_cell: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for e in eps:
            by_cell[(run, e["scenario"], e["arm"], int(e["seed"]))].append(e)
        reports = _reports(run)
        for key, items in by_cell.items():
            m = cell_metrics(items)
            m["change_at"] = int(items[0].get("change_at") or 0)
            m["done"] = len(items) >= max_episode
            rep = next((r for r in reports if r.get("seed") == key[3] and r.get("arm") == key[2]
                        and str(r.get("cell", "")).startswith(key[1] + "|")), None)
            if rep is not None and key[2].startswith("senselab-rsi"):
                m["rsi"] = rsi_metrics(rep, m["change_at"])
            cells[key] = m
    pooled: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (run, fam, arm, _seed), m in cells.items():
        p = pooled.setdefault((run, fam, arm), Counter())
        for k in ("n", "ok", "late_n", "late_ok", "changed_post_n", "changed_post_ok", "unchanged_post_n",
                  "unchanged_post_ok", "classes_changed", "recovered_le3", "searched", "rec_present",
                  "act_n", "act_followed", "explore_n", "explore_followed", "attempts", "retries",
                  "repeated", "first_attempt_ok"):
            p[k] += m[k]
        p["cells"] += 1
        p["cost_usd"] += m["cost_usd"]
        for mode, c in m["modes"].items():
            p[f"mode_{mode}"] += c
    return {"cells": {"|".join(map(str, k)): v for k, v in cells.items()},
            "pooled": {"|".join(k): dict(v) for k, v in pooled.items()}}


def render(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("Amendment (h) development readout — pooled per run × family × arm (episodes < window)")
    lines.append(f"{'run|family|arm':52} cells    n  success  1st-try   late(20+)  chg-post  unchg-post  rec-present  act-follow  expl-follow  retry  repeat  rec<=3")
    for key, p in sorted(summary["pooled"].items()):
        lines.append(
            f"{key:52} {p['cells']:5d} {p['n']:4d}  {_pct(p['ok'], p['n'])}  {_pct(p['first_attempt_ok'], p['n'])}"
            f"  {_pct(p['late_ok'], p['late_n'])}   {_pct(p['changed_post_ok'], p['changed_post_n'])}"
            f"   {_pct(p['unchanged_post_ok'], p['unchanged_post_n'])}    {_pct(p['rec_present'], p['searched'])}"
            f"    {_pct(p['act_followed'], p['act_n'])}    {_pct(p['explore_followed'], p['explore_n'])}"
            f"  {_pct(p['retries'], p['attempts'])} {_pct(p['repeated'], p['attempts'])}"
            f"  {p['recovered_le3']}/{p['classes_changed']}"
        )
        modes = {k[5:]: v for k, v in p.items() if k.startswith("mode_")}
        if modes:
            lines.append(f"{'':52}   modes: " + ", ".join(f"{k} {v}" for k, v in sorted(modes.items())))
    lines.append("")
    lines.append("Per cell — changed classes after the change (failed episodes before first success / failed total / episodes):")
    for key, m in sorted(summary["cells"].items()):
        status = "done" if m.get("done") else f"ep {m['n']}"
        rec = m.get("recovery") or {}
        parts = [f"{cls}: {r['failed_before_first_success']}/{r['failed_total']}/{r['episodes']}"
                 for cls, r in sorted(rec.items())]
        lines.append(f"  {key:50} [{status:7}] late {_pct(m['late_ok'], m['late_n'])}  labels {m['labels']:2d}  "
                     + ("; ".join(parts) if parts else "(no post-change episodes yet)"))
        if m.get("rsi"):
            r = m["rsi"]
            lines.append(f"  {'':50}  RSI: fixes {r['fixes']} first-lag {r['first_fix_lag']} sources {r['sources']} "
                         f"levers {r['levers']} statuses {r['statuses']} shipped {r['shipped']} ship-lag {r['ship_lag']} "
                         f"incidents {r['incidents']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[11, 23])
    ap.add_argument("--fleet", type=int, default=1)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=60, help="window: episodes < this are read")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    summary = summarise(args.runs, seeds=set(args.seeds) if args.seeds else None, fleet=args.fleet,
                        scenarios=set(args.scenarios) if args.scenarios else None, max_episode=args.episodes)
    summary["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    if args.json:
        print(json.dumps(summary, indent=1, default=str))
    else:
        print(render(summary))


if __name__ == "__main__":
    main()
