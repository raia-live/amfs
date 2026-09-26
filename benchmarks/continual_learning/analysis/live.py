"""Live scoreboard for runs in progress: SenseLab arms against the alternatives.

Reads ``results/<run>/episodes.jsonl`` for the runs named (or every run whose
episodes file changed in the last day), and prints per-arm success, first-attempt
success, attempts, escalations, cost and latency, plus a matched head-to-head
over the cells every listed arm has finished the same (scenario, model, seed,
fleet) for — the only comparison that is fair while a grid is still filling in.

    python -m continual_learning.analysis.live --runs e1-A-baselines rsi-pilot-4
    python -m continual_learning.analysis.live --watch 60 --html results/live.html

``--html`` writes a self-refreshing page (open it in a browser and leave it);
``--watch`` re-renders every N seconds. Nothing here is used by the paper's
analysis — it is an operator's view, not a result.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent.parent / "results"
LATE_FROM = 30  # episodes >= this count as "late" for the early→late learning delta


def _load(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = run_dir / "episodes.jsonl"
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a row still being written
    return rows


def _cost(r: dict[str, Any]) -> float:
    u = r.get("usage") or {}
    ru = r.get("reflection_usage") or {}
    m = r.get("memory") or {}
    return float(u.get("cost_usd") or 0) + float(ru.get("cost_usd") or 0) + float(m.get("arm_extra_cost_usd") or 0)


def _key(r: dict[str, Any]) -> tuple:
    return (r.get("scenario"), r.get("model"), r.get("seed"), r.get("fleet"))


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    succ = sum(1 for r in rows if r.get("success"))
    first = sum(1 for r in rows if r.get("first_attempt_success"))
    esc = sum(1 for r in rows if r.get("escalated"))
    late = [r for r in rows if int(r.get("episode", 0)) >= LATE_FROM]
    early = [r for r in rows if int(r.get("episode", 0)) < LATE_FROM]
    rate = lambda xs: (sum(1 for r in xs if r.get("success")) / len(xs)) if xs else None  # noqa: E731
    return {
        "n": n,
        "cells": len({r.get("cell") for r in rows}),
        "success": succ / n,
        "first": first / n,
        "attempts": statistics.fmean(int(r.get("attempts") or 1) for r in rows),
        "escalated": esc / n,
        "cost": statistics.fmean(_cost(r) for r in rows),
        "wall_s": statistics.fmean(float(r.get("wall_ms") or 0) for r in rows) / 1000,
        "memory_s": statistics.fmean(float((r.get("memory") or {}).get("memory_ms") or 0) for r in rows) / 1000,
        "early": rate(early),
        "late": rate(late),
    }


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:5.1f}%"


def _delta(a: dict[str, Any]) -> str:
    if a.get("early") is None or a.get("late") is None:
        return "-"
    return f"{100 * (a['late'] - a['early']):+5.1f}"


def _progress(run_dir: Path, rows: list[dict[str, Any]]) -> str:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        total = len(manifest.get("cells") or [])
        started = manifest.get("started")
    except Exception:  # noqa: BLE001
        total, started = 0, None
    done_cells = 0
    cells_path = run_dir / "cells.json"
    if cells_path.exists():
        try:
            done_cells = len(json.loads(cells_path.read_text()))
        except Exception:  # noqa: BLE001
            done_cells = 0
    last = max((float(r.get("ts") or 0) for r in rows), default=0)
    age = f"{(time.time() - last) / 60:.0f} min ago" if last else "no episodes yet"
    hours = f"{(time.time() - started) / 3600:.1f} h" if started else "?"
    return f"{run_dir.name}: {len(rows)} episodes, {done_cells}/{total or '?'} cells finished, running {hours}, last episode {age}"


def render(run_dirs: list[Path], *, scenario: str | None = None, fleet: int | None = None) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    progress: list[str] = []
    for d in run_dirs:
        rows = _load(d)
        progress.append(_progress(d, rows))
        for r in rows:
            r["_run"] = d.name
        all_rows.extend(rows)
    if scenario:
        all_rows = [r for r in all_rows if r.get("scenario") == scenario]
    if fleet is not None:
        all_rows = [r for r in all_rows if int(r.get("fleet") or 1) == fleet]

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scen_arm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_arm[r["arm"]].append(r)
        by_scen_arm[(r["scenario"], r["arm"])].append(r)

    # Matched head-to-head: cells (scenario, model, seed, fleet) that every arm
    # with any data has finished at least LATE_FROM episodes of.
    arms = sorted(by_arm)
    done_keys: dict[str, set[tuple]] = {}
    for arm, rows in by_arm.items():
        per_key: dict[tuple, int] = defaultdict(int)
        for r in rows:
            per_key[_key(r)] += 1
        done_keys[arm] = {k for k, n in per_key.items() if n >= LATE_FROM}
    common: set[tuple] = set.intersection(*done_keys.values()) if done_keys else set()
    matched = {arm: _agg([r for r in rows if _key(r) in common]) for arm, rows in by_arm.items()} if common else {}

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "progress": progress,
        "filters": {"scenario": scenario, "fleet": fleet},
        "arms": {arm: _agg(rows) for arm, rows in by_arm.items()},
        "scenario_arm": {f"{s}|{a}": _agg(rows) for (s, a), rows in sorted(by_scen_arm.items())},
        "matched": matched,
        "matched_cells": sorted(common, key=str),
        "arm_order": arms,
    }


def _table(title: str, rows: list[tuple[str, dict[str, Any]]], *, out: list[str]) -> None:
    if not rows:
        return
    out.append("")
    out.append(title)
    out.append(f"  {'arm':28s} {'n':>5s} {'cells':>5s} {'success':>8s} {'1st-try':>8s} {'att':>5s} {'escal':>7s} "
               f"{'$/ep':>7s} {'wall s':>7s} {'mem s':>6s} {'early':>7s} {'late':>7s} {'Δ pts':>6s}")
    for name, a in rows:
        if not a.get("n"):
            continue
        out.append(
            f"  {name[:28]:28s} {a['n']:5d} {a['cells']:5d} {_pct(a['success']):>8s} {_pct(a['first']):>8s} "
            f"{a['attempts']:5.2f} {_pct(a['escalated']):>7s} {a['cost']:7.3f} {a['wall_s']:7.1f} {a['memory_s']:6.1f} "
            f"{_pct(a['early']):>7s} {_pct(a['late']):>7s} {_delta(a):>6s}"
        )


def as_text(view: dict[str, Any]) -> str:
    out: list[str] = [f"Continual-learning benchmark — live scoreboard ({view['generated']})"]
    for p in view["progress"]:
        out.append(f"  {p}")
    f = view["filters"]
    if f["scenario"] or f["fleet"] is not None:
        out.append(f"  filters: scenario={f['scenario'] or 'all'} fleet={f['fleet'] if f['fleet'] is not None else 'all'}")
    _table("All episodes so far, by arm (not matched — arms may be on different families/seeds):",
           [(a, view["arms"][a]) for a in view["arm_order"]], out=out)
    if view["matched"]:
        _table(f"Matched head-to-head — {len(view['matched_cells'])} (scenario, model, seed, fleet) cells every arm has "
               f">= {LATE_FROM} episodes of:", [(a, view["matched"][a]) for a in view["arm_order"]], out=out)
    else:
        out.append("")
        out.append(f"  No matched head-to-head yet: no (scenario, model, seed, fleet) cell has >= {LATE_FROM} episodes in every arm.")
    by_scen: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for k, a in view["scenario_arm"].items():
        s, arm = k.split("|", 1)
        by_scen[s].append((arm, a))
    for s in sorted(by_scen):
        _table(f"Family: {s}", sorted(by_scen[s]), out=out)
    out.append("")
    out.append(f"  early = episodes < {LATE_FROM}, late = episodes >= {LATE_FROM}; Δ pts = late − early success, "
               "the learning signal. $/ep includes reflection and arm overhead.")
    return "\n".join(out)


def as_html(view: dict[str, Any], refresh: int) -> str:
    body = html.escape(as_text(view))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta http-equiv='refresh' content='{int(refresh)}'>"
        "<title>CL benchmark — live</title>"
        "<style>body{background:#111;color:#ddd;font:13px/1.45 ui-monospace,Menlo,monospace;padding:18px}"
        "pre{white-space:pre;margin:0}</style></head><body>"
        f"<pre>{body}</pre></body></html>"
    )


def _default_runs() -> list[Path]:
    cutoff = time.time() - 24 * 3600
    runs = []
    for d in RESULTS.iterdir():
        ep = d / "episodes.jsonl"
        if ep.exists() and ep.stat().st_mtime >= cutoff:
            runs.append(d)
    return sorted(runs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="*", help="run names under results/ (default: runs active in the last 24 h)")
    ap.add_argument("--scenario", help="restrict to one family")
    ap.add_argument("--fleet", type=int, help="restrict to one fleet size")
    ap.add_argument("--html", help="also write a self-refreshing HTML page here")
    ap.add_argument("--watch", type=int, default=0, help="re-render every N seconds")
    ap.add_argument("--json", action="store_true", help="print the view as JSON instead of text")
    args = ap.parse_args()

    run_dirs = [RESULTS / r for r in args.runs] if args.runs else _default_runs()
    while True:
        view = render(run_dirs, scenario=args.scenario, fleet=args.fleet)
        text = json.dumps(view, indent=1, default=str) if args.json else as_text(view)
        if args.html:
            Path(args.html).write_text(as_html(view, max(args.watch, 30)))
        if args.watch:
            print("\033[2J\033[H", end="")
        print(text)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
