"""Exploratory (not preregistered; amendment (t)): what happened to traffic after the
no-canary ablation shipped a fix, against the paired SenseLab cell over the same episodes.

For every cell of ``senselab-rsi-nocanary`` with at least one shipped fix, the window starts
at the first episode sealed after the earliest ``shipped_at`` and runs to the end of the cell.
Recurrence is the canary's reading of a session — not a first-attempt success (a failed
outcome or a recorded failed attempt) — split by population: ``kept`` (post-change, in the
class, rule held) and ``moved`` (post-change, in the class, rule changed). The paired
``senselab-rsi`` cell (same family, seed, engine cut) shipped nothing in any dev run, so the
same episode window there is the no-ship counterfactual. Diff-in-diff is
(after - before)_nocanary - (after - before)_senselab; negative = the ships were associated
with less recurrence than not shipping.

    .venv/bin/python -m benchmarks.continual_learning.analysis.rsi_posthoc_ships rsi-c1-rsi rsi-c2 rsi-c3
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
SHIPPED = ("memory_shipped", "verified", "regressed")
ABLATION, SYSTEM = "senselab-rsi-nocanary", "senselab-rsi"


def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def population(e: dict) -> str:
    t = e.get("tags", {})
    if not t.get("post_change"):
        return "pre"
    if not t.get("change_class"):
        return "unchanged"
    return "moved" if t.get("change_cond") else "kept"


def recurred(e: dict) -> bool:
    return not e.get("first_attempt_success")


def load(run: str) -> tuple[dict[str, list[dict]], list[dict]]:
    eps: dict[str, list[dict]] = collections.defaultdict(list)
    for line in open(RESULTS / run / "episodes.jsonl"):
        e = json.loads(line)
        if not e.get("error"):
            eps[e["cell"]].append(e)
    for c in eps:
        eps[c].sort(key=lambda e: e["episode"])
    reports = [json.loads(line) for line in open(RESULTS / run / "arm_reports.jsonl")]
    return eps, reports


def ships(run: str) -> list[dict]:
    eps, reports = load(run)
    out = []
    for r in reports:
        if r["arm"] != ABLATION:
            continue
        cell = r["cell"]
        for f in r["report"].get("fixes", []):
            if f.get("status") not in SHIPPED or not f.get("shipped_at"):
                continue
            t0 = _ts(f["shipped_at"])
            after = [e for e in eps[cell] if e["ts"] >= t0]
            corr = f.get("correction") or f.get("corrective_entry") or {}
            out.append({
                "run": run, "cell": cell, "family": cell.split("|")[0], "seed": cell.split("|")[3],
                "key": corr.get("key") or f.get("lever") or "?", "fix_id": f["id"],
                "ship_episode": after[0]["episode"] if after else None,
            })
    return out


def windows(runs: list[str]) -> dict[str, dict[str, list[int]]]:
    """Per population -> arm_when -> [recurred, n], first ship per cell, windows once."""
    agg: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))
    cells = 0
    for run in runs:
        eps, _ = load(run)
        first: dict[str, int] = {}
        for s in ships(run):
            if s["ship_episode"] is None:
                continue
            first[s["cell"]] = min(first.get(s["cell"], 10**9), s["ship_episode"])
        for cell, k in first.items():
            cells += 1
            pair = cell.replace(ABLATION, SYSTEM)
            for arm, c in (("nocanary", cell), ("senselab", pair)):
                for e in eps[c]:
                    p = population(e)
                    if p in ("pre", "unchanged"):
                        continue
                    when = "after" if e["episode"] >= k else "before"
                    slot = agg[p][f"{arm}_{when}"]
                    slot[0] += recurred(e)
                    slot[1] += 1
    agg["_cells"]["n"] = [cells, cells]
    return agg


def report(runs: list[str]) -> str:
    lines = ["# What the replay gate alone let through (exploratory)", ""]
    allships = [s for run in runs for s in ships(run)]
    lines.append(f"Runs: {', '.join(runs)}. Ships by `{ABLATION}`: {len(allships)}.")
    lines += ["", "## Ships", ""]
    lines.append("| run | family | seed | key | ship at episode |")
    lines.append("|---|---|---|---|---|")
    for s in allships:
        lines.append(f"| {s['run']} | {s['family']} | {s['seed']} | `{s['key'][:48]}` | "
                     f"{s['ship_episode'] if s['ship_episode'] is not None else 'after end'} |")
    agg = windows(runs)
    n_cells = agg.pop("_cells")["n"][0]
    lines += ["", "## Windows", "",
              f"Cells with a ship: {n_cells}; window = first ship to end of cell, paired "
              f"`{SYSTEM}` cell over the same episodes. Recurrence = not a first-attempt success.", "",
              "| population | −canary before | −canary after | SenseLab before | SenseLab after | DiD |",
              "|---|---|---|---|---|---|"]
    for p in ("kept", "moved"):
        d = agg[p]

        def rate(k: str) -> float:
            return d[k][0] / d[k][1] if d[k][1] else float("nan")

        did = (rate("nocanary_after") - rate("nocanary_before")) - (
            rate("senselab_after") - rate("senselab_before"))
        cell = lambda k: f"{d[k][0]}/{d[k][1]} ({rate(k):.0%})"  # noqa: E731
        lines.append(f"| {p} | {cell('nocanary_before')} | {cell('nocanary_after')} | "
                     f"{cell('senselab_before')} | {cell('senselab_after')} | {did:+.0%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    runs = sys.argv[1:] or ["rsi-c1-rsi", "rsi-c2", "rsi-c3"]
    text = report(runs)
    out = RESULTS / "rsi-report" / "POSTHOC_SHIPS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
