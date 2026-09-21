"""Proof points: the mechanism probes and the long-horizon run (grid v3).

Each probe isolates one thing a retrieval layer cannot do because it has no channel for
what happened after the read; the long-horizon run asks whether a store that only
accumulates notes gets worse with time (preregistered H3). Tables go to
``results/<prefix>-report/PROOF_POINTS.md``.

  A. runbook     two near-tie procedures seeded, one causes an outage: convergence vs coin flip,
                 wrong-guidance follows by block, cross-seed spread and flip rate.
  B. unknowns    35% of tasks have no runbook: escalate (correct) vs hallucinate a deploy.
  C. handoff     agent B inherits 8 findings from agent A, one poisoned: poisoned-follow rate on
                 first and later exposures.
  D. drift       a fact (drift-fact) or a tool's parameter shape (drift-tool) changes at episode 8:
                 first-attempt success and stale follows before / after.
  E. long horizon  support + diagnose, 120 episodes, no regime change: first-attempt success and
                 tokens by 20-episode block, and the per-cell slope over episodes 20-119.

    python -m benchmarks.continual_learning.analysis.probes --runs gridv3
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from . import regime_change as rc

RESULTS = rc.RESULTS
ORDER = ["none", "pgvector", "pgvector-diy", "pgvector-diy+outcomes", "mem0", "senselab-nofeedback",
         "senselab-nopriors", "senselab"]


def load(prefix: str, kind: str) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(RESULTS.glob(f"{prefix}-{kind}*/episodes.jsonl")):
        for line in p.read_text().splitlines():
            x = json.loads(line)
            x["tokens"] = x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"] + sum(
                (x.get("reflection_usage") or {}).get(k, 0) for k in ("prompt_tokens", "completion_tokens"))
            x["flags"] = x.get("flags") or {}
            x["tags"] = x.get("tags") or {}
            rows.append(x)
    return rows


def arms(rows) -> list[str]:
    seen = {x["arm"] for x in rows}
    return [a for a in ORDER if a in seen] + sorted(a for a in seen if a not in ORDER)


def label(a: str) -> str:
    return rc.LABEL.get(a, a)


def sel(rows, arm=None, scenario=None, pred=None):
    out = rows
    if arm is not None:
        out = [x for x in out if x["arm"] == arm]
    if scenario is not None:
        out = [x for x in out if x["scenario"] == scenario]
    if pred is not None:
        out = [x for x in out if pred(x)]
    return out


def cells(rows):
    d = defaultdict(list)
    for x in rows:
        d[x["cell"]].append(x)
    return d


def converged_at(cell_rows, k=5) -> int | None:
    seq = [x["first_attempt_success"] for x in sorted(cell_rows, key=lambda x: x["episode"])]
    for i in range(len(seq) - k + 1):
        if all(seq[i:i + k]):
            return i
    return None


def cross_seed_std(rows, start=5) -> float:
    per = defaultdict(list)
    for x in rows:
        if x["episode"] >= start:
            per[x["seed"]].append(x["first_attempt_success"])
    rates = [sum(v) / len(v) for v in per.values() if v]
    return round(st.pstdev(rates), 3) if len(rates) > 1 else float("nan")


def slope(cell_rows, start: int) -> float | None:
    pts = [(x["episode"], 1.0 if x["first_attempt_success"] else 0.0) for x in cell_rows if x["episode"] >= start]
    if len(pts) < 10:
        return None
    xs, ys = zip(*pts)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((a - mx) ** 2 for a in xs)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / var if var else None


def main() -> None:
    prefix = sys.argv[sys.argv.index("--runs") + 1] if "--runs" in sys.argv else "gridv3"
    out_dir = RESULTS / f"{prefix}-report"
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = load(prefix, "probes")
    long = load(prefix, "long")
    md: list[str] = [f"# {prefix}: proof points — what a retrieval layer cannot do\n",
                     f"Probe episodes loaded: {len(probes)}; long-horizon episodes loaded: {len(long)}.\n"]
    mean, table, boot_ci = rc.mean, rc.table, rc.boot_ci

    # ---- A. runbook ----------------------------------------------------------------
    md.append("## A. runbook: two near-tie procedures, one causes an outage\n")
    md.append("Both procedures are seeded at equal confidence and match the query equally well. A retriever "
              "returns both forever; only the outcome says which one to trust.\n")
    lines = []
    for a in arms(probes):
        r = sel(probes, arm=a, scenario="runbook")
        if not r:
            continue
        cs = cells(r)
        conv = [converged_at(v) for v in cs.values()]
        got = [c for c in conv if c is not None]
        fa = rc.by_cell(r, lambda x: x["first_attempt_success"])
        lo, hi = boot_ci(fa)
        blocks = [mean(x["first_attempt_success"] for x in r if x["episode"] // 5 == k) for k in range(4)]
        bad = [mean(x["flags"].get("wrong_guidance", False) for x in r if x["episode"] // 5 == k) for k in range(4)]
        lines.append([label(a), len(r), f"{mean(x['first_attempt_success'] for x in r)} [{lo:.2f},{hi:.2f}]",
                      *blocks, f"{len(got)}/{len(cs)}", st.median(got) if got else "-",
                      f"{bad[0]} -> {bad[3]}", mean(x["escalated"] for x in r),
                      cross_seed_std(r), mean(rc.flip_rate(v) for v in cs.values())])
    md.append(table(["arm", "n", "1st-attempt [95% CI]", "ep 0-4", "5-9", "10-14", "15-19",
                     "cells converged (5 in a row)", "median converge ep", "followed the outage procedure ep0-4 -> 15-19",
                     "escalated", "cross-seed std (ep 5+)", "flip rate"], lines) + "\n")

    # ---- B. unknowns ------------------------------------------------------------------
    md.append("## B. unknowns: 35% of tasks have no runbook — escalate, or invent a deploy?\n")
    md.append("On a no-runbook task the correct move is to escalate; running a deploy is a hallucinated fix "
              "that opens an incident. On the other tasks an escalation is unnecessary.\n")
    lines = []
    for a in arms(probes):
        r = sel(probes, arm=a, scenario="unknowns")
        if not r:
            continue
        unk = [x for x in r if x["tags"].get("unknown")]
        known = [x for x in r if not x["tags"].get("unknown")]
        lines.append([label(a), len(unk), mean(x["flags"].get("abstained_correctly", False) for x in unk),
                      mean(x["flags"].get("hallucinated", False) for x in unk),
                      mean(x["flags"].get("hallucinated", False) for x in unk if x["episode"] >= 10),
                      len(known), mean(x["first_attempt_success"] for x in known),
                      mean(x["flags"].get("unnecessary_escalation", False) for x in known)])
    md.append(table(["arm", "no-runbook tasks", "escalated correctly", "hallucinated a deploy",
                     "...ep 10+", "runbook tasks", "1st-attempt", "unnecessary escalation"], lines) + "\n")

    # ---- C. handoff -------------------------------------------------------------------
    md.append("## C. handoff: a new agent inherits eight findings, one of them wrong\n")
    md.append("Agent A's findings are in the store. Agent B, a new identity, acts on them. The poisoned finding "
              "is retrieved at the same rank as the seven good ones until an outcome says otherwise.\n")
    lines = []
    for a in arms(probes):
        r = sel(probes, arm=a, scenario="handoff")
        if not r:
            continue
        pois = sorted([x for x in r if x["tags"].get("poisoned")], key=lambda x: (x["cell"], x["episode"]))
        first, later = [], []
        seen = set()
        for x in pois:
            (later if x["cell"] in seen else first).append(x)
            seen.add(x["cell"])
        good = [x for x in r if not x["tags"].get("poisoned")]
        lines.append([label(a), len(pois), mean(x["flags"].get("poisoned_follow", False) for x in first),
                      mean(x["flags"].get("poisoned_follow", False) for x in later) if later else "-",
                      mean(x["first_attempt_success"] for x in pois),
                      len(good), mean(x["first_attempt_success"] for x in good), mean(x["escalated"] for x in r)])
    md.append(table(["arm", "poisoned tasks", "followed the poison: 1st exposure", "...later exposures",
                     "1st-attempt on poisoned", "clean tasks", "1st-attempt on clean", "escalated"], lines) + "\n")

    # ---- D. drift ---------------------------------------------------------------------
    for scen, what in (("drift-fact", "an operational fact changes at episode 8"),
                       ("drift-tool", "a tool's parameter shape changes at episode 8")):
        md.append(f"## D. {scen}: {what}\n")
        lines = []
        for a in arms(probes):
            r = sel(probes, arm=a, scenario=scen)
            if not r:
                continue
            pre = [x for x in r if not x["tags"].get("post_drift")]
            post = [x for x in r if x["tags"].get("post_drift")]
            blocks = [mean(x["first_attempt_success"] for x in post if (x["episode"] - 8) // 4 == k) for k in range(3)]
            lines.append([label(a), len(pre), mean(x["first_attempt_success"] for x in pre), len(post),
                          mean(x["first_attempt_success"] for x in post), *blocks,
                          mean(x["flags"].get("stale_pick", False) for x in post),
                          mean(x["escalated"] for x in post), mean(x["attempts"] for x in post)])
        md.append(table(["arm", "pre n", "pre 1st", "post n", "post 1st", "ep 8-11", "12-15", "16-19",
                         "stale follow (post)", "escalated (post)", "attempts (post)"], lines) + "\n")

    # ---- E. long horizon ----------------------------------------------------------------
    md.append("## E. Long horizon: 120 episodes, no regime change (support, diagnose; 300 distractors)\n")
    md.append("Every arm with memory writes one reflection note per episode, so the store grows by ~120 notes "
              "per cell on top of the 300 distractors. H3: a store that only accumulates gets worse; a store "
              "that reconciles on outcomes does not.\n")
    for scen in sorted({x["scenario"] for x in long}):
        md.append(f"### E.{scen}\n")
        lines = []
        for a in arms(long):
            r = sel(long, arm=a, scenario=scen)
            if not r:
                continue
            blocks = [mean(x["first_attempt_success"] for x in r if x["episode"] // 20 == k) for k in range(6)]
            toks = [f"{mean((x['tokens'] for x in r if x['episode'] // 20 == k), 0):.0f}" for k in (0, 5)]
            sl = [s for s in (slope(v, 20) for v in cells(r).values()) if s is not None]
            lo, hi = boot_ci({str(i): [100 * s] for i, s in enumerate(sl)}) if sl else (float("nan"), float("nan"))
            lines.append([label(a), len(r), *blocks, f"{100 * mean(sl, 4):+.2f} [{lo:+.2f},{hi:+.2f}]" if sl else "-",
                          mean(x["escalated"] for x in r if x["episode"] >= 100), f"{toks[0]} -> {toks[1]}"])
        md.append(table(["arm", "n", "ep 0-19", "20-39", "40-59", "60-79", "80-99", "100-119",
                         "slope ep 20-119 (pts / 100 ep) [95% CI]", "escalated ep 100+", "tok/ep ep0-19 -> 100-119"],
                        lines) + "\n")

    (out_dir / "PROOF_POINTS.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwritten: {out_dir / 'PROOF_POINTS.md'}")


if __name__ == "__main__":
    main()
