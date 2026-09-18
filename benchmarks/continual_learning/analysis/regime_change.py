"""Grid v2 analysis: does continual learning handle a changing world better than memory?

Assembles one episode table from the grid v2 runs (baselines and Mem0 from ``gridv2*``, the
outcome-learning SenseLab arms from the ``gridv2b*`` rerun with precise blame) and produces
the tables and figures the paper leads with:

  1. Pre-change learning curve (first-attempt success by episode block).
  2. Regime change: recovery curve on changed-class tasks, stale-pick rate, episodes to
     recover, escalations.
  3. Cost of not learning: tokens and seconds per successful task, wasted tokens on failed
     attempts, escalations per 100 tasks.
  4. Predictability: cross-seed spread and flip rate after the change.
  5. Fleet-6 propagation.
  6. Mechanism: what happened to the stale entries' confidence (SenseLab only).

Every headline delta carries a cell-level bootstrap 95% interval (a cell = one arm x
scenario x seed store; episodes inside a cell are not independent).

    python -m benchmarks.continual_learning.analysis.regime_change
    python -m benchmarks.continual_learning.analysis.regime_change --runs gridv3

With ``--runs <prefix>`` every arm is read from every ``results/<prefix>*/episodes.jsonl``
(the grid v3 layout, one launcher per arm family) and the report goes to
``results/<prefix>-report``; two tables are added that only grid v3 records carry — the
actions each fleet tried per changed class, and the recommendation / outcome cross-tab.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "gridv2-report"
RUNS_PREFIX: str | None = None

# Where each arm's episodes come from. The SenseLab outcome-learning arms were rerun as
# gridv2b with precise blame (cited keys, else top hit since the last attempt boundary);
# their v2 cells are kept only for the blame-precision ablation table.
SOURCES = {
    "none": ["gridv2", "gridv2-fleet"],
    "pgvector": ["gridv2", "gridv2-fleet"],
    "pgvector-diy": ["gridv2"],
    "pgvector-diy+outcomes": ["gridv2", "gridv2-fleet"],
    "mem0": ["gridv2-mem0"],
    "senselab-nofeedback": ["gridv2", "gridv2-fleet"],
    "senselab-episode": ["gridv2b"],
    "senselab": ["gridv2b", "gridv2b-fleet"],
}
ARMS = list(SOURCES)
HEAD = ["none", "pgvector", "pgvector-diy+outcomes", "mem0", "senselab-nofeedback", "senselab-episode", "senselab"]
LABEL = {
    "none": "no memory", "pgvector": "pgvector RAG", "pgvector-diy": "pgvector + reflection + consolidation",
    "pgvector-diy+outcomes": "pgvector DIY + outcome counter", "mem0": "Mem0", "senselab-nofeedback": "SenseLab, no outcomes",
    "senselab-episode": "SenseLab, episode-level outcomes", "senselab": "SenseLab, continual learning",
    "senselab-nopriors": "SenseLab, no action priors",
}


def use_runs(prefix: str) -> None:
    """Read every arm from every run directory starting with *prefix* (grid v3 layout)."""
    global SOURCES, ARMS, HEAD, OUT, RUNS_PREFIX
    RUNS_PREFIX = prefix
    runs = sorted(p.name for p in RESULTS.glob(f"{prefix}*") if (p / "episodes.jsonl").exists()
                  and not p.name.endswith("-report"))
    arms: list[str] = []
    for run in runs:
        for line in (RESULTS / run / "episodes.jsonl").read_text().splitlines():
            a = json.loads(line)["arm"]
            if a not in arms:
                arms.append(a)
    SOURCES = {a: runs for a in arms}
    ARMS = list(SOURCES)
    order = ["none", "pgvector", "pgvector-diy", "pgvector-diy+outcomes", "mem0", "senselab-nofeedback",
             "senselab-episode", "senselab-nopriors", "senselab"]
    HEAD = [a for a in order if a in arms] + [a for a in arms if a not in order]
    for a in arms:
        LABEL.setdefault(a, a)
    OUT = RESULTS / f"{prefix}-report"


def load() -> list[dict]:
    rows: list[dict] = []
    for arm, runs in SOURCES.items():
        for run in runs:
            p = RESULTS / run / "episodes.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                x = json.loads(line)
                if x["arm"] != arm:
                    continue
                if RUNS_PREFIX and x.get("fleet_mode", "rr") != "rr":
                    continue  # transfer-protocol cells belong to analysis/transfer.py
                x["run"] = run
                x["tokens"] = x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"] + sum(
                    (x.get("reflection_usage") or {}).get(k, 0) for k in ("prompt_tokens", "completion_tokens"))
                x["cost"] = x["usage"]["cost_usd"] + (x.get("reflection_usage") or {}).get("cost_usd", 0.0) \
                    + (x["memory"].get("arm_extra_cost_usd") or 0.0)
                x["wall_s"] = x["wall_ms"] / 1000
                x["changed"] = bool((x.get("tags") or {}).get("change_class"))
                x["post"] = x["episode"] >= (x.get("change_at") or 20)
                x["stale"] = bool((x.get("flags") or {}).get("stale_pick"))
                rows.append(x)
    return rows


def mean(xs, nd=2):
    xs = list(xs)
    return round(sum(xs) / len(xs), nd) if xs else float("nan")


def boot_ci(cells: dict[str, list[float]], n: int = 2000, seed: int = 7) -> tuple[float, float]:
    """95% interval for the mean of a per-episode statistic, resampling cells."""
    keys = [k for k, v in cells.items() if v]
    if not keys:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        pick = [cells[rng.choice(keys)] for _ in keys]
        flat = [v for c in pick for v in c]
        means.append(sum(flat) / len(flat))
    means.sort()
    return (round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3))


def by_cell(rows, f):
    d = defaultdict(list)
    for x in rows:
        d[x["cell"]].append(float(f(x)))
    return d


def sel(rows, arm=None, fleet=None, pre=None, changed=None, scenarios=None):
    out = rows
    if arm is not None:
        out = [x for x in out if x["arm"] == arm]
    if fleet is not None:
        out = [x for x in out if x["fleet"] == fleet]
    if pre is True:
        out = [x for x in out if not x["post"]]
    if pre is False:
        out = [x for x in out if x["post"]]
    if changed is not None:
        out = [x for x in out if x["changed"] == changed]
    if scenarios:
        out = [x for x in out if x["scenario"] in scenarios]
    return out


def table(headers, lines) -> str:
    if not lines:
        return "_(no rows)_"
    w = [max(len(str(h)), *(len(str(l[i])) for l in lines)) for i, h in enumerate(headers)]
    fmt = "| " + " | ".join(f"{{:<{w[0]}}}" if i == 0 else f"{{:>{w[i]}}}" for i in range(len(headers))) + " |"
    out = [fmt.format(*headers), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += [fmt.format(*[str(c) for c in l]) for l in lines]
    return "\n".join(out)


def episodes_to_recover(cell_rows, k=3) -> int | None:
    """First changed-class exposure index after the change from which k consecutive
    first-attempt successes follow; None if never."""
    seq = [x["first_attempt_success"] for x in sorted(cell_rows, key=lambda x: x["episode"])]
    for i in range(len(seq) - k + 1):
        if all(seq[i:i + k]):
            return i + 1
    return None


def flip_rate(cell_rows) -> float:
    seq = [x["first_attempt_success"] for x in sorted(cell_rows, key=lambda x: x["episode"])]
    return sum(a != b for a, b in zip(seq, seq[1:])) / max(1, len(seq) - 1)


def main() -> None:
    if "--runs" in sys.argv:
        use_runs(sys.argv[sys.argv.index("--runs") + 1])
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    md: list[str] = [f"# {RUNS_PREFIX or 'Grid v2'}: continual learning vs memory under regime change\n"]
    n_by = {a: len(sel(rows, arm=a)) for a in ARMS}
    md.append("Episodes per arm: " + ", ".join(f"{a} {n}" for a, n in n_by.items() if n) + ".\n")
    mem0_scen = sorted({x["scenario"] for x in sel(rows, arm="mem0")})
    md.append(f"Mem0 ran on {', '.join(mem0_scen)} only; tables marked (M) restrict every arm to those scenarios.\n")

    # ---- 1. headline: post-change on changed classes -------------------------------------
    def headline(scenarios, tag):
        lines = []
        for a in HEAD:
            r = sel(rows, arm=a, fleet=1, pre=False, changed=True, scenarios=scenarios)
            if not r:
                continue
            pre = sel(rows, arm=a, fleet=1, pre=True, scenarios=scenarios)
            fa = by_cell(r, lambda x: x["first_attempt_success"])
            lo, hi = boot_ci(fa)
            lines.append([
                LABEL[a], len(r), mean(x["first_attempt_success"] for x in pre),
                f"{mean(x['first_attempt_success'] for x in r)} [{lo:.2f},{hi:.2f}]",
                mean(x["success"] for x in r), mean(x["stale"] for x in r),
                mean(x["escalated"] for x in r), mean(x["attempts"] for x in r),
                f"{mean((x['tokens'] for x in r), 0):.0f}", f"{mean((x['wall_s'] for x in r), 1)}",
            ])
        return table(["arm", "n", "pre 1st", "post-change 1st-attempt [95% CI]", "success", "stale-pick",
                      "escalated", "attempts", "tok/ep", "wall s"], lines)

    md.append("## 1. After the world changes: changed-class tasks, episodes 20-59, single agent\n")
    md.append(headline(None, "all") + "\n")
    md.append("### 1b. Same, restricted to the Mem0 scenarios (M)\n")
    md.append(headline(mem0_scen, "M") + "\n")

    # ---- 2. recovery curve -----------------------------------------------------------
    md.append("## 2. Recovery curve: first-attempt success on changed classes by 10-episode block\n")
    lines = []
    for a in HEAD:
        r = sel(rows, arm=a, fleet=1, pre=False, changed=True)
        if not r:
            continue
        blocks = [mean(x["first_attempt_success"] for x in r if (x["episode"] - 20) // 10 == k) for k in range(4)]
        etr = [episodes_to_recover([x for x in r if x["cell"] == c]) for c in {x["cell"] for x in r}]
        rec = [e for e in etr if e is not None]
        lines.append([LABEL[a], *blocks, f"{len(rec)}/{len(etr)}", mean(rec, 1) if rec else "-"])
    md.append(table(["arm", "ep 20-29", "30-39", "40-49", "50-59", "cells recovered (3 in a row)", "median-ish exposures"], lines) + "\n")

    md.append("### 2b. Stale-pick rate (agent applied the pre-change fix) by block\n")
    lines = []
    for a in HEAD:
        r = sel(rows, arm=a, fleet=1, pre=False, changed=True)
        if not r:
            continue
        lines.append([LABEL[a], *[mean(x["stale"] for x in r if (x["episode"] - 20) // 10 == k) for k in range(4)]])
    md.append(table(["arm", "ep 20-29", "30-39", "40-49", "50-59"], lines) + "\n")

    # ---- 3. pre-change learning ----------------------------------------------------
    md.append("## 3. Before the change: learning curve on all tasks (episodes 0-19)\n")
    lines = []
    for a in HEAD:
        r = sel(rows, arm=a, fleet=1, pre=True)
        if not r:
            continue
        lines.append([LABEL[a], *[mean(x["first_attempt_success"] for x in r if x["episode"] // 5 == k) for k in range(4)],
                      mean(x["escalated"] for x in r), f"{mean((x['tokens'] for x in r), 0):.0f}"])
    md.append(table(["arm", "ep 0-4", "5-9", "10-14", "15-19", "escalated", "tok/ep"], lines) + "\n")

    # ---- 4. cost of not learning ----------------------------------------------------
    md.append("## 4. Cost of not learning (post-change, changed-class tasks, single agent)\n")
    lines = []
    for a in HEAD:
        r = sel(rows, arm=a, fleet=1, pre=False, changed=True)
        if not r:
            continue
        succ = [x for x in r if x["success"]]
        tok_per_success = sum(x["tokens"] for x in r) / max(1, len(succ))
        sec_per_success = sum(x["wall_s"] for x in r) / max(1, len(succ))
        # tokens spent on attempts that did not end the task: approximated as the share of
        # attempts beyond the first on tasks that eventually succeeded plus all tokens of
        # escalated tasks (the whole chain was wasted, a human was paged).
        wasted = sum(x["tokens"] * (1 - 1 / x["attempts"]) for x in succ) + sum(x["tokens"] for x in r if x["escalated"])
        lines.append([LABEL[a], f"{tok_per_success:.0f}", f"{sec_per_success:.1f}", f"{wasted / len(r):.0f}",
                      f"{100 * mean(x['escalated'] for x in r):.0f}", f"{sum(x['cost'] for x in r) / max(1, len(succ)):.4f}"])
    md.append(table(["arm", "tokens / successful task", "seconds / successful task", "wasted tok / task",
                     "escalations / 100 tasks", "$ / successful task"], lines) + "\n")

    # ---- 5. predictability ------------------------------------------------------------
    md.append("## 5. Predictability after the change (changed-class tasks)\n")
    lines = []
    for a in HEAD:
        r = sel(rows, arm=a, fleet=1, pre=False, changed=True)
        if not r:
            continue
        per_seed = defaultdict(list)
        for x in r:
            per_seed[(x["scenario"], x["seed"])].append(x["first_attempt_success"])
        per_scen = defaultdict(list)
        for (s, _), v in per_seed.items():
            per_scen[s].append(sum(v) / len(v))
        spread = mean((st.pstdev(v) for v in per_scen.values() if len(v) > 1), 3)
        flips = mean((flip_rate([x for x in r if x["cell"] == c]) for c in {x["cell"] for x in r}), 2)
        lines.append([LABEL[a], spread, flips])
    md.append(table(["arm", "cross-seed std of 1st-attempt rate", "flip rate (episode to episode)"], lines) + "\n")

    # ---- 6. fleet -------------------------------------------------------------------
    md.append("## 6. Fleet of 6 agents sharing one store (support, ci-fix)\n")
    lines = []
    for a in HEAD:
        r6 = sel(rows, arm=a, fleet=6, pre=False, changed=True)
        r1 = sel(rows, arm=a, fleet=1, pre=False, changed=True, scenarios=["support", "ci-fix"])
        if not r6:
            continue
        # propagation: per cell, the first post-change changed-class exposure of each agent
        first = []
        for c in {x["cell"] for x in r6}:
            seen = set()
            for x in sorted((y for y in r6 if y["cell"] == c), key=lambda y: y["episode"]):
                if x["agent_id"] not in seen:
                    seen.add(x["agent_id"])
                    first.append(x)
        lines.append([LABEL[a], len(r6), mean(x["first_attempt_success"] for x in r1), mean(x["first_attempt_success"] for x in r6),
                      mean(x["stale"] for x in r6), mean(x["escalated"] for x in r6),
                      mean(x["first_attempt_success"] for x in first), mean(x["stale"] for x in first)])
    md.append(table(["arm", "n", "single-agent 1st", "fleet 1st", "fleet stale", "fleet escalated",
                     "each agent's FIRST post-change exposure: 1st-attempt", "...stale"], lines) + "\n")

    # ---- 7. mechanism ---------------------------------------------------------------
    md.append("## 7. Mechanism: what the loop did to the entries (SenseLab arms)\n")
    lines = []
    mech = [(a, run) for a in HEAD if a.startswith("senselab") for run in SOURCES.get(a, [])] if RUNS_PREFIX else \
        [("senselab", "gridv2b"), ("senselab-episode", "gridv2b"), ("senselab-nofeedback", "gridv2")]
    for a, run in mech:
        p = RESULTS / run / "confidence_traces.jsonl"
        if not p.exists():
            continue
        vs = [v for l in p.read_text().splitlines() for t in [json.loads(l)] if f"|{a}|" in t["cell"] for v in t["versions"]]
        if not vs:
            continue
        status = defaultdict(int)
        for v in vs:
            status[v.get("evidence_status", "untested")] += 1
        latest = {}
        for v in vs:
            if v["key"] not in latest or v["version"] > latest[v["key"]]["version"]:
                latest[v["key"]] = v
        lines.append([LABEL[a], len(latest), len(vs), status["validated"], status["contested"], status["discredited"],
                      sum(1 for v in latest.values() if v.get("evidence_status") == "discredited")])
    md.append(table(["arm", "keys", "versions", "validated", "contested", "discredited", "keys ending discredited"], lines) + "\n")

    # ---- 8. blame precision ablation ----------------------------------------------------
    v2 = RESULTS / "gridv2" / "episodes.jsonl"
    if v2.exists():
        old = [json.loads(l) for l in v2.read_text().splitlines()]
        old = [x for x in old if x["arm"] in ("senselab", "senselab-episode")]
        md.append("## 8. Ablation: blame everything read (v2) vs cited/top-hit blame (v2b)\n")
        lines = []
        for a in ("senselab", "senselab-episode"):
            for label, src in (("blame whole read window", old), ("cited keys, else top hit", sel(rows, arm=a, fleet=1))):
                r = [x for x in src if x["arm"] == a and x["fleet"] == 1]
                pre = [x for x in r if x["episode"] < 20]
                post = [x for x in r if x["episode"] >= 20 and (x.get("tags") or {}).get("change_class")]
                lines.append([f"{LABEL[a]} — {label}", len(r), mean(x["first_attempt_success"] for x in pre),
                              mean(x["first_attempt_success"] for x in post), mean(x["escalated"] for x in post)])
        md.append(table(["arm — blame rule", "n", "pre 1st", "post-change 1st", "post esc"], lines) + "\n")

    # ---- 9. actions tried per changed class per fleet ----------------------------------
    # The grid-v2 finding this plan set out to fix: every fleet spent its attempts on the
    # same two failing actions and never tried the new one. Per arm: over (cell, changed
    # class) pairs after the change, how many attempts went to the pre-change fix, how
    # many distinct actions were tried, and how many pairs ever tried the action that
    # turned out to work (the modal final answer of successful post-change episodes on
    # that class, across every arm).
    md.append("## 9. Actions tried per changed class, after the change (all fleet sizes)\n")
    post_changed = [x for x in rows if x["post"] and x["changed"] and x.get("answers")]
    winning: dict[tuple[str, str], str] = {}
    for (scen, cls), grp in _group(post_changed, lambda x: (x["scenario"], _task_class(x))).items():
        wins = Counter(x["final_answer"] for x in grp if x["success"] and x["final_answer"])
        if wins:
            winning[(scen, cls)] = wins.most_common(1)[0][0]
    lines = []
    for a in HEAD:
        pairs = _group([x for x in post_changed if x["arm"] == a], lambda x: (x["cell"], _task_class(x)))
        if not pairs:
            continue
        stale_attempts = tried_new = solved = 0
        distinct, attempts_per = [], []
        for (cell, cls), grp in pairs.items():
            scen = grp[0]["scenario"]
            new = winning.get((scen, cls))
            acts = [ans for x in grp for ans in x["answers"]]
            attempts_per.append(len(acts))
            distinct.append(len(set(acts)))
            stale_attempts += sum(1 for x in grp if x["stale"])
            tried_new += bool(new and new in acts)
            solved += any(x["success"] for x in grp)
        n = len(pairs)
        lines.append([LABEL[a], n, mean(attempts_per, 1), mean(distinct, 1), f"{stale_attempts / n:.1f}",
                      f"{tried_new}/{n}", f"{solved}/{n}"])
    md.append(table(["arm", "(cell, class) pairs", "attempts / pair", "distinct actions / pair",
                     "stale final picks / pair", "pairs that ever tried the new fix", "pairs solved at least once"],
                    lines) + "\n")

    # ---- 10. recommendation followed / outcome ----------------------------------------
    # Only arms that return a recommendation log one per search. The last recommendation
    # before the first terminal action is what the agent had in hand; "followed" means the
    # first answer equals the suggested action (act / explore) or the agent escalated on the
    # first attempt (escalate).
    recs = [x for x in rows if any(sl.get("recommendation") for sl in x.get("searches_log") or [])]
    if recs:
        md.append("## 10. Recommendation in hand vs what the agent did (first attempt)\n")
        lines = []
        for a in HEAD:
            r = [x for x in recs if x["arm"] == a]
            if not r:
                continue
            cross: dict[tuple[str, bool], list[bool]] = defaultdict(list)
            for x in r:
                first = [sl["recommendation"] for sl in x["searches_log"] if sl.get("recommendation") and sl.get("attempt", 1) == 1]
                if not first or not x["answers"]:
                    continue
                rec = first[-1]
                mode, sug = rec.get("mode"), rec.get("suggested_action")
                first_ans = x["answers"][0]
                if mode == "escalate":
                    followed = x["escalated"] and x["attempts"] <= 1
                else:
                    followed = bool(sug) and sug.split(":", 1)[-1] == first_ans
                cross[(mode, followed)].append(bool(x["first_attempt_success"]))
            for (mode, followed), v in sorted(cross.items(), key=lambda kv: (str(kv[0][0]), not kv[0][1])):
                lines.append([LABEL[a], mode, "followed" if followed else "deviated", len(v), mean(v)])
        md.append(table(["arm", "recommendation", "agent", "n", "1st-attempt success"], lines) + "\n")
        lines = []
        for a in HEAD:
            r = [x for x in recs if x["arm"] == a and x["post"] and x["changed"]]
            if not r:
                continue
            modes = Counter(sl["recommendation"]["mode"] for x in r for sl in x["searches_log"] if sl.get("recommendation"))
            tot = sum(modes.values()) or 1
            lines.append([LABEL[a], tot, *(f"{100 * modes[m] / tot:.0f}%" for m in ("act", "explore", "escalate"))])
        md.append("### 10b. Recommendation mix on changed-class tasks after the change\n")
        md.append(table(["arm", "searches", "act", "explore", "escalate"], lines) + "\n")

    (OUT / "REPORT.md").write_text("\n".join(md))
    print("\n".join(md))
    _figures(rows)
    print(f"\nwritten: {OUT}")


def _task_class(x: dict) -> str:
    t = x.get("tags") or {}
    for k in ("issue", "failure", "request", "expected"):
        if t.get(k):
            return str(t[k])
    return "?"


def _group(rows, key):
    d = defaultdict(list)
    for x in rows:
        d[key(x)].append(x)
    return d


def _figures(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    colors = {"none": "#999999", "pgvector": "#1f77b4", "pgvector-diy+outcomes": "#17becf", "mem0": "#ff7f0e",
              "senselab-nofeedback": "#bcbd22", "senselab-episode": "#9467bd", "senselab": "#d62728",
              "senselab-nopriors": "#e377c2", "pgvector-diy": "#8c564b"}
    # `use_runs` adds whatever arms the run directories hold to HEAD, so any arm
    # without a colour above gets one from the cycle rather than a KeyError
    # after the markdown report has already been written.
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, a in enumerate(a for a in HEAD if a not in colors):
        colors[a] = fallback[i % len(fallback)]
    # Fig 1: first-attempt success by 5-episode block across the whole horizon, changed-class after 20
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for a in HEAD:
        r = sel(rows, arm=a, fleet=1)
        if not r:
            continue
        xs, ys = [], []
        for k in range(12):
            blk = [x for x in r if x["episode"] // 5 == k and (x["episode"] < 20 or x["changed"])]
            if blk:
                xs.append(5 * k + 2.5)
                ys.append(sum(x["first_attempt_success"] for x in blk) / len(blk))
        ax.plot(xs, ys, marker="o", lw=2 if a == "senselab" else 1.2, color=colors[a], label=LABEL[a])
    ax.axvline(20, color="k", ls="--", lw=1)
    ax.text(20.5, 0.05, "regime change", fontsize=9)
    ax.set_xlabel("episode (after 20: changed-class tasks only)")
    ax.set_ylabel("first-attempt success")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Learning, then unlearning: first-attempt success across a regime change (gpt-5.5, 3 seeds x 5 scenarios)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_learning_curve.png", dpi=150)
    # Fig 2: tokens per successful task, post-change changed-class
    fig, ax = plt.subplots(figsize=(7, 4))
    names, vals, bar_colors = [], [], []
    for a in HEAD:
        r = sel(rows, arm=a, fleet=1, pre=False, changed=True)
        if not r:
            continue
        names.append(LABEL[a])
        vals.append(sum(x["tokens"] for x in r) / max(1, sum(x["success"] for x in r)))
        bar_colors.append(colors[a])
    ax.barh(names, vals, color=bar_colors)
    ax.set_xlabel("tokens per successful changed-class task (post-change)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tokens_per_success.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    main()
