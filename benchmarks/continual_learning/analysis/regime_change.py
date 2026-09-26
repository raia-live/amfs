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
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .. import config

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
# Canonical arm order for tables and figures: baselines, then the SenseLab ablations, then
# the full continual-learning arm last. Arms not listed are appended alphabetically.
ARM_ORDER = ["none", "pgvector", "pgvector-diy", "pgvector-diy+outcomes", "mem0", "zep", "senselab-nofeedback",
             "senselab-episode", "senselab-attempts", "senselab-nopriors", "senselab-lean", "senselab"]
LABEL = {
    "none": "no memory", "pgvector": "pgvector RAG", "pgvector-diy": "pgvector + reflection + consolidation",
    "pgvector-diy+outcomes": "pgvector DIY + outcome counter", "mem0": "Mem0", "zep": "Zep",
    "senselab-nofeedback": "SenseLab, no outcomes",
    "senselab-episode": "SenseLab, episode-level outcomes", "senselab-attempts": "SenseLab, attempt-level outcomes",
    "senselab": "SenseLab, outcome loop on",
    "senselab-nopriors": "SenseLab, no action priors", "senselab-lean": "SenseLab, lean briefing",
}

# The tag that names a task's class in each scenario (what ``first_exposures`` and the
# actions-per-class table key on). Scenarios not listed fall back to ``tags["class"]``.
CLASS_TAG = {"support": "issue", "ci-fix": "failure", "order-ops": "request", "retention": "expected",
             "diagnose": "rule", "concierge": "kind", "analytics": "kind",
             "policy-qa": "topic", "accounts": "class", "dispatch": "task_class"}


def class_of(x: dict) -> str | None:
    """The task class of an episode row, or None when the scenario's class tag is absent."""
    tags = x.get("tags") or {}
    v = tags.get(CLASS_TAG.get(x.get("scenario", ""), "class"))
    return None if v is None else str(v)


# Infrastructure failures: an episode that died on the network or a 5xx from a hosted memory
# service (or the model API) rather than on the task. An episode that never reached an
# attempt for that reason (``attempts == 0``; in practice an Anthropic 529 "overloaded" on
# the first model call) is dropped by ``load`` for every arm alike — it carries no information
# about the memory layer and would otherwise be scored as an escalated failure. The count
# dropped is reported (``DROPPED_NO_ATTEMPT``). Rows whose error came after a verdict
# ("reflection: ReadTimeout ...", a failed commit) stay in every metric; the ``infra_err``
# share and the "excl. infra" success rate are added alongside so the learning signal can be
# read net of them too.
INFRA_RE = re.compile(r"timeout|connection|ReadTimeout|ConnectError|RemoteProtocolError|5\d\d|"
                      r"Server disconnected|ServiceUnavailable|OverloadedError|overloaded", re.I)
#: (arm, scenario) -> episodes dropped by ``load`` because an infrastructure error preceded any attempt.
DROPPED_NO_ATTEMPT: dict[tuple[str, str], int] = {}
#: ``--paper``: (arm, scenario) -> the one run the paper reads for it (see paper_sources.py).
SCENARIO_RUN: dict[tuple[str, str], str] = {}


def is_infra(error) -> bool:
    return bool(error) and bool(INFRA_RE.search(str(error)))


def infra_rate(rows) -> float:
    return mean((is_infra(x.get("error")) for x in rows), 3)


def success_excl(rows, field: str = "success"):
    """Success rate over the rows without an infrastructure error (nan if none remain)."""
    return mean(x[field] for x in rows if not is_infra(x.get("error")))


def order_arms(arms) -> list[str]:
    seen = list(dict.fromkeys(arms))
    return [a for a in ARM_ORDER if a in seen] + sorted(a for a in seen if a not in ARM_ORDER)


def order_scenarios(scenarios) -> list[str]:
    """Study order (config core then probe scenarios) first, anything else alphabetical."""
    seen = set(scenarios)
    known = [s for s in config.STUDY.scenarios if s in seen]
    return known + sorted(seen - set(known))


def use_runs(prefix: str) -> None:
    """Read every arm from every run directory starting with *prefix* (grid v3 layout)."""
    global SOURCES, ARMS, HEAD, OUT, RUNS_PREFIX
    RUNS_PREFIX = prefix
    # The transfer protocols, the probes and the long-horizon run have their own analyses
    # (transfer.py, probes.py); mixing them in here would pool 20- and 120-episode cells with
    # the 60-episode regime-change cells.
    skip = ("-report", "-xfer-", "-probes", "-long", "-dry", "-sweep-")
    runs = sorted(p.name for p in RESULTS.glob(f"{prefix}*") if (p / "episodes.jsonl").exists()
                  and not any(s in p.name[len(prefix):] or p.name.endswith(s) for s in skip))
    arms: list[str] = []
    for run in runs:
        for line in (RESULTS / run / "episodes.jsonl").read_text().splitlines():
            a = json.loads(line)["arm"]
            if a not in arms:
                arms.append(a)
    SOURCES = {a: runs for a in arms}
    ARMS = list(SOURCES)
    HEAD = order_arms(arms)
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
                if SCENARIO_RUN and not _paper_keep(arm, x["scenario"], run):
                    continue  # the paper reads one build per scenario for the SenseLab arms
                x["run"] = run
                x["tokens"] = x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"] + sum(
                    (x.get("reflection_usage") or {}).get(k, 0) for k in ("prompt_tokens", "completion_tokens"))
                x["cost"] = x["usage"]["cost_usd"] + (x.get("reflection_usage") or {}).get("cost_usd", 0.0) \
                    + (x["memory"].get("arm_extra_cost_usd") or 0.0)
                x["wall_s"] = x["wall_ms"] / 1000
                x["changed"] = bool((x.get("tags") or {}).get("change_class"))
                x["post"] = x["episode"] >= (x.get("change_at") or 20)
                x["stale"] = bool((x.get("flags") or {}).get("stale_pick"))
                x["infra"] = is_infra(x.get("error"))
                if x["infra"] and not x.get("attempts"):
                    k = (x["arm"], x["scenario"])
                    DROPPED_NO_ATTEMPT[k] = DROPPED_NO_ATTEMPT.get(k, 0) + 1
                    continue
                rows.append(x)
    if SCENARIO_RUN:
        # --paper: every arm is read on the (scenario, fleet size) pairs the SenseLab arm has cells
        # for, so the pooled rows of a table compare the same task mix across arms. Per fleet size,
        # not per scenario: the fleet-of-10 run covers scenarios the single-agent grid does not, and
        # its rows must not admit other arms' single-agent cells on those scenarios.
        have = {(x["scenario"], x["fleet"]) for x in rows if x["arm"] == "senselab"}
        rows = [x for x in rows if (x["scenario"], x["fleet"]) in have]
    return rows


def _paper_keep(arm: str, scenario: str, run: str) -> bool:
    from . import paper_sources
    return paper_sources.keep_row(arm, scenario, run)


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
    if "--paper" in sys.argv:
        from . import paper_sources
        paper_sources.configure(sys.modules[__name__])
    elif "--runs" in sys.argv:
        use_runs(sys.argv[sys.argv.index("--runs") + 1])
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    md: list[str] = [f"# {RUNS_PREFIX or 'Grid v2'}: continual learning vs memory under regime change\n"]
    if SCENARIO_RUN:
        from . import paper_sources
        md.append("```\n" + paper_sources.describe() + "\n```\n")
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
                mean(x["success"] for x in r), success_excl(r), infra_rate(r), mean(x["stale"] for x in r),
                mean(x["escalated"] for x in r), mean(x["attempts"] for x in r),
                f"{mean((x['tokens'] for x in r), 0):.0f}", f"{mean((x['wall_s'] for x in r), 1)}",
            ])
        return table(["arm", "n", "pre 1st", "post-change 1st-attempt [95% CI]", "success", "success excl. infra",
                      "infra_err", "stale-pick", "escalated", "attempts", "tok/ep", "wall s"], lines)

    md.append("## 1. After the world changes: changed-class tasks, episodes 20-59, single agent\n")
    md.append("`infra_err` = share of episodes whose `error` is an infrastructure failure (timeout / connection / "
              "5xx) after a verdict was reached; `success excl. infra` = success rate over the other episodes. "
              "Episodes that never reached an attempt because of such an error (a model-API 529 on the first "
              "call) are excluded from every column for every arm: "
              f"{sum(DROPPED_NO_ATTEMPT.values())} episodes"
              + (" (" + ", ".join(f"{a}/{sc} {n}" for (a, sc), n in sorted(DROPPED_NO_ATTEMPT.items())) + ")"
                 if DROPPED_NO_ATTEMPT else "") + ".\n")
    md.append(headline(None, "all") + "\n")
    if set(mem0_scen) != {x["scenario"] for x in rows}:
        md.append("### 1b. Same, restricted to the Mem0 scenarios (M)\n")
        md.append(headline(mem0_scen, "M") + "\n")

    # ---- 1c. paired deltas: same scenario x seed cell, senselab minus each other arm ------
    # A cell is one store; pairing on (scenario, seed) removes the scenario mix and the seed's
    # task order from the comparison. Interval: bootstrap over the paired cells.
    def paired(ref: str, other: str, *, episodes=None, scenarios=None, fleet=1):
        def rate(a):
            r = sel(rows, arm=a, fleet=fleet, pre=False, changed=True, scenarios=scenarios)
            if episodes:
                r = [x for x in r if episodes[0] <= x["episode"] < episodes[1]]
            d = defaultdict(list)
            for x in r:
                d[(x["scenario"], x["seed"])].append(x["first_attempt_success"])
            return {k: sum(v) / len(v) for k, v in d.items()}
        ra, rb = rate(ref), rate(other)
        keys = sorted(set(ra) & set(rb))
        if not keys:
            return None
        deltas = [ra[k] - rb[k] for k in keys]
        lo, hi = boot_ci({str(k): [d] for k, d in zip(keys, deltas)})
        return mean(deltas, 3), lo, hi, sum(d > 0 for d in deltas), len(keys)

    md.append("### 1c. Paired cell deltas: SenseLab CL minus each arm (same scenario x seed), "
              "post-change changed-class first-attempt success\n")
    lines = []
    for a in HEAD:
        if a == "senselab":
            continue
        all_ep = paired("senselab", a)
        late = paired("senselab", a, episodes=(40, 60))
        if not all_ep:
            continue
        d, lo, hi, pos, n = all_ep
        dl, llo, lhi, _, _ = late if late else (float("nan"),) * 5
        lines.append([f"vs {LABEL[a]}", f"{d:+.3f} [{lo:+.2f},{hi:+.2f}]", f"{pos}/{n}",
                      f"{dl:+.3f} [{llo:+.2f},{lhi:+.2f}]" if late else "-"])
    md.append(table(["comparison", "ep 20-59 delta [95% CI]", "cells with senselab ahead",
                     "ep 40-59 delta [95% CI]"], lines) + "\n")

    # ---- 1d. per scenario --------------------------------------------------------------
    md.append("### 1d. Post-change changed-class first-attempt success by scenario (single agent)\n")
    scen_all = order_scenarios(x["scenario"] for x in rows)
    lines = []
    for a in HEAD:
        cells = []
        for s in scen_all:
            r = sel(rows, arm=a, fleet=1, pre=False, changed=True, scenarios=[s])
            cells.append(mean((x["first_attempt_success"] for x in r), 2) if r else "-")
        if any(c != "-" for c in cells):
            lines.append([LABEL[a], *cells])
    md.append(table(["arm", *scen_all], lines) + "\n")
    md.append("### 1e. Post-change changed-class escalation rate by scenario (single agent)\n")
    lines = []
    for a in HEAD:
        cells = []
        for s in scen_all:
            r = sel(rows, arm=a, fleet=1, pre=False, changed=True, scenarios=[s])
            cells.append(mean((x["escalated"] for x in r), 2) if r else "-")
        if any(c != "-" for c in cells):
            lines.append([LABEL[a], *cells])
    md.append(table(["arm", *scen_all], lines) + "\n")

    # ---- 1f. infrastructure errors by arm x scenario -----------------------------------
    md.append("### 1f. Infrastructure-error share by scenario (all episodes, all fleet sizes)\n")
    md.append("Share of episodes per arm x scenario whose `error` matches the infra pattern, with the "
              "eventual-success rate raw and excluding those episodes (all episodes of the cell, not only "
              "changed-class).\n")
    lines = []
    for a in HEAD:
        r_all = sel(rows, arm=a)
        if not r_all:
            continue
        cells = []
        for s in scen_all:
            r = sel(r_all, scenarios=[s])
            cells.append(infra_rate(r) if r else "-")
        lines.append([LABEL[a], *cells, infra_rate(r_all), mean(x["success"] for x in r_all), success_excl(r_all)])
    md.append(table(["arm", *scen_all, "all", "success", "success excl. infra"], lines) + "\n")

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

    # ---- 4b. whole horizon: every task, episodes 0-59 -----------------------------------
    # The operator's view: over the whole run, changed classes and stable classes alike,
    # what did each arm deliver and what did it cost. Per 1000 tasks so the ROI section
    # can attach a price to an escalation and a token.
    md.append("## 4b. Whole horizon, every task (episodes 0-59, single agent): outcomes and cost per 1000 tasks\n")
    lines = []
    for a in HEAD:
        r = sel(rows, arm=a, fleet=1)
        if not r:
            continue
        n = len(r)
        fa = by_cell(r, lambda x: x["first_attempt_success"])
        lo, hi = boot_ci(fa)
        lines.append([LABEL[a], n, f"{mean(x['first_attempt_success'] for x in r)} [{lo:.2f},{hi:.2f}]",
                      mean(x["success"] for x in r), success_excl(r), infra_rate(r),
                      f"{1000 * mean(x['escalated'] for x in r):.0f}",
                      mean(x["attempts"] for x in r), f"{sum(x['tokens'] for x in r) / n * 1000 / 1e6:.2f}M",
                      f"{sum(x['cost'] for x in r) / n * 1000:.0f}", f"{sum(x['wall_s'] for x in r) / n:.1f}",
                      f"{sum(x['memory']['memory_ms'] for x in r) / n / 1000:.1f}"])
    md.append(table(["arm", "n", "1st-attempt [95% CI]", "eventual success", "success excl. infra", "infra_err",
                     "escalations / 1000", "attempts / task", "tokens / 1000 tasks", "$ / 1000 tasks",
                     "wall s / task", "memory s / task"], lines) + "\n")

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
    # One table per fleet size present in the data (6 in grid v3; 10 in grid v4). The single-
    # agent reference is restricted to the scenarios that fleet actually ran, derived from
    # the rows rather than a fixed list.
    fleet_sizes = sorted({x["fleet"] for x in rows if x["fleet"] > 1})
    if not fleet_sizes:
        md.append("## 6. Fleet of agents sharing one store\n\n_(no fleet cells)_\n")
    for fi, n_fleet in enumerate(fleet_sizes):
        fleet_scen = order_scenarios(x["scenario"] for x in sel(rows, fleet=n_fleet))
        md.append(f"## 6{'' if fi == 0 else chr(ord('a') + fi)}. Fleet of {n_fleet} agents sharing one store "
                  f"({', '.join(fleet_scen)})\n")
        lines = []
        for a in HEAD:
            r6 = sel(rows, arm=a, fleet=n_fleet, pre=False, changed=True)
            r1 = sel(rows, arm=a, fleet=1, pre=False, changed=True, scenarios=fleet_scen)
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
                          mean(x["stale"] for x in r6), mean(x["escalated"] for x in r6), infra_rate(r6),
                          mean(x["first_attempt_success"] for x in first), mean(x["stale"] for x in first)])
        md.append(table(["arm", "n", "single-agent 1st", "fleet 1st", "fleet stale", "fleet escalated", "infra_err",
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
            if v["key"] not in latest or v.get("version", 0) > latest[v["key"]].get("version", 0):
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
    c = class_of(x)
    if c is not None:
        return c
    t = x.get("tags") or {}
    for k in ("issue", "failure", "request", "expected", "rule", "topic", "class", "task_class", "kind"):
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
              "zep": "#2ca02c", "senselab-nofeedback": "#bcbd22", "senselab-episode": "#9467bd",
              "senselab-attempts": "#7f7f7f", "senselab": "#d62728", "senselab-nopriors": "#e377c2",
              "senselab-lean": "#8c6d31", "pgvector-diy": "#8c564b"}
    single = sel(rows, fleet=1)
    n_models = len({x["model"] for x in single})
    n_seeds = len({x["seed"] for x in single})
    n_scen = len({x["scenario"] for x in single})
    setting = (f"{', '.join(sorted({x['model'] for x in single}))}; {n_seeds} seeds x {n_scen} scenarios"
               if n_models <= 2 else f"{n_models} models; {n_seeds} seeds x {n_scen} scenarios")
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
    ax.set_title(f"Learning, then unlearning: first-attempt success across a regime change ({setting})")
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
