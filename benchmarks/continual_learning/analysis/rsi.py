"""Dev-seed readout for the RSI paper (HYPOTHESIS_RSI.md, set C, seeds 11 / 23).

Reads the set-C runs under ``results/`` and writes ``results/rsi-report/REPORT.md`` — the
headed markdown tables ``fill_rsi_paper.py`` turns into the paper's tables — and
``report.json`` with the same numbers. Every table follows the presentation rule of
amendment (o): one system, *SenseLab* (harness arm ``senselab-rsi``), against the frontier
model alone, the retrieval baselines and the context window; the ablations (``senselab``,
``senselab-rsi-nocanary``) appear in the attribution table only. Harness arm names are
mapped to system names here and appear nowhere in the output.

Set C moves a class's correct action *conditionally* on an instance attribute, so every
post-change changed-class episode is either in the **moved half** (the rule changed for this
instance — ``tags.change_cond``) or the **kept half** (it did not). Recovery is measured on
the moved half; collateral damage on the kept half and on the unchanged classes. All rates
are first-attempt success unless a column says otherwise; the window is every post-change
episode (episodes 30–99).

    python -m benchmarks.continual_learning.analysis.rsi --senselab-run rsi-c3 --paper

Nothing here is a confirmatory statistic: seeds 11 / 23 are the development seeds, and the
confirmatory seeds (37 / 41 / 59 / 73) run once after ``analysis-freeze-rsi``.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parents[1] / "results"
REPORT_DIR = RESULTS / "rsi-report"

#: system name -> (run, harness arm). The SenseLab row is taken from ``--senselab-run``.
BASELINES: dict[str, tuple[str, str]] = {
    "Frontier alone": ("rsi-c1-tier1", "none"),
    "Context window": ("rsi-c1-baselines-cw", "raw-traces"),
    "pgvector": ("rsi-c1-baselines", "pgvector"),
    "pgvector + rerank": ("rsi-c1-tier1", "pgvector+rerank"),
    "pgvector + outcomes (DIY)": ("rsi-c1-baselines", "pgvector-diy+outcomes"),
}
SENSELAB_ARM = "senselab-rsi"
ABLATIONS: dict[str, tuple[str | None, str]] = {
    # None = the same run as SenseLab
    "SenseLab − canary": (None, "senselab-rsi-nocanary"),
    "SenseLab − repair loop": ("rsi-c1-tier1", "senselab"),
}
#: SenseLab runs by engine cut, oldest first (HYPOTHESIS_RSI.md cut records).
CUTS: list[tuple[str, str, str]] = [
    ("rsi-c1-rsi", "pre-seam4", "#787/#788: per-agent auto-propose, situation contract"),
    ("rsi-c2", "post-seam5", "#793 case sets for regime fixes; #794 situation identity"),
    ("rsi-c3", "post-seam5b", "#795 canary re-stamps replays, post-shift cases; #796 embedding stage"),
    ("rsi-c4", "post-seam8", "#797 replays are not traffic; proof cases need a clean success"),
]
SCENARIOS = ("support", "ci-fix")
SEEDS = (11, 23)
BOOT = 2000


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def episodes(run: str) -> list[dict[str, Any]]:
    path = RESULTS / run / "episodes.jsonl"
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


def reports(run: str) -> list[dict[str, Any]]:
    path = RESULTS / run / "arm_reports.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def population(e: dict[str, Any]) -> str:
    t = e.get("tags") or {}
    if not t.get("post_change"):
        return "pre"
    if t.get("change_class"):
        return "moved" if t.get("change_cond") else "kept"
    return "unchanged"


# ---------------------------------------------------------------------------
# cells and rates
# ---------------------------------------------------------------------------

class Cell:
    def __init__(self) -> None:
        self.n: Counter[str] = Counter()
        self.ok: Counter[str] = Counter()
        self.episodes = 0
        self.success = 0
        self.escalated = 0
        self.errors = 0
        self.cost = 0.0
        self.extra_cost = 0.0
        self.tokens = 0
        self.memory_ms = 0.0
        self.wall_ms = 0.0

    def add(self, e: dict[str, Any]) -> None:
        if e.get("error"):
            self.errors += 1
            return  # infra errors are excluded from rates (HYPOTHESIS.md purge rule)
        p = population(e)
        self.n[p] += 1
        self.ok[p] += 1 if e.get("first_attempt_success") else 0
        self.episodes += 1
        self.success += 1 if e.get("success") else 0
        self.escalated += 1 if e.get("escalated") else 0
        usage = e.get("usage") or {}
        mem = e.get("memory") or {}
        after = mem.get("after_episode") or {}
        self.cost += float(usage.get("cost_usd") or 0.0)
        self.extra_cost += float(mem.get("arm_extra_cost_usd") or 0.0) + float(after.get("arm_extra_cost_usd") or 0.0)
        self.tokens += int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        self.memory_ms += float(mem.get("memory_ms") or 0.0)
        self.wall_ms += float(e.get("wall_ms") or 0.0)

    def rate(self, p: str) -> float | None:
        return self.ok[p] / self.n[p] if self.n[p] else None


CELL_EPISODES = 100


def cells_of(run: str, arm: str) -> dict[tuple[str, int], Cell]:
    """The finished cells of *arm* in *run*; a cell still running (or aborted) is left out,
    so a table never mixes a complete cell with a partial one."""
    out: dict[tuple[str, int], Cell] = defaultdict(Cell)
    for e in episodes(run):
        if e.get("arm") != arm or e.get("scenario") not in SCENARIOS or e.get("seed") not in SEEDS:
            continue
        out[(e["scenario"], int(e["seed"]))].add(e)
    return {k: c for k, c in out.items() if c.episodes + c.errors >= CELL_EPISODES}


def pooled(cells: list[Cell], p: str) -> tuple[int, int]:
    return sum(c.ok[p] for c in cells), sum(c.n[p] for c in cells)


def boot_ci(cells: list[Cell], p: str, seed: int = 11) -> tuple[float, float] | None:
    """Cell-level bootstrap (2,000 resamples of cells with replacement) of the pooled rate."""
    cells = [c for c in cells if c.n[p]]
    if len(cells) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(BOOT):
        pick = [cells[rng.randrange(len(cells))] for _ in cells]
        ok, n = pooled(pick, p)
        vals.append(ok / n)
    vals.sort()
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def fmt_rate(cells: list[Cell], p: str, ci: bool = False) -> str:
    ok, n = pooled(cells, p)
    if not n:
        return "—"
    s = f"{100 * ok / n:.0f} % ({ok}/{n})"
    if ci:
        b = boot_ci(cells, p)
        if b:
            s += f" [{100 * b[0]:.0f}, {100 * b[1]:.0f}]"
    return s


def fmt_share(num: int, den: int) -> str:
    return f"{100 * num / den:.0f} % ({num}/{den})" if den else "—"


# ---------------------------------------------------------------------------
# fixes
# ---------------------------------------------------------------------------

def fix_rows(run: str, arm: str) -> list[dict[str, Any]]:
    out = []
    for r in reports(run):
        cell = r.get("cell", "")
        parts = cell.split("|")
        if len(parts) < 4 or parts[1] != arm:
            continue
        scen, seed = parts[0], int(parts[3].lstrip("s"))
        for fx in (r.get("report") or {}).get("fixes") or []:
            out.append({"scenario": scen, "seed": seed, **fx})
    return out


def lifecycle(fixes: list[dict[str, Any]]) -> dict[str, int]:
    """Counts over the arm report's fix records. Tier 1 ``test.status`` is passed /
    inconclusive / failed / skipped; Tier 2 ``replay.status`` is passed / failed / requested /
    expired (or absent when Tier 1 did not pass); a canary is ``running`` or ``stopped`` and
    none reached a decision inside a 100-episode cell (HEADROOM_RSI.md, K1–K4)."""
    c: Counter[str] = Counter()
    for fx in fixes:
        c["proposed"] += 1
        regime = fx.get("judge_id") is None and fx.get("behavior_id") is None
        c["regime" if regime else "judge"] += 1
        c[f"tier1:{(fx.get('test') or {}).get('status') or 'none'}"] += 1
        c[f"tier2:{(fx.get('replay') or {}).get('status') or 'none'}"] += 1
        cn = fx.get("canary") or {}
        if cn:
            c["canary_started"] += 1
            c[f"canary:{cn.get('status')}"] += 1
        if fx.get("shipped_at"):
            c["shipped"] += 1
        if fx.get("reverted_at"):
            c["reverted"] += 1
        if fx.get("source") == "repair_agent":
            c["by_repair_agent"] += 1
        c[f"lever:{fx.get('lever') or '?'}"] += 1
        c[f"status:{fx.get('status')}"] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_(no rows)_"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def system_rows(systems: dict[str, tuple[str, str]], ci: bool) -> tuple[list[list[str]], dict[str, Any]]:
    rows, data = [], {}
    for name, (run, arm) in systems.items():
        cells = list(cells_of(run, arm).values())
        if not cells:
            rows.append([name, "0", "—", "—", "—", "—", "—", "—", "—", "—"])
            data[name] = {"run": run, "cells": 0}
            continue
        eps = sum(c.episodes for c in cells)
        rows.append([
            name, str(len(cells)),
            fmt_rate(cells, "moved", ci), fmt_rate(cells, "kept", ci), fmt_rate(cells, "unchanged"),
            fmt_rate(cells, "pre"),
            fmt_share(sum(c.success for c in cells), eps),
            fmt_share(sum(c.escalated for c in cells), eps),
            str(sum(c.errors for c in cells)),
            f"{sum(c.cost + c.extra_cost for c in cells) / eps:.3f}" if eps else "—",
        ])
        data[name] = {
            "run": run, "cells": len(cells), "episodes": eps,
            **{p: pooled(cells, p) for p in ("moved", "kept", "unchanged", "pre")},
            "moved_ci": boot_ci(cells, "moved"), "kept_ci": boot_ci(cells, "kept"),
            "success": sum(c.success for c in cells), "escalated": sum(c.escalated for c in cells),
            "errors": sum(c.errors for c in cells),
            "usd_per_episode": (sum(c.cost + c.extra_cost for c in cells) / eps) if eps else None,
        }
    return rows, data


HEAD = ["system", "cells", "moved half", "kept half", "unchanged classes", "pre-change",
        "episode success", "escalated", "infra err", "$/ep"]


def paired_rows(systems: dict[str, tuple[str, str]], senselab: tuple[str, str], p: str) -> list[list[str]]:
    sl = cells_of(*senselab)
    rows = []
    for name, (run, arm) in systems.items():
        other = cells_of(run, arm)
        deltas, cols = [], []
        for scen in SCENARIOS:
            for seed in SEEDS:
                a, b = sl.get((scen, seed)), other.get((scen, seed))
                if a and b and a.rate(p) is not None and b.rate(p) is not None:
                    d = a.rate(p) - b.rate(p)
                    deltas.append(d)
                    cols.append(f"{100 * d:+.0f}")
                else:
                    cols.append("—")
        if deltas:
            mean = statistics.mean(deltas)
            wins = sum(1 for d in deltas if d > 0)
            rows.append([f"SenseLab − {name}", *cols, f"{100 * mean:+.0f}", f"{wins}/{len(deltas)}"])
    return rows


def by_family_rows(systems: dict[str, tuple[str, str]]) -> list[list[str]]:
    rows = []
    for name, (run, arm) in systems.items():
        cells = cells_of(run, arm)
        row = [name]
        for scen in SCENARIOS:
            cs = [c for (s, _), c in cells.items() if s == scen]
            row += [fmt_rate(cs, "moved"), fmt_rate(cs, "kept"), fmt_rate(cs, "unchanged")]
        rows.append(row)
    return rows


def lifecycle_row(label: str, run: str, arm: str) -> tuple[list[str], dict[str, int]]:
    lc = lifecycle(fix_rows(run, arm))
    g = lc.get
    return [label, str(g("proposed", 0)), f"{g('regime', 0)} / {g('judge', 0)}",
            f"{g('tier1:passed', 0)} / {g('tier1:inconclusive', 0)} / {g('tier1:failed', 0) + g('tier1:skipped', 0)}",
            f"{g('tier2:passed', 0)} / {g('tier2:failed', 0)} / {g('tier2:requested', 0) + g('tier2:expired', 0)}",
            f"{g('canary_started', 0)} / {g('canary:running', 0)}",
            str(g("shipped", 0)), str(g("reverted", 0)),
            fmt_share(g("by_repair_agent", 0), g("proposed", 0))], lc


LIFECYCLE_HEAD = ["arm", "proposed", "regime / judge", "T1 pass / inc. / fail",
                  "T2 pass / fail / open", "canaries run / undecided", "shipped", "reverted",
                  "drafted by the repair agent"]


def cost_rows(systems: dict[str, tuple[str, str]]) -> list[list[str]]:
    rows = []
    for name, (run, arm) in systems.items():
        cells = list(cells_of(run, arm).values())
        eps = sum(c.episodes for c in cells)
        if not eps:
            continue
        rows.append([name, f"{sum(c.tokens for c in cells) / eps:,.0f}",
                     f"{sum(c.cost for c in cells) / eps:.3f}",
                     f"{sum(c.extra_cost for c in cells) / eps:.3f}",
                     f"{sum(c.memory_ms for c in cells) / eps / 1000:.1f}",
                     f"{sum(c.wall_ms for c in cells) / eps / 1000:.0f}"])
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def newest_complete_senselab_run() -> str:
    for run, _, _ in reversed(CUTS):
        eps = [e for e in episodes(run) if e.get("arm") == SENSELAB_ARM]
        if len(eps) >= 400:
            return run
    return CUTS[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--senselab-run", default=None, help="the SenseLab run for the headline (default: newest complete)")
    ap.add_argument("--paper", action="store_true", help="write results/rsi-report/REPORT.md and report.json")
    args = ap.parse_args()

    sl_run = args.senselab_run or newest_complete_senselab_run()
    senselab = (sl_run, SENSELAB_ARM)
    systems: dict[str, tuple[str, str]] = {**BASELINES, "SenseLab": senselab}
    data: dict[str, Any] = {"senselab_run": sl_run, "seeds": list(SEEDS), "scenarios": list(SCENARIOS)}
    md: list[str] = [f"# RSI paper — development-seed readout (set C, seeds {', '.join(map(str, SEEDS))}; "
                     f"SenseLab from `{sl_run}`)", ""]

    md += ["## 1. After the rule changes conditionally", "",
           "First-attempt success on every post-change episode (30–99), pooled over the four cells "
           "(support / ci-fix × seeds 11 / 23), GPT-5.5. Moved half: changed-class instances whose rule "
           "changed; kept half: changed-class instances whose rule did not. Brackets: cell-level bootstrap "
           "95 % interval. Infra errors are excluded from every rate.", ""]
    rows, d = system_rows(systems, ci=True)
    md += md_table(HEAD, rows) + [""]
    data["headline"] = d

    md += ["## 1b. Paired cell deltas, moved half", "",
           "SenseLab minus each system on the moved half, per cell (percentage points), paired on (family, seed).", ""]
    cols = [f"{s} s{seed}" for s in SCENARIOS for seed in SEEDS]
    md += md_table(["pair", *cols, "mean", "cells won"], paired_rows(BASELINES, senselab, "moved")) + [""]

    md += ["## 1c. By family", "", "First-attempt success per family: moved half, kept half, unchanged classes.", ""]
    fam_head = ["system"] + [f"{s}: {p}" for s in SCENARIOS for p in ("moved", "kept", "unchanged")]
    md += md_table(fam_head, by_family_rows(systems)) + [""]

    md += ["## 2. SenseLab by engine cut", "",
           "Every SenseLab run on set C, oldest engine first. The engine changed between runs "
           "(HYPOTHESIS_RSI.md cut records); each row is the same protocol on a different cut.", ""]
    rows, d = [], {}
    for run, cut, what in CUTS:
        cells = list(cells_of(run, SENSELAB_ARM).values())
        if not cells:
            continue
        eps = sum(c.episodes for c in cells)
        lc = lifecycle(fix_rows(run, SENSELAB_ARM))
        rows.append([f"{cut} (`{run}`)", str(len(cells)), fmt_rate(cells, "moved"), fmt_rate(cells, "kept"),
                     fmt_rate(cells, "unchanged"), fmt_share(sum(c.escalated for c in cells), eps),
                     str(lc.get("proposed", 0)), str(lc.get("tier2:passed", 0)), str(lc.get("shipped", 0)),
                     what])
        d[run] = {"cut": cut, "cells": len(cells), **{p: pooled(cells, p) for p in ("moved", "kept", "unchanged")},
                  "lifecycle": lc}
    md += md_table(["engine cut", "cells", "moved half", "kept half", "unchanged", "escalated",
                    "fixes proposed", "Tier 2 passed", "shipped", "what changed"], rows) + [""]
    data["cuts"] = d

    md += ["## 3. Attribution", "",
           f"SenseLab (`{sl_run}`) against its own ablations on the same protocol: without the canary "
           f"(same run) and without the repair loop (outcome learning alone, `rsi-c1-tier1`).", ""]
    abl: dict[str, tuple[str, str]] = {"SenseLab": senselab}
    for name, (run, arm) in ABLATIONS.items():
        abl[name] = (run or sl_run, arm)
    rows, d = system_rows(abl, ci=True)
    md += md_table(HEAD, rows) + [""]
    data["attribution"] = d

    md += ["## 4. Fix lifecycle", "",
           "Every fix the loop proposed in the SenseLab cells and the no-canary cells, by run. Tier 1 is the "
           "stand-in replay on the fix branch; Tier 2 the customer replay of the fix's case set; a canary "
           "grades live exposure over the sessions after it starts. `open` Tier 2 replays were still awaited "
           "when the cell ended; no canary reached a decision inside a cell (HEADROOM K1–K4).", ""]
    rows, d = [], {}
    for run, cut, _ in CUTS:
        for arm, label in ((SENSELAB_ARM, "SenseLab"), ("senselab-rsi-nocanary", "SenseLab − canary")):
            if not fix_rows(run, arm):
                continue
            row, lc = lifecycle_row(f"{label}, {cut}", run, arm)
            rows.append(row)
            d[f"{run}/{arm}"] = lc
    md += md_table(LIFECYCLE_HEAD, rows) + [""]
    data["lifecycle"] = d

    md += ["## 5. Cost and latency", "",
           "Per episode, pooled over the four cells. Agent $ is the agent's own calls; arm $ is every other "
           "call the arm caused (reflection, repair agent, stand-in replay, judges). Memory s is time in memory "
           "calls per episode against the production API.", ""]
    md += md_table(["system", "tokens/ep", "agent $/ep", "arm $/ep", "memory s/ep", "wall s/ep"],
                   cost_rows({**systems, **{k: (r or sl_run, a) for k, (r, a) in ABLATIONS.items()}})) + [""]

    text = "\n".join(md)
    print(text)
    if args.paper:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "REPORT.md").write_text(text + "\n")
        (REPORT_DIR / "report.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
        print(f"\nwrote {REPORT_DIR / 'REPORT.md'} and report.json")


if __name__ == "__main__":
    main()
