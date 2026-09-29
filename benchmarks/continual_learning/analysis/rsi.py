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
    # rsi-c4 (post-seam8) was stopped at 323 episodes when #798 merged (HYPOTHESIS (t)); it is
    # kept on disk and used for no prediction, so it is not a row here.
    ("rsi-c5", "post-clean-success",
     "#797 replays are not traffic; #797/#798 proof cases need a clean success, whatever grades them"),
    ("rsi-c6", "post-canary-ni",
     "#799 canary grades runs in the situation or that read an implicated entry; non-inferiority primary; "
     "regime draft sees the situation's passing runs"),
]
SCENARIOS = ("support", "ci-fix")
#: development seeds and model; ``--seeds 37 41 59 73 --models gpt-5.5 claude-opus-5`` is the
#: confirmatory read (HYPOTHESIS_RSI.md, amendment (w)). A cell is (family, seed, model).
SEEDS: tuple[int, ...] = (11, 23)
MODELS: tuple[str, ...] = ("gpt-5.5",)
BOOT = 2000
CellKey = tuple[str, int, str]


def cell_keys() -> list[CellKey]:
    return [(s, seed, m) for s in SCENARIOS for seed in SEEDS for m in MODELS]


def key_of_cell_id(cell: str) -> CellKey | None:
    """``family|arm|model|sNN|...`` -> (family, seed, model)."""
    parts = cell.split("|")
    if len(parts) < 4:
        return None
    return parts[0], int(parts[3].lstrip("s")), parts[2]


def cell_label(k: CellKey) -> str:
    fam, seed, model = k
    return f"{fam} s{seed}" if len(MODELS) == 1 else f"{fam} s{seed} {model}"


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


def cells_of(run: str, arm: str) -> dict[CellKey, Cell]:
    """The finished cells of *arm* in *run*; a cell still running (or aborted) is left out,
    so a table never mixes a complete cell with a partial one."""
    out: dict[CellKey, Cell] = defaultdict(Cell)
    for e in episodes(run):
        if (e.get("arm") != arm or e.get("scenario") not in SCENARIOS or e.get("seed") not in SEEDS
                or e.get("model") not in MODELS):
            continue
        out[(e["scenario"], int(e["seed"]), e["model"])].add(e)
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
        key = key_of_cell_id(cell)
        if key is None or cell.split("|")[1] != arm or key[1] not in SEEDS or key[2] not in MODELS:
            continue
        for fx in (r.get("report") or {}).get("fixes") or []:
            out.append({"scenario": key[0], "seed": key[1], "model": key[2], **fx})
    return out


CALIBRATION = REPORT_DIR / "platform_calibration.json"


def per_draft_usd() -> float | None:
    """The account-ledger remainder per fix the repair agent drafted (repair agent, Tier 1
    stand-in replay, lever, discovery), from ``analysis/platform_calibration.py``; ``None``
    when the calibration has not been taken."""
    if not CALIBRATION.exists():
        return None
    return json.loads(CALIBRATION.read_text()).get("per_draft_usd")


def platform_costs(run: str, arm: str) -> dict[tuple[str, int], dict[str, float]]:
    """What the loop spent on the platform side per cell, from the arm report: every judge
    verdict's ``cost_usd`` (exact), every Tier 2 replay the harness re-ran for the fix's case
    set (exact; the agent's own calls on the branch), and the number of fixes the repair
    agent drafted (priced at :func:`per_draft_usd`, an estimate)."""
    out: dict[CellKey, dict[str, float]] = {}
    for r in reports(run):
        cell = r.get("cell", "")
        key = key_of_cell_id(cell)
        if key is None or cell.split("|")[1] != arm:
            continue
        rep = r.get("report") or {}
        d = out.setdefault(key, {"judge_usd": 0.0, "replay_usd": 0.0, "drafts": 0, "fixes": 0, "verdicts": 0})
        for v in (rep.get("verdicts") or {}).get("items") or []:
            d["judge_usd"] += float(v.get("cost_usd") or 0.0)
            d["verdicts"] += 1
        for rp in (rep.get("tier2") or {}).get("replays") or []:
            d["replay_usd"] += float((rp.get("usage") or {}).get("cost_usd") or 0.0)
        for fx in rep.get("fixes") or []:
            d["fixes"] += 1
            d["drafts"] += 1 if (fx.get("manifest") or {}).get("model") else 0
    if out:
        return out
    # Arms that write no arm report (``senselab`` with the loop held at review, the oracle
    # arms) still cause platform spend — regime proposals drafted and held. Their per-cell
    # counts are dumped from the account by ``analysis/platform_calibration.py --runs``.
    ledger = RESULTS / run / "platform_ledger.json"
    if ledger.exists():
        for cell, d in (json.loads(ledger.read_text()).get("cells") or {}).items():
            key = key_of_cell_id(cell)
            if key is not None and cell.split("|")[1] == arm:
                out[key] = {
                    "judge_usd": float(d.get("judge_usd", 0.0)), "replay_usd": 0.0,
                    "drafts": int(d.get("drafts", 0)), "fixes": int(d.get("fixes", 0)),
                    "verdicts": int(d.get("verdicts", 0))}
    return out


def ledger_delta(run: str) -> dict[str, Any] | None:
    """The run's platform spend as the account ledger saw it: last reading minus first, from
    ``budget_snapshots.jsonl`` (runs started after 2026-09-29). Shared with any run on the
    account at the same time, which the caller says."""
    path = RESULTS / run / "budget_snapshots.jsonl"
    if not path.exists():
        return None
    snaps = [json.loads(line) for line in path.open() if line.strip()]
    snaps = [s for s in snaps if s.get("spent_usd") is not None]
    if len(snaps) < 2:
        return None
    return {"start": snaps[0]["at_iso"], "end": snaps[-1]["at_iso"],
            "delta_usd": round(float(snaps[-1]["spent_usd"]) - float(snaps[0]["spent_usd"]), 4),
            "complete": snaps[-1].get("label") == "run_end"}


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


def sign_flip_p(deltas: list[float], seed: int = 11) -> float | None:
    """Two-sided p-value of the mean paired delta under the null that each cell's delta is
    symmetric about zero: exact enumeration of all 2^n sign assignments up to n = 20, Monte
    Carlo (20,000 draws) beyond. Needs no distributional assumption and is the paired test
    the preregistration's cell-level design calls for; with n = 4 (dev seeds) the smallest
    attainable p is 0.125, which is why dev tables carry no stars."""
    n = len(deltas)
    if n < 2:
        return None
    obs = abs(sum(deltas))
    if n <= 20:
        hits = total = 0
        for mask in range(1 << n):
            s = sum(d if mask >> i & 1 else -d for i, d in enumerate(deltas))
            hits += abs(s) >= obs - 1e-12
            total += 1
        return hits / total
    rng = random.Random(seed)
    draws = 20000
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in deltas)) >= obs - 1e-12 for _ in range(draws))
    return (hits + 1) / (draws + 1)


def holm(pvals: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjustment over the family of tests given (None passes through)."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float | None] = {k: None for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def paired_deltas(systems: dict[str, tuple[str, str]], senselab: tuple[str, str], p: str
                  ) -> dict[str, dict[CellKey, float]]:
    """SenseLab minus each system on population *p*, per cell, over the cells both have."""
    sl = cells_of(*senselab)
    out: dict[str, dict[CellKey, float]] = {}
    for name, (run, arm) in systems.items():
        other = cells_of(run, arm)
        d: dict[CellKey, float] = {}
        for k in cell_keys():
            a, b = sl.get(k), other.get(k)
            if a and b and a.rate(p) is not None and b.rate(p) is not None:
                d[k] = a.rate(p) - b.rate(p)
        out[name] = d
    return out


def paired_rows(systems: dict[str, tuple[str, str]], senselab: tuple[str, str], p: str
                ) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    """Per-cell paired deltas (percentage points) when the design has at most eight cells;
    per family × model means beyond that. Always: the mean, cells won, the exact sign-flip p
    and its Holm adjustment across the systems compared."""
    keys = cell_keys()
    per_cell = len(keys) <= 8
    deltas = paired_deltas(systems, senselab, p)
    pv = {name: sign_flip_p(list(d.values())) for name, d in deltas.items()}
    adj = holm(pv)
    if per_cell:
        cols = [cell_label(k) for k in keys]
    else:
        cols = [f"{fam} {m}" for fam in SCENARIOS for m in MODELS]
    rows, data = [], {}
    for name, d in deltas.items():
        if not d:
            continue
        if per_cell:
            cells = [f"{100 * d[k]:+.0f}" if k in d else "—" for k in keys]
        else:
            cells = []
            for fam in SCENARIOS:
                for m in MODELS:
                    v = [x for (f, _, mm), x in d.items() if f == fam and mm == m]
                    cells.append(f"{100 * statistics.mean(v):+.0f}" if v else "—")
        mean = statistics.mean(d.values())
        wins = sum(1 for x in d.values() if x > 0)
        rows.append([f"SenseLab − {name}", *cells, f"{100 * mean:+.0f}", f"{wins}/{len(d)}",
                     f"{pv[name]:.3f}" if pv[name] is not None else "—",
                     f"{adj[name]:.3f}" if adj[name] is not None else "—"])
        data[name] = {"deltas": {cell_label(k): v for k, v in d.items()}, "mean": mean, "wins": wins,
                      "n": len(d), "p_sign_flip": pv[name], "p_holm": adj[name]}
    return ["pair", *cols, "mean", "cells won", "p", "p (Holm)"], rows, data


def by_family_rows(systems: dict[str, tuple[str, str]]) -> list[list[str]]:
    rows = []
    for name, (run, arm) in systems.items():
        cells = cells_of(run, arm)
        row = [name]
        for scen in SCENARIOS:
            cs = [c for (s, _, _), c in cells.items() if s == scen]
            row += [fmt_rate(cs, "moved"), fmt_rate(cs, "kept"), fmt_rate(cs, "unchanged")]
        rows.append(row)
    return rows


def by_model_rows(systems: dict[str, tuple[str, str]]) -> list[list[str]]:
    rows = []
    for name, (run, arm) in systems.items():
        cells = cells_of(run, arm)
        row = [name]
        for m in MODELS:
            cs = [c for (_, _, mm), c in cells.items() if mm == m]
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


COST_HEAD = ["system", "tokens/ep", "agent $/ep", "arm $/ep", "judges $/ep", "replays $/ep",
             "drafts/100 ep", "drafts $/ep (est.)", "total $/ep", "memory s/ep", "wall s/ep"]


def cost_rows(systems: dict[str, tuple[str, str]]) -> tuple[list[list[str]], dict[str, Any]]:
    """Per-episode spend by where it is charged. *agent*: the agent's own calls (episode
    records). *arm*: reflection / consolidation the arm ran per episode (episode records).
    *judges*, *replays*: the loop's platform-side judge verdicts and the Tier 2 replays the
    harness re-ran, both exact from the arm reports. *drafts*: fixes the repair agent
    drafted, priced at the account ledger's remainder per draft (``platform_calibration.json``)
    — the repair agent, Tier 1 stand-in replay, lever and discovery are not persisted per fix,
    so this column is an estimate and says so. *total* sums the five."""
    rows, data = [], {}
    draft_usd = per_draft_usd()
    for name, (run, arm) in systems.items():
        cells = cells_of(run, arm)
        eps = sum(c.episodes for c in cells.values())
        if not eps:
            continue
        plat = platform_costs(run, arm)
        judge = sum(plat.get(k, {}).get("judge_usd", 0.0) for k in cells)
        replay = sum(plat.get(k, {}).get("replay_usd", 0.0) for k in cells)
        drafts = sum(int(plat.get(k, {}).get("drafts", 0)) for k in cells)
        drafts_usd = drafts * draft_usd if draft_usd is not None else None
        agent = sum(c.cost for c in cells.values())
        extra = sum(c.extra_cost for c in cells.values())
        total = agent + extra + judge + replay + (drafts_usd or 0.0)
        rows.append([name, f"{sum(c.tokens for c in cells.values()) / eps:,.0f}",
                     f"{agent / eps:.3f}", f"{extra / eps:.3f}", f"{judge / eps:.3f}", f"{replay / eps:.3f}",
                     f"{100 * drafts / eps:.1f}",
                     f"{drafts_usd / eps:.3f}" if drafts_usd is not None else ("—" if drafts else "0.000"),
                     f"{total / eps:.3f}" + ("+" if drafts and drafts_usd is None else ""),
                     f"{sum(c.memory_ms for c in cells.values()) / eps / 1000:.1f}",
                     f"{sum(c.wall_ms for c in cells.values()) / eps / 1000:.0f}"])
        data[name] = {"run": run, "episodes": eps, "agent_usd": agent, "arm_usd": extra, "judge_usd": judge,
                      "replay_usd": replay, "drafts": drafts, "drafts_usd_est": drafts_usd, "total_usd": total,
                      "usd_per_episode": total / eps, "ledger": ledger_delta(run)}
    return rows, data


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
    global SEEDS, MODELS, REPORT_DIR  # noqa: PLW0603 — module constants the tables close over
    ap = argparse.ArgumentParser()
    ap.add_argument("--senselab-run", default=None, help="the SenseLab run for the headline (default: newest complete)")
    ap.add_argument("--paper", action="store_true", help="write results/rsi-report/REPORT.md and report.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                    help="cell seeds (dev: 11 23; confirmatory: 37 41 59 73)")
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="agent models the cells were run on")
    ap.add_argument("--report-dir", type=Path, default=None,
                    help="where REPORT.md / report.json go (default results/rsi-report; the confirmatory read "
                         "uses results/rsi-report-confirmatory so it never overwrites the dev readout)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="restrict every system to these runs (the confirmatory read names its own runs; "
                         "a system whose run is not listed is reported with 0 cells)")
    args = ap.parse_args()
    SEEDS, MODELS = tuple(args.seeds), tuple(args.models)
    if args.report_dir:
        REPORT_DIR = args.report_dir
    confirmatory = set(SEEDS) & {37, 41, 59, 73} and not set(SEEDS) & {11, 23}

    def resolve(arm: str, default_run: str) -> str:
        """With ``--runs``, the listed run that holds finished cells of *arm* at these seeds and
        models (the confirmatory runs have their own names); otherwise the dev default."""
        if args.runs is None:
            return default_run
        for run in args.runs:
            if cells_of(run, arm):
                return run
        return "__not_run__"

    sl_run = args.senselab_run or (resolve(SENSELAB_ARM, "") if args.runs else newest_complete_senselab_run())
    senselab = (sl_run, SENSELAB_ARM)
    systems: dict[str, tuple[str, str]] = {**{k: (resolve(a, r), a) for k, (r, a) in BASELINES.items()},
                                           "SenseLab": senselab}
    ablations = {k: (resolve(a, r or sl_run), a) for k, (r, a) in ABLATIONS.items()}
    data: dict[str, Any] = {"senselab_run": sl_run, "seeds": list(SEEDS), "models": list(MODELS),
                            "scenarios": list(SCENARIOS), "confirmatory": bool(confirmatory)}
    n_cells = len(cell_keys())
    kind = "confirmatory readout" if confirmatory else "development-seed readout"
    md: list[str] = [f"# RSI paper — {kind} (set C, seeds {', '.join(map(str, SEEDS))}, "
                     f"{' / '.join(MODELS)}; SenseLab from `{sl_run}`)", ""]

    md += ["## 1. After the rule changes conditionally", "",
           f"First-attempt success on every post-change episode (30–99), pooled over the {n_cells} cells "
           f"(support / ci-fix × seeds {' / '.join(map(str, SEEDS))} × {' / '.join(MODELS)}). Moved half: "
           "changed-class instances whose rule changed; kept half: changed-class instances whose rule did not. "
           "Brackets: cell-level bootstrap 95 % interval. Infra errors are excluded from every rate.", ""]
    rows, d = system_rows(systems, ci=True)
    md += md_table(HEAD, rows) + [""]
    data["headline"] = d

    md += ["## 1b. Paired cell deltas, moved half", "",
           "SenseLab minus each system on the moved half (percentage points), paired on (family, seed, model). "
           "p: exact two-sided sign-flip test of the mean paired delta over the cells; Holm: adjusted across "
           "the systems in this table. "
           + ("Four cells cannot reach p < 0.125; the dev read carries no significance claim."
              if n_cells <= 4 else f"{n_cells} cells; the smallest attainable p is {2 / 2 ** n_cells:.4f}."), ""]
    head, rows, d = paired_rows(BASELINES, senselab, "moved")
    md += md_table(head, rows) + [""]
    data["paired_moved"] = d
    secondary: dict[str, Any] = {}
    for p in ("kept", "unchanged"):
        _, _, dd = paired_rows(BASELINES, senselab, p)
        secondary[p] = dd
    data["paired_secondary"] = secondary
    if confirmatory:
        for p, title in (("kept", "kept half"), ("unchanged", "unchanged classes")):
            head, rows, _ = paired_rows(BASELINES, senselab, p)
            md += [f"### 1b-{p}. Secondary endpoint, paired: {title}", "",
                   "Holm within this endpoint.", ""] + md_table(head, rows) + [""]

    md += ["## 1c. By family", "", "First-attempt success per family: moved half, kept half, unchanged classes.", ""]
    fam_head = ["system"] + [f"{s}: {p}" for s in SCENARIOS for p in ("moved", "kept", "unchanged")]
    md += md_table(fam_head, by_family_rows(systems)) + [""]
    if len(MODELS) > 1:
        md += ["## 1d. By model", "", "First-attempt success per agent model: moved half, kept half, unchanged classes.", ""]
        model_head = ["system"] + [f"{m}: {p}" for m in MODELS for p in ("moved", "kept", "unchanged")]
        md += md_table(model_head, by_model_rows(systems)) + [""]

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
           f"(same run) and without the repair loop (outcome learning alone, "
           f"`{ablations['SenseLab − repair loop'][0]}`).", ""]
    abl: dict[str, tuple[str, str]] = {"SenseLab": senselab, **ablations}
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

    cal = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
    md += ["## 5. Cost and latency", "",
           "Per episode, pooled over the four cells, by where the call is charged. Agent $: the agent's own "
           "calls. Arm $: reflection / consolidation the arm ran per episode. Judges $ and replays $: the "
           "loop's judge verdicts and the Tier 2 replays the harness re-ran on the fix branch, exact from the "
           "arm reports. Drafts: fixes the repair agent drafted per 100 episodes, priced at the account "
           "ledger's remainder per draft (repair agent, Tier 1 stand-in replay, lever, discovery — not "
           "persisted per fix, so an estimate: "
           + (f"${cal.get('per_draft_usd', 0):.3f} per draft from ${cal.get('spent_usd', 0):.2f} spent, "
              f"${cal.get('verdict_usd', 0):.2f} of it judges, over {cal.get('drafted_fixes', 0)} drafts on the "
              f"account at {cal.get('taken_at', '?')}" if cal else "calibration not taken")
           + "). Total sums the five. Memory s: time in memory calls per episode against the production API.",
           ""]
    rows, d = cost_rows({**systems, **ablations})
    md += md_table(COST_HEAD, rows) + [""]
    data["cost"] = {"calibration": cal, "systems": d}
    ledgers = {k: v["ledger"] for k, v in d.items() if v.get("ledger")}
    if ledgers:
        md += ["Account-ledger check (runs whose runner snapshotted the ledger; a delta is shared with any "
               "other run on the account in the same window):", ""]
        md += [f"- {k}: `{v['run']}` ledger delta ${ledgers[k]['delta_usd']:.2f} from {ledgers[k]['start']} to "
               f"{ledgers[k]['end']}{'' if ledgers[k]['complete'] else ' (run not finished)'}; estimate for the "
               f"run's cells ${v['judge_usd'] + (v['drafts_usd_est'] or 0):.2f}"
               for k, v in d.items() if v.get("ledger")] + [""]

    text = "\n".join(md)
    print(text)
    if args.paper:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "REPORT.md").write_text(text + "\n")
        (REPORT_DIR / "report.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
        print(f"\nwrote {REPORT_DIR / 'REPORT.md'} and report.json")


if __name__ == "__main__":
    main()
