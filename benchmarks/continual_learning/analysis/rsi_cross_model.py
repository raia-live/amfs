"""The regime-change study across two agent models (HYPOTHESIS_RSI.md, amendment (ac)).

Four systems, each from its own model, on identical task instances: the open model alone, the
open model with SenseLab, the frontier model alone, the frontier model with its own history in
context. Cells are keyed on ``(family, seed)`` — the model is part of the system, not of the
cell — so a system on one model pairs with a system on the other over the same tasks.

Emits only what the amendment names: the headline table, the paired deltas of the open model +
SenseLab against each other system with the bootstrap interval the publish gate reads, the
pre-change column, the fix lifecycle, model cost, tokens and the turn diagnostics. No engine-cut
or attribution sections; no ablation arm is read. One system is called SenseLab; the others are
external baselines.

    python -m continual_learning.analysis.rsi --cross-model --seeds 37 41 59 73 \\
        --models Beam-1.0-501B-A23B claude-opus-5-5 --runs rsi-conf-f1-beam-s37 ... --paper

``--models`` is ``<open model> <frontier model>``. ``--runs`` names the runs the cells live in;
a system resolves to the listed run that holds finished cells of its (arm, model). The readout
refuses to run for a model without a ``PRICES`` row (``config.cost_usd`` is silently 0.0
otherwise, and the cost column would be a lie).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .. import config
from .rsi import (
    BOOT,
    CELL_EPISODES,
    RESULTS,
    SCENARIOS,
    SENSELAB_ARM,
    Cell,
    boot_ci,
    episodes,
    fmt_rate,
    fmt_share,
    key_of_cell_id,
    lifecycle,
    md_table,
    reports,
    sign_flip_p,
)

REPORT_DIR = RESULTS / "rsi-report-beam"
FamilySeed = tuple[str, int]

#: Short display names; anything else is shown by its model string.
MODEL_LABELS = {"Beam-1.0-501B-A23B": "Beam", "claude-opus-5-5": "Opus 5.5", "claude-opus-5": "Opus 5",
                "gpt-5.5": "GPT-5.5"}

#: The publish gate's margins (amendment (ac)); on the paired delta's 95 % bootstrap interval.
GATE_MARGINS = {"together_moved": 0.0, "vs_frontier_cw_moved": -0.05, "collateral": -0.02, "cost_ratio": 1 / 3}

DIAG_KEYS = ("truncated_turns", "no_tool_call_turns", "malformed_tool_args", "unknown_tool_names", "client_retries")


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


class XCell(Cell):
    """A cell with the turn diagnostics, reasoning tokens and the Tier 2 replay cost (the
    agent's own model re-running its failing cases on the fix branch) on top of the base rates."""

    def __init__(self) -> None:
        super().__init__()
        self.diag: Counter[str] = Counter()
        self.reasoning = 0
        self.replay_cost = 0.0
        self.replay_tokens = 0
        self.replays = 0

    def add(self, e: dict[str, Any]) -> None:
        super().add(e)
        if e.get("error"):
            return
        for k, v in (e.get("diagnostics") or {}).items():
            self.diag[k] += int(v)
        self.reasoning += int((e.get("usage") or {}).get("reasoning_tokens") or 0)

    @property
    def model_cost(self) -> float:
        """The agent model's own calls: episodes plus Tier 2 replays."""
        return self.cost + self.replay_cost


def cells_of(runs: list[str], arm: str, model: str, seeds: tuple[int, ...]) -> tuple[str, dict[FamilySeed, XCell]]:
    """The finished cells of (*arm*, *model*) at *seeds*, gathered over every listed run: the
    study launches one run per seed (``rsi-conf-f1-beam-s37`` ...), so a system's cells are spread
    across runs. Each (family, seed) is taken from the first run that finished it; a cell still
    running is left out so no table mixes a complete cell with a partial one. The run name
    returned is the comma-joined list of the runs the cells came from."""
    done: dict[FamilySeed, XCell] = {}
    sources: list[str] = []
    for run in runs:
        out: dict[FamilySeed, XCell] = defaultdict(XCell)
        for e in episodes(run):
            if e.get("arm") != arm or e.get("model") != model or e.get("scenario") not in SCENARIOS \
                    or e.get("seed") not in seeds:
                continue
            out[(e["scenario"], int(e["seed"]))].add(e)
        here = {k: c for k, c in out.items() if k not in done and c.episodes + c.errors >= CELL_EPISODES}
        if not here:
            continue
        for r in reports(run):
            cell = r.get("cell", "")
            key = key_of_cell_id(cell)
            if key is None or cell.split("|")[1] != arm or key[2] != model or (key[0], key[1]) not in here:
                continue
            c = here[(key[0], key[1])]
            for rp in ((r.get("report") or {}).get("tier2") or {}).get("replays") or []:
                u = rp.get("usage") or {}
                c.replay_cost += float(u.get("cost_usd") or 0.0)
                c.replay_tokens += int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
                c.replays += 1
        done.update(here)
        sources.append(run)
    return (",".join(sources) if sources else "__not_run__"), done


def fixes_of(runs: list[str] | str, arm: str, model: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    """Every fix the engine proposed in the (*arm*, *model*) cells of *runs* (a list, or the
    comma-joined name ``cells_of`` returns)."""
    if isinstance(runs, str):
        runs = [r for r in runs.split(",") if r and r != "__not_run__"]
    out = []
    for run in runs:
        for r in reports(run):
            cell = r.get("cell", "")
            key = key_of_cell_id(cell)
            if key is None or cell.split("|")[1] != arm or key[2] != model or key[1] not in seeds:
                continue
            for fx in (r.get("report") or {}).get("fixes") or []:
                out.append({"scenario": key[0], "seed": key[1], "model": key[2], **fx})
    return out


# ---------------------------------------------------------------------------
# paired deltas with the interval the gate reads
# ---------------------------------------------------------------------------

def delta_ci(deltas: list[float], seed: int = 11) -> tuple[float, float] | None:
    """Cell-level bootstrap (2,000 resamples with replacement) of the mean paired delta."""
    if len(deltas) < 2:
        return None
    rng = random.Random(seed)
    vals = sorted(statistics.mean(rng.choice(deltas) for _ in deltas) for _ in range(BOOT))
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def paired(a: dict[FamilySeed, XCell], b: dict[FamilySeed, XCell], p: str,
           keys: list[FamilySeed]) -> dict[FamilySeed, float]:
    out: dict[FamilySeed, float] = {}
    for k in keys:
        x, y = a.get(k), b.get(k)
        if x and y and x.rate(p) is not None and y.rate(p) is not None:
            out[k] = x.rate(p) - y.rate(p)
    return out


def paired_row(name: str, d: dict[FamilySeed, float], keys: list[FamilySeed]) -> tuple[list[str], dict[str, Any]]:
    vals = list(d.values())
    mean = statistics.mean(vals) if vals else None
    ci = delta_ci(vals)
    pv = sign_flip_p(vals)
    row = [name, *[f"{100 * d[k]:+.0f}" if k in d else "—" for k in keys],
           f"{100 * mean:+.1f}" if mean is not None else "—",
           f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else "—",
           f"{sum(1 for v in vals if v > 0)}/{len(vals)}" if vals else "—",
           f"{pv:.3f}" if pv is not None else "—"]
    return row, {"deltas": {f"{f} s{s}": v for (f, s), v in d.items()}, "mean": mean, "ci95": ci,
                 "wins": sum(1 for v in vals if v > 0), "n": len(vals), "p_sign_flip": pv}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cross-model", action="store_true", help="(the dispatch flag; implied here)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[37, 41, 59, 73])
    ap.add_argument("--models", nargs=2, metavar=("OPEN", "FRONTIER"), required=True,
                    help="the open model (alone and with SenseLab) and the frontier model (alone and with context)")
    ap.add_argument("--runs", nargs="+", required=True, help="the runs the cells live in")
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--paper", action="store_true", help="write REPORT.md and report.json to --report-dir")
    args = ap.parse_args(argv)
    seeds = tuple(args.seeds)
    open_model, frontier = args.models
    for m in (open_model, frontier):
        if m not in config.PRICES:
            raise SystemExit(f"no PRICES row for {m!r} in config.py; the model-cost column would read 0.0. Refusing.")

    keys: list[FamilySeed] = [(s, seed) for s in SCENARIOS for seed in seeds]
    systems: dict[str, tuple[str, str]] = {   # display name -> (arm, model)
        f"{label(open_model)} alone": ("none", open_model),
        f"{label(open_model)} + SenseLab": (SENSELAB_ARM, open_model),
        f"{label(frontier)} alone": ("none", frontier),
        f"{label(frontier)} + context window": ("raw-traces", frontier),
    }
    sl_name = f"{label(open_model)} + SenseLab"
    resolved: dict[str, tuple[str, dict[FamilySeed, XCell]]] = {
        name: cells_of(args.runs, arm, model, seeds) for name, (arm, model) in systems.items()}
    sl_run, sl = resolved[sl_name]

    data: dict[str, Any] = {"study": "regime-change, cross-model (amendment (ac))", "seeds": list(seeds),
                            "open_model": open_model, "frontier_model": frontier, "scenarios": list(SCENARIOS),
                            "runs": {name: run for name, (run, _) in resolved.items()},
                            "prices": {m: config.PRICES[m].__dict__ if hasattr(config.PRICES[m], "__dict__")
                                       else str(config.PRICES[m]) for m in (open_model, frontier)},
                            "gate_margins": GATE_MARGINS}
    md: list[str] = [f"# The regime-change study on {label(open_model)} and {label(frontier)} "
                     f"(set C, seeds {', '.join(map(str, seeds))})", "",
                     f"Four systems, each from its own model, on identical task instances; cells are (family, seed), "
                     f"{len(keys)} per system. {label(open_model)}: `{open_model}`. {label(frontier)}: `{frontier}`. "
                     "SenseLab is one system and is always on; the other three are external baselines. "
                     "Infra errors are excluded from every rate. Brackets: cell-level bootstrap 95 % intervals.", ""]

    # 1. headline
    md += ["## 1. Headline", "",
           "First-attempt success on post-change episodes (30–99): moved half (changed-class instances whose "
           "rule changed), kept half (changed-class instances whose rule did not), unchanged classes; "
           "pre-change (0–29) is raw capability on the same tasks (Q1). Model $/ep is the agent model's own "
           "token cost — episodes plus, on the SenseLab cells, the Tier 2 replays the agent ran on the fix "
           "branch — at the rate in `config.PRICES`; nothing else is priced in it.", ""]
    head = ["system", "cells", "moved half", "kept half", "unchanged classes", "pre-change", "episode success",
            "escalated", "infra err", "tokens/ep", "model $/ep"]
    rows, hd = [], {}
    for name, (run, cells) in resolved.items():
        cs = list(cells.values())
        if not cs:
            rows.append([name, "0"] + ["—"] * (len(head) - 2))
            hd[name] = {"run": run, "cells": 0}
            continue
        eps = sum(c.episodes for c in cs)
        cost = sum(c.model_cost for c in cs)
        rows.append([name, str(len(cs)), fmt_rate(cs, "moved", True), fmt_rate(cs, "kept", True),
                     fmt_rate(cs, "unchanged"), fmt_rate(cs, "pre"),
                     fmt_share(sum(c.success for c in cs), eps), fmt_share(sum(c.escalated for c in cs), eps),
                     str(sum(c.errors for c in cs)), f"{sum(c.tokens for c in cs) / eps:,.0f}",
                     f"{cost / eps:.4f}"])
        hd[name] = {"run": run, "arm": systems[name][0], "model": systems[name][1], "cells": len(cs),
                    "episodes": eps, **{p: list(map(int, (sum(c.ok[p] for c in cs), sum(c.n[p] for c in cs))))
                                        for p in ("moved", "kept", "unchanged", "pre")},
                    "moved_ci": boot_ci(cs, "moved"), "kept_ci": boot_ci(cs, "kept"),
                    "success": sum(c.success for c in cs), "escalated": sum(c.escalated for c in cs),
                    "errors": sum(c.errors for c in cs), "tokens_per_episode": sum(c.tokens for c in cs) / eps,
                    "model_usd_per_episode": cost / eps, "agent_usd": sum(c.cost for c in cs),
                    "replay_usd": sum(c.replay_cost for c in cs), "replays": sum(c.replays for c in cs)}
    md += md_table(head, rows) + [""]
    data["headline"] = hd

    # 2. paired deltas (the gate's quantities)
    md += ["## 2. Paired deltas", "",
           f"{sl_name} minus each system (percentage points), paired on (family, seed) over the cells both have. "
           "CI: cell-level bootstrap 95 % interval of the mean paired delta — the quantity the publish gate "
           "reads. p: exact two-sided sign-flip test, descriptive.", ""]
    phead = ["pair", *[f"{f} s{s}" for f, s in keys], "mean", "95 % CI", "cells won", "p"]
    pd: dict[str, dict[str, Any]] = {}
    for p, title in (("moved", "moved half"), ("kept", "kept half"), ("unchanged", "unchanged classes")):
        rows = []
        pd[p] = {}
        for name, (_, other) in resolved.items():
            if name == sl_name:
                continue
            d = paired(sl, other, p, keys)
            if not d:
                continue
            row, dd = paired_row(f"{sl_name} − {name}", d, keys)
            rows.append(row)
            pd[p][name] = dd
        md += [f"### 2{'abc'[['moved', 'kept', 'unchanged'].index(p)]}. {title}", ""] + md_table(phead, rows) + [""]
    data["paired"] = pd

    # 3. the gate
    open_alone, fr_alone, fr_cw = (f"{label(open_model)} alone", f"{label(frontier)} alone",
                                   f"{label(frontier)} + context window")

    def lower(p: str, other: str) -> float | None:
        ci = (pd.get(p, {}).get(other) or {}).get("ci95")
        return ci[0] if ci else None

    def upper(p: str, other: str) -> float | None:
        ci = (pd.get(p, {}).get(other) or {}).get("ci95")
        return ci[1] if ci else None

    sl_cost = (hd.get(sl_name) or {}).get("model_usd_per_episode")
    cw_cost = (hd.get(fr_cw) or {}).get("model_usd_per_episode")
    cost_ratio = (sl_cost / cw_cost) if sl_cost is not None and cw_cost else None
    gate = {
        "G1_together_moved": {"lower": lower("moved", open_alone), "margin": GATE_MARGINS["together_moved"]},
        "G2_vs_frontier_cw_moved": {"lower": lower("moved", fr_cw), "upper": upper("moved", fr_cw),
                                    "margin": GATE_MARGINS["vs_frontier_cw_moved"]},
        "G3_collateral_kept": {"lower": lower("kept", fr_alone), "margin": GATE_MARGINS["collateral"]},
        "G3_collateral_unchanged": {"lower": lower("unchanged", fr_alone), "margin": GATE_MARGINS["collateral"]},
        "G4_cost_ratio": {"ratio": cost_ratio, "margin": GATE_MARGINS["cost_ratio"]},
    }
    for k, g in gate.items():
        if k == "G4_cost_ratio":
            g["pass"] = g["ratio"] is not None and g["ratio"] <= g["margin"]
        else:
            g["pass"] = g["lower"] is not None and g["lower"] > g["margin"]
    g1, g2, g3k, g3u, g4 = (gate[k]["pass"] for k in ("G1_together_moved", "G2_vs_frontier_cw_moved",
                                                     "G3_collateral_kept", "G3_collateral_unchanged", "G4_cost_ratio"))
    g2_up = gate["G2_vs_frontier_cw_moved"]["upper"]
    complete = all(len(cells) == len(keys) for _, cells in resolved.values())
    # Amendment (ac), in order of precedence: an interval entirely below 0 against the frontier's
    # best setup is "no post" whatever the other gates say (G2's −5 pp margin is a non-inferiority
    # floor, so an interval such as [−4, −1] clears it while still being worse); then the full
    # claim; then the "matches" claim when G2's interval covers 0.
    if not complete:
        verdict = "incomplete: not every system has all cells; no verdict"
    elif g2_up is not None and g2_up < 0:
        verdict = "no post: below the frontier model's best setup on recovery; readout to HEADROOM_RSI.md"
    elif g1 and g2 and g3k and g3u and g4:
        verdict = "full claim: better together, better than the frontier model, cheaper"
    elif g1 and g3k and g3u and g4 and g2_up is not None and g2_up >= 0:
        verdict = "matches the frontier model's best setup at a fraction of the model cost"
    else:
        verdict = "no post: gate not met"
    gate["verdict"] = verdict
    gate["complete"] = complete
    data["gate"] = gate
    md += ["## 3. Publish gate (amendment (ac), margins fixed before launch)", "",
           f"- G1 together: {sl_name} − {open_alone}, moved half, CI lower bound > 0: "
           f"{_fmt_bound(gate['G1_together_moved'])}",
           f"- G2 vs the frontier at its best: {sl_name} − {fr_cw}, moved half, lower bound > −5 pp: "
           f"{_fmt_bound(gate['G2_vs_frontier_cw_moved'])}",
           f"- G3 collateral: {sl_name} − {fr_alone}, kept half, lower bound > −2 pp: "
           f"{_fmt_bound(gate['G3_collateral_kept'])}; unchanged classes: {_fmt_bound(gate['G3_collateral_unchanged'])}",
           f"- G4 cheaper: model $/ep ratio ({sl_name} / {fr_cw}) ≤ 1/3: "
           + (f"{cost_ratio:.3f} → {'PASS' if g4 else 'FAIL'}" if cost_ratio is not None else "— (no cost)"),
           "", f"**Verdict: {verdict}.**", ""]

    # 4. by family
    md += ["## 4. By family", "", "First-attempt success per family: moved half, kept half, unchanged classes.", ""]
    fam_head = ["system"] + [f"{s}: {p}" for s in SCENARIOS for p in ("moved", "kept", "unchanged")]
    rows = []
    for name, (_, cells) in resolved.items():
        row = [name]
        for scen in SCENARIOS:
            cs = [c for (s, _), c in cells.items() if s == scen]
            row += [fmt_rate(cs, "moved"), fmt_rate(cs, "kept"), fmt_rate(cs, "unchanged")]
        rows.append(row)
    md += md_table(fam_head, rows) + [""]

    # 5. fix lifecycle (the SenseLab cells)
    md += ["## 5. Fix lifecycle", "",
           f"Every fix the engine proposed in the {sl_name} cells (runs `{sl_run.replace(',', '`, `')}`): Tier 1 is the stand-in replay on "
           "the fix branch; Tier 2 the agent's own replay of the fix's case set; a canary grades live exposure "
           "over the sessions after it starts — on this engine admitting only sessions in the fix's situation. "
           "`open` Tier 2 replays were still awaited when the cell ended.", ""]
    lc = lifecycle(fixes_of(sl_run, SENSELAB_ARM, open_model, seeds)) if sl else {}
    g = lc.get
    lrow = [[sl_name, str(g("proposed", 0)), f"{g('regime', 0)} / {g('judge', 0)}",
             f"{g('tier1:passed', 0)} / {g('tier1:inconclusive', 0)} / {g('tier1:failed', 0) + g('tier1:skipped', 0)}",
             f"{g('tier2:passed', 0)} / {g('tier2:failed', 0)} / {g('tier2:requested', 0) + g('tier2:expired', 0)}",
             f"{g('canary_started', 0)} / {g('canary:running', 0)}",
             str(g("shipped", 0)), str(g("reverted", 0)), fmt_share(g("by_repair_agent", 0), g("proposed", 0))]]
    md += md_table(["system", "proposed", "regime / judge", "T1 pass / inc. / fail", "T2 pass / fail / open",
                    "canaries run / undecided", "shipped", "reverted", "drafted by the repair agent"], lrow) + [""]
    data["lifecycle"] = lc

    # 6. tokens, cost, diagnostics
    md += ["## 6. Tokens, model cost and turn diagnostics", "",
           "Per episode. Tokens: the agent's prompt + completion tokens (reasoning included; on the SenseLab "
           "cells the retrieved context is in the prompt). Reasoning: tokens the model spent reasoning. "
           "Model $: the agent model's own calls at its `PRICES` rate — episodes, plus the Tier 2 replays on the "
           "SenseLab cells (replay $, counted in model $). Diagnostics are per 100 episodes: turns cut off at the "
           "output cap, turns with no tool call, tool calls whose arguments did not parse, calls to tools that do "
           "not exist, client-side retries (the frontier model's soft-forced tool call).", ""]
    chead = ["system", "tokens/ep", "reasoning/ep", "model $/ep", "of which replay $/ep", "replays",
             "truncated /100", "no tool call /100", "malformed args /100", "unknown tool /100", "retries /100",
             "wall s/ep"]
    rows, cd = [], {}
    for name, (run, cells) in resolved.items():
        cs = list(cells.values())
        eps = sum(c.episodes for c in cs)
        if not eps:
            continue
        diag = Counter()
        for c in cs:
            diag.update(c.diag)
        rows.append([name, f"{sum(c.tokens for c in cs) / eps:,.0f}", f"{sum(c.reasoning for c in cs) / eps:,.0f}",
                     f"{sum(c.model_cost for c in cs) / eps:.4f}", f"{sum(c.replay_cost for c in cs) / eps:.4f}",
                     str(sum(c.replays for c in cs)),
                     *[f"{100 * diag.get(k, 0) / eps:.1f}" for k in DIAG_KEYS],
                     f"{sum(c.wall_ms for c in cs) / eps / 1000:.0f}"])
        cd[name] = {"episodes": eps, "tokens_per_episode": sum(c.tokens for c in cs) / eps,
                    "reasoning_per_episode": sum(c.reasoning for c in cs) / eps,
                    "model_usd_per_episode": sum(c.model_cost for c in cs) / eps,
                    "replay_usd_per_episode": sum(c.replay_cost for c in cs) / eps,
                    "replays": sum(c.replays for c in cs), "diagnostics_per_100": {k: 100 * diag.get(k, 0) / eps
                                                                                   for k in DIAG_KEYS}}
    md += md_table(chead, rows) + [""]
    data["cost_and_diagnostics"] = cd

    text = "\n".join(md)
    print(text)
    if args.paper:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "REPORT.md").write_text(text + "\n")
        (args.report_dir / "report.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
        print(f"\nwrote {args.report_dir / 'REPORT.md'} and report.json")


def _fmt_bound(g: dict[str, Any]) -> str:
    if g.get("lower") is None:
        return "— (no paired cells)"
    return f"lower {100 * g['lower']:+.1f} pp → {'PASS' if g['pass'] else 'FAIL'}"


if __name__ == "__main__":
    main()
