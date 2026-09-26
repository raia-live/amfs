"""Which runs and arms the paper reads (``--paper`` on every analysis).

The paper compares three things: no memory (``none``), a hosted memory product (``mem0``) and
continual learning (``senselab``), with the platform's own no-outcome ablation
(``senselab-nofeedback``: identical retrieval and briefing, the outcome loop never called) as
the control that separates "memory" from "learning" on one codebase. pgvector arms are not in
the paper's tables (2026-09-20 decision); they stay in the grid v4 report.

SenseLab is reported as one system at the latest production build. The SenseLab arm was
re-run per scenario after each production fix, so for every (SenseLab arm, scenario) the
paper reads the *newest* run that has a usable share of that scenario's cells finished, in
``SL_RUN_PRECEDENCE`` order, and ignores the same scenario in the older runs. Baseline arms
come from the grid v4 family runs unchanged.

    python -m benchmarks.continual_learning.analysis.regime_change --paper
    python -m benchmarks.continual_learning.analysis.transfer --paper
    python -m benchmarks.continual_learning.analysis.probes --paper
    python -m benchmarks.continual_learning.analysis.figures --paper

Output goes to ``results/paper-report``.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
FAMILY_PREFIX = "gridv4"
OUT = RESULTS / "paper-report"

PAPER_ARMS = ["none", "mem0", "senselab-nofeedback", "senselab"]
SL_ARMS = {"senselab", "senselab-nofeedback", "senselab-nopriors", "senselab-induce"}

#: Newest first. A scenario is taken from the first run in which at least ``MIN_CELL_SHARE``
#: of the cells that run has for the scenario are finished (``cells.json``), so a re-run in
#: progress does not mix half of its cells with an older build's.
SL_RUN_PRECEDENCE = ["gridv5c-core-sl", "gridv5b-concierge-sl", "gridv5-core-sl", "gridv4-core-sl"]
MIN_CELL_SHARE = 0.5
MIN_CELLS_DONE = 12     # half of a scenario's 24 core cells (6 seeds x 2 models x fleet 1/6)

#: (arm, scenario) -> run the paper reads for the 60-episode regime-change cells.
SCENARIO_RUN: dict[tuple[str, str], str] = {}


def _cells_by_scenario(run: str) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """(arm, scenario) -> cells with any rows, and -> cells finished, for one run."""
    d = RESULTS / run
    seen: dict[tuple[str, str], set[str]] = {}
    if (d / "episodes.jsonl").exists():
        for line in (d / "episodes.jsonl").read_text().splitlines():
            x = json.loads(line)
            if x.get("fleet_mode", "rr") != "rr":
                continue
            seen.setdefault((x["arm"], x["scenario"]), set()).add(x["cell"])
    done: dict[tuple[str, str], set[str]] = {}
    if (d / "cells.json").exists():
        for c in json.loads((d / "cells.json").read_text()):
            parts = c["cell"].split("|")
            if len(parts) >= 2:
                done.setdefault((parts[1], parts[0]), set()).add(c["cell"])
    return {k: len(v) for k, v in seen.items()}, {k: len(v) for k, v in done.items()}


def resolve_scenario_runs() -> dict[tuple[str, str], str]:
    """Pick the newest usable run per (SenseLab arm, scenario)."""
    picked: dict[tuple[str, str], str] = {}
    for run in SL_RUN_PRECEDENCE:
        seen, done = _cells_by_scenario(run)
        for key, n_seen in seen.items():
            if key in picked or key[0] not in SL_ARMS:
                continue
            if done.get(key, 0) >= max(MIN_CELLS_DONE, MIN_CELL_SHARE * n_seen):
                picked[key] = run
    return picked


def configure(rc=None) -> None:
    """Point regime_change (and through it transfer/figures) at the paper's sources.

    ``rc`` is the regime_change module object to configure. It must be passed when
    regime_change itself is the ``__main__`` module (``python -m ...regime_change``), since a
    relative import from here would configure a second copy of it."""
    if rc is None:
        from . import regime_change as rc

    global SCENARIO_RUN
    SCENARIO_RUN = resolve_scenario_runs()
    skip = ("-report", "-xfer-", "-probes", "-long", "-dry", "-sweep-")
    family = sorted(p.name for p in RESULTS.glob(f"{FAMILY_PREFIX}*") if (p / "episodes.jsonl").exists()
                    and not any(s in p.name[len(FAMILY_PREFIX):] or p.name.endswith(s) for s in skip))
    sl_runs = [r for r in SL_RUN_PRECEDENCE if (RESULTS / r / "episodes.jsonl").exists()]
    rc.SOURCES = {a: (sorted(set(family) | set(sl_runs)) if a in SL_ARMS else family) for a in PAPER_ARMS}
    rc.ARMS = list(rc.SOURCES)
    rc.HEAD = rc.order_arms(rc.ARMS)
    rc.RUNS_PREFIX = "paper"
    rc.OUT = OUT
    rc.SCENARIO_RUN = SCENARIO_RUN
    rc.LABEL.update({"none": "no memory", "mem0": "memory (Mem0)",
                     "senselab-nofeedback": "SenseLab, outcome loop off", "senselab": "SenseLab, outcome loop on"})


def keep_row(arm: str, scenario: str, run: str) -> bool:
    """False when a SenseLab-arm row comes from a run the paper does not read for that scenario.

    The choice applies to the 60-episode core cells only; the family's other runs (fleet-10
    and the like) have no re-run and are read as they are."""
    if arm not in SL_ARMS:
        return True
    if run.startswith(FAMILY_PREFIX) and "-core-" not in run:
        return True
    chosen = SCENARIO_RUN.get((arm, scenario))
    if chosen is None:
        # No re-run for this scenario: read the family run only, never a re-run's stray cells.
        return run.startswith(FAMILY_PREFIX)
    return run == chosen


def describe() -> str:
    lines = ["SenseLab-arm sources by scenario (newest usable run):"]
    for (arm, sc), run in sorted(SCENARIO_RUN.items()):
        lines.append(f"  {arm:22s} {sc:12s} <- {run}")
    return "\n".join(lines)
