"""Distillation grid: the students, their controls and the teachers on held-out seeds.

    python -m benchmarks.continual_learning.student.grid --dry            # size every phase
    python -m benchmarks.continual_learning.student.grid --launch main    # one phase, detached
    python -m benchmarks.continual_learning.student.grid --status

Every phase is one ``runner`` invocation (resumable). Seeds are new — none of the training
cells (grid v4: 11, 23, 37, 41, 59, 73) is reused — and every condition runs the same
scenarios, episodes, regime change and distractor load as the teacher grid, so the numbers
sit in the same tables as the frontier arms'. Conditions, per scenario with a trained student:

  teachers    gpt-5.5 / claude-opus-5, full tool loop, arms none + senselab      (the bar)
  students    <alias>  x arms {none, senselab}                                    (T2, T1)
              <alias>~plain x senselab   — context, no avoid rows / priors         (T3)
              <alias>~k0    x senselab   — the weights alone, nothing served
              <alias>+notes=gpt-5.4-mini x senselab — a cheap note-writer reflects (T5)
  controls    gpt-5.5@decision=<alias> x {none, senselab} — identical prompt, frontier answers
              <base alias> x senselab   — the untuned Gemini base behind the same prompt
  cross       students on the other teacher's scenario are not run: a student is per scenario.

The phases split by what they talk to: ``main`` (students and base controls: gateway only,
cheap, fast), ``ctrl`` (frontier decision controls), ``teach`` (frontier full loops). Cost
per episode from grid v4 for the frontier phases; students are ~40x cheaper per call.

``novel*`` phases run the same conditions under regime change set B (``--change-variant B``,
see Scenario.change_variant): a change the students' training traces never contain. Under A a
student carries both regimes in its weights and needs memory only to tell which applies; under
B whatever it recovers after the change is what memory gave it. ``novel`` is the students
(S1, S0 and the context-only S1p), ``novel-teach`` the frontier bar, ``novel-ctrl`` the
frontier on the served prompt for one student per scenario.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .. import config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOGS = config.RESULTS_DIR / "logs"

HELDOUT_SEEDS = ["101", "113", "127"]
# A self-retrained student learned from its parent's cells on the held-out seeds, so it and
# its parent are compared on a third set of seeds.
SELF_SEEDS = ["131", "137", "149"]
NOTE_WRITER = os.environ.get("CL_NOTE_WRITER_MODEL", "gpt-5.4-mini")
COST_PER_EP = {"main": 0.004, "ctrl": 0.020, "teach": 0.046,
               "novel": 0.004, "novel-ctrl": 0.020, "novel-teach": 0.046,
               "self": 0.004, "self-teach": 0.046}
PHASES = tuple(COST_PER_EP)


def registry() -> dict:
    try:
        return json.loads(config.STUDENT_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def students_by_scenario() -> dict[str, dict[str, list[str]]]:
    """scenario -> {"students": [alias...], "bases": [alias...]} for every live/base row."""
    out: dict[str, dict[str, list[str]]] = {}
    for alias, e in registry().items():
        sc = ((e.get("teacher") or {}).get("scenario")) or e.get("scenario")
        if not sc:
            continue
        slot = out.setdefault(sc, {"students": [], "bases": [], "self": []})
        if e.get("serve_base"):
            slot["bases"].append(alias)
        elif not (e.get("is_live") or e.get("state") in ("live", "ready")):
            continue
        elif e.get("parent"):
            # Trained on its parent's evaluation cells: kept out of every phase that runs on
            # those seeds and compared with the parent on SELF_SEEDS only.
            slot["self"].append((alias, e["parent"]))
        else:
            slot["students"].append(alias)
    return out


def phases(name: str) -> list[tuple[str, list[str]]]:
    # --opaque-ids on every distillation cell: counted ticket/PR numbers encode the episode
    # index and so the regime, and the first student learned that shortcut from its training
    # traces (0.81 post-change success with no memory). The runs before the fix are kept
    # under results/*-leaky for the record.
    base = ["--seeds", *HELDOUT_SEEDS, "--fleet", "1", "6", "--opaque-ids"]
    out: list[tuple[str, list[str]]] = []
    for sc, slot in sorted(students_by_scenario().items()):
        if name == "main":
            # A scenario with more than four students is split over two runner processes
            # (``main`` takes the first three in registry order, ``main2`` the rest) so the
            # students' cells, the long pole of the grid, run at twice the concurrency. The
            # analysis loads every ``distill-*`` run, so the split is invisible to it.
            halves = [slot["students"]] if len(slot["students"]) <= 4 else [slot["students"][:3], slot["students"][3:]]
            for i, students in enumerate(halves):
                models = []
                for a in students:
                    models += [a, f"{a}~plain", f"{a}~k0", f"{a}+notes={NOTE_WRITER}"]
                if models:
                    out.append((f"distill-main{i + 1 if i else ''}-{sc}", ["--scenarios", sc, "--arms", "none", "senselab",
                                                                          "--models", *models, *base]))
            if slot["bases"]:
                out.append((f"distill-base-{sc}", ["--scenarios", sc, "--arms", "senselab",
                                                   "--models", *slot["bases"], *base]))
        elif name == "ctrl":
            models = [f"gpt-5.5@decision={a}" for a in slot["students"]]
            if models:
                out.append((f"distill-ctrl-{sc}", ["--scenarios", sc, "--arms", "none", "senselab",
                                                   "--models", *models, *base]))
        elif name == "teach":
            out.append((f"distill-teach-{sc}", ["--scenarios", sc, "--arms", "none", "senselab",
                                                "--models", "gpt-5.5", "claude-opus-5", *base]))
        elif name == "novel":
            novel = [*base, "--change-variant", "B"]
            if slot["students"]:
                out.append((f"distill-novel-{sc}", ["--scenarios", sc, "--arms", "none", "senselab",
                                                    "--models", *slot["students"], *novel]))
                out.append((f"distill-novel-plain-{sc}", ["--scenarios", sc, "--arms", "senselab",
                                                          "--models", *[f"{a}~plain" for a in slot["students"]], *novel]))
        elif name == "novel-ctrl":
            # One student per scenario carries the served prompt: the v2 gpt-5.5 student, the
            # one trained on opaque-ids traces, else whichever is first.
            pick = sorted(slot["students"], key=lambda a: (not a.endswith("gpt55-v2"), a))[:1]
            if pick:
                out.append((f"distill-novel-ctrl-{sc}", ["--scenarios", sc, "--arms", "senselab",
                                                         "--models", f"gpt-5.5@decision={pick[0]}",
                                                         *base, "--change-variant", "B"]))
        elif name == "novel-teach":
            out.append((f"distill-novel-teach-{sc}", ["--scenarios", sc, "--arms", "none", "senselab",
                                                      "--models", "gpt-5.5", "claude-opus-5",
                                                      *base, "--change-variant", "B"]))
        elif name == "self" and slot["self"]:
            models = sorted({m for child, parent in slot["self"] for m in (child, parent)})
            out.append((f"distill-self-{sc}", ["--scenarios", sc, "--arms", "none", "senselab", "--models", *models,
                                               "--seeds", *SELF_SEEDS, "--fleet", "1", "6", "--opaque-ids"]))
        elif name == "self-teach" and slot["self"]:
            out.append((f"distill-self-teach-{sc}", ["--scenarios", sc, "--arms", "none", "senselab",
                                                     "--models", "gpt-5.5", "claude-opus-5",
                                                     "--seeds", *SELF_SEEDS, "--fleet", "1", "6", "--opaque-ids"]))
    return out


def runner_cmd(run: str, args: list[str], workers: int, dry: bool = False) -> list[str]:
    cmd = [PY, "-m", "benchmarks.continual_learning.runner", "--run", run, *args, "--workers", str(workers)]
    return cmd + (["--dry"] if dry else [])


def launch(name: str, workers: int) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("AMFS_RPM", "600")
    env.setdefault("CL_SENSELAB_CONCURRENCY", "16")
    chain = " && ".join(" ".join(f"'{a}'" for a in runner_cmd(run, args, workers)) for run, args in phases(name))
    if not chain:
        sys.exit(f"phase {name}: nothing to run (registry has no live students)")
    log = (LOGS / f"distill-{name}.log").open("a")
    p = subprocess.Popen(["/bin/sh", "-c", chain], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(f"launched {name}: pid {p.pid}, {len(phases(name))} runs, log {LOGS}/distill-{name}.log")


def dry() -> None:
    total_e = 0
    for name in PHASES:
        for run, args in phases(name):
            res = subprocess.run(runner_cmd(run, args, 1, dry=True), cwd=ROOT, capture_output=True, text=True)
            line = [ln for ln in res.stdout.splitlines() if "episodes" in ln]
            print(f"  {run:32s} {line[-1] if line else res.stdout.strip()[-120:]}")
            try:
                e = int(line[-1].split("episodes")[0].split()[-1])
                total_e += e
                print(f"  {'':32s} ~${e * COST_PER_EP[name]:,.0f}")
            except (IndexError, ValueError):
                pass
    print(f"TOTAL ~{total_e} episodes")


def status() -> None:
    for name in PHASES:
        for run, _ in phases(name):
            d = config.RESULTS_DIR / run
            eps = sum(1 for _ in (d / "episodes.jsonl").open()) if (d / "episodes.jsonl").exists() else 0
            cells = len(json.loads((d / "cells.json").read_text())) if (d / "cells.json").exists() else 0
            print(f"  {run:32s} cells {cells:4d} episodes {eps:6d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--launch", choices=list(PHASES))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.dry:
        dry()
    elif a.launch:
        launch(a.launch, a.workers)
    elif a.status:
        status()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
