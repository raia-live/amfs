"""Resumable benchmark runner.

A *cell* is (scenario, arm, model, seed, sweep). Each cell gets its own isolated memory
scope, is seeded identically, and runs ``episodes`` episodes in order. Records stream to
``results/<run>/episodes.jsonl`` and finished cells are skipped on restart.

    python -m benchmarks.continual_learning.runner --run pilot --scenarios runbook diagnose \
        --arms none senselab --models gpt-5.5 --seeds 11 --episodes 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .agent import get_llm, run_episode
from .arms import make_arm
from .scenarios import make_scenario

# Hosted arms share one production database; 12 concurrent SenseLab cells at 1000 RPM saturated a
# 2-vCPU Cloud SQL instance on 2026-09-16. The default stays at 2 (override with
# CL_SENSELAB_CONCURRENCY; gridv4 sets 24 against the resized instance and 12-instance API tier);
# the client-side limiter (AMFS_RPM) caps total request rate. Mem0 likewise (CL_MEM0_CONCURRENCY).
# The semaphore is per arm, so --workers is the binding cap when several arms run in one process.
_SL = int(os.environ.get("CL_SENSELAB_CONCURRENCY", "2"))
_M0 = int(os.environ.get("CL_MEM0_CONCURRENCY", "6"))
_ARM_CONCURRENCY = {"mem0": _M0, "zep": 8, "senselab": _SL, "senselab-episode": _SL,
                    "senselab-nofeedback": _SL, "senselab-attempts": _SL, "senselab-nopriors": _SL,
                    "senselab-lean": _SL, "senselab-induce": _SL, "skillfile": _SL}
_sems: dict[str, threading.Semaphore] = {}
_write_lock = threading.Lock()


def _sem(arm: str) -> threading.Semaphore:
    if arm not in _sems:
        _sems[arm] = threading.Semaphore(_ARM_CONCURRENCY.get(arm, 6))
    return _sems[arm]


@dataclass(frozen=True)
class Cell:
    scenario: str
    arm: str
    model: str
    seed: int
    episodes: int
    distractors: int = 0
    label_noise: float = 0.0
    feedback_delay: int = 0
    fleet: int = 1
    change_at: int = 0
    change_focus: float = 0.0
    fleet_mode: str = "rr"     # rr | pioneer | newcomer (see Scenario)
    join_at: int = 0
    # Cross-model onboarding (newcomer protocol only): the model the arriving agent N runs on.
    # "" = same model as the veterans. Veterans keep ``model``; the newcomer's episode rows
    # carry ``newcomer_model`` in their ``model`` field because EpisodeRecord.model is llm.model.
    newcomer_model: str = ""
    # Task identifiers drawn from the instance rng instead of counted from the episode index
    # (see Scenario.opaque_ids). Part of the cell id (``-oid``) so results never mix with
    # cells whose ticket numbers encoded the regime.
    opaque_ids: bool = False
    # Which regime change set applies at ``change_at`` (see Scenario.change_variant). "A" is
    # the original and adds nothing to ids; any other value is part of the cell id and scope
    # (``-vB``) so a novel-change evaluation never mixes with the change the students trained on.
    change_variant: str = "A"
    # Fleet members act as replicas of ONE hosted agent identity (HYPOTHESIS_RSI.md): the
    # repair loop's fixes, judges, incidents and canary slot are scoped per agent, so a fleet
    # that shares one loop must share one identity. Part of the cell id (``-rep``) so fleet
    # rows never mix with the v7 one-identity-per-member protocol. Repair-loop arms force it.
    fleet_replicas: bool = False

    @property
    def _variant(self) -> str:
        return f"-v{self.change_variant}" if self.change_at and self.change_variant != "A" else ""

    @property
    def _replicas(self) -> str:
        return "-rep" if self.fleet > 1 and self.fleet_replicas else ""

    @property
    def _transfer(self) -> str:
        if not (self.fleet > 1 and self.fleet_mode != "rr"):
            return ""
        seg = f"-{self.fleet_mode}{self.join_at}"
        if self.newcomer_model:
            seg += f"-nm{self.newcomer_model}"
        return seg

    @property
    def id(self) -> str:
        sweep = f"-d{self.distractors}-n{self.label_noise}-f{self.feedback_delay}-a{self.fleet}{self._transfer}"
        if self.change_at:
            sweep += f"-c{self.change_at}"
            if self.change_focus:
                sweep += f"-x{self.change_focus}"
        if self.opaque_ids:
            sweep += "-oid"
        sweep += self._variant + self._replicas
        return f"{self.scenario}|{self.arm}|{self.model}|s{self.seed}|e{self.episodes}{sweep}"

    @property
    def scope(self) -> str:
        raw = (f"{self.scenario}-{self.arm}-{self.model}-s{self.seed}-d{self.distractors}"
               f"-n{int(self.label_noise * 100)}-f{self.feedback_delay}-a{self.fleet}{self._transfer}"
               + (f"-c{self.change_at}" if self.change_at else "")
               + (f"-x{int(self.change_focus * 100)}" if self.change_at and self.change_focus else "")
               + ("-oid" if self.opaque_ids else "") + self._variant + self._replicas)
        slug = re.sub(r"[^a-z0-9-]", "-", raw.lower())
        return f"{config.SCOPE_ROOT}/{slug}"


JUDGE_PROMPT = """You are grading an incident-response agent's explanation. Be strict and blind: you do not know which system produced it.

True root cause: {cause}
Agent's stated root cause and explanation: {explanation}
Fix the agent applied: {root_cause}

Rubric: {rubric}

Reply with a single integer 0-3 and nothing else."""


def judge(scenario, task, rec) -> int | None:
    if not scenario.uses_judge:
        return None
    try:
        text, _ = get_llm(config.JUDGE_MODEL).complete_text(JUDGE_PROMPT.format(
            cause=task.truth.get("cause", ""), root_cause=rec.final_answer, explanation=rec.explanation,
            rubric=scenario.judge_rubric), max_tokens=64)
        m = re.search(r"[0-3]", text)
        return int(m.group()) if m else None
    except Exception:  # noqa: BLE001
        return None


def run_cell(cell: Cell, out_path: Path, *, keep_transcript: bool, retry_budget: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    scenario = make_scenario(cell.scenario, cell.seed, cell.episodes, distractors=cell.distractors,
                             label_noise=cell.label_noise, feedback_delay=cell.feedback_delay, fleet=cell.fleet,
                             change_at=cell.change_at, change_focus=cell.change_focus,
                             fleet_mode=cell.fleet_mode, join_at=cell.join_at, opaque_ids=cell.opaque_ids,
                             change_variant=cell.change_variant)
    arm = make_arm(cell.arm)
    llm_main = get_llm(cell.model)
    if hasattr(arm, "attach"):   # skillfile: validation replays need same-shaped instances of another seed
        arm.attach(scenario, lambda s: make_scenario(cell.scenario, s, cell.episodes, distractors=cell.distractors,
                                                     label_noise=cell.label_noise, feedback_delay=cell.feedback_delay,
                                                     fleet=cell.fleet, change_at=cell.change_at,
                                                     change_focus=cell.change_focus, fleet_mode=cell.fleet_mode,
                                                     join_at=cell.join_at, opaque_ids=cell.opaque_ids,
                                                     change_variant=cell.change_variant),
                   llm_main, retry_budget=retry_budget)
    llm_new = get_llm(cell.newcomer_model) if cell.newcomer_model else llm_main
    n_ok = 0
    errors = 0
    read_keys: dict[str, None] = {}   # every key any search served this cell, in first-seen order
    suffix = f"{int(time.time()) % 1_000_000:06d}"
    scope = f"{cell.scope}-{suffix}"  # fresh scope per attempt
    # Agent identities are per cell as well. Hosted systems compile agent-centric context (e.g.
    # SenseLab's agent brief / briefing pulls every entity the agent has touched), so a shared
    # "support-agent" across seeds and arms would leak knowledge between cells and contaminate
    # the senselab vs senselab-nofeedback ablation. Found on 2026-09-16; SenseLab cells run
    # before this change are kept separately as the "shared-identity" variant.
    scenario.agent_base = f"{scenario.agent_base}-{suffix}"
    # The newcomer identity (agent N in ``newcomer`` mode); computed after the suffix so it
    # matches the ids Scenario.agent_for() hands out. Only this agent runs on newcomer_model.
    newcomer_id = f"{scenario.agent_base}-{scenario.fleet_size}"
    # What the arm needs to know about the cell before it opens: the agent model (stamped on
    # hosted traces as the ``model`` confounder) and whether fleet members are replicas.
    arm.cell_model = cell.model
    if cell.fleet_replicas and hasattr(arm, "fleet_replicas"):
        arm.fleet_replicas = True
    with _sem(cell.arm):
        arm.open(scope)
        arm.configure(scenario)
        seed_ms = 0.0
        try:
            if arm.has_memory:
                ts = time.perf_counter()
                arm.seed(scenario.seed_entries())
                seed_ms = (time.perf_counter() - ts) * 1000
            for ep in range(cell.episodes):
                arm.drain_deferred(ep)
                task = scenario.task(ep)
                llm = llm_new if (cell.newcomer_model and task.agent_id == newcomer_id) else llm_main
                rec = run_episode(scenario, arm, llm, task, seed=cell.seed, retry_budget=retry_budget,
                                  keep_transcript=keep_transcript)
                rec.judge_score = judge(scenario, task, rec) if rec.final_answer else None
                d = rec.as_dict()
                d.update({"cell": cell.id, "scope": scope, "distractors": cell.distractors,
                          "label_noise": cell.label_noise, "feedback_delay": cell.feedback_delay,
                          "fleet": cell.fleet, "change_at": cell.change_at, "change_focus": cell.change_focus,
                          "fleet_mode": scenario.fleet_mode, "join_at": cell.join_at,
                          "newcomer_model": cell.newcomer_model, "opaque_ids": cell.opaque_ids,
                          "change_variant": cell.change_variant,
                          "fleet_replicas": bool(getattr(arm, "fleet_replicas", False)) if cell.fleet > 1 else False,
                          "harness_version": config.HARNESS_VERSION,
                          "ts": time.time()})
                for sl in rec.searches_log:
                    for h in sl["hits"]:
                        read_keys.setdefault(h["key"], None)
                extra = arm.after_episode(ep, llm_main)  # store maintenance stays on the veterans' model
                if extra is not None:
                    d["memory"]["after_episode"] = extra.as_dict()
                with _write_lock, out_path.open("a") as f:
                    f.write(json.dumps(d, default=str) + "\n")
                n_ok += int(rec.success)
                errors += int(bool(rec.error))
                print(f"  {cell.id} ep{ep:02d} {'OK ' if rec.success else 'FAIL'} attempts={rec.attempts} "
                      f"tok={rec.usage['prompt_tokens'] + rec.usage['completion_tokens']} "
                      f"${rec.usage['cost_usd']:.3f}" + (f" err={rec.error}" if rec.error else ""), flush=True)
            # Confidence trajectories for what the agent actually read, for arms that keep
            # version history. Non-seed keys first: those are the lessons whose rise and fall
            # is the story; seed entries come after, up to the arm's cap.
            try:
                seed_keys = {k for k, _, _ in scenario.seed_entries()}
                ordered = [k for k in read_keys if k not in seed_keys] + [k for k in read_keys if k in seed_keys]
                hist = arm.confidence_history(ordered)
                if hist:
                    with _write_lock, (out_path.parent / "confidence_traces.jsonl").open("a") as f:
                        f.write(json.dumps({"cell": cell.id, "scope": scope, "change_at": cell.change_at,
                                            "versions": hist}, default=str) + "\n")
            except Exception as e:  # noqa: BLE001
                print(f"  {cell.id} confidence history skipped: {e}", flush=True)
            # Arm-side record of the cell (the repair arms' fix lifecycle, verdicts, incidents;
            # the oracle-fix arm's write). One line per cell, next to the episode rows.
            try:
                rep = arm.report()
                if rep:
                    with _write_lock, (out_path.parent / "arm_reports.jsonl").open("a") as f:
                        f.write(json.dumps({"cell": cell.id, "scope": scope, "arm": cell.arm, "model": cell.model,
                                            "seed": cell.seed, "report": rep}, default=str) + "\n")
            except Exception as e:  # noqa: BLE001
                print(f"  {cell.id} arm report skipped: {e}", flush=True)
        finally:
            try:
                arm.drain_deferred(10**9)  # flush any outcomes still pending at the end of the cell
            except Exception:  # noqa: BLE001
                pass
            arm.close()
    return {"cell": cell.id, "scope": scope, "success": n_ok, "episodes": cell.episodes, "errors": errors,
            "seed_s": round(seed_ms / 1000, 1), "wall_s": round(time.perf_counter() - t0, 1)}


def done_cells(out_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                counts[json.loads(line)["cell"]] = counts.get(json.loads(line)["cell"], 0) + 1
            except Exception:  # noqa: BLE001
                continue
    return counts


def build_cells(args) -> list[Cell]:
    """Per-tier defaults from config unless overridden on the command line:
    core scenarios: 60 episodes, regime change at 20 with change_focus 0.5, 6 seeds, 300
    distractors, fleet sizes (1, 6); probes: 20 episodes, 5 seeds, 0 distractors, single agent
    (fleet probe keeps its own 4), no change."""
    S = config.STUDY
    cells = []
    # --newcomer-model is a cell attribute, not a loop dimension: it only applies to fleet
    # cells in ``newcomer`` mode (single-agent and rr/pioneer cells have no newcomer).
    newcomer_model = getattr(args, "newcomer_model", "") or ""
    if newcomer_model and args.fleet_mode != "newcomer":
        print(f"--newcomer-model {newcomer_model} ignored: only meaningful with --fleet-mode newcomer")
        newcomer_model = ""
    for sc in args.scenarios:
        core = sc in S.core_scenarios
        eps = args.episodes or (S.core_episodes if core else S.episodes)
        seeds = args.seeds or list(S.seeds if core else S.probe_seeds)
        dists = args.distractors or [S.core_distractors if core else 0]
        fleets = args.fleet or (list(S.fleet_sizes) if core else [1])
        changes = args.change_at if args.change_at is not None else [S.core_change_at if core else 0]
        for arm in args.arms:
            for model in args.models:
                for seed in seeds:
                    for d in dists:
                        for n in args.label_noise:
                            for f in args.feedback_delay:
                                for a in fleets:
                                    for ch in changes:
                                        focus = (args.change_focus if args.change_focus is not None
                                                 else S.core_change_focus) if ch else 0.0
                                        mode = args.fleet_mode if a > 1 else "rr"
                                        replicas = bool(getattr(args, "fleet_replicas", False)) or arm.startswith("senselab-rsi")
                                        cells.append(Cell(sc, arm, model, seed, eps, d, n, f, a, ch, focus,
                                                          mode, args.join_at if mode != "rr" else 0,
                                                          newcomer_model if mode == "newcomer" else "",
                                                          bool(getattr(args, "opaque_ids", False)),
                                                          getattr(args, "change_variant", "A"),
                                                          replicas and a > 1))
    # interleave arms so slow hosted arms never monopolise the worker pool
    by_arm: dict[str, list[Cell]] = {}
    for c in cells:
        by_arm.setdefault(c.arm, []).append(c)
    out: list[Cell] = []
    while any(by_arm.values()):
        for arm in list(by_arm):
            if by_arm[arm]:
                out.append(by_arm[arm].pop(0))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--scenarios", nargs="+", default=list(config.STUDY.scenarios))
    p.add_argument("--arms", nargs="+", default=list(config.STUDY.arms))
    p.add_argument("--models", nargs="+", default=config.AGENT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: per-tier from config")
    p.add_argument("--episodes", type=int, default=0, help="default: per-tier from config")
    p.add_argument("--distractors", nargs="+", type=int, default=None, help="default: per-tier from config")
    p.add_argument("--fleet", nargs="+", type=int, default=None, help="agent identities per store; default per-tier")
    p.add_argument("--change-at", nargs="+", type=int, default=None,
                   help="regime change episode (0 = none); default: core scenarios 20, probes 0")
    p.add_argument("--change-focus", type=float, default=None,
                   help="share of post-change tasks drawn from changed classes; default 0.5 when a change is set")
    p.add_argument("--fleet-mode", choices=["rr", "pioneer", "newcomer"], default="rr",
                   help="knowledge-transfer protocol for fleet cells (see Scenario.fleet_mode)")
    p.add_argument("--join-at", type=int, default=0,
                   help="pioneer: episode the other agents join; newcomer: episode the new agent takes over")
    p.add_argument("--newcomer-model", default="",
                   help="newcomer mode only: run the arriving agent on this model while the veterans keep "
                        "--models (cross-model onboarding); adds -nm<model> to the cell id and scope")
    p.add_argument("--opaque-ids", action="store_true",
                   help="ticket/PR numbers from the instance rng, not the episode index (see Scenario.opaque_ids); "
                        "adds -oid to the cell id and scope")
    p.add_argument("--change-variant", default="A", choices=["A", "B"],
                   help="which regime change set applies at --change-at (see Scenario.change_variant); "
                        "B adds -vB to the cell id and scope")
    p.add_argument("--fleet-replicas", action="store_true",
                   help="fleet members share one hosted agent identity (HYPOTHESIS_RSI.md); adds -rep to fleet "
                        "cell ids. Always on for senselab-rsi* arms")
    p.add_argument("--label-noise", nargs="+", type=float, default=[0.0])
    p.add_argument("--feedback-delay", nargs="+", type=int, default=[0])
    p.add_argument("--retry-budget", type=int, default=config.STUDY.retry_budget)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--transcripts", action="store_true")
    p.add_argument("--dry", action="store_true", help="list cells and exit")
    args = p.parse_args()

    out_dir = config.RESULTS_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    cells = build_cells(args)
    done = done_cells(out_path)
    todo = [c for c in cells if done.get(c.id, 0) < c.episodes]
    partial = {c.id for c in todo if done.get(c.id, 0) > 0}
    if partial:
        # a half-finished cell has a polluted scope: drop its rows and rerun it in a fresh scope
        keep = [l for l in out_path.read_text().splitlines() if json.loads(l).get("cell") not in partial]
        out_path.write_text("\n".join(keep) + ("\n" if keep else ""))
        print(f"re-running {len(partial)} partial cells from scratch")
    manifest = {"run": args.run, "cells": [c.id for c in cells], "models": args.models,
                "newcomer_model": args.newcomer_model if args.fleet_mode == "newcomer" else "",
                "judge": config.JUDGE_MODEL, "openai_effort": config.OPENAI_REASONING_EFFORT,
                "anthropic_effort": config.ANTHROPIC_EFFORT, "retry_budget": args.retry_budget,
                "harness_version": config.HARNESS_VERSION, "amfs_http_url": config.AMFS_HTTP_URL,
                "fleet_replicas": bool(getattr(args, "fleet_replicas", False)),
                "study": config.STUDY.__dict__, "started": time.time()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"{len(cells)} cells, {len(todo)} to run, {sum(c.episodes for c in todo)} episodes")
    if args.dry:
        for c in todo:
            print(" ", c.id)
        return
    # cells that already have partial rows are re-run from scratch in a fresh scope suffix
    cells_path = out_dir / "cells.json"
    summaries: dict[str, dict] = {}
    if cells_path.exists():
        summaries = {s["cell"]: s for s in json.loads(cells_path.read_text()) if "cell" in s}

    def _run_with_retry(c: Cell) -> dict:
        # a cell that dies on a transient arm/API failure (timeout, 5xx) is retried once from
        # scratch in a fresh scope; its partial rows are dropped first so the JSONL stays clean
        for attempt in range(2):
            try:
                return run_cell(c, out_path, keep_transcript=args.transcripts, retry_budget=args.retry_budget)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]
                if attempt == 1:
                    return {"cell": c.id, "error": err}
                print(f"  {c.id} cell error, retrying in 60s: {err}", flush=True)
                with _write_lock:
                    if out_path.exists():  # first cells of a run may fail before any row is written
                        keep = [l for l in out_path.read_text().splitlines() if json.loads(l).get("cell") != c.id]
                        out_path.write_text("\n".join(keep) + ("\n" if keep else ""))
                time.sleep(60)
        return {"cell": c.id, "error": "unreachable"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_with_retry, c): c for c in todo}
        for fut in as_completed(futs):
            s = fut.result()
            summaries[s["cell"]] = s
            print("CELL DONE", json.dumps(s), flush=True)
            with _write_lock:
                cells_path.write_text(json.dumps(list(summaries.values()), indent=2))
    _drop_sdk_backlog()


def _drop_sdk_backlog() -> None:
    """Let the process exit once the cells are done.

    The AMFS SDK logs an event and materialises graph edges after every write and commit
    on a single background thread. At 48 cells in flight that thread falls hours behind,
    and Python joins it at exit: gridv5-core-sl finished its 120 cells at 00:43 and was
    still posting edges at 07:10, holding the family's chain on a phase that was over.
    Events and edges are not read by any arm or metric here, so the backlog is dropped."""
    try:
        from amfs import memory as _sdk_memory
    except Exception:  # noqa: BLE001 — local-only runs never import the SDK
        return
    ex = getattr(_sdk_memory, "_sdk_bg_executor", None)
    if ex is not None:
        pending = getattr(getattr(ex, "_work_queue", None), "qsize", lambda: -1)()
        ex.shutdown(wait=False, cancel_futures=True)
        print(f"dropped {pending} queued SDK background tasks (events/edges) so the run can exit",
              flush=True)


if __name__ == "__main__":
    main()
