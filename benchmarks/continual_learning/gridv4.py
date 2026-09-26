"""Grid v4 launcher: the complete study as runner invocations, grouped by arm family.

Every phase is one ``runner`` invocation (resumable: finished cells are skipped, partial
cells re-run in a fresh scope). Phases of one *family* run sequentially in one detached
process chain so that at most one runner per family talks to a hosted API at a time; the
three families run in parallel:

  sl     senselab, senselab-nofeedback, senselab-nopriors, senselab-lean   (production API; critical path)
  mem0   mem0                                                              (Mem0 Pro API)
  local  none, pgvector, pgvector-diy, pgvector-diy+outcomes               (local Postgres; fast)

    python -m benchmarks.continual_learning.gridv4 --dry               # size every phase
    python -m benchmarks.continual_learning.gridv4 --launch sl         # start one family's chain
    python -m benchmarks.continual_learning.gridv4 --status            # episodes done per run
    python -m benchmarks.continual_learning.gridv4 --stop sl           # kill one family's chain
    python -m benchmarks.continual_learning.gridv4 --purge-infra sl [--apply]
        # list (or, with --apply, drop) every cell that recorded a memory-API failure so the
        # next --launch re-runs it in a fresh scope; see purge_infra()

Hosted-arm pacing is set here through the environment the chain inherits (AMFS_RPM and
CL_SENSELAB_CONCURRENCY for the SenseLab family, MEM0_RPM for Mem0); the key allows 1000
requests/min and the API tier runs 12 instances since 2026-09-18, so the v3 defaults (100
RPM, 2 concurrent cells, set for a 2-vCPU database) are raised.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOGS = config.RESULTS_DIR / "logs"

MODELS = ["gpt-5.5", "claude-opus-5"]
CORE = ["support", "ci-fix", "analytics", "concierge", "retention", "order-ops", "diagnose",
        "policy-qa", "accounts", "dispatch"]
PROBES = ["runbook", "drift-fact", "drift-tool", "triage", "handoff", "unknowns", "fleet", "sizing"]
FLEET_SC = ["support", "ci-fix", "dispatch", "policy-qa"]      # many-writer collisions most plausible
XMODEL_SC = ["support", "diagnose"]
LONG_SC = ["support", "diagnose"]

ARMS = {
    "sl":    {"core": ["senselab", "senselab-nofeedback", "senselab-nopriors"],
              "probe": ["senselab", "senselab-lean"], "other": ["senselab"]},
    "mem0":  {"core": ["mem0"], "probe": ["mem0"], "other": ["mem0"]},
    "local": {"core": ["none", "pgvector", "pgvector-diy", "pgvector-diy+outcomes"],
              "probe": ["none", "pgvector", "pgvector-diy+outcomes"],
              "other": ["none", "pgvector", "pgvector-diy+outcomes"]},
}
WORKERS = {"sl": 48, "mem0": 12, "local": 24}
ENV = {
    "sl": {"AMFS_RPM": "800", "CL_SENSELAB_CONCURRENCY": "48"},
    "mem0": {"MEM0_RPM": "300", "CL_MEM0_CONCURRENCY": "12"},
    "local": {},
}
# (cost/episode observed on grid v3, both models priced alike) for the estimate in --dry
COST_PER_EP = {"sl": 0.046, "mem0": 0.036, "local": 0.031}


def phases(family: str) -> list[tuple[str, list[str]]]:
    a = ARMS[family]
    m = ["--models", *MODELS]
    out: list[tuple[str, list[str]]] = [
        # 1. core: 10 scenarios x fleet {1, 6}; 60 episodes, change at 20, focus 0.5, 300 distractors, 6 seeds
        (f"gridv4-core-{family}", ["--scenarios", *CORE, "--arms", *a["core"], *m, "--fleet", "1", "6"]),
        # 2. fleet 10 at 100 episodes so each agent still sees ~10; change at 30
        (f"gridv4-fleet10-{family}", ["--scenarios", *FLEET_SC, "--arms", *a["core"], *m, "--fleet", "10",
                                      "--episodes", "100", "--change-at", "30"]),
        # 3. probes: 8 mechanisms; 20 episodes, 5 seeds, clean store (fleet probe keeps its own 4 agents)
        (f"gridv4-probes-{family}", ["--scenarios", *PROBES, "--arms", *a["probe"], *m]),
        # 4. transfer, pioneer: agent 1 alone to 20, five join; 50 episodes, no change
        (f"gridv4-xfer-pioneer-{family}", ["--scenarios", *FLEET_SC, "--arms", *a["other"], *m, "--fleet", "6",
                                           "--fleet-mode", "pioneer", "--join-at", "20", "--episodes", "50",
                                           "--change-at", "0"]),
        # 5. transfer, newcomer after a regime change: 5 share 0..39 with the change at 20; agent 6 takes 40..59
        (f"gridv4-xfer-newcomer-{family}", ["--scenarios", *FLEET_SC, "--arms", *a["other"], *m, "--fleet", "6",
                                            "--fleet-mode", "newcomer", "--join-at", "40"]),
        # 6. cold start onto a healthy store: same protocol, no change
        (f"gridv4-xfer-cold-{family}", ["--scenarios", *XMODEL_SC, "--arms", *a["other"], *m, "--fleet", "6",
                                        "--fleet-mode", "newcomer", "--join-at", "40", "--change-at", "0"]),
        # 7. cross-model onboarding: the store is built by one model, the newcomer runs the other
        (f"gridv4-xfer-xmodel-g2c-{family}", ["--scenarios", *XMODEL_SC, "--arms", *a["other"], "--models", "gpt-5.5",
                                              "--newcomer-model", "claude-opus-5", "--fleet", "6", "--fleet-mode",
                                              "newcomer", "--join-at", "40", "--change-at", "0"]),
        (f"gridv4-xfer-xmodel-c2g-{family}", ["--scenarios", *XMODEL_SC, "--arms", *a["other"], "--models",
                                              "claude-opus-5", "--newcomer-model", "gpt-5.5", "--fleet", "6",
                                              "--fleet-mode", "newcomer", "--join-at", "40", "--change-at", "0"]),
        # 8. long horizon: 120 episodes, no change, 3 seeds
        (f"gridv4-long-{family}", ["--scenarios", *LONG_SC, "--arms", *a["other"], *m, "--episodes", "120",
                                   "--change-at", "0", "--seeds", "11", "23", "37", "--fleet", "1"]),
    ]
    # 9. sensitivity sweeps on runbook (3 seeds): distractor volume, label noise, feedback delay
    sweep_arms = [x for x in a["other"] if x in config.STUDY.sweep_arms]
    if sweep_arms:
        base = ["--scenarios", "runbook", "--arms", *sweep_arms, *m, "--seeds", "11", "23", "37"]
        out += [
            (f"gridv4-sweep-dist-{family}", [*base, "--distractors", "300", "1000"]),
            (f"gridv4-sweep-noise-{family}", [*base, "--label-noise", "0.1", "0.2"]),
            (f"gridv4-sweep-delay-{family}", [*base, "--feedback-delay", "3", "6"]),
        ]
    return out


def runner_cmd(run: str, args: list[str], workers: int, dry: bool = False) -> list[str]:
    cmd = [PY, "-m", "benchmarks.continual_learning.runner", "--run", run, *args, "--workers", str(workers)]
    return cmd + (["--dry"] if dry else [])


def size(family: str) -> tuple[int, int]:
    cells = eps = 0
    for run, args in phases(family):
        out = subprocess.run(runner_cmd(run, args, 1, dry=True), cwd=ROOT, capture_output=True, text=True)
        head = out.stdout.splitlines()[0] if out.stdout else out.stderr.strip().splitlines()[-1]
        try:  # "N cells, M to run, E episodes"
            c, e = int(head.split()[0]), int(head.split()[5])
        except (ValueError, IndexError):
            print(f"  {run}: {head}")
            continue
        cells += c
        eps += e
        print(f"  {run:36s} {c:5d} cells {e:7d} episodes  ~${e * COST_PER_EP[family]:,.0f}")
        # --dry writes a manifest; leave it (harmless, overwritten at launch)
    return cells, eps


CONCURRENCY_ENV = {"sl": "CL_SENSELAB_CONCURRENCY", "mem0": "CL_MEM0_CONCURRENCY", "local": None}


#: Core scenarios whose SenseLab cells completed before the #426 build (items 1-3 of the
#: late-phase plan) reached production on 2026-09-19 19:03 JST. ``--recheck-core`` re-runs
#: the full ``senselab`` arm on these as ``gridv5-core-sl`` (fresh scopes) for a paired
#: before/after on the same cells; the ablation arms are unaffected by the change
#: (``nofeedback`` never touches outcomes) and are not re-run.
RECHECK_SC = ["support", "ci-fix", "analytics", "concierge", "retention"]


def launch(family: str, concurrency: int | None = None, defer_core: bool = False,
           recheck_core: bool = False, recheck_run: str | None = None,
           recheck_scenarios: list[str] | None = None, first: list[str] | None = None) -> None:
    """Start the family's phase chain. ``concurrency`` caps the cells in flight (runner
    --workers, which is the binding cap across arms, and the per-arm semaphore env) below the
    family default — used when the hosted API cannot take the full load (2026-09-19: the
    SenseLab database lock queue saturated at 48 cells; relaunched at 24).

    ``defer_core`` moves the core phase to the end of the chain: the secondary phases (fleet
    of 10, probes, transfer, long horizon, sweeps) run first and the unfinished core cells
    last. Used when the core scenarios already complete are enough for the main result and
    the other sections are what the paper is waiting on; nothing is skipped, only reordered,
    and the runner's resume logic still skips the core cells already on file."""
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("CL_ENV_FILE",)}
    fam_env = dict(ENV[family])
    workers = WORKERS[family]
    if concurrency:
        workers = concurrency
        if CONCURRENCY_ENV[family]:
            fam_env[CONCURRENCY_ENV[family]] = str(concurrency)
    env.update(fam_env)
    order = phases(family)
    if defer_core:
        order = order[1:] + order[:1]
    if first:
        # Phases whose run name contains one of the substrings go to the front, in the order
        # given (2026-09-20: the paper wants the transfer sections before fleet-10).
        front = [ph for key in first for ph in order if key in ph[0]]
        order = front + [ph for ph in order if ph not in front]
    if recheck_core or recheck_run:
        if family != "sl":
            raise SystemExit("--recheck-core / --recheck-run apply to the sl family only")
        run = recheck_run or f"gridv5-core-{family}"
        scs = recheck_scenarios or RECHECK_SC
        order = [(run, ["--scenarios", *scs, "--arms", "senselab",
                        "--models", *MODELS, "--fleet", "1", "6"])] + order
    chain = " ; ".join(
        " ".join(f"'{x}'" if " " in x else x for x in runner_cmd(run, args, workers))
        + f" >> {LOGS / (run + '.log')} 2>&1"
        for run, args in order
    )
    log = open(LOGS / f"gridv4-chain-{family}.log", "ab")
    p = subprocess.Popen(["/bin/sh", "-c", chain], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    (LOGS / f"gridv4-chain-{family}.pid").write_text(str(p.pid))
    print(f"{family}: chain pid {p.pid}, {len(order)} phases starting with {order[0][0]}, "
          f"workers {workers}, env {fam_env}")


def chain_pid(family: str) -> int | None:
    """Pid of the family's chain shell if it is still alive (the runner is its child)."""
    f = LOGS / f"gridv4-chain-{family}.pid"
    if not f.exists():
        return None
    try:
        pid = int(f.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def stop(family: str) -> None:
    """Kill the family's chain and its runner (one process group, see launch). The cells in
    flight keep their partial rows, which the next --launch drops and re-runs in fresh scopes."""
    pid = chain_pid(family)
    if pid is None:
        print(f"{family}: no live chain")
        return
    os.killpg(pid, signal.SIGTERM)
    for _ in range(60):
        if chain_pid(family) is None:
            break
        time.sleep(0.5)
    else:
        os.killpg(pid, signal.SIGKILL)
        time.sleep(1)
    print(f"{family}: chain {pid} {'stopped' if chain_pid(family) is None else 'STILL ALIVE'}")


# Failures of the hosted memory API itself, as the arm's own HTTP client reports them
# (httpx for SenseLab, the mem0 SDK's NetworkError). They hit only the arm that talks to that
# API, so a cell that recorded one is not comparable with its `none`/`pgvector` twins and is
# re-run. LLM-provider incidents (Anthropic 529 OverloadedError, the SDKs' APIConnectionError)
# hit every arm alike and stay in the data; analysis.regime_change.is_infra excludes them
# from the rates.
MEMORY_INFRA_RE = re.compile(r"ReadTimeout|ConnectError|ConnectTimeout|HTTPStatusError|NetworkError|"
                             r"RemoteProtocolError|Server disconnected|ServiceUnavailable")
LLM_INFRA_RE = re.compile(r"OverloadedError|APIConnectionError|APITimeoutError|RateLimitError|InternalServerError")


def is_memory_infra(error) -> bool:
    e = str(error or "")
    return bool(MEMORY_INFRA_RE.search(e)) and not LLM_INFRA_RE.search(e)


def purge_infra(family: str, apply: bool, min_hits: int = 1) -> None:
    """Drop every cell of the family's runs that recorded >= min_hits episodes with a memory-API
    failure, so the next --launch re-runs it from scratch in a fresh scope.

    Nothing is deleted: the cell's rows move to purged-episodes.jsonl, its confidence traces to
    purged-confidence_traces.jsonl, and purged.jsonl gets one audit record per cell (rows, hit
    count, error kinds, the cells.json summary). Refuses while the family's chain is alive: the
    runner keeps its own todo list and appends to episodes.jsonl, so edits under it are racy
    and would not be picked up anyway. Without --apply this only lists what would go."""
    if (pid := chain_pid(family)) is not None:
        sys.exit(f"{family}: chain {pid} is alive; run --stop {family} first")
    total_cells = total_rows = 0
    for run, _ in phases(family):
        d = config.RESULTS_DIR / run
        ep_path = d / "episodes.jsonl"
        if not ep_path.exists():
            continue
        lines = [l for l in ep_path.read_text().splitlines() if l.strip()]
        rows = [json.loads(l) for l in lines]
        hits: dict[str, int] = {}
        kinds: dict[str, dict[str, int]] = {}
        for r in rows:
            if is_memory_infra(r.get("error")):
                c = r["cell"]
                hits[c] = hits.get(c, 0) + 1
                k = re.sub(r"^reflection: ", "", str(r["error"])).split(":")[0]
                kinds.setdefault(c, {})[k] = kinds.get(c, {}).get(k, 0) + 1
        victims = {c for c, n in hits.items() if n >= min_hits}
        if not victims:
            print(f"  {run}: no cell with >= {min_hits} memory-API failures")
            continue
        n_rows = sum(1 for r in rows if r["cell"] in victims)
        total_cells += len(victims)
        total_rows += n_rows
        print(f"  {run}: {len(victims)} cells, {n_rows} rows" + ("" if apply else " (dry run)"))
        for c in sorted(victims, key=lambda c: -hits[c]):
            print(f"    {hits[c]:3d}  {c}  {kinds[c]}")
        if not apply:
            continue
        when = time.time()
        summaries = {}
        cells_path = d / "cells.json"
        if cells_path.exists():
            summaries = {s["cell"]: s for s in json.loads(cells_path.read_text()) if "cell" in s}
        with (d / "purged-episodes.jsonl").open("a") as f:
            f.write("".join(l + "\n" for l, r in zip(lines, rows) if r["cell"] in victims))
        ep_path.write_text("".join(l + "\n" for l, r in zip(lines, rows) if r["cell"] not in victims))
        ct_path = d / "confidence_traces.jsonl"
        if ct_path.exists():
            ct = [l for l in ct_path.read_text().splitlines() if l.strip()]
            gone = [l for l in ct if json.loads(l).get("cell") in victims]
            with (d / "purged-confidence_traces.jsonl").open("a") as f:
                f.write("".join(l + "\n" for l in gone))
            ct_path.write_text("".join(l + "\n" for l in ct if json.loads(l).get("cell") not in victims))
        with (d / "purged.jsonl").open("a") as f:
            for c in sorted(victims):
                f.write(json.dumps({"cell": c, "when": when, "reason": "memory-api-infra",
                                    "min_hits": min_hits, "rows": sum(1 for r in rows if r["cell"] == c),
                                    "infra_rows": hits[c], "kinds": kinds[c],
                                    "scopes": sorted({r.get("scope") for r in rows if r["cell"] == c}),
                                    "summary": summaries.pop(c, None)}) + "\n")
        if cells_path.exists():
            cells_path.write_text(json.dumps(list(summaries.values()), indent=2))
    print(f"{family}: {total_cells} cells / {total_rows} rows " + ("purged" if apply else "would be purged"))


def status() -> None:
    for family in ARMS:
        for run, _ in phases(family):
            d = config.RESULTS_DIR / run
            m = d / "manifest.json"
            if not m.exists():
                continue
            planned = len(json.loads(m.read_text())["cells"])
            done_eps = sum(1 for _ in (d / "episodes.jsonl").open()) if (d / "episodes.jsonl").exists() else 0
            done_cells = len(json.loads((d / "cells.json").read_text())) if (d / "cells.json").exists() else 0
            print(f"  {run:36s} cells {done_cells:4d}/{planned:<4d} episodes {done_eps:7d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--launch", choices=list(ARMS))
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", choices=list(ARMS), help="kill one family's chain and runner")
    p.add_argument("--purge-infra", choices=list(ARMS), metavar="FAMILY",
                   help="drop cells with memory-API failures so --launch re-runs them (dry run unless --apply)")
    p.add_argument("--apply", action="store_true", help="with --purge-infra: write the changes")
    p.add_argument("--min-hits", type=int, default=1, help="with --purge-infra: failed episodes per cell to purge")
    p.add_argument("--concurrency", type=int, default=None,
                   help="with --launch: cap cells in flight below the family default (see launch())")
    p.add_argument("--defer-core", action="store_true",
                   help="with --launch: run the secondary phases first and the unfinished core cells last")
    p.add_argument("--first", nargs="+", default=None, metavar="SUBSTR",
                   help="with --launch: run the phases whose name contains these substrings first, in this order")
    p.add_argument("--recheck-core", action="store_true",
                   help="with --launch sl: first re-run the senselab arm on RECHECK_SC as gridv5-core-sl")
    p.add_argument("--recheck-run", default=None,
                   help="with --launch sl: name of a one-off senselab re-run to put first (fresh scopes)")
    p.add_argument("--recheck-scenarios", nargs="+", default=None,
                   help="with --recheck-run: its scenarios (default RECHECK_SC)")
    args = p.parse_args()
    if args.dry:
        total_c = total_e = 0
        for fam in ARMS:
            print(f"[{fam}]")
            c, e = size(fam)
            total_c += c
            total_e += e
        print(f"TOTAL {total_c} cells, {total_e} episodes")
    if args.stop:
        stop(args.stop)
    if args.purge_infra:
        purge_infra(args.purge_infra, apply=args.apply, min_hits=args.min_hits)
    if args.launch:
        launch(args.launch, concurrency=args.concurrency, defer_core=args.defer_core,
               recheck_core=args.recheck_core, recheck_run=args.recheck_run,
               recheck_scenarios=args.recheck_scenarios, first=args.first)
    if args.status:
        status()
    if not (args.dry or args.launch or args.status or args.stop or args.purge_infra):
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
