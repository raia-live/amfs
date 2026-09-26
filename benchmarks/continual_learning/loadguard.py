"""Keep the SenseLab API under the load it can take while several runs share it.

2026-09-19 the database lock queue saturated with 48 cells in flight from one process; on
2026-09-20 morning three processes (the family chain, a concierge redo, the post-#430 core
re-run) plus another session's runs put ~50 cells and ~2,400 RPM of client budget on it and
``commit_outcome`` started timing out — a lost commit is a lost learning signal for the
SenseLab arm only, so it biases the comparison as well as the run.

This runs detached and, every ``PERIOD`` seconds:

1. Counts SenseLab-side failures in the last ``WINDOW`` seconds across every live run's
   ``episodes.jsonl`` (``commit_error`` / ``briefing_error`` / ``end_error`` in the arm notes,
   or an ``error`` naming the memory API) and Cloud Run 5xx for the API service.
2. If either exceeds its threshold, stops the *managed* runs (the ones this script launched;
   never the family chain, whose runner resumes on its own) and waits for a clean window.
3. When clean, (re)launches the managed runs at ``WORKERS`` cells each. The runner skips cells
   already on file and restarts partial ones in a fresh scope, so a stop costs at most the
   in-flight cells.

Managed runs, in order: the gridv5b concierge GPT-5.5 redo (waits for the Opus redo to exit),
then the gridv5c chain (core re-run, then the induction arm).

Usage: ``python -m benchmarks.continual_learning.loadguard`` (detached by the caller).
State and decisions go to ``results/logs/loadguard.log``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/bin/python")
RESULTS = config.RESULTS_DIR
LOGS = RESULTS / "logs"
LOG = LOGS / "loadguard.log"

PERIOD = 120           # seconds between checks
WINDOW = 300           # seconds of history each check looks at
MAX_SL_ERRORS = 4      # SenseLab-side failed episodes in WINDOW across all live runs
MAX_5XX = 15           # Cloud Run 5xx responses in WINDOW
CLEAN_CHECKS = 2       # consecutive clean checks before (re)launching
WORKERS = int(os.environ.get("LOADGUARD_WORKERS", "10"))
RPM = os.environ.get("LOADGUARD_RPM", "400")
OPUS_REDO_PID = int(os.environ.get("LOADGUARD_OPUS_REDO_PID", "0") or 0)

LIVE_RUNS = ["gridv4-fleet10-sl", "gridv4-probes-sl", "gridv4-xfer-pioneer-sl", "gridv4-xfer-newcomer-sl",
             "gridv4-xfer-cold-sl", "gridv4-xfer-xmodel-g2c-sl", "gridv4-xfer-xmodel-c2g-sl", "gridv4-long-sl",
             "gridv4-sweep-dist-sl", "gridv4-sweep-noise-sl", "gridv4-sweep-delay-sl", "gridv4-core-sl",
             "gridv5b-concierge-sl", "gridv5c-core-sl", "gridv5c-induce-sl"]
SL_ERR = ("ReadTimeout", "ConnectError", "ConnectTimeout", "HTTPStatusError", "RemoteProtocolError",
          "Server disconnected", "ServiceUnavailable", "gateway 5")


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def sl_errors(since: float) -> tuple[int, int, list[str]]:
    """(SenseLab-side failed episodes, episodes total) in the window, with a few samples."""
    bad = total = 0
    samples: list[str] = []
    for run in LIVE_RUNS:
        p = RESULTS / run / "episodes.jsonl"
        if not p.exists() or p.stat().st_mtime < since - 60:
            continue
        with p.open("rb") as f:          # only the tail can be recent
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 6_000_000))
            tail = f.read().decode(errors="ignore").splitlines()[1:]
        for line in tail:
            try:
                e = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if float(e.get("ts") or 0) < since:
                continue
            total += 1
            notes = (e.get("memory") or {}).get("arm_notes") or {}
            err = str(e.get("error") or "")
            hit = any(notes.get(k) for k in ("commit_error", "briefing_error", "end_error", "record_attempt_error")) \
                or (err and any(s in err for s in SL_ERR) and "OverloadedError" not in err)
            if hit:
                bad += 1
                if len(samples) < 3:
                    samples.append(f"{run}: " + (str(notes.get("commit_error") or notes.get("briefing_error") or err))[:80])
    return bad, total, samples


def cloud_run_5xx(minutes: int) -> int | None:
    try:
        out = subprocess.run(
            ["gcloud", "logging", "read",
             'resource.type="cloud_run_revision" AND resource.labels.service_name="amfs-api" AND httpRequest.status>=500',
             f"--freshness={minutes}m", "--limit", "200", "--format", "value(timestamp)"],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return None
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:  # noqa: BLE001
        return None


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def launch(cmd: str, logfile: str) -> int:
    # 90 s read timeout: under load a slow /retrieve or /outcomes is a slow episode, not a lost
    # reflection note or commit (the SDK default of 30 s lost 2% of episodes at 12 req/s).
    env = dict(os.environ, AMFS_RPM=RPM, CL_SENSELAB_CONCURRENCY=str(WORKERS),
               CL_SENSELAB_TIMEOUT=os.environ.get("CL_SENSELAB_TIMEOUT", "90"))
    p = subprocess.Popen(["/bin/sh", "-c", cmd], cwd=ROOT, env=env, start_new_session=True,
                         stdout=open(LOGS / logfile, "ab"), stderr=subprocess.STDOUT)
    return p.pid


def stop(pid: int) -> None:
    try:
        os.killpg(pid, 15)
    except OSError:
        pass


def runner(run: str, *args: str) -> str:
    return " ".join([PY, "-m", "benchmarks.continual_learning.runner", "--run", run, *args,
                     "--workers", str(WORKERS)]) + f" >> {LOGS / (run + '.log')} 2>&1"


def gpt_redo_needed() -> bool:
    p = RESULTS / "gridv5b-concierge-sl" / "episodes.jsonl"
    if not p.exists():
        return False
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return any(r["model"] == "gpt-5.5" and ((r["memory"].get("arm_notes") or {}).get("commit_error")
                                            or (r["memory"].get("arm_notes") or {}).get("end_error")) for r in rows)


def strip_gpt_lost_commits() -> int:
    d = RESULTS / "gridv5b-concierge-sl"
    rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
    bad = {r["cell"] for r in rows if r["model"] == "gpt-5.5" and
           ((r["memory"].get("arm_notes") or {}).get("commit_error") or (r["memory"].get("arm_notes") or {}).get("end_error"))}
    if not bad:
        return 0
    (d / "episodes.jsonl.with-gpt-lost-commits.bak").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    (d / "episodes.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows if r["cell"] not in bad))
    cp = d / "cells.json"
    if cp.exists():
        cp.write_text(json.dumps([s for s in json.loads(cp.read_text()) if s.get("cell") not in bad], indent=2))
    log(f"gridv5b: stripped {len(bad)} gpt-5.5 cells with lost commits for redo")
    return len(bad)


GPT_REDO = runner("gridv5b-concierge-sl", "--scenarios", "concierge", "--arms", "senselab",
                  "--models", "gpt-5.5", "--fleet", "1", "6")
V5C_CHAIN = (runner("gridv5c-core-sl", "--scenarios", "support", "ci-fix", "retention", "--arms", "senselab",
                    "--models", "gpt-5.5", "claude-opus-5", "--fleet", "1", "6")
             + " ; " +
             runner("gridv5c-induce-sl", "--scenarios", "retention", "--arms", "senselab-induce",
                    "--models", "gpt-5.5", "claude-opus-5", "--fleet", "1", "6"))


def main() -> None:
    log(f"start: workers={WORKERS} rpm={RPM} opus_redo_pid={OPUS_REDO_PID}")
    managed: dict[str, int] = {}       # name -> pgid/pid
    clean = 0
    gpt_redo_done = False
    gpt_redo_launches = 0
    halted_until_clean = False
    while True:
        since = time.time() - WINDOW
        bad, total, samples = sl_errors(since)
        n5 = cloud_run_5xx(WINDOW // 60)
        tripped = bad > MAX_SL_ERRORS or (n5 is not None and n5 > MAX_5XX)
        state = {k: alive(v) for k, v in managed.items()}
        log(f"check: sl_errors={bad}/{total} 5xx={n5} managed={state}" + (f" samples={samples}" if samples else ""))
        if tripped:
            clean = 0
            if any(state.values()):
                for name, pid in managed.items():
                    if alive(pid):
                        stop(pid)
                        log(f"TRIPPED: stopped {name} (pid {pid})")
                halted_until_clean = True
        else:
            clean += 1
        # The gridv5c chain finished on its own?
        if "v5c" in managed and not alive(managed["v5c"]) and not halted_until_clean:
            log("gridv5c chain finished; loadguard exiting")
            return
        if clean >= CLEAN_CHECKS:
            halted_until_clean = False
            # 1. concierge GPT redo, once the Opus redo has exited and nothing else of ours runs
            if not gpt_redo_done and not alive(OPUS_REDO_PID) and not alive(managed.get("gpt_redo", 0)):
                if gpt_redo_needed() and gpt_redo_launches < 2:
                    gpt_redo_launches += 1
                    strip_gpt_lost_commits()
                    managed["gpt_redo"] = launch(GPT_REDO, "gridv5b-concierge-sl-gpt-redo.log")
                    log(f"launched gpt redo pid {managed['gpt_redo']}")
                else:
                    gpt_redo_done = True
            if "gpt_redo" in managed and not alive(managed["gpt_redo"]) and not gpt_redo_needed():
                gpt_redo_done = True
            # 2. the gridv5c chain (runs alongside the redo; both at WORKERS)
            if not alive(managed.get("v5c", 0)):
                managed["v5c"] = launch(V5C_CHAIN, "gridv5c-chain.log")
                log(f"launched gridv5c chain pid {managed['v5c']}")
        time.sleep(PERIOD)


if __name__ == "__main__":
    sys.exit(main())
