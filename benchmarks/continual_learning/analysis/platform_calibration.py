"""The platform side of the loop's spend, read from the account's own ledger.

Every platform-side LLM call the benchmark account causes — judges, the repair agent, the
stand-in (Tier 1) replay, lever classification, behaviour discovery — is settled into
``amfs_eval_budgets.spent_usd`` as it happens (HYPOTHESIS_RSI.md, E6: "judges from the
account's eval budget ledger"). Judge verdicts also carry their own ``cost_usd``, and the arm
reports hold every verdict of every cell, so the judge share is exact per cell. The rest is
not persisted per fix; it is recovered here as the ledger's remainder over the account,
divided by the fixes the repair agent drafted:

    per_draft_usd = (spent_usd - sum(verdict cost_usd)) / drafted_fixes

and applied per cell as ``drafts x per_draft_usd`` by ``analysis/rsi.py``, labelled an
estimate. Runs started after 2026-09-29 also carry ``budget_snapshots.jsonl`` (the runner
reads the ledger at start, after each cell and at the end), which gives the run's platform
spend as a difference of two readings — the check on the estimate.

Reads the production database through the RLS-scoped account; the DSN is taken from the
environment variable named by ``--dsn-env`` and never printed. Writes
``results/rsi-report/platform_calibration.json``.

    PGPASS_DSN=... .venv/bin/python -m benchmarks.continual_learning.analysis.platform_calibration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ACCOUNT_ID = "fdba89a6-4303-4489-95b0-5630808df6aa"  # clbench-rsi (HYPOTHESIS_RSI.md)
RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "rsi-report" / "platform_calibration.json"

SQL = [
    ("verdicts", "SELECT count(*), coalesce(sum(cost_usd), 0), coalesce(sum(input_tokens), 0), "
                 "coalesce(sum(output_tokens), 0) FROM amfs_eval_verdicts WHERE account_id = %s"),
    ("fixes", "SELECT count(*), count(*) FILTER (WHERE coalesce(manifest->>'model', '') <> '') "
              "FROM amfs_eval_fixes WHERE account_id = %s"),
    ("budget", "SELECT spent_usd, monthly_usd_cap, period_start, updated_at FROM amfs_eval_budgets "
               "WHERE account_id = %s"),
]


PER_AGENT_SQL = {
    "verdicts": "SELECT agent_id, count(*), coalesce(sum(cost_usd), 0) FROM amfs_eval_verdicts "
                "WHERE account_id = %s AND agent_id = ANY(%s) GROUP BY 1",
    "fixes": "SELECT agent_id, count(*), count(*) FILTER (WHERE coalesce(manifest->>'model', '') <> '') "
             "FROM amfs_eval_fixes WHERE account_id = %s AND agent_id = ANY(%s) GROUP BY 1",
}


def run_agents(run: str) -> dict[str, list[str]]:
    """cell id -> hosted agent ids, from the episode records (any ``senselab*`` arm)."""
    path = RESULTS / run / "episodes.jsonl"
    out: dict[str, set[str]] = {}
    if not path.exists():
        return {}
    for line in path.open():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(e.get("arm", "")).startswith("senselab") and e.get("agent_id"):
            out.setdefault(e["cell"], set()).add(e["agent_id"])
    return {k: sorted(v) for k, v in out.items()}


def per_run_ledger(cur: object, run: str) -> dict[str, object]:
    """For a run whose arms wrote no arm report (the ``senselab`` arm with the loop held at
    review, the oracle arms): each cell's judge verdicts and fixes on the platform, by agent.
    Written to ``results/<run>/platform_ledger.json`` and read by ``analysis/rsi.py`` where the
    arm report is missing."""
    cells = run_agents(run)
    agents = sorted({a for v in cells.values() for a in v})
    found: dict[str, dict[str, float]] = {a: {"verdicts": 0, "judge_usd": 0.0, "fixes": 0, "drafts": 0}
                                          for a in agents}
    cur.execute(PER_AGENT_SQL["verdicts"], [ACCOUNT_ID, agents])  # type: ignore[attr-defined]
    for agent, n, usd in cur.fetchall():  # type: ignore[attr-defined]
        found[agent].update(verdicts=int(n), judge_usd=float(usd))
    cur.execute(PER_AGENT_SQL["fixes"], [ACCOUNT_ID, agents])  # type: ignore[attr-defined]
    for agent, n, drafted in cur.fetchall():  # type: ignore[attr-defined]
        found[agent].update(fixes=int(n), drafts=int(drafted))
    return {"run": run, "taken_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cells": {cell: {"agents": ids, **{k: sum(found[a][k] for a in ids)
                                                for k in ("verdicts", "judge_usd", "fixes", "drafts")}}
                      for cell, ids in cells.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn-env", default="PGPASS_DSN", help="environment variable holding the DSN")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--runs", nargs="*", default=[],
                    help="runs without arm reports whose per-cell platform spend to dump")
    args = ap.parse_args()
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        sys.exit(f"{args.dsn_env} is not set")
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET statement_timeout = '60s'")
        cur.execute("SELECT set_config('amfs.current_account_id', %s, false)", [ACCOUNT_ID])
        rows = {}
        for name, sql in SQL:
            cur.execute(sql, [ACCOUNT_ID])
            rows[name] = cur.fetchone()
        for run in args.runs:
            ledger = per_run_ledger(cur, run)
            (RESULTS / run / "platform_ledger.json").write_text(json.dumps(ledger, indent=1) + "\n")
            print(f"{run}: {len(ledger['cells'])} cells -> results/{run}/platform_ledger.json")
    verdicts, verdict_usd, in_tok, out_tok = rows["verdicts"]
    fixes, drafted = rows["fixes"]
    budget = rows["budget"]
    spent = float(budget[0]) if budget else 0.0
    remainder = spent - float(verdict_usd)
    cal = {
        "taken_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": "clbench-rsi",
        "period_start": str(budget[2]) if budget else None,
        "ledger_updated_at": str(budget[3]) if budget else None,
        "spent_usd": round(spent, 6),
        "verdicts": int(verdicts),
        "verdict_usd": round(float(verdict_usd), 6),
        "verdict_tokens": int(in_tok) + int(out_tok),
        "fixes": int(fixes),
        "drafted_fixes": int(drafted),
        "remainder_usd": round(remainder, 6),
        "per_draft_usd": round(remainder / drafted, 6) if drafted else None,
        "method": "per_draft_usd = (ledger spent_usd - sum of verdict cost_usd) / fixes drafted by the "
                  "repair agent; the remainder is the repair agent, the Tier 1 stand-in replay, lever "
                  "classification and discovery, none of which is persisted per fix",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cal, indent=1) + "\n")
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
