"""P12 / P13 / P14 of amendment (v): did canaries reach the floor, decide, and pass no harm?

Reads every canary of a run from the arm reports (``fix.canary``: arms, graded counts, status,
reason) and — for P14 — joins the run's episode records to the sealed traces behind them in the
production database, where the router stamped each trace with ``attributes.canary_arm`` and
``attributes.canary_fix_id``. The join key is the trace's ``attributes.replica`` (the harness
agent id, replica suffix included) and ``attributes.bench_episode``; the engine's own tally is
not used for P14, as the preregistration requires.

    P12  ≥ 1 canary in 2 reaches ``min_outcomes`` graded on both arms; graded/exposed ≥ 0.8 per arm.
         Falsified by 0 of ≥ 6.
    P13  ≥ 1 canary in 3 reaches a verdict (passed or failed) within the cell; read separately for
         canaries started from ``test_passed`` (pct 50) and ``test_inconclusive`` (pct 10).
         Falsified by 0 verdicts in ≥ 6.
    P14  of canaries that reach a verdict on a fix whose replay set contains kept cases, none
         *passes* with kept-class post-change first-attempt success in the canary arm below the
         control arm by more than 10 points over the cell. One such pass falsifies.

    PGPASS_DSN=... .venv/bin/python -m benchmarks.continual_learning.analysis.rsi_canary_readout --run rsi-e2b-c6

Writes ``results/<run>/canary_readout.json`` and prints the tables. The DSN is read from the
environment variable named by ``--dsn-env`` and never printed; ``--no-db`` skips P14.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import rsi
from .platform_calibration import ACCOUNT_ID
from .rsi_coding_sheet import RSI_ARMS

MARGIN = 0.10
TRACES_SQL = (
    "SELECT attributes->>'replica', (attributes->>'bench_episode')::int, attributes->>'canary_arm', "
    "attributes->>'canary_fix_id', outcome_type FROM amfs_immutable_traces "
    "WHERE account_id = %s AND agent_id = ANY(%s) AND attributes ? 'canary_arm'"
)


def _hosted_agent(agent_id: str) -> str:
    """``support-agent-810550451-7`` (harness replica) -> ``support-agent-810550451`` (hosted)."""
    parts = agent_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2 else agent_id


def _kept_moved(cases: list[dict[str, Any]], cell_eps: dict[str, dict[str, Any]]) -> tuple[int, int]:
    kept = moved = 0
    for c in cases:
        e = cell_eps.get(c.get("task_input") or "")
        t = (e or {}).get("tags") or {}
        if e is not None and t.get("post_change") and t.get("change_class"):
            if t.get("change_cond"):
                moved += 1
            else:
                kept += 1
    return kept, moved


def _kept_rate(eps: list[dict[str, Any]]) -> tuple[int, int]:
    """(kept-class post-change episodes, first-attempt successes among them)."""
    kept = [e for e in eps if (e.get("tags") or {}).get("post_change") and (e.get("tags") or {}).get("change_class")
            and not (e.get("tags") or {}).get("change_cond") and not e.get("error")]
    return len(kept), sum(1 for e in kept if e.get("first_attempt_success"))


def readout(run: str, dsn: str | None, arms: tuple[str, ...]) -> dict[str, Any]:
    eps = rsi.episodes(run)
    eps_by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    ep_index: dict[tuple[str, int], dict[str, Any]] = {}
    for e in eps:
        eps_by_cell.setdefault(e.get("cell", ""), {})[e.get("task_prompt", "")] = e
        ep_index[(e.get("agent_id", ""), int(e.get("episode", -1)))] = e

    # traces stamped by the router, keyed by fix id -> arm -> episodes
    stamped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    unjoined = 0
    if dsn:
        import psycopg

        agents = sorted({_hosted_agent(e["agent_id"]) for e in eps if e.get("agent_id")
                         and str(e.get("arm", "")).startswith("senselab")})
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SET statement_timeout = '120s'")
            cur.execute("SELECT set_config('amfs.current_account_id', %s, false)", [ACCOUNT_ID])
            cur.execute(TRACES_SQL, [ACCOUNT_ID, agents])
            for replica, bench_ep, arm, fix_id, _outcome in cur.fetchall():
                e = ep_index.get((replica or "", int(bench_ep) if bench_ep is not None else -1))
                if e is None:
                    unjoined += 1
                    continue
                stamped.setdefault(fix_id or "", {}).setdefault(arm or "", []).append(e)

    canaries = []
    for r in rsi.reports(run):
        cell = r.get("cell", "")
        arm = cell.split("|")[1] if "|" in cell else ""
        if arm not in arms:
            continue
        cell_eps = eps_by_cell.get(cell, {})
        for f in (r.get("report") or {}).get("fixes") or []:
            c = f.get("canary")
            if not c:
                continue
            ca, co = c.get("canary") or {}, c.get("control") or {}
            floor = int(c.get("min_outcomes") or 8)
            kept_cases, moved_cases = _kept_moved((f.get("replay") or {}).get("cases") or [], cell_eps)
            st = stamped.get(f.get("id") or "", {})
            k_can, s_can = _kept_rate(st.get("canary", []))
            k_con, s_con = _kept_rate(st.get("control", []))
            rate_can = s_can / k_can if k_can else None
            rate_con = s_con / k_con if k_con else None
            harm = (rate_con - rate_can) if rate_can is not None and rate_con is not None else None
            canaries.append({
                "cell": cell, "arm": arm, "fix_id": f.get("id"), "title": f.get("title"),
                "regime": not (f.get("judge_id") or f.get("behavior_id")),
                "started_from": "test_passed" if int(c.get("pct") or 0) >= 50 else "test_inconclusive",
                "pct": c.get("pct"), "min_outcomes": floor, "status": c.get("status"), "reason": c.get("reason"),
                "note": c.get("note"), "evaluations": c.get("evaluations"), "started_at": c.get("started_at"),
                "decided_at": c.get("decided_at"),
                "canary_traces": ca.get("traces"), "canary_graded": ca.get("graded"), "canary_recurred": ca.get("recurred"),
                "control_traces": co.get("traces"), "control_graded": co.get("graded"), "control_recurred": co.get("recurred"),
                "reached_floor": (ca.get("graded") or 0) >= floor and (co.get("graded") or 0) >= floor,
                "graded_over_exposed": {
                    "canary": (ca.get("graded") or 0) / ca["traces"] if ca.get("traces") else None,
                    "control": (co.get("graded") or 0) / co["traces"] if co.get("traces") else None},
                "kept_cases": kept_cases, "moved_cases": moved_cases,
                "p14": {"stamped_canary": len(st.get("canary", [])), "stamped_control": len(st.get("control", [])),
                        "kept_post_change": {"canary": [k_can, s_can], "control": [k_con, s_con]},
                        "kept_fas": {"canary": rate_can, "control": rate_con},
                        "control_minus_canary": harm},
            })

    decided = [c for c in canaries if c["status"] in ("passed", "failed")]
    p12_ok = [c for c in canaries if c["reached_floor"]]
    p14_pop = [c for c in decided if c["kept_cases"] > 0]
    p14_viol = [c for c in p14_pop if c["status"] == "passed" and c["p14"]["control_minus_canary"] is not None
                and c["p14"]["control_minus_canary"] > MARGIN]
    by_start: dict[str, dict[str, int]] = {}
    for c in canaries:
        d = by_start.setdefault(c["started_from"], {"canaries": 0, "decided": 0, "passed": 0, "failed": 0})
        d["canaries"] += 1
        d["decided"] += c["status"] in ("passed", "failed")
        d["passed"] += c["status"] == "passed"
        d["failed"] += c["status"] == "failed"
    n = len(canaries)
    return {
        "run": run, "canaries": n, "unjoined_stamped_traces": unjoined,
        "p12": {"reached_floor": len(p12_ok), "share": len(p12_ok) / n if n else None,
                "met": n > 0 and len(p12_ok) / n >= 0.5, "falsified": n >= 6 and len(p12_ok) == 0},
        "p13": {"decided": len(decided), "share": len(decided) / n if n else None,
                "met": n > 0 and len(decided) / n >= 1 / 3, "falsified": n >= 6 and len(decided) == 0,
                "by_started_from": by_start},
        "p14": {"population": len(p14_pop), "passes": sum(1 for c in p14_pop if c["status"] == "passed"),
                "violations": [c["fix_id"] for c in p14_viol], "met": len(p14_pop) > 0 and not p14_viol,
                "undefined": len(p14_pop) == 0, "margin": MARGIN},
        "rows": canaries,
    }


def print_tables(out: dict[str, Any]) -> None:
    print(f"## {out['run']}: {out['canaries']} canaries")
    print("| cell | fix | from | status | reason | canary graded/exposed | control graded/exposed | kept cases | "
          "kept FAS canary | kept FAS control | ctrl−can |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in out["rows"]:
        p = c["p14"]
        fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"| {c['cell'].split('|')[0]} {c['arm']} {c['cell'].split('|')[3]} | {str(c['fix_id'])[:8]} | "
              f"{c['started_from']} | {c['status']} | {c['reason'] or ''} | "
              f"{c['canary_graded']}/{c['canary_traces']} | {c['control_graded']}/{c['control_traces']} | "
              f"{c['kept_cases']} | {fmt(p['kept_fas']['canary'])} ({p['kept_post_change']['canary'][0]}) | "
              f"{fmt(p['kept_fas']['control'])} ({p['kept_post_change']['control'][0]}) | {fmt(p['control_minus_canary'])} |")
    for k in ("p12", "p13", "p14"):
        print(f"{k}: {json.dumps({a: b for a, b in out[k].items() if a != 'by_started_from'})}")
    print(f"p13 by start: {json.dumps(out['p13']['by_started_from'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", nargs="+", default=list(RSI_ARMS))
    ap.add_argument("--dsn-env", default="PGPASS_DSN")
    ap.add_argument("--no-db", action="store_true", help="skip the trace join (P14 undefined)")
    args = ap.parse_args()
    dsn = None if args.no_db else os.environ.get(args.dsn_env)
    if not args.no_db and not dsn:
        sys.exit(f"{args.dsn_env} is not set (use --no-db to skip P14)")
    out = readout(args.run, dsn, tuple(args.arms))
    (rsi.RESULTS / args.run / "canary_readout.json").write_text(json.dumps(out, indent=1) + "\n")
    print_tables(out)


if __name__ == "__main__":
    main()
