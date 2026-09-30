"""Amendment (z)/(z′) canary readout: three classes, P13′, P13″, P15′, P16, independent read counts.

Supplementary to the frozen analysis/rsi_canary_readout.py (P12/P13/P14), which it imports and does
not modify. Reads arm reports + episodes (searches_log top-5 hits, cited_keys), the live canary
records in amfs_eval_fixes (final state, by fix id), the router-stamped traces (canary/control arm
per episode) and amfs_reuse_events (independent read count, version-exact, diag agent excluded).

  PGPASS_DSN=... python /tmp/canary_classes.py rsi-conf-e2b-s37 rsi-conf-e2b-s41 ...

Never prints the DSN. Output: JSON per run to results/<run>/canary_classes.json and a summary table.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from typing import Any

sys.path.insert(0, os.path.expanduser("~/Desktop/projects/amfs"))
import psycopg  # noqa: E402

from benchmarks.continual_learning.analysis import rsi  # noqa: E402
from benchmarks.continual_learning.analysis.platform_calibration import ACCOUNT_ID  # noqa: E402
from benchmarks.continual_learning.analysis.rsi_canary_readout import (  # noqa: E402
    TRACES_SQL, _hosted_agent, _kept_moved, _kept_rate,
)

DIAG_AGENTS = ("diag-branch-treatment",)
CUT = os.environ.get("CL_CUT", "2026-09-30T08:21:05Z")  # post-canary-visible; CL_CUT=2026-09-30T12:52:04Z for post-replay-evidence
MARGIN = 0.10


def fix_keys(f: dict[str, Any]) -> set[str]:
    ks = set()
    for part in ("correction", "corrective_entry"):
        k = (f.get(part) or {}).get("key")
        if k:
            ks.add(str(k))
    return ks


def episode_reads(e: dict[str, Any], keys: set[str], branch_versions: set[tuple[str, int]]) -> tuple[bool, bool, bool]:
    """(fix key name in top-5 hits of any search [name only, not version-exact],
    fix key cited by the harness [name only],
    the session's trace causal reads include a platform-authored version on the fix branch [version-exact])."""
    hit = False
    for s in e.get("searches_log") or []:
        for h in (s.get("hits") or [])[:5]:
            if str((h or {}).get("key")) in keys:
                hit = True
    cited = any(str(k) in keys for k in (e.get("cited_keys") or []))
    exact = any((str(c.get("key")), int(c.get("version") or -1)) in branch_versions for c in e.get("_causal") or [])
    return hit, cited, exact


def readout(run: str, dsn: str) -> dict[str, Any]:
    eps = rsi.episodes(run)
    eps_by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    ep_index: dict[tuple[str, int], dict[str, Any]] = {}
    for e in eps:
        eps_by_cell.setdefault(e.get("cell", ""), {})[e.get("task_prompt", "")] = e
        ep_index[(e.get("agent_id", ""), int(e.get("episode", -1)))] = e
    agents = sorted({_hosted_agent(e["agent_id"]) for e in eps if e.get("agent_id")
                     and str(e.get("arm", "")).startswith("senselab")})

    fixes: list[tuple[str, dict[str, Any]]] = []
    for r in rsi.reports(run):
        cell = r.get("cell", "")
        if "|senselab-rsi|" not in f"|{cell}|" and not cell.split("|")[1:2] == ["senselab-rsi"]:
            pass
        for f in (r.get("report") or {}).get("fixes") or []:
            if f.get("canary"):
                fixes.append((cell, f))
    fix_ids = [f["id"] for _, f in fixes]

    stamped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    live: dict[str, dict[str, Any]] = {}
    reuse: dict[str, dict[str, int]] = {}
    branch_versions: dict[str, set[tuple[str, int]]] = {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET statement_timeout = '120s'")
        cur.execute("SELECT set_config('amfs.current_account_id', %s, false)", [ACCOUNT_ID])
        # fleet-1 traces carry no replica attribute (one agent per cell): fall back to the agent id
        cur.execute(TRACES_SQL.replace("outcome_type FROM", "outcome_type, causal_entries, agent_id FROM"), [ACCOUNT_ID, agents])
        for replica, bench_ep, arm, fix_id, _o, causal, agent_id in cur.fetchall():
            e = ep_index.get((replica or agent_id or "", int(bench_ep) if bench_ep is not None else -1))
            if e is not None:
                e = dict(e, _causal=causal or [])
                stamped.setdefault(fix_id or "", {}).setdefault(arm or "", []).append(e)
        if fix_ids:
            cur.execute("SELECT branch, key, version FROM amfs_memory_entries WHERE branch = ANY(%s) "
                        "AND agent_id IN ('amfs-server','system','amfs')",
                        [[f"repair/{fid[:8]}" for fid in fix_ids]])
            for branch, key, version in cur.fetchall():
                branch_versions.setdefault(branch, set()).add((str(key), int(version)))
            cur.execute("SELECT id::text, status, shipped_at, reverted_at, canary FROM amfs_eval_fixes WHERE id = ANY(%s::uuid[])",
                        [fix_ids])
            for fid, status, shipped_at, reverted_at, canary in cur.fetchall():
                live[fid] = {"status": status, "shipped_at": str(shipped_at) if shipped_at else None,
                             "reverted_at": str(reverted_at) if reverted_at else None, "canary": canary or {}}
            # independent read count: version-exact reuse events on the fix branch of the fix's keys,
            # by non-diagnostic agents, since the cut
            cur.execute(
                "SELECT r.branch, r.key, count(*), count(DISTINCT r.reused_by) FROM amfs_reuse_events r "
                "WHERE r.created_at >= %s AND r.branch = ANY(%s) AND r.reused_by <> ALL(%s) "
                "AND r.written_by IN ('amfs-server','system','amfs') GROUP BY 1, 2",
                [CUT, [f"repair/{fid[:8]}" for fid in fix_ids], list(DIAG_AGENTS)])
            for branch, key, n, n_agents in cur.fetchall():
                reuse.setdefault(branch, {})[key] = int(n)

    rows = []
    for cell, f in fixes:
        fid = f["id"]
        lv = live.get(fid, {})
        c = lv.get("canary") or f.get("canary") or {}
        ca, co = c.get("canary") or {}, c.get("control") or {}
        floor = int(c.get("min_outcomes") or 8)
        status = lv.get("status") or f.get("status")
        cstatus = c.get("status")
        read = int(ca.get("read") or 0)
        decided = cstatus in ("passed", "failed")
        closed = cstatus not in (None, "running")
        reason = str(c.get("reason") or "")
        if decided and read >= floor:
            klass = "decided_treated"
        elif closed and (("unexercised" in reason) or (not decided and read < floor)):
            klass = "unexercised"
        elif decided:
            klass = "decided_untreated"
        else:
            klass = "other"
        keys = fix_keys(f)
        st = stamped.get(fid, {})
        can_eps, con_eps = st.get("canary", []), st.get("control", [])
        bv = branch_versions.get(f"repair/{fid[:8]}", set())
        # P16 (z′): canary-arm sessions in the fix's situation after the retrieve change (amfs#454) that
        # retrieved the fix's own version. Population = canary-arm sessions the engine graded (its own
        # in-situation-or-read population, `graded_by`); numerator = version-exact reads, from the trace's
        # causal reads naming a platform-authored version on the fix branch (cross-checked against the
        # engine's `read`). `top5`/`cited` are key-name matches from the harness and cannot tell the branch
        # version from main's; they are reported, not used.
        pc = [e for e in can_eps if not e.get("error")]
        hits = [episode_reads(e, keys, bv) for e in pc]
        top5 = sum(1 for h, _, _ in hits if h)
        cited = sum(1 for _, c_, _ in hits if c_)
        exact = sum(1 for _, _, x in hits if x)
        graded_n = int(ca.get("graded") or 0)
        # P15′: kept-class FAS in canary-arm sessions that read the fix (version-exact) vs control-arm
        treated = [e for e in can_eps if episode_reads(e, keys, bv)[2] and not e.get("error")]
        k_tr, s_tr = _kept_rate(treated)
        k_co, s_co = _kept_rate(con_eps)
        r_tr = s_tr / k_tr if k_tr else None
        r_co = s_co / k_co if k_co else None
        kept_cases, moved_cases = _kept_moved((f.get("replay") or {}).get("cases") or [], eps_by_cell.get(cell, {}))
        rows.append({
            "cell": cell, "fix_id": fid, "title": f.get("title"), "situation": f.get("situation"),
            "lever": f.get("lever"), "keys": sorted(keys), "fix_status": status, "shipped_at": lv.get("shipped_at"),
            "reverted_at": lv.get("reverted_at"), "canary_status": cstatus, "reason": reason, "note": c.get("note"),
            "started_at": c.get("started_at"), "decided_at": c.get("decided_at"), "min_outcomes": floor,
            "canary": {k: ca.get(k) for k in ("traces", "outcomes", "graded", "read", "recurred", "succeeded")},
            "control": {k: co.get(k) for k in ("traces", "outcomes", "graded", "read", "recurred", "succeeded")},
            "class": klass, "zero_count_bound": decided and int(co.get("recurred") or 0) == 0,
            "kept_cases": kept_cases, "moved_cases": moved_cases,
            "stamped": {"canary": len(can_eps), "control": len(con_eps)},
            "p16": {"canary_sessions": len(pc), "engine_read": ca.get("read"), "engine_graded": graded_n,
                    "top5_name": top5, "cited_name": cited, "exact": exact,
                    "share_exact_of_graded": exact / graded_n if graded_n else None,
                    "share_exact_of_sessions": exact / len(pc) if pc else None},
            "branch_platform_versions": len(bv),
            "p15prime": {"treated_sessions": len(treated), "kept": {"treated": [k_tr, s_tr], "control": [k_co, s_co]},
                         "fas": {"treated": r_tr, "control": r_co},
                         "control_minus_treated": (r_co - r_tr) if r_tr is not None and r_co is not None else None,
                         "underpowered": len(treated) < 8},
            "reuse_events": reuse.get(f"repair/{fid[:8]}", {}),
        })

    n = len(rows)
    decided_rows = [r for r in rows if r["canary_status"] in ("passed", "failed")]
    reads = [int(r["canary"]["read"] or 0) for r in rows]
    classes = {k: sum(1 for r in rows if r["class"] == k) for k in ("decided_treated", "unexercised", "decided_untreated", "other")}
    passed_shipped = [r for r in rows if r["canary_status"] == "passed"]
    p16_rows = [r for r in rows if r["p16"]["engine_graded"] > 0]
    p16_share = (sum(r["p16"]["exact"] for r in p16_rows) / sum(r["p16"]["engine_graded"] for r in p16_rows)
                 if p16_rows else None)
    return {
        "run": run, "canaries": n, "cut": CUT, "classes": classes,
        "p13prime": {"decided": len(decided_rows), "share": len(decided_rows) / n if n else None,
                     "met": n > 0 and len(decided_rows) / n >= 1 / 3, "falsified": n >= 6 and not decided_rows,
                     "zero_count_bound": sum(1 for r in decided_rows if r["zero_count_bound"]),
                     "control_recurring": sum(1 for r in decided_rows if not r["zero_count_bound"])},
        "p13doubleprime": {"median_read": statistics.median(reads) if reads else None,
                           "reads": sorted(reads), "floors": sorted({r["min_outcomes"] for r in rows}),
                           "any_reached_floor": any(int(r["canary"]["read"] or 0) >= r["min_outcomes"] for r in rows)},
        "p15prime": {"passed": len(passed_shipped),
                     "harmful": [r["fix_id"] for r in passed_shipped if r["p15prime"]["control_minus_treated"] is not None
                                 and r["p15prime"]["control_minus_treated"] > MARGIN and not r["p15prime"]["underpowered"]],
                     "underpowered": [r["fix_id"] for r in passed_shipped if r["p15prime"]["underpowered"]]},
        "p16": {"canaries_with_graded_canary_sessions": len(p16_rows), "share_exact_of_graded_pooled": p16_share,
                "per_canary": {r["fix_id"][:8]: r["p16"]["share_exact_of_graded"] for r in p16_rows},
                "exact_vs_engine_read": {r["fix_id"][:8]: [r["p16"]["exact"], r["p16"]["engine_read"]] for r in rows},
                "met": p16_share is not None and p16_share >= 0.5, "falsified": p16_share is not None and p16_share < 0.2},
        "rows": rows,
    }


def main() -> None:
    dsn = os.environ.get("PGPASS_DSN")
    if not dsn:
        sys.exit("PGPASS_DSN is not set")
    runs = sys.argv[1:] or sorted(p.name for p in rsi.RESULTS.glob("rsi-conf-e2b-*"))
    for run in runs:
        out = readout(run, dsn)
        (rsi.RESULTS / run / "canary_classes.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"## {run}: {out['canaries']} canaries  classes={out['classes']}")
        print("| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | "
              "P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in out["rows"]:
            ca, co, p = r["canary"], r["control"], r["p15prime"]
            fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
            print(f"| {r['cell'].split('|')[0]} {r['cell'].split('|')[3] if r['cell'].count('|') >= 3 else ''} | {r['fix_id'][:8]} | "
                  f"{r['situation']} | {r['lever']} | {r['canary_status']} | {r['reason'][:40]} | {r['class']} | "
                  f"{ca['read']}/{ca['graded']}/{ca['traces']} | {co['read']}/{co['graded']}/{co['traces']} | "
                  f"{r['p16']['exact']}/{r['p16']['engine_graded']} ({r['p16']['canary_sessions']}) | "
                  f"{fmt(p['fas']['treated'])} ({p['kept']['treated'][0]}) | {fmt(p['fas']['control'])} ({p['kept']['control'][0]}) | "
                  f"{sum(r['reuse_events'].values())} |")
        for k in ("p13prime", "p13doubleprime", "p15prime", "p16"):
            print(f"{k}: {json.dumps(out[k])}")
        print()


if __name__ == "__main__":
    main()
