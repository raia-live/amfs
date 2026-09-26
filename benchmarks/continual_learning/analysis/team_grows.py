"""The team grows: what a joining agent inherits from a team that has been repairing itself
(prereg amendment (i), HYPOTHESIS_RSI.md).

The transfer protocols (``analysis/transfer.py``) ask how fast a joiner reaches the veterans.
This module asks what the joiner is handed on arrival, and whether it is any good:

  A. Joiner first exposures after ``join_at`` — first-attempt success on changed and on
     unchanged classes, stale-pick on changed classes, escalation (transfer.py's measure,
     restated here per protocol and arm on the probe's own cells).
  B. Inheritance quality — what sat in the joiner's context on those first exposures, from the
     per-hit evidence the searches logged: validated hits, untested peer notes, discredited
     entries served in the avoid list, stale entries; the share of first exposures that were
     *poisoned* (failed first attempt with an untested peer note in context and no validated
     hit) and *guided* (succeeded with a validated hit in context). The veterans' own late
     episodes are given as the reference: an inherited context should look like a veteran's.
  C. Onboarding velocity — episodes and tokens to parity with the veterans, cold-start curve
     (transfer.py's units, restated on the probe's cells).
  D. Disagreement — on each post-join task class, the share of joiner episodes whose first
     answer differs from the veterans' modal post-change answer on that class. A team whose
     memory carries one repaired answer disagrees less than one whose members each rediscover.
  E. Cost — prompt+completion tokens per joiner episode and per joiner-solved task.
  F. RSI arm only — fixes proposed and shipped before the join (what existed to inherit), the
     levers, and per joiner identity: verdicts and fixes recorded on ITS identity after arrival
     (fixes are scoped per agent on the platform, so a joiner's own loop starts from zero; what
     it inherits arrives through the shared memory, not through the veterans' fix list).

The design (amendment (i)): veterans are replicas of ONE hosted identity (``--fleet-replicas``,
always on for ``senselab-rsi*``), so the team's repair loop has a single fix list and canary;
each joiner is a NEW identity (``Scenario.is_joiner``), so nothing reaches it by construction.

    python -m continual_learning.analysis.team_grows --runs rsi-teamgrows-dev
    python -m continual_learning.analysis.team_grows --runs rsi-teamgrows-dev --scenarios support ci-fix

``--runs <prefix>`` reads the pioneer / newcomer cells from every ``results/<prefix>*/`` and
writes ``results/<prefix>-report/TEAM_GROWS.md`` and ``TEAM_GROWS.json``.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import regime_change as rc
from . import transfer as tr
from .regime_change import LABEL, RESULTS, boot_ci, mean, table

PROTOCOLS = ("pioneer", "newcomer")
LATE_WINDOW = 10   # veterans' reference: their last N episodes before the join


# ---- loading ----------------------------------------------------------------------------

def load_rows(prefix: str, scenarios: set[str] | None) -> dict[str, list[dict]]:
    """protocol -> rows, from every run under ``results/<prefix>*``."""
    tr.use_runs(prefix)
    out: dict[str, list[dict]] = {}
    for proto in PROTOCOLS:
        rows = tr.load(proto)
        if scenarios:
            rows = [x for x in rows if x["scenario"] in scenarios]
        out[proto] = rows
    return out


def load_reports(prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(RESULTS.glob(f"{prefix}*")):
        f = p / "arm_reports.jsonl"
        if p.name.endswith("-report") or not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["_run"] = p.name
            out.append(r)
    return out


def report_for(reports: list[dict[str, Any]], cell_rows: list[dict]) -> dict[str, Any] | None:
    x = cell_rows[0]
    for r in reports:
        if r.get("_run") == x["run"] and r.get("arm") == x["arm"] and r.get("seed") == x["seed"] \
                and str(r.get("cell", "")).startswith(x["scenario"] + "|"):
            return r
    return None


# ---- per-episode context quality ----------------------------------------------------------

def context_of(x: dict, authors: dict[str, tuple[str, int]]) -> dict[str, Any]:
    """What the episode's searches served, classified by evidence and authorship.

    ``authors`` maps a key to (agent, episode) of its first write in the cell (``written_keys``);
    a key nobody in the cell wrote is *platform*-authored (lesson-contrast entries, fix
    memories, seeds) — that is where the repair loop's output would show up."""
    validated = untested_peer = untested_own = platform = avoid = contested = 0
    for s in x.get("searches_log") or []:
        for h in s.get("hits") or []:
            key = str(h.get("key", "")).split("/")[-1]
            if h.get("avoid"):
                avoid += 1
                continue
            ev = h.get("evidence_status")
            if ev == "validated":
                validated += 1
            elif ev == "contested":
                contested += 1
            elif ev in ("untested", None):
                who = authors.get(key)
                if who is None:
                    platform += 1
                elif who[0] == x["agent"]:
                    untested_own += 1
                else:
                    untested_peer += 1
    return {"validated": validated, "untested_peer": untested_peer, "untested_own": untested_own,
            "platform": platform, "avoid": avoid, "contested": contested,
            "stale": int(x.get("stale_in_context") or 0)}


def poisoned(x: dict, ctx: dict[str, Any]) -> bool:
    """Failed at first attempt with an untested peer note in context and nothing validated."""
    return (not x["first_attempt_success"]) and ctx["untested_peer"] > 0 and ctx["validated"] == 0


def guided(x: dict, ctx: dict[str, Any]) -> bool:
    return bool(x["first_attempt_success"]) and ctx["validated"] > 0


# ---- per cell -----------------------------------------------------------------------------

def cell_summary(cr: list[dict], protocol: str, report: dict[str, Any] | None) -> dict[str, Any]:
    cr = sorted(cr, key=lambda r: (r["episode"], r["agent"]))
    join = cr[0]["join"]
    change = int(cr[0].get("change_at") or 0)
    joiners = tr.joiner_agents(cr, protocol)
    veterans = tr.veteran_agents(cr, protocol)
    authors = tr.authorship(cr)

    post = [x for x in cr if x["episode"] >= join]
    joiner_rows = [x for x in post if x["agent"] in joiners]
    vet_pre_late = [x for x in cr if x["agent"] in veterans and join - LATE_WINDOW <= x["episode"] < join]
    vet_post_change = [x for x in cr if x["agent"] in veterans and change <= x["episode"] < join]

    fe = tr.first_exposures(cr, joiners, join)
    fe_changed = [x for x in fe if x["changed"]]
    fe_unchanged = [x for x in fe if not x["changed"]]

    def ctx_block(rows: list[dict]) -> dict[str, Any]:
        ctxs = [context_of(x, authors) for x in rows]
        n = len(rows)
        tot = Counter()
        for c in ctxs:
            tot.update(c)
        return {"n": n, "per_ep": {k: (v / n if n else None) for k, v in tot.items()},
                "any_validated": mean((c["validated"] > 0 for c in ctxs), 3) if n else None,
                "any_untested_peer": mean((c["untested_peer"] > 0 for c in ctxs), 3) if n else None,
                "any_avoid": mean((c["avoid"] > 0 for c in ctxs), 3) if n else None,
                "any_platform": mean((c["platform"] > 0 for c in ctxs), 3) if n else None,
                "poisoned": sum(poisoned(x, c) for x, c in zip(rows, ctxs)),
                "guided": sum(guided(x, c) for x, c in zip(rows, ctxs))}

    # D. disagreement: the veterans' modal first answer per class over their LAST pre-join
    # episodes (what the team had settled on by the join) vs the joiner's first answer.
    modal_vet: dict[str, str] = {}
    by_cls: dict[str, Counter] = defaultdict(Counter)
    for x in vet_pre_late:
        if x.get("answers"):
            by_cls[x["cls"] or "?"][str(x["answers"][0])[:200]] += 1
    for cls, c in by_cls.items():
        modal_vet[cls] = c.most_common(1)[0][0]
    disagree = [str(x["answers"][0])[:200] != modal_vet[x["cls"] or "?"] for x in joiner_rows
                if x.get("answers") and (x["cls"] or "?") in modal_vet]

    # C. onboarding velocity (transfer.py's units)
    units = tr.onboarding_units(cr, protocol)

    # E. cost
    tok = [x["tokens"] for x in joiner_rows]
    solved = sum(1 for x in joiner_rows if x["success"])

    out: dict[str, Any] = {
        "cell": cr[0]["cell"], "run": cr[0]["run"], "arm": cr[0]["arm"], "scenario": cr[0]["scenario"],
        "seed": cr[0]["seed"], "protocol": protocol, "join": join, "change_at": change,
        "n_agents": cr[0]["n_agents"], "joiners": sorted(joiners, key=int), "episodes": len(cr),
        "complete": max(x["episode"] for x in cr) + 1,
        "vet_late_first": mean((x["first_attempt_success"] for x in vet_pre_late), 3) if vet_pre_late else None,
        "vet_post_changed_first": mean((x["first_attempt_success"] for x in vet_post_change if x["changed"]), 3)
        if any(x["changed"] for x in vet_post_change) else None,
        "vet_late_ctx": ctx_block(vet_pre_late),
        "fe": {"all": len(fe), "changed": len(fe_changed), "unchanged": len(fe_unchanged),
               "first_all": mean((x["first_attempt_success"] for x in fe), 3) if fe else None,
               "first_changed": mean((x["first_attempt_success"] for x in fe_changed), 3) if fe_changed else None,
               "first_unchanged": mean((x["first_attempt_success"] for x in fe_unchanged), 3) if fe_unchanged else None,
               "stale_changed": mean((x["stale"] for x in fe_changed), 3) if fe_changed else None,
               "escalated": mean((x["escalated"] for x in fe), 3) if fe else None,
               "ctx": ctx_block(fe), "ctx_changed": ctx_block(fe_changed)},
        "joiner_all": {"n": len(joiner_rows),
                       "first": mean((x["first_attempt_success"] for x in joiner_rows), 3) if joiner_rows else None,
                       "success": mean((x["success"] for x in joiner_rows), 3) if joiner_rows else None,
                       "stale_changed": mean((x["stale"] for x in joiner_rows if x["changed"]), 3)
                       if any(x["changed"] for x in joiner_rows) else None,
                       "ctx": ctx_block(joiner_rows)},
        "disagree": {"n": len(disagree), "rate": mean(disagree, 3) if disagree else None},
        "parity": [{"agent": u["agent"], "episodes": u["parity"], "tokens": u["tokens"], "window": u["window"]}
                   for u in units],
        "cold_start": [mean((u["episodes"][k]["first_attempt_success"] for u in units if len(u["episodes"]) > k), 3)
                       if any(len(u["episodes"]) > k for u in units) else None for k in range(tr.COLD_START_N)],
        "cost": {"tokens_per_ep": mean(tok, 0) if tok else None,
                 "tokens_per_solved": (sum(tok) / solved) if solved else None},
    }
    if cr[0]["arm"].startswith("senselab-rsi"):
        out["rsi"] = rsi_block(report, cr, join, change)
    return out


def rsi_block(report: dict[str, Any] | None, cr: list[dict], join: int, change: int) -> dict[str, Any]:
    """F. What the veterans' loop produced before the join, and the joiners' own identities."""
    if report is None:
        return {"report": "missing"}
    rep = report.get("report") or {}
    ledger = sorted((rep.get("ledger") or []), key=lambda r: float(r.get("sealed_at") or 0))
    fixes = rep.get("fixes") or []
    agents = rep.get("agents") or ([rep.get("agent_id")] if rep.get("agent_id") else [])
    primary = rep.get("agent_id")

    def episode_at(iso: str | None) -> int | None:
        if not iso:
            return None
        try:
            t = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
        ep = None
        for r in ledger:
            if float(r.get("sealed_at") or 0) <= t:
                ep = int(r.get("episode", -1))
        return ep

    proposed = [episode_at(f.get("created_at")) for f in fixes]
    shipped = [episode_at(f.get("shipped_at")) for f in fixes if f.get("shipped_at")]
    before_join_proposed = sum(1 for e in proposed if e is not None and e < join)
    before_join_shipped = sum(1 for e in shipped if e is not None and e < join)
    fix_agents = Counter(str(f.get("agent_id") or primary or "?") for f in fixes)
    peers = rep.get("peers") or {}
    per_peer = {}
    for a, got in peers.items():
        verdicts = got.get("verdicts")
        items = (verdicts.get("items") if isinstance(verdicts, dict) else verdicts) or []
        inc = (got.get("incidents") or {}).get("items") if isinstance(got.get("incidents"), dict) else got.get("incidents")
        per_peer[a] = {"verdicts": len(items), "incidents": len(inc or []),
                       "fixes": fix_agents.get(a, 0),
                       "errors": sorted(k for k in got if k.endswith("_error"))}
    sealed_by_agent = Counter(str(r.get("agent") or primary or "?") for r in ledger)
    return {
        "agents": agents, "primary": primary, "fixes": len(fixes),
        "proposed_before_join": before_join_proposed, "shipped_before_join": before_join_shipped,
        "shipped_total": len(shipped),
        "first_fix_lag": (min(e for e in proposed if e is not None) - change) if any(e is not None for e in proposed) else None,
        "levers": dict(Counter(str(f.get("lever") or "?") for f in fixes)),
        "sources": dict(Counter(str(f.get("source") or "?") for f in fixes)),
        "statuses": dict(Counter(str(f.get("status") or "?") for f in fixes)),
        "fixes_by_agent": dict(fix_agents), "sealed_by_agent": dict(sealed_by_agent),
        "peers": per_peer,
    }


# ---- pooling and rendering ----------------------------------------------------------------

def _f(v, nd=2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pool_rate(cells: list[dict], path: tuple[str, ...], n_path: tuple[str, ...]) -> tuple[float | None, int]:
    """Weighted pooled rate across cells from a per-cell (rate, n) pair."""
    num = den = 0.0
    for c in cells:
        r, n = c, c
        for k in path:
            r = r.get(k) if isinstance(r, dict) else None
        for k in n_path:
            n = n.get(k) if isinstance(n, dict) else None
        if r is None or not n:
            continue
        num += r * n
        den += n
    return (num / den if den else None), int(den)


def _ci(cells: list[dict], path: tuple[str, ...], n_path: tuple[str, ...]) -> str:
    """Pooled rate with a bootstrap CI over cells (cell-level resampling, as transfer.py)."""
    rate, n = _pool_rate(cells, path, n_path)
    if rate is None:
        return "-"
    per_cell: dict[str, list[float]] = {}
    for c in cells:
        r, k = c, c
        for p in path:
            r = r.get(p) if isinstance(r, dict) else None
        for p in n_path:
            k = k.get(p) if isinstance(k, dict) else None
        if r is None or not k:
            continue
        per_cell[c["cell"]] = [1.0] * round(r * k) + [0.0] * (k - round(r * k))
    if len(per_cell) < 2:
        return f"{rate:.2f} (n={n})"
    lo, hi = boot_ci(per_cell)
    return f"{rate:.2f} [{lo:.2f},{hi:.2f}] (n={n})"


def render(summaries: dict[str, list[dict]], prefix: str) -> str:
    md = [f"# The team grows: what a joiner inherits (`{prefix}`)\n",
          "Veterans = replicas of one hosted identity; each joiner = a new identity. First exposure = the "
          "joiner's first episode on a task class after joining. *Poisoned* = failed first attempt with an "
          "untested peer note in context and no validated hit; *guided* = succeeded with a validated hit. "
          "*Platform* hits = keys no agent in the cell wrote (lesson-contrast entries, fix memories, seeds). "
          "A hit with no evidence status at all (arms that track no outcomes) counts as untested. "
          "*Disagreement* = the joiner's first answer differs from the veterans' modal answer on that class "
          f"over their last {LATE_WINDOW} pre-join episodes; read it next to the veterans' late success — "
          "disagreeing with a team that had settled on the wrong answer is not a defect. CIs are bootstrap over cells.\n"]
    for proto in PROTOCOLS:
        cells = summaries.get(proto) or []
        if not cells:
            md.append(f"## {proto}: no cells\n")
            continue
        join = Counter(c["join"] for c in cells).most_common(1)[0][0]
        change = Counter(c["change_at"] for c in cells).most_common(1)[0][0]
        n_agents = Counter(c["n_agents"] for c in cells).most_common(1)[0][0]
        who = (f"agent 1 works alone until {join}; agents 2..{n_agents} join" if proto == "pioneer"
               else f"agents 1..{n_agents - 1} work until {join}; agent {n_agents} joins")
        md.append(f"## {proto}: {who}; regime change at {change}\n")
        arms = rc.order_arms({c["arm"] for c in cells})
        scen = rc.order_scenarios({c["scenario"] for c in cells})
        md.append(f"Cells: {len(cells)} ({', '.join(scen)}; seeds "
                  f"{', '.join(str(s) for s in sorted({c['seed'] for c in cells}))}); "
                  f"complete: {sum(1 for c in cells if c['complete'] >= c['episodes'] and c['complete'] >= join + 5)}"
                  f"/{len(cells)} reached ep {join + 5}+.\n")

        # A
        md.append(f"### A. Joiner first exposures after ep {join}\n")
        lines = []
        for a in arms:
            ac = [c for c in cells if c["arm"] == a]
            lines.append([LABEL.get(a, a), len(ac),
                          _f(_pool_rate(ac, ("vet_late_first",), ("vet_late_ctx", "n"))[0]),
                          _f(_pool_rate(ac, ("vet_post_changed_first",), ("fe", "changed"))[0]),
                          _ci(ac, ("fe", "first_all"), ("fe", "all")),
                          _ci(ac, ("fe", "first_changed"), ("fe", "changed")),
                          _ci(ac, ("fe", "first_unchanged"), ("fe", "unchanged")),
                          _f(_pool_rate(ac, ("fe", "stale_changed"), ("fe", "changed"))[0]),
                          _f(_pool_rate(ac, ("fe", "escalated"), ("fe", "all"))[0]),
                          _ci(ac, ("joiner_all", "first"), ("joiner_all", "n")),
                          _f(_pool_rate(ac, ("joiner_all", "stale_changed"), ("joiner_all", "n"))[0])])
        md.append(table(["arm", "cells", f"veterans ep{join - LATE_WINDOW}-{join - 1} 1st", "veterans changed post 1st",
                         "joiner 1st-exp all", "1st-exp changed", "1st-exp unchanged", "stale-pick (changed 1st-exp)",
                         "escalated", "joiner all eps 1st", "joiner stale (changed, all eps)"], lines) + "\n")

        # B
        md.append("### B. Inheritance quality: what sat in the joiner's context on its first exposures\n")
        lines = []
        for a in arms:
            ac = [c for c in cells if c["arm"] == a]
            n_fe = sum(c["fe"]["all"] for c in ac)
            pois = sum(c["fe"]["ctx"]["poisoned"] for c in ac)
            guid = sum(c["fe"]["ctx"]["guided"] for c in ac)
            n_ch = sum(c["fe"]["changed"] for c in ac)
            pois_ch = sum(c["fe"]["ctx_changed"]["poisoned"] for c in ac)

            def per(k: str, _ac=ac) -> str:
                num = sum((c["fe"]["ctx"]["per_ep"].get(k) or 0) * c["fe"]["all"] for c in _ac)
                den = sum(c["fe"]["all"] for c in _ac)
                return _f(num / den if den else None)

            def vet_per(k: str, _ac=ac) -> str:
                num = sum((c["vet_late_ctx"]["per_ep"].get(k) or 0) * c["vet_late_ctx"]["n"] for c in _ac)
                den = sum(c["vet_late_ctx"]["n"] for c in _ac)
                return _f(num / den if den else None)

            lines.append([LABEL.get(a, a), n_fe,
                          _f(_pool_rate(ac, ("fe", "ctx", "any_validated"), ("fe", "all"))[0]),
                          _f(_pool_rate(ac, ("fe", "ctx", "any_untested_peer"), ("fe", "all"))[0]),
                          _f(_pool_rate(ac, ("fe", "ctx", "any_avoid"), ("fe", "all"))[0]),
                          _f(_pool_rate(ac, ("fe", "ctx", "any_platform"), ("fe", "all"))[0]),
                          f"{per('validated')} / {per('untested_peer')} / {per('platform')} / {per('avoid')} / {per('stale')}",
                          f"{vet_per('validated')} / {vet_per('untested_peer')} / {vet_per('platform')} / {vet_per('avoid')} / {vet_per('stale')}",
                          f"{pois}/{n_fe}" + (f" ({pois / n_fe:.2f})" if n_fe else ""),
                          f"{pois_ch}/{n_ch}" + (f" ({pois_ch / n_ch:.2f})" if n_ch else ""),
                          f"{guid}/{n_fe}" + (f" ({guid / n_fe:.2f})" if n_fe else "")])
        md.append(table(["arm", "1st-exposures", "any validated hit", "any untested peer note", "any discredited (avoid)",
                         "any platform-authored", "hits/ep validated / untested-peer / platform / avoid / stale",
                         f"veterans' own ep{join - LATE_WINDOW}-{join - 1} (same)",
                         "poisoned", "poisoned (changed)", "guided"], lines) + "\n")

        # C
        md.append("### C. Onboarding velocity\n")
        lines = []
        for a in arms:
            ac = [c for c in cells if c["arm"] == a]
            units = [u for c in ac for u in c["parity"]]
            hit = [u["episodes"] for u in units if u["episodes"] is not None]
            toks = [u["tokens"] for u in units if u["episodes"] is not None]
            window = max((u["window"] for u in units), default=0)
            cold = []
            for k in range(tr.COLD_START_N):
                vals = [(c["cold_start"][k], len(c["parity"])) for c in ac if c["cold_start"][k] is not None]
                cold.append(_f(sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else "-")
            lines.append([LABEL.get(a, a), len(units), f"{len(hit)}/{len(units)}",
                          f"{st.median(hit):.1f}" if hit else "-",
                          f"{(len(units) - len(hit)) / len(units):.0%} (n/a within {window})" if units else "-",
                          f"{st.median(toks):.0f}" if toks else "-", *cold])
        md.append(table(["arm", "joiners", "reached parity", "median episodes to parity", "never",
                         "median tokens to parity", *[f"cold ep {k + 1}" for k in range(tr.COLD_START_N)]], lines) + "\n")

        # D + E
        md.append("### D/E. Disagreement with the veterans and cost\n")
        lines = []
        for a in arms:
            ac = [c for c in cells if c["arm"] == a]
            tok_ep = [c["cost"]["tokens_per_ep"] for c in ac if c["cost"]["tokens_per_ep"] is not None]
            tok_solved = [c["cost"]["tokens_per_solved"] for c in ac if c["cost"]["tokens_per_solved"] is not None]
            lines.append([LABEL.get(a, a), _ci(ac, ("disagree", "rate"), ("disagree", "n")),
                          _ci(ac, ("joiner_all", "success"), ("joiner_all", "n")),
                          f"{mean(tok_ep, 0):.0f}" if tok_ep else "-", f"{mean(tok_solved, 0):.0f}" if tok_solved else "-"])
        md.append(table(["arm", f"joiner answers differing from veterans' modal ep{join - LATE_WINDOW}-{join - 1} answer",
                         "joiner success (all eps)", "tokens / joiner ep", "tokens / joiner-solved task"], lines) + "\n")

        # F
        rsi_cells = [c for c in cells if "rsi" in c]
        if rsi_cells:
            md.append("### F. RSI arm: what the veterans' loop had produced by the join, and the joiners' own identities\n")
            lines = []
            for c in rsi_cells:
                r = c["rsi"]
                if r.get("report") == "missing":
                    lines.append([c["cell"], "report missing", "", "", "", "", ""])
                    continue
                peers = r.get("peers") or {}
                lines.append([c["cell"], r["fixes"], f"{r['proposed_before_join']} / {r['shipped_before_join']}",
                              _f(r["first_fix_lag"]), json.dumps(r["levers"], sort_keys=True),
                              json.dumps(r["sources"], sort_keys=True),
                              "; ".join(f"{a.rsplit('-', 1)[-1]}: {p['verdicts']} verdicts, {p['fixes']} fixes"
                                        + (f", errors {p['errors']}" if p["errors"] else "") for a, p in sorted(peers.items()))
                              or "no joiner identity in report"])
            md.append(table(["cell", "fixes (all)", "proposed / shipped before join", "first fix lag (eps after change)",
                             "levers", "sources", "joiner identities"], lines) + "\n")
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, help="results/<prefix>* to read")
    ap.add_argument("--scenarios", nargs="*", default=None)
    args = ap.parse_args()
    scen = set(args.scenarios) if args.scenarios else None
    rows = load_rows(args.runs, scen)
    reports = load_reports(args.runs)
    summaries: dict[str, list[dict]] = {}
    for proto in PROTOCOLS:
        summaries[proto] = [cell_summary(cr, proto, report_for(reports, cr))
                            for cr in tr.cells(rows[proto]).values() if cr]
    out = RESULTS / f"{args.runs}-report"
    out.mkdir(parents=True, exist_ok=True)
    md = render(summaries, args.runs)
    (out / "TEAM_GROWS.md").write_text(md)
    (out / "TEAM_GROWS.json").write_text(json.dumps(summaries, indent=1, default=str))
    print(md)
    print(f"\nwrote {out / 'TEAM_GROWS.md'}")


if __name__ == "__main__":
    main()
