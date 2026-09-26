"""Knowledge transfer between agents: do peers build on each other, and does continual learning
make that transfer faster and safer than a shared memory store?

Two protocols (``Scenario.fleet_mode``):

pioneer   agent 1 works alone for episodes 0..join_at-1; agents 2..N join at ``join_at`` and the
          N share the queue. The measurement is each follower's FIRST exposure to a task class
          it has never handled: the pioneer had to discover the Acme quirk by failing; a follower
          either inherits it from the store or repeats the discovery.
            transfer ratio = follower first-exposure success / pioneer first-exposure success
          (1.0 = the follower is exactly as good on its first try as the pioneer was on its first
          try; > 1.0 = the follower starts ahead of where the pioneer started; a solo agent's
          curve from the single-agent cells gives the "no peers" reference).

newcomer  agents 1..N-1 share episodes 0..join_at-1 (with a regime change in the middle, or
          not); agent N arrives at ``join_at`` and takes every episode after. With a change the
          measurement is the newcomer's first exposures to the changed classes: does it apply the
          stale fix its peers already burned on (stale-pick), or the new one? Without a change
          (``change_at == 0``) it is a cold start onto a healthy store. With ``--newcomer-model``
          the arriving agent runs on a different model than the veterans (cross-model onboarding).

Onboarding velocity (both protocols):
  episodes_to_parity  the joining agent's k-th episode after joining at which its trailing-5
                      first-attempt success rate first reaches the veterans' reference. Pioneer:
                      the pioneer's own trailing-5 at ITS k-th episode (same offset from start).
                      Newcomer: the fleet's trailing-5 over the last 5 pre-join episodes (a
                      constant). Needs a full window, so parity cannot be read before episode 5.
  tokens_to_parity    prompt+completion tokens the joining agent consumed up to and including
                      the parity episode.
  cold-start curve    the joining agent's first-attempt success on its 1st..5th episode after
                      joining, pooled over cells.

Both protocols also report how much each episode leaned on peer-authored entries: the share of
cited keys written by a different agent, using ``written_keys`` for authorship.

    python -m benchmarks.continual_learning.analysis.transfer
    python -m benchmarks.continual_learning.analysis.transfer --runs gridv3

``--runs <prefix>`` reads the protocol cells (``fleet_mode`` pioneer / newcomer) from every
``results/<prefix>*/episodes.jsonl`` and writes to ``results/<prefix>-report``.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import regime_change as rc
from .regime_change import CLASS_TAG, LABEL, RESULTS, boot_ci, class_of, is_infra, mean, order_arms, sel, table

OUT = RESULTS / "gridv2-report"
RUNS = {"pioneer": ["gridv2-xfer-pioneer", "gridv2-xfer-pioneer-mem0"], "newcomer": ["gridv2-xfer-newcomer"]}
ARMS = ["none", "pgvector", "pgvector-diy+outcomes", "mem0", "senselab-nofeedback", "senselab"]
PARITY_WINDOW = 5      # trailing window for episodes_to_parity
COLD_START_N = 5       # episodes shown in the cold-start curve
DEFAULT_JOIN = {"pioneer": 20, "newcomer": 40}

__all__ = ["CLASS_TAG", "load", "cells", "first_exposures", "joiner_agents", "veteran_agents", "use_runs"]


def use_paper() -> None:
    """``--paper``: the paper's arms from the grid v4 transfer runs (see paper_sources.py)."""
    global OUT, ARMS
    from . import paper_sources
    paper_sources.configure()
    OUT = rc.OUT
    runs = sorted(p.name for p in RESULTS.glob(f"{paper_sources.FAMILY_PREFIX}*") if (p / "episodes.jsonl").exists()
                  and not p.name.endswith("-report"))
    RUNS["pioneer"] = runs
    RUNS["newcomer"] = runs
    ARMS = list(rc.HEAD)


def use_runs(prefix: str) -> None:
    global OUT, ARMS
    rc.use_runs(prefix)
    OUT = rc.OUT
    runs = sorted(p.name for p in RESULTS.glob(f"{prefix}*") if (p / "episodes.jsonl").exists()
                  and not p.name.endswith("-report"))
    RUNS["pioneer"] = runs
    RUNS["newcomer"] = runs
    ARMS = list(rc.HEAD)


def load(protocol: str) -> list[dict]:
    rows = []
    for run in RUNS[protocol]:
        p = RESULTS / run / "episodes.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            x = json.loads(line)
            if x.get("fleet_mode", "rr") != protocol:
                continue
            if rc.SCENARIO_RUN and x["arm"] not in ARMS:
                continue   # --paper: arms outside the paper's set are not read
            x["run"] = run
            x["cls"] = class_of(x)
            x["agent"] = x["agent_id"].rsplit("-", 1)[-1]
            x["tokens"] = x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"]
            x["changed"] = bool((x.get("tags") or {}).get("change_class"))
            x["stale"] = bool((x.get("flags") or {}).get("stale_pick"))
            x["join"] = x.get("join_at") or DEFAULT_JOIN[protocol]
            x["n_agents"] = x.get("fleet") or 6
            x["newcomer_model"] = x.get("newcomer_model") or ""
            x["infra"] = is_infra(x.get("error"))
            if x["infra"] and not x.get("attempts"):
                continue   # no attempt reached: see regime_change.DROPPED_NO_ATTEMPT
            rows.append(x)
    return rows


def arms_of(rows: list[dict]) -> list[str]:
    """Report order: the configured ARMS first, then any other arm present in canonical order."""
    present = {x["arm"] for x in rows}
    return [a for a in ARMS if a in present] + [a for a in order_arms(present) if a not in ARMS]


def joiner_agents(cell_rows: list[dict], protocol: str) -> set[str]:
    """Agents that join at ``join_at``: the newcomer (agent N) or the pioneer's followers (2..N)."""
    n = cell_rows[0]["n_agents"]
    return {str(n)} if protocol == "newcomer" else {str(i) for i in range(2, n + 1)}


def veteran_agents(cell_rows: list[dict], protocol: str) -> set[str]:
    """Agents active before ``join_at``: agents 1..N-1 (newcomer) or the pioneer alone."""
    n = cell_rows[0]["n_agents"]
    return {str(i) for i in range(1, n)} if protocol == "newcomer" else {"1"}


def authorship(cell_rows: list[dict]) -> dict[str, tuple[str, int]]:
    """key -> (agent, episode) of the first write of that key in the cell."""
    first: dict[str, tuple[str, int]] = {}
    for x in sorted(cell_rows, key=lambda r: r["episode"]):
        for k in x.get("written_keys") or []:
            first.setdefault(k, (x["agent"], x["episode"]))
    return first


def peer_share(x: dict, authors: dict[str, tuple[str, int]]) -> float | None:
    """Share of the keys this episode cited that a different agent wrote (None if nothing cited)."""
    cited = [k.split("/")[-1] for k in x.get("cited_keys") or []]
    known = [k for k in cited if k in authors]
    if not known:
        return None
    return sum(1 for k in known if authors[k][0] != x["agent"]) / len(known)


def first_exposures(cell_rows: list[dict], agents: set[str] | None = None, after: int = 0) -> list[dict]:
    """Each agent's first episode on each task class, restricted to ``agents`` and episodes >= after."""
    seen: set[tuple[str, str | None]] = set()
    out = []
    for x in sorted(cell_rows, key=lambda r: r["episode"]):
        if x["episode"] < after or (agents is not None and x["agent"] not in agents):
            continue
        k = (x["agent"], x["cls"])
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def cells(rows: list[dict]) -> dict[str, list[dict]]:
    d = defaultdict(list)
    for x in rows:
        d[x["cell"]].append(x)
    return d


def modal(rows: list[dict], key: str, default):
    c = Counter(x.get(key) for x in rows)
    return c.most_common(1)[0][0] if c else default


# ---- onboarding velocity -------------------------------------------------------------

def trailing(seq: list[float], i: int, w: int = PARITY_WINDOW) -> float | None:
    """Mean of seq[i-w+1 .. i]; None until a full window is available."""
    if i + 1 < w:
        return None
    return sum(seq[i - w + 1:i + 1]) / w


def parity(joiner_rows: list[dict], reference) -> tuple[int | None, int | None, int]:
    """(episodes_to_parity, tokens_to_parity, window) for one joining agent.

    ``joiner_rows`` are the agent's episodes after joining, in order; ``reference(k)`` is the
    veterans' trailing-w first-attempt rate to beat at the agent's k-th episode (None = not
    defined). Parity is the first k with a full trailing window at which the joiner's trailing
    rate >= the reference; tokens are cumulative prompt+completion through that episode.
    ``window`` is the number of episodes the agent had (what "n/a within N" refers to)."""
    seq = [1.0 if x["first_attempt_success"] else 0.0 for x in joiner_rows]
    toks = 0
    for i, x in enumerate(joiner_rows):
        toks += x["tokens"]
        mine, ref = trailing(seq, i), reference(i + 1)
        if mine is None or ref is None:
            continue
        if mine >= ref:
            return i + 1, toks, len(joiner_rows)
    return None, None, len(joiner_rows)


def onboarding_units(cell_rows: list[dict], protocol: str) -> list[dict]:
    """One record per joining agent in the cell: its ordered post-join episodes, its parity
    reading and the veterans' reference it was measured against."""
    join = cell_rows[0]["join"]
    out = []
    if protocol == "pioneer":
        pio = sorted((x for x in cell_rows if x["agent"] == "1" and x["episode"] < join), key=lambda r: r["episode"])
        pseq = [1.0 if x["first_attempt_success"] else 0.0 for x in pio]

        def ref(k: int, _p=pseq):
            return trailing(_p, min(k, len(_p)) - 1) if _p else None
    else:
        vets = [x for x in cell_rows if x["episode"] < join and x["episode"] >= join - PARITY_WINDOW]
        const = mean((x["first_attempt_success"] for x in vets), 4) if len(vets) >= PARITY_WINDOW else None

        def ref(k: int, _c=const):
            return None if _c is None or _c != _c else _c  # nan-safe
    for agent in sorted(joiner_agents(cell_rows, protocol), key=int):
        mine = sorted((x for x in cell_rows if x["agent"] == agent and x["episode"] >= join), key=lambda r: r["episode"])
        if not mine:
            continue
        k, toks, window = parity(mine, ref)
        out.append({"cell": cell_rows[0]["cell"], "agent": agent, "episodes": mine, "parity": k,
                    "tokens": toks, "window": window, "reference": ref(PARITY_WINDOW)})
    return out


def _fmt_na(units: list[dict]) -> str:
    miss = [u for u in units if u["parity"] is None]
    if not units:
        return "-"
    window = max(u["window"] for u in units)
    return f"{len(miss) / len(units):.0%} (n/a within {window})"


def onboarding_tables(md: list[str], rows: list[dict], protocol: str, letter: str, n_from: int, *,
                      who: str, flt=None) -> None:
    """The two onboarding-velocity tables for one protocol block."""
    md.append(f"### {letter}{n_from}. Onboarding velocity: episodes and tokens to parity with the veterans\n")
    md.append(f"Unit = one joining agent in one cell ({who}). Parity = first episode after joining at which the "
              f"agent's trailing-{PARITY_WINDOW} first-attempt success rate reaches the veterans' reference "
              + ("(the pioneer's own trailing-5 at the same episode offset from its start)"
                 if protocol == "pioneer" else
                 f"(the fleet's first-attempt rate over its last {PARITY_WINDOW} pre-join episodes)")
              + f"; it needs a full window, so the earliest reading is episode {PARITY_WINDOW}. "
              "Tokens = the agent's cumulative prompt+completion tokens through the parity episode. "
              "Units that never reach parity within their episodes are reported as n/a.\n")
    lines = []
    all_units: dict[str, list[dict]] = {}
    for a in arms_of(rows):
        ar = [x for x in rows if x["arm"] == a]
        if not ar:
            continue
        units = [u for cr in cells(ar).values() for u in onboarding_units(cr, protocol)]
        all_units[a] = units
        hit = [u for u in units if u["parity"] is not None]
        eps = [u["parity"] for u in hit]
        toks = [u["tokens"] for u in hit]
        refs = [u["reference"] for u in units if u["reference"] is not None]
        lines.append([LABEL.get(a, a), len(units), mean(refs, 2) if refs else "-", f"{len(hit)}/{len(units)}",
                      f"{st.median(eps):.1f}" if eps else "-", mean(eps, 1) if eps else "-", _fmt_na(units),
                      f"{st.median(toks):.0f}" if toks else "-", f"{mean(toks, 0):.0f}" if toks else "-"])
    md.append(table(["arm", "units", "veterans' reference", "reached parity", "median episodes to parity",
                     "mean", "never (share)", "median tokens to parity", "mean tokens"], lines) + "\n")

    md.append(f"### {letter}{n_from + 1}. Cold start: the joining agent's first-attempt success on its "
              f"1st..{COLD_START_N}th episode after joining\n")
    md.append("Pooled over cells" + (" and followers" if protocol == "pioneer" else "")
              + ("; all task classes" if flt is None else "; changed-class episodes only are marked (ch)") + ".\n")
    lines = []
    for a, units in all_units.items():
        if not units:
            continue
        cols = []
        for k in range(COLD_START_N):
            kth = [u["episodes"][k] for u in units if len(u["episodes"]) > k]
            cols.append(mean(x["first_attempt_success"] for x in kth) if kth else "-")
        if flt is not None:
            ch = [mean(x["first_attempt_success"] for u in units if len(u["episodes"]) > k
                       for x in [u["episodes"][k]] if flt(x)) for k in range(COLD_START_N)]
            cols += [c if c == c else "-" for c in ch]
        lines.append([LABEL.get(a, a), len(units), *cols])
    head = ["arm", "units", *[f"ep {k + 1}" for k in range(COLD_START_N)]]
    if flt is not None:
        head += [f"ep {k + 1} (ch)" for k in range(COLD_START_N)]
    md.append(table(head, lines) + "\n")


# ---- pioneer -------------------------------------------------------------------------

def pioneer_tables(md: list[str]) -> None:
    rows = load("pioneer")
    if not rows:
        md.append("_pioneer protocol: no data yet_\n")
        return
    join = modal(rows, "join", 20)
    n_agents = modal(rows, "n_agents", 6)
    md.append(f"## A. Pioneer protocol: one agent learns alone for {join} episodes, {n_agents - 1} peers join\n")
    md.append("First exposure = an agent's first episode on a task class it has never handled. The pioneer's "
              "first exposures are the discovery cost; the followers' first exposures show what transferred.\n")
    lines = []
    for a in arms_of(rows):
        ar = [x for x in rows if x["arm"] == a]
        if not ar:
            continue
        pio, fol, fol_cells, peer, fol_all, tok_p, tok_f = [], [], defaultdict(list), [], [], [], []
        for cid, cr in cells(ar).items():
            cj = cr[0]["join"]
            authors = authorship(cr)
            pio += first_exposures(cr, {"1"}, 0)
            f = first_exposures(cr, joiner_agents(cr, "pioneer"), cj)
            fol += f
            fol_cells[cid] += [float(x["first_attempt_success"]) for x in f]
            for x in cr:
                if x["episode"] >= cj:
                    fol_all.append(x)
                    ps = peer_share(x, authors)
                    if ps is not None:
                        peer.append(ps)
            tok_p += [x["tokens"] for x in cr if x["episode"] < cj]
            tok_f += [x["tokens"] for x in cr if x["episode"] >= cj]
        p1 = mean(x["first_attempt_success"] for x in pio)
        f1 = mean(x["first_attempt_success"] for x in fol)
        lo, hi = boot_ci(fol_cells)
        lines.append([LABEL.get(a, a), len(pio), p1, mean(x["escalated"] for x in pio), len(fol),
                      f"{f1} [{lo:.2f},{hi:.2f}]", mean(x["escalated"] for x in fol),
                      round(f1 / p1, 2) if p1 else "-", mean(peer) if peer else "-",
                      f"{mean(tok_p, 0):.0f}", f"{mean(tok_f, 0):.0f}"])
    md.append(table(["arm", "pioneer 1st-exposures", "1st-attempt", "escalated", "follower 1st-exposures",
                     "1st-attempt [95% CI]", "escalated", "transfer ratio", "peer-authored share of cited keys",
                     "tok/ep pioneer", "tok/ep followers"], lines) + "\n")

    # followers' learning curve after joining, by 10-episode block, all episodes (not only first exposures)
    md.append("### A2. Followers after joining: first-attempt success by block (all their episodes)\n")
    lines = []
    for a in arms_of(rows):
        ar = [x for x in rows if x["arm"] == a and x["episode"] >= x["join"] and x["agent"] != "1"]
        if not ar:
            continue
        n_blocks = max((x["episode"] - x["join"]) // 10 for x in ar) + 1
        lines.append([LABEL.get(a, a), *[mean(x["first_attempt_success"] for x in ar if (x["episode"] - x["join"]) // 10 == k)
                                          for k in range(n_blocks)]])
    nb = max((len(l) - 1 for l in lines), default=0)
    lines = [l + ["-"] * (nb + 1 - len(l)) for l in lines]
    md.append(table(["arm", *[f"ep {join + 10 * k}-{join + 10 * k + 9}" for k in range(nb)]], lines) + "\n")

    # quirk classes only: where the textbook answer is wrong and only the store can help
    md.append("### A3. Follower first exposures on Acme-quirk classes only (textbook answer is wrong)\n")
    lines = []
    for a in arms_of(rows):
        ar = [x for x in rows if x["arm"] == a]
        if not ar:
            continue
        fol = [x for cr in cells(ar).values() for x in first_exposures(cr, joiner_agents(cr, "pioneer"), cr[0]["join"])
               if (x.get("tags") or {}).get("quirk") or x["scenario"] == "order-ops"]
        pio = [x for cr in cells(ar).values() for x in first_exposures(cr, {"1"}, 0)
               if (x.get("tags") or {}).get("quirk") or x["scenario"] == "order-ops"]
        lines.append([LABEL.get(a, a), len(pio), mean(x["first_attempt_success"] for x in pio), len(fol),
                      mean(x["first_attempt_success"] for x in fol), mean(x["success"] for x in fol),
                      mean(x["escalated"] for x in fol), mean(x["attempts"] for x in fol)])
    md.append(table(["arm", "pioneer n", "pioneer 1st", "follower n", "follower 1st", "follower success",
                     "follower escalated", "follower attempts"], lines) + "\n")

    onboarding_tables(md, rows, "pioneer", "A", 4, who="cell x follower")


# ---- newcomer ------------------------------------------------------------------------

def _newcomer_first_exposure_line(ar: list[dict], label: str, scen_label: str, flt) -> list:
    fe = [x for cr in cells(ar).values() for x in first_exposures(cr, joiner_agents(cr, "newcomer"), cr[0]["join"])
          if flt(x)]
    fe_cells = defaultdict(list)
    for x in fe:
        fe_cells[x["cell"]].append(float(x["first_attempt_success"]))
    lo, hi = boot_ci(fe_cells)
    new_ch = [x for x in ar if x["episode"] >= x["join"] and flt(x)]
    return [label, scen_label, len(fe_cells), len(fe),
            f"{mean(x['first_attempt_success'] for x in fe)} [{lo:.2f},{hi:.2f}]",
            mean(x["stale"] for x in fe), mean(x["escalated"] for x in fe),
            mean(x["first_attempt_success"] for x in new_ch)]


def _matched_scenario_table(md: list[str], rows: list[dict], letter: str, join: int, flt) -> None:
    """The three-arm comparison on the scenarios every arm completed, so that no arm's row mixes a
    different scenario set from the others; below it, the arms that ran every scenario on the full set.
    A scenario counts as completed by an arm when the arm has as many cells on it as the arm with the
    most cells there (2026-09-20: Mem0 finished support and ci-fix; dispatch and policy-qa are pending)."""
    arms = arms_of(rows)
    eps = defaultdict(set)
    for x in rows:
        eps[(x["arm"], x["scenario"], x["cell"])].add((x["agent"], x["episode"]))
    scenarios = sorted({x["scenario"] for x in rows})
    # A cell counts only when it is complete: as many episodes as the fullest cell of its scenario.
    full_len = {s: max(len(v) for (a, sc, c), v in eps.items() if sc == s) for s in scenarios}
    per = defaultdict(set)
    for (a, s, c), v in eps.items():
        if len(v) >= full_len[s]:
            per[(a, s)].add(c)
    full = {s: max(len(per[(a, s)]) for a in arms) for s in scenarios}
    matched = [s for s in scenarios if all(len(per[(a, s)]) >= full[s] for a in arms)]
    complete_arms = [a for a in arms if all(len(per[(a, s)]) >= full[s] for s in scenarios)]
    if not matched or len(matched) == len(scenarios):
        return   # every arm ran everything: the main table already is the matched comparison
    md.append(f"### {letter}1. Newcomer protocol on the scenarios every arm completed\n")
    md.append(f"Matched rows: {', '.join(matched)} only, so every arm's figure covers the same cells, seeds and "
              f"models. The rows marked 'all' cover every scenario in the protocol and are given for the arms "
              f"that ran all of them ({', '.join(LABEL.get(a, a) for a in complete_arms)}). Pending for the "
              f"other arms: {', '.join(s for s in scenarios if s not in matched)}.\n")
    lines = []
    m_lbl = " + ".join(matched)
    for a in arms:
        ar = [x for x in rows if x["arm"] == a and x["scenario"] in matched]
        lines.append(_newcomer_first_exposure_line(ar, LABEL.get(a, a), m_lbl, flt))
    for a in complete_arms:
        ar = [x for x in rows if x["arm"] == a]
        lines.append(_newcomer_first_exposure_line(ar, LABEL.get(a, a), f"all ({len(scenarios)})", flt))
    md.append(table(["arm", "scenarios", "cells", "1st-exposures (changed)", "1st-attempt [95% CI]",
                     "stale-pick", "escalated", f"newcomer all changed ep{join}+ 1st"], lines) + "\n")


def _newcomer_block(md: list[str], rows: list[dict], letter: str, changed_only: bool) -> None:
    join = modal(rows, "join", 40)
    change = modal(rows, "change_at", 0)
    n = modal(rows, "n_agents", 6)
    flt = (lambda x: x["changed"]) if changed_only else (lambda x: True)
    if changed_only:
        md.append(f"## {letter}. Newcomer protocol: {n - 1} agents live through a regime change at {change}; "
                  f"agent {n} arrives at {join}\n")
        md.append("The newcomer has never seen these tasks. On changed classes the store holds both the old fix "
                  "(which its peers learned before the change and burned on after it) and whatever replaced it.\n")
        what = "changed"
    else:
        md.append(f"## {letter}. Cold start onto a healthy store (no regime change): {n - 1} agents work for "
                  f"{join} episodes; agent {n} arrives at {join}\n")
        md.append("The newcomer has never seen these tasks and nothing in the store is stale: what it inherits "
                  "is whatever its peers learned. Tables cover every task class.\n")
        what = "all"
    lines = []
    for a in arms_of(rows):
        ar = [x for x in rows if x["arm"] == a]
        if not ar:
            continue
        new = [x for x in ar if x["episode"] >= x["join"]]
        vets_post = [x for x in ar if (x.get("change_at") or 0) <= x["episode"] < x["join"]]
        new_ch = [x for x in new if flt(x)]
        vets_ch_late = [x for x in vets_post if flt(x) and x["episode"] >= x["join"] - 10]
        fe = [x for cr in cells(ar).values() for x in first_exposures(cr, joiner_agents(cr, "newcomer"), cr[0]["join"])
              if flt(x)]
        fe_cells = defaultdict(list)
        for x in fe:
            fe_cells[x["cell"]].append(float(x["first_attempt_success"]))
        lo, hi = boot_ci(fe_cells)
        peer = []
        for cr in cells(ar).values():
            authors = authorship(cr)
            for x in cr:
                if x["episode"] >= x["join"]:
                    ps = peer_share(x, authors)
                    if ps is not None:
                        peer.append(ps)
        lines.append([LABEL.get(a, a), mean(x["first_attempt_success"] for x in vets_ch_late), mean(x["stale"] for x in vets_ch_late),
                      len(fe), f"{mean(x['first_attempt_success'] for x in fe)} [{lo:.2f},{hi:.2f}]",
                      mean(x["stale"] for x in fe), mean(x["escalated"] for x in fe),
                      mean(x["first_attempt_success"] for x in new_ch), mean(x["stale"] for x in new_ch),
                      mean(peer) if peer else "-"])
    md.append(table(["arm", f"veterans ep{join - 10}-{join - 1} 1st", "veterans stale", f"newcomer 1st-exposures ({what})",
                     "1st-attempt [95% CI]", "stale-pick", "escalated",
                     f"newcomer all {'changed ' if changed_only else ''}ep{join}+ 1st", "stale",
                     "peer-authored share"], lines) + "\n")
    if changed_only:
        _matched_scenario_table(md, rows, letter, join, flt)
    md.append(f"### {letter}2. Newcomer on {what} classes by 5-episode block after arrival\n")
    lines = []
    post = [x for x in rows if x["episode"] >= x["join"] and flt(x)]
    n_blocks = (max((x["episode"] - x["join"] for x in post), default=0) // 5 + 1) if post else 0
    for a in arms_of(rows):
        ar = [x for x in post if x["arm"] == a]
        if not ar:
            continue
        lines.append([LABEL.get(a, a),
                      *[mean(x["first_attempt_success"] for x in ar if (x["episode"] - x["join"]) // 5 == k) for k in range(n_blocks)],
                      *[mean(x["stale"] for x in ar if (x["episode"] - x["join"]) // 5 == k) for k in range(n_blocks)]])
    blocks = [f"{join + 5 * k}-{join + 5 * k + 4}" for k in range(n_blocks)]
    md.append(table(["arm", *(["1st " + blocks[0]] + blocks[1:] if blocks else []),
                     *(["stale " + blocks[0]] + blocks[1:] if blocks else [])], lines) + "\n")

    onboarding_tables(md, rows, "newcomer", letter, 3, who="one newcomer per cell",
                      flt=(lambda x: x["changed"]) if changed_only else None)


def cross_model_tables(md: list[str], rows: list[dict], letter: str) -> None:
    """Newcomer cells whose arriving agent ran on a different model than the veterans, grouped by
    (arm, veteran model -> newcomer model). Same-model cells of the same arm are listed alongside
    as the reference; the section says so when there is nothing cross-model to show."""
    md.append(f"## {letter}. Cross-model onboarding\n")
    groups: dict[tuple[str, str, str, bool], list[list[dict]]] = defaultdict(list)
    cross = False
    for cr in cells(rows).values():
        newc = joiner_agents(cr, "newcomer")
        vet_models = sorted({x["model"] for x in cr if x["agent"] not in newc})
        new_models = sorted({x["model"] for x in cr if x["agent"] in newc})
        nm = cr[0].get("newcomer_model") or ""
        is_cross = bool(nm) or (bool(new_models) and set(new_models) != set(vet_models))
        cross |= is_cross
        has_change = any(x["changed"] for x in cr)
        groups[(cr[0]["arm"], ",".join(vet_models), nm or ",".join(new_models) or "-", has_change)].append(cr)
    if not cross:
        md.append("_no cross-model cells found: in every newcomer cell the arriving agent ran on the veterans' "
                  "model (no `newcomer_model` on the rows, and no model difference between agents)_\n")
        return
    md.append("Grouped by the veterans' model and the newcomer's model (same-model cells of the same arm are the "
              "reference rows). First-exposure success is the newcomer's first episode on each task class after "
              "joining (changed classes when the cell has a regime change, all classes on a healthy store); parity "
              "as in the onboarding tables above; stale-pick over all the newcomer's episodes.\n")
    lines = []
    order = {a: i for i, a in enumerate(arms_of(rows))}
    for (arm, vet, new, has_change), crs in sorted(groups.items(),
                                                   key=lambda kv: (order.get(kv[0][0], 99), not kv[0][3], kv[0][1], kv[0][2])):
        flt = (lambda x: x["changed"]) if has_change else (lambda x: True)
        fe = [x for cr in crs for x in first_exposures(cr, joiner_agents(cr, "newcomer"), cr[0]["join"]) if flt(x)]
        new_rows = [x for cr in crs for x in cr if x["agent"] in joiner_agents(cr, "newcomer") and x["episode"] >= cr[0]["join"]]
        units = [u for cr in crs for u in onboarding_units(cr, "newcomer")]
        hit = [u["parity"] for u in units if u["parity"] is not None]
        toks = [u["tokens"] for u in units if u["parity"] is not None]
        lines.append([LABEL.get(arm, arm), vet, new + ("" if vet != new else " (same)"),
                      "regime change" if has_change else "healthy", len(crs), len(fe),
                      mean(x["first_attempt_success"] for x in fe), mean(x["escalated"] for x in fe),
                      mean(x["stale"] for x in new_rows), f"{len(hit)}/{len(units)}",
                      f"{st.median(hit):.1f}" if hit else "-", _fmt_na(units),
                      f"{st.median(toks):.0f}" if toks else "-", f"{mean((x['tokens'] for x in new_rows), 0):.0f}"])
    md.append(table(["arm", "veterans' model", "newcomer model", "store", "cells", "1st-exposures", "1st-attempt",
                     "escalated", "stale-pick", "reached parity", "median episodes to parity", "never (share)",
                     "median tokens to parity", "tok/ep newcomer"], lines) + "\n")


def newcomer_tables(md: list[str]) -> None:
    rows = load("newcomer")
    if not rows:
        md.append("_newcomer protocol: no data yet_\n")
        return
    with_change = [x for x in rows if x.get("change_at")]
    healthy = [x for x in rows if not x.get("change_at")]
    letters = iter("BCDEFG")
    if with_change:
        # a change_at without any changed-class tag (scenario has no CHANGES) is a healthy store too
        _newcomer_block(md, with_change, next(letters), changed_only=any(x["changed"] for x in with_change))
    if healthy:
        _newcomer_block(md, healthy, next(letters), changed_only=False)
    cross_model_tables(md, rows, next(letters))


def main() -> None:
    if "--paper" in sys.argv:
        use_paper()
    elif "--runs" in sys.argv:
        use_runs(sys.argv[sys.argv.index("--runs") + 1])
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Knowledge transfer between agents: pioneer and newcomer protocols\n"]
    for proto in ("pioneer", "newcomer"):
        n = len(load(proto))
        md.append(f"- {proto}: {n} episodes loaded")
    md.append("")
    pioneer_tables(md)
    newcomer_tables(md)
    (OUT / "TRANSFER.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
