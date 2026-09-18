"""Knowledge transfer between agents: do peers build on each other, and does continual learning
make that transfer faster and safer than a shared memory store?

Two protocols (``Scenario.fleet_mode``):

pioneer   agent 1 works alone for episodes 0..join_at-1; agents 2..N join at ``join_at`` and the
          six share the queue. The measurement is each follower's FIRST exposure to a task class
          it has never handled: the pioneer had to discover the Acme quirk by failing; a follower
          either inherits it from the store or repeats the discovery.
            transfer ratio = follower first-exposure success / pioneer first-exposure success
          (1.0 = the follower is exactly as good on its first try as the pioneer was on its first
          try; > 1.0 = the follower starts ahead of where the pioneer started; a solo agent's
          curve from the single-agent cells gives the "no peers" reference).

newcomer  agents 1..N-1 share episodes 0..join_at-1 with a regime change in the middle; agent N
          arrives at ``join_at`` and takes every episode after. The measurement is the newcomer's
          first exposures to the changed classes: does it apply the stale fix its peers already
          burned on (stale-pick), or the new one?

Both protocols also report how much each episode leaned on peer-authored entries: the share of
cited keys written by a different agent, using ``written_keys`` for authorship.

    python -m benchmarks.continual_learning.analysis.transfer
    python -m benchmarks.continual_learning.analysis.transfer --runs gridv3

``--runs <prefix>`` reads the protocol cells (``fleet_mode`` pioneer / newcomer) from every
``results/<prefix>*/episodes.jsonl`` and writes to ``results/<prefix>-report``.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from . import regime_change as rc
from .regime_change import LABEL, RESULTS, boot_ci, mean, sel, table

OUT = RESULTS / "gridv2-report"
RUNS = {"pioneer": ["gridv2-xfer-pioneer", "gridv2-xfer-pioneer-mem0"], "newcomer": ["gridv2-xfer-newcomer"]}
CLASS_TAG = {"support": "issue", "ci-fix": "failure", "order-ops": "request", "retention": "expected", "diagnose": "rule"}
ARMS = ["none", "pgvector", "pgvector-diy+outcomes", "mem0", "senselab-nofeedback", "senselab"]


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
            x["run"] = run
            x["cls"] = (x.get("tags") or {}).get(CLASS_TAG.get(x["scenario"], "issue"))
            x["agent"] = x["agent_id"].rsplit("-", 1)[-1]
            x["tokens"] = x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"]
            x["changed"] = bool((x.get("tags") or {}).get("change_class"))
            x["stale"] = bool((x.get("flags") or {}).get("stale_pick"))
            rows.append(x)
    return rows


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
    seen: set[tuple[str, str]] = set()
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


def pioneer_tables(md: list[str]) -> None:
    rows = load("pioneer")
    if not rows:
        md.append("_pioneer protocol: no data yet_\n")
        return
    join = rows[0].get("join_at") or 20
    md.append(f"## A. Pioneer protocol: one agent learns alone for {join} episodes, five peers join\n")
    md.append("First exposure = an agent's first episode on a task class it has never handled. The pioneer's "
              "first exposures are the discovery cost; the followers' first exposures show what transferred.\n")
    lines = []
    ci_lines = []
    for a in ARMS:
        ar = [x for x in rows if x["arm"] == a]
        if not ar:
            continue
        pio, fol, fol_cells, peer, fol_all, tok_p, tok_f = [], [], defaultdict(list), [], [], [], []
        for cid, cr in cells(ar).items():
            authors = authorship(cr)
            pio += first_exposures(cr, {"1"}, 0)
            f = first_exposures(cr, {"2", "3", "4", "5", "6"}, join)
            fol += f
            fol_cells[cid] += [float(x["first_attempt_success"]) for x in f]
            for x in cr:
                if x["episode"] >= join:
                    fol_all.append(x)
                    ps = peer_share(x, authors)
                    if ps is not None:
                        peer.append(ps)
            tok_p += [x["tokens"] for x in cr if x["episode"] < join]
            tok_f += [x["tokens"] for x in cr if x["episode"] >= join]
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

    # followers' learning curve after joining, by 5-episode block, all episodes (not only first exposures)
    md.append("### A2. Followers after joining: first-attempt success by block (all their episodes)\n")
    lines = []
    for a in ARMS:
        ar = [x for x in rows if x["arm"] == a and x["episode"] >= join and x["agent"] != "1"]
        if not ar:
            continue
        n_blocks = (max(x["episode"] for x in ar) - join) // 10 + 1
        lines.append([LABEL.get(a, a), *[mean(x["first_attempt_success"] for x in ar if (x["episode"] - join) // 10 == k)
                                          for k in range(n_blocks)]])
    nb = (len(lines[0]) - 1) if lines else 0
    md.append(table(["arm", *[f"ep {join + 10 * k}-{join + 10 * k + 9}" for k in range(nb)]], lines) + "\n")

    # quirk classes only: where the textbook answer is wrong and only the store can help
    md.append("### A3. Follower first exposures on Acme-quirk classes only (textbook answer is wrong)\n")
    lines = []
    for a in ARMS:
        ar = [x for x in rows if x["arm"] == a]
        if not ar:
            continue
        fol = [x for cr in cells(ar).values() for x in first_exposures(cr, {"2", "3", "4", "5", "6"}, join)
               if (x.get("tags") or {}).get("quirk") or x["scenario"] == "order-ops"]
        pio = [x for cr in cells(ar).values() for x in first_exposures(cr, {"1"}, 0)
               if (x.get("tags") or {}).get("quirk") or x["scenario"] == "order-ops"]
        lines.append([LABEL.get(a, a), len(pio), mean(x["first_attempt_success"] for x in pio), len(fol),
                      mean(x["first_attempt_success"] for x in fol), mean(x["success"] for x in fol),
                      mean(x["escalated"] for x in fol), mean(x["attempts"] for x in fol)])
    md.append(table(["arm", "pioneer n", "pioneer 1st", "follower n", "follower 1st", "follower success",
                     "follower escalated", "follower attempts"], lines) + "\n")


def newcomer_tables(md: list[str]) -> None:
    rows = load("newcomer")
    if not rows:
        md.append("_newcomer protocol: no data yet_\n")
        return
    join = rows[0].get("join_at") or 40
    change = rows[0].get("change_at") or 20
    md.append(f"## B. Newcomer protocol: five agents live through a regime change at {change}; a sixth arrives at {join}\n")
    md.append("The newcomer has never seen these tasks. On changed classes the store holds both the old fix "
              "(which its peers learned before the change and burned on after it) and whatever replaced it.\n")
    lines = []
    for a in ARMS:
        ar = [x for x in rows if x["arm"] == a]
        if not ar:
            continue
        new = [x for x in ar if x["episode"] >= join]
        vets_post = [x for x in ar if change <= x["episode"] < join]
        new_ch = [x for x in new if x["changed"]]
        vets_ch_late = [x for x in vets_post if x["changed"] and x["episode"] >= join - 10]
        fe = [x for cr in cells(ar).values() for x in first_exposures(cr, {"6"}, join) if x["changed"]]
        fe_cells = defaultdict(list)
        for x in fe:
            fe_cells[x["cell"]].append(float(x["first_attempt_success"]))
        lo, hi = boot_ci(fe_cells)
        peer = []
        for cr in cells(ar).values():
            authors = authorship(cr)
            for x in cr:
                if x["episode"] >= join:
                    ps = peer_share(x, authors)
                    if ps is not None:
                        peer.append(ps)
        lines.append([LABEL.get(a, a), mean(x["first_attempt_success"] for x in vets_ch_late), mean(x["stale"] for x in vets_ch_late),
                      len(fe), f"{mean(x['first_attempt_success'] for x in fe)} [{lo:.2f},{hi:.2f}]",
                      mean(x["stale"] for x in fe), mean(x["escalated"] for x in fe),
                      mean(x["first_attempt_success"] for x in new_ch), mean(x["stale"] for x in new_ch),
                      mean(peer) if peer else "-"])
    md.append(table(["arm", f"veterans ep{join - 10}-{join - 1} 1st", "veterans stale", "newcomer 1st-exposures (changed)",
                     "1st-attempt [95% CI]", "stale-pick", "escalated", f"newcomer all changed ep{join}+ 1st", "stale",
                     "peer-authored share"], lines) + "\n")
    md.append("### B2. Newcomer on changed classes by 5-episode block after arrival\n")
    lines = []
    for a in ARMS:
        ar = [x for x in rows if x["arm"] == a and x["episode"] >= join and x["changed"]]
        if not ar:
            continue
        lines.append([LABEL.get(a, a), *[mean(x["first_attempt_success"] for x in ar if (x["episode"] - join) // 5 == k) for k in range(4)],
                      *[mean(x["stale"] for x in ar if (x["episode"] - join) // 5 == k) for k in range(4)]])
    md.append(table(["arm", "1st 40-44", "45-49", "50-54", "55-59", "stale 40-44", "45-49", "50-54", "55-59"], lines) + "\n")


def main() -> None:
    if "--runs" in sys.argv:
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
