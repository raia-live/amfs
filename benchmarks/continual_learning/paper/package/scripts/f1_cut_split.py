"""Fleet-1 confirmatory: episodes before/after the post-canary-zero cut (03:40:19Z) per cell,
and the headline with post-cut episodes excluded vs the full window. Temp script; reads only
the episode files through the frozen analysis helpers. Prints counts and rates only."""
import collections
import sys

sys.path.insert(0, ".")
from benchmarks.continual_learning.analysis import rsi  # noqa: E402

CUT = 1790739619  # 2026-09-30T03:40:19Z
RUNS = [f"rsi-conf-f1-s{s}" for s in (37, 41, 59, 73)]
LABEL = {
    "none": "Frontier alone", "raw-traces": "Context window", "pgvector": "pgvector",
    "pgvector+rerank": "pgvector + rerank", "pgvector-diy+outcomes": "pgvector + outcomes (DIY)",
    "senselab": "SenseLab - repair loop", "senselab-rsi": "SenseLab", "senselab-rsi-nocanary": "SenseLab - canary",
}

eps = []
for r in RUNS:
    eps.extend(rsi.episodes(r))
eps = [e for e in eps if not e.get("error")]
print(f"episodes: {len(eps)}; before cut {sum(1 for e in eps if e['ts'] < CUT)}, after {sum(1 for e in eps if e['ts'] >= CUT)}")

# per cell: before / after, and the post-change episodes (>= 30) split the same way
print("\n== per cell (family|arm|model|seed): episodes before/after the cut; post-change episodes before/after")
cells = collections.defaultdict(list)
for e in eps:
    cells[(e["scenario"], e["arm"], e["model"], int(e["seed"]))].append(e)
for k in sorted(cells):
    rows = cells[k]
    b = sum(1 for e in rows if e["ts"] < CUT); a = len(rows) - b
    pc = [e for e in rows if int(e["episode"]) >= 30]
    pb = sum(1 for e in pc if e["ts"] < CUT); pa = len(pc) - pb
    print(f"{k[0]:8s} {k[1]:22s} {k[2]:14s} s{k[3]}  {b:3d}/{a:3d}   post-change {pb:2d}/{pa:2d}")


def fas(rows, pop):
    rows = [e for e in rows if rsi.population(e) == pop and int(e["episode"]) >= 30]
    ok = sum(1 for e in rows if e.get("first_attempt_success"))
    return ok, len(rows)


def fmt(ok, n):
    return f"{100 * ok / n:3.0f}% ({ok}/{n})" if n else "   —"


print("\n== headline by system, post-change window: full  |  pre-cut episodes only")
for arm in ["none", "raw-traces", "pgvector", "pgvector+rerank", "pgvector-diy+outcomes", "senselab", "senselab-rsi", "senselab-rsi-nocanary"]:
    rows = [e for e in eps if e["arm"] == arm]
    pre = [e for e in rows if e["ts"] < CUT]
    parts = []
    for pop in ("moved", "kept", "unchanged"):
        parts.append(f"{pop} {fmt(*fas(rows, pop))} | {fmt(*fas(pre, pop))}")
    print(f"{LABEL[arm]:26s} " + "   ".join(parts))

print("\n== P12 direction: SenseLab minus baseline on the moved half (pp), full | pre-cut")
def rate(rows, pop):
    ok, n = fas(rows, pop)
    return 100 * ok / n if n else float("nan")
sl = [e for e in eps if e["arm"] == "senselab-rsi"]
for arm in ["none", "raw-traces", "pgvector", "pgvector+rerank", "pgvector-diy+outcomes", "senselab", "senselab-rsi-nocanary"]:
    b = [e for e in eps if e["arm"] == arm]
    full = rate(sl, "moved") - rate(b, "moved")
    pre = rate([e for e in sl if e["ts"] < CUT], "moved") - rate([e for e in b if e["ts"] < CUT], "moved")
    same = "same sign" if (full >= 0) == (pre >= 0) else "DIFFERENT SIGN"
    print(f"vs {LABEL[arm]:26s} {full:+6.1f} | {pre:+6.1f}   {same}")
