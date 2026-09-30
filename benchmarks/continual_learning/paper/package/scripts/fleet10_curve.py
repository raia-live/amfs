"""Supplementary read for the confirmatory fleet-10 cells: first-attempt success by 10-episode
block, pooled over the cells of several runs (one per seed), with a cell-level bootstrap
interval, for the moved half, the kept half and the unchanged classes. Descriptive, per
amendment (x). Reads the frozen harness records through the frozen reader (`analysis.rsi`):
no metric, population rule or test is redefined here.

usage: python /tmp/fleet10_curve.py <run> [<run> ...]   (e.g. rsi-conf-e2b-r2-s37 ...)
"""
from __future__ import annotations

import random
import sys

from benchmarks.continual_learning.analysis import rsi

ARM = "senselab-rsi"
BLOCK = 10
# the confirmatory seeds and the E2b cell length, as `analysis.rsi --seeds` would set them
rsi.SEEDS = (37, 41, 59, 73)
rsi.CELL_EPISODES = 200


def block_rates(runs: list[str], pop: str) -> list[tuple[int, int, int, float, float, float]]:
    per_cell: dict[tuple, dict[int, list[int]]] = {}
    for run in runs:
        finished = rsi.cells_of(run, ARM)
        for e in rsi.episodes(run):
            if e.get("arm") != ARM or e.get("error"):
                continue
            k = (run, e.get("scenario"), int(e.get("seed", -1)), e.get("model"))
            if (e.get("scenario"), int(e.get("seed", -1)), e.get("model")) not in finished:
                continue
            t = e.get("tags") or {}
            if pop == "unchanged" and t.get("change_class"):
                continue
            if pop in ("moved", "kept") and rsi.population(e) != pop:
                continue
            b = int(e["episode"]) // BLOCK * BLOCK
            per_cell.setdefault(k, {}).setdefault(b, []).append(1 if e.get("first_attempt_success") else 0)
    cells = list(per_cell.values())
    rng = random.Random(11)
    out = []
    for b in sorted({b for c in cells for b in c}):
        have = [c for c in cells if c.get(b)]
        ok = sum(sum(c[b]) for c in have)
        n = sum(len(c[b]) for c in have)
        if not n:
            continue
        vals = []
        for _ in range(2000):
            pick = [have[rng.randrange(len(have))] for _ in have]
            nn = sum(len(c[b]) for c in pick)
            vals.append(sum(sum(c[b]) for c in pick) / nn if nn else 0.0)
        vals.sort()
        out.append((b, ok, n, ok / n, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]))
    return out, len(cells)


def main() -> None:
    runs = sys.argv[1:]
    for pop in ("moved", "kept", "unchanged"):
        rows, ncells = block_rates(runs, pop)
        print(f"\n== {pop} ({ncells} cells; runs {' '.join(runs)})")
        print("block   ok/n     rate   [95% cell bootstrap]")
        for b, ok, n, r, lo, hi in rows:
            print(f"{b:>3}-{b + BLOCK - 1:<3} {ok:>4}/{n:<4} {r:6.2f}  [{lo:.2f}, {hi:.2f}]")
        # the windows the preregistration names: post-change 50-99 (E2b change at 50), 100-149, 150-199
        for lo_, hi_ in ((50, 99), (100, 149), (150, 199)):
            ok = sum(ok for b, ok, n, *_ in rows if lo_ <= b <= hi_)
            n = sum(n for b, ok, n, *_ in rows if lo_ <= b <= hi_)
            if n:
                print(f"   episodes {lo_}-{hi_}: {ok}/{n} = {ok / n:.2f}")


if __name__ == "__main__":
    main()
