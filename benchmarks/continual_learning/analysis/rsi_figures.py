"""Figures for the RSI paper, drawn from the same runs and rules as ``analysis/rsi.py``.

  fig_headline.png   first-attempt success after the conditional change — moved half, kept half,
                     unchanged classes — one bar per system, cell-level bootstrap 95 % intervals
  fig_curves.png     first-attempt success by 10-episode block across the change (episode 30):
                     moved half and unchanged classes, one line per system
  fig_cost.png       recovery on the moved half against all-in USD per episode (the loop's spend
                     included), one point per system
  fig_cuts.png       SenseLab by engine cut: moved / kept half and the fix lifecycle per run

    python -m benchmarks.continual_learning.analysis.rsi_figures [--senselab-run rsi-c5]
                                                                  [--seeds ...] [--models ...]

Writes ``paper/figures/``. Systems are named as in the paper (amendment (o)); the ablations
appear in no figure. Nothing here is a confirmatory statistic on the development seeds.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import rsi  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
POPS = (("moved", "moved half"), ("kept", "kept half"), ("unchanged", "unchanged classes"))
#: system -> colour; SenseLab is always red, the frontier model grey.
COLORS = {"Frontier alone": "#8c8c8c", "Context window": "#7f4f9f", "pgvector": "#1f77b4",
          "pgvector + rerank": "#17becf", "pgvector + outcomes (DIY)": "#8c564b", "SenseLab": "#d62728"}
BLOCK = 10


def color(name: str) -> str:
    return COLORS.get(name, "#333333")


def block_rates(run: str, arm: str, pop: str) -> list[tuple[int, float, float, float]]:
    """(block start, rate, lo, hi) per 10-episode block, pooled over the finished cells, with a
    cell-level bootstrap interval; *pop* is a population of :func:`rsi.population` or ``all``."""
    per_cell: dict[rsi.CellKey, dict[int, list[int]]] = {}
    finished = rsi.cells_of(run, arm)
    for e in rsi.episodes(run):
        if e.get("arm") != arm or e.get("error"):
            continue
        k = (e.get("scenario"), int(e.get("seed", -1)), e.get("model"))
        if k not in finished:
            continue
        t = e.get("tags") or {}
        if pop == "changed" and not t.get("change_class"):
            continue  # the classes whose rule changes at the change, before and after it
        if pop == "unchanged" and t.get("change_class"):
            continue
        if pop in ("moved", "kept") and rsi.population(e) != pop:
            continue
        b = int(e["episode"]) // BLOCK * BLOCK
        per_cell.setdefault(k, {}).setdefault(b, []).append(1 if e.get("first_attempt_success") else 0)
    out = []
    cells = list(per_cell.values())
    if not cells:
        return out
    rng = random.Random(11)
    for b in sorted({b for c in cells for b in c}):
        have = [c for c in cells if c.get(b)]
        ok = sum(sum(c[b]) for c in have)
        n = sum(len(c[b]) for c in have)
        if not n:
            continue
        vals = []
        for _ in range(500):
            pick = [have[rng.randrange(len(have))] for _ in have]
            nn = sum(len(c[b]) for c in pick)
            vals.append(sum(sum(c[b]) for c in pick) / nn if nn else 0.0)
        vals.sort()
        out.append((b, ok / n, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]))
    return out


def fig_headline(systems: dict[str, tuple[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    names = [n for n in systems if rsi.cells_of(*systems[n])]
    for ax, (pop, title) in zip(axes, POPS, strict=True):
        for i, name in enumerate(names):
            cells = list(rsi.cells_of(*systems[name]).values())
            ok, n = rsi.pooled(cells, pop)
            rate = ok / n if n else 0.0
            ci = rsi.boot_ci(cells, pop) or (rate, rate)
            ax.bar(i, 100 * rate, color=color(name), width=0.72)
            ax.errorbar(i, 100 * rate, yerr=[[100 * (rate - ci[0])], [100 * (ci[1] - rate)]],
                        fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
            ax.text(i, 100 * rate + 2 + 100 * (ci[1] - rate), f"{100 * rate:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" (DIY)", "\n(DIY)").replace("Frontier alone", "Frontier\nalone")
                            .replace("Context window", "Context\nwindow") for n in names], fontsize=7)
        ax.set_ylim(0, 112)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("first-attempt success, episodes 30–99 (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_headline.png", dpi=200)
    plt.close(fig)


def fig_curves(systems: dict[str, tuple[str, str]], out: Path, change_at: int = 30) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (pop, title) in zip(axes, (("changed", "changed classes (moved and kept halves)"),
                                       ("unchanged", "unchanged classes")), strict=True):
        for name, (run, arm) in systems.items():
            pts = block_rates(run, arm, pop)
            if not pts:
                continue
            xs = [b + BLOCK / 2 for b, *_ in pts]
            lw = 2.0 if name == "SenseLab" else 1.2
            ax.plot(xs, [100 * r for _, r, _, _ in pts], marker="o", ms=3, lw=lw, color=color(name), label=name,
                    zorder=3 if name == "SenseLab" else 2)
            if name == "SenseLab":
                ax.fill_between(xs, [100 * lo for *_, lo, _ in pts], [100 * hi for *_, hi in pts],
                                color=color(name), alpha=0.12, lw=0)
        ax.axvline(change_at, color="black", lw=0.8, ls="--")
        ax.text(change_at + 1, 3, "rule changes\nconditionally", fontsize=7, va="bottom")
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("episode", fontsize=8)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("first-attempt success (%)", fontsize=8)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png", dpi=200)
    plt.close(fig)


def fig_cost(systems: dict[str, tuple[str, str]], out: Path) -> None:
    _, data = rsi.cost_rows(systems)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, d in data.items():
        cells = list(rsi.cells_of(*systems[name]).values())
        ok, n = rsi.pooled(cells, "moved")
        if not n:
            continue
        rate = 100 * ok / n
        ci = rsi.boot_ci(cells, "moved")
        x = d["usd_per_episode"]
        ax.scatter([x], [rate], s=46, color=color(name), zorder=3)
        if ci:
            ax.plot([x, x], [100 * ci[0], 100 * ci[1]], color=color(name), lw=0.9, zorder=2)
        ax.annotate(name, (x, rate), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("USD per episode, all in (agent + loop)", fontsize=8)
    ax.set_ylabel("moved-half first-attempt success, ep. 30–99 (%)", fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 80)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_cost.png", dpi=200)
    plt.close(fig)


def fig_cuts(out: Path) -> None:
    rows = []
    for run, cut, _ in rsi.CUTS:
        cells = list(rsi.cells_of(run, rsi.SENSELAB_ARM).values())
        if not cells:
            continue
        lc = rsi.lifecycle(rsi.fix_rows(run, rsi.SENSELAB_ARM))
        rows.append((cut, run, cells, lc))
    if not rows:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    xs = range(len(rows))
    for pop, mk, lab in (("moved", "o", "moved half"), ("kept", "s", "kept half"), ("unchanged", "^", "unchanged")):
        ys, lo, hi = [], [], []
        for _, _, cells, _ in rows:
            ok, n = rsi.pooled(cells, pop)
            ci = rsi.boot_ci(cells, pop) or (ok / n, ok / n)
            ys.append(100 * ok / n)
            lo.append(100 * (ok / n - ci[0]))
            hi.append(100 * (ci[1] - ok / n))
        ax.errorbar(list(xs), ys, yerr=[lo, hi], marker=mk, ms=4, lw=1.2, capsize=2, label=lab)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{c}\n({r})" for c, r, _, _ in rows], fontsize=7)
    ax.set_ylabel("first-attempt success, ep. 30–99 (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("SenseLab outcomes by engine cut", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    keys = (("proposed", "proposed"), ("tier1:passed", "Tier 1 passed"), ("tier2:passed", "Tier 2 passed"),
            ("canary_started", "canary started"), ("shipped", "shipped"))
    w = 0.16
    for j, (k, lab) in enumerate(keys):
        ax2.bar([x + (j - 2) * w for x in xs], [lc.get(k, 0) for *_, lc in rows], width=w, label=lab)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([c for c, *_ in rows], fontsize=7)
    ax2.set_ylabel("fixes, four cells", fontsize=8)
    ax2.legend(fontsize=7, frameon=False)
    ax2.set_title("the loop's own lifecycle by engine cut", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_cuts.png", dpi=200)
    plt.close(fig)


def fig_canary(run: str, out: Path, horizon: int = 200) -> bool:
    """Every canary of a fleet run: exposure per arm against the cell's episode index, from the
    sealed traces' stamps (``analysis/rsi_canary_readout.py`` -> ``canary_readout.json``), with
    the graded floor and the verdict. Returns False when the readout is missing."""
    path = rsi.RESULTS / run / "canary_readout.json"
    if not path.exists():
        return False
    rows = [r for r in json.loads(path.read_text())["rows"] if r["p14"]["stamped_canary"] + r["p14"]["stamped_control"]]
    if not rows:
        return False
    rows.sort(key=lambda r: (r["cell"], r["started_at"] or ""))
    ncol = 4
    nrow = (len(rows) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.3 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, r in zip(axes.flat, rows, strict=False):
        for arm, col in (("canary", "#d62728"), ("control", "#8c8c8c")):
            eps = r["p14"]["episodes"][arm]
            if eps:
                ax.step(eps, range(1, len(eps) + 1), where="post", color=col, lw=1.3,
                        label=f"{arm} ({r[f'{arm}_graded']} graded)")
        ax.axhline(r["min_outcomes"], color="k", lw=0.6, ls=":")
        fam, arm_name, _, seed, _ = r["cell"].split("|")
        verdict = r["status"] + (f" ({r['reason']})" if r.get("reason") else "")
        ax.set_title(f"{fam} {arm_name.replace('senselab-rsi', 'SL')} {seed} · from {r['started_from'].split('_')[1]}"
                     f" · {verdict}", fontsize=6.5)
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, horizon)
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("cell episode", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("exposed runs", fontsize=8)
    fig.suptitle(f"Canaries of {run}: runs stamped to each arm by the router, from the sealed traces; "
                 "dotted line = graded floor per arm", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_canary.png", dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--senselab-run", default=None)
    ap.add_argument("--fleet-run", default=None, help="fleet-10 run with a canary_readout.json (fig_canary)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(rsi.SEEDS))
    ap.add_argument("--models", nargs="+", default=list(rsi.MODELS))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rsi.SEEDS, rsi.MODELS = tuple(args.seeds), tuple(args.models)
    sl_run = args.senselab_run or rsi.newest_complete_senselab_run()
    systems = {**rsi.BASELINES, "SenseLab": (sl_run, rsi.SENSELAB_ARM)}
    args.out.mkdir(parents=True, exist_ok=True)
    fig_headline(systems, args.out)
    fig_curves(systems, args.out)
    fig_cost(systems, args.out)
    fig_cuts(args.out)
    canary_drawn = bool(args.fleet_run) and fig_canary(args.fleet_run, args.out)
    (args.out / "provenance.json").write_text(json.dumps(
        {"senselab_run": sl_run, "fleet_run": args.fleet_run if canary_drawn else None,
         "seeds": list(rsi.SEEDS), "models": list(rsi.MODELS),
         "systems": {k: list(v) for k, v in systems.items()}}, indent=1) + "\n")
    print(f"wrote {sorted(p.name for p in args.out.glob('fig_*.png'))} to {args.out}")


if __name__ == "__main__":
    main()
