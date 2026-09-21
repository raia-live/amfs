"""Paper figures for grid v3 (beyond the two regime_change.py draws).

  fig3_multi_agent.png   first-attempt success where knowledge is shared: fleet-6 post-change,
                         newcomer first exposures on changed classes, pioneer followers' first exposures
  fig4_paired_deltas.png SenseLab CL minus each arm, paired on (scenario, seed), episodes 40-59
  fig5_long_horizon.png  120 episodes without a regime change, first-attempt success by 20-episode block
  fig6_recovery.png      post-change recovery on changed classes by 10-episode block (cleaner than fig1)

    python -m benchmarks.continual_learning.analysis.figures --runs gridv3
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import probes as pb  # noqa: E402
from . import regime_change as rc  # noqa: E402
from . import transfer as tr  # noqa: E402

COLORS = {"none": "#999999", "pgvector": "#1f77b4", "pgvector-diy": "#8c564b", "pgvector-diy+outcomes": "#17becf",
          "mem0": "#ff7f0e", "senselab-nofeedback": "#bcbd22", "senselab-nopriors": "#e377c2", "senselab-lean": "#9467bd",
          "senselab": "#d62728"}
SHORT = {"none": "no memory", "pgvector": "pgvector RAG", "pgvector-diy": "pgvector DIY", "pgvector-diy+outcomes": "pgvector DIY\n+ outcome counter",
         "mem0": "Mem0", "senselab-nofeedback": "SenseLab\nno outcomes", "senselab-nopriors": "SenseLab\nno priors",
         "senselab-lean": "SenseLab CL\nlean briefing", "senselab": "SenseLab CL"}


def paired(rows, ref: str, other: str, *, episodes=None, fleet=1):
    """SenseLab-minus-arm delta of post-change changed-class first-attempt success, paired on
    (scenario, seed): mean, cell-bootstrap lo, hi. Mirrors regime_change.py section 1c."""
    from collections import defaultdict

    def rate(a):
        r = rc.sel(rows, arm=a, fleet=fleet, pre=False, changed=True)
        if episodes:
            r = [x for x in r if episodes[0] <= x["episode"] < episodes[1]]
        d = defaultdict(list)
        for x in r:
            d[(x["scenario"], x["seed"])].append(x["first_attempt_success"])
        return {k: sum(v) / len(v) for k, v in d.items()}

    ra, rb = rate(ref), rate(other)
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return None
    deltas = [ra[k] - rb[k] for k in keys]
    lo, hi = rc.boot_ci({str(k): [d] for k, d in zip(keys, deltas)})
    return {"mean": sum(deltas) / len(deltas), "lo": lo, "hi": hi}


def _rate(rows) -> float:
    return sum(x["first_attempt_success"] for x in rows) / len(rows) if rows else float("nan")


def _ci(rows):
    lo, hi = rc.boot_ci(rc.by_cell(rows, lambda x: x["first_attempt_success"])) if rows else (float("nan"),) * 2
    return lo, hi


def main() -> None:
    prefix = sys.argv[sys.argv.index("--runs") + 1] if "--runs" in sys.argv else "gridv3"
    rc.use_runs(prefix)
    tr.use_runs(prefix)
    out = rc.OUT
    out.mkdir(parents=True, exist_ok=True)
    rows = rc.load()
    arms = [a for a in ("none", "pgvector", "pgvector-diy+outcomes", "mem0", "senselab") if a in rc.HEAD]

    # ---- fig3: multi-agent -------------------------------------------------------------
    pioneer = tr.load("pioneer")
    newcomer = tr.load("newcomer")
    panels = []
    fleet = {a: rc.sel(rows, arm=a, fleet=6, pre=False, changed=True) for a in arms}
    panels.append(("Fleet of 6, one store:\nchanged classes after the change", fleet))
    nc = {}
    join = (newcomer[0].get("join_at") or 40) if newcomer else 40
    for a in arms:
        ar = [x for x in newcomer if x["arm"] == a]
        nc[a] = [x for x in ar if x["episode"] >= join and x["agent"] == "6" and x["changed"]]
    panels.append(("Newcomer arriving after the change:\nall its changed-class tasks", nc))
    pf = {}
    pjoin = (pioneer[0].get("join_at") or 20) if pioneer else 20
    for a in arms:
        ar = [x for x in pioneer if x["arm"] == a]
        pf[a] = [x for cr in tr.cells(ar).values() for x in tr.first_exposures(cr, {"2", "3", "4", "5", "6"}, pjoin)]
    panels.append(("Pioneer + 5 followers:\nfollowers' first exposure to each class", pf))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (title, data) in zip(axes, panels):
        names, vals, err, cols = [], [], [], []
        for a in arms:
            r = data.get(a) or []
            if not r:
                continue
            m = _rate(r)
            lo, hi = _ci(r)
            names.append(SHORT[a]); vals.append(m); err.append([max(0, m - lo), max(0, hi - m)]); cols.append(COLORS[a])
        ax.bar(names, vals, color=cols, yerr=list(zip(*err)) if err else None, capsize=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelsize=7.5)
        for i, v in enumerate(vals):
            ax.text(i, v + err[i][1] + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("first-attempt success")
    fig.suptitle("Where knowledge is shared between agents (gpt-5.5; bars: cell bootstrap 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig3_multi_agent.png", dpi=150)

    # ---- fig4: paired deltas -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.8))
    comps = [a for a in ("none", "pgvector", "pgvector-diy", "pgvector-diy+outcomes", "mem0", "senselab-nofeedback",
                         "senselab-nopriors") if a in rc.HEAD]
    ys, labels = [], []
    for i, a in enumerate(comps):
        for j, (eps, col, mk) in enumerate((((20, 60), "#bbbbbb", "o"), ((40, 60), "#d62728", "s"))):
            d = paired(rows, "senselab", a, episodes=eps, fleet=1)
            if d is None:
                continue
            m, lo, hi = d["mean"], d["lo"], d["hi"]
            y = i + (0.18 if j else -0.18)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=mk, color=col, capsize=3,
                        label=("episodes 20-59" if j == 0 else "episodes 40-59") if i == 0 else None)
        ys.append(i); labels.append(rc.LABEL[a])
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("SenseLab CL minus arm: post-change changed-class first-attempt success (paired on scenario x seed)")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "fig4_paired_deltas.png", dpi=150)

    # ---- fig6: recovery by 10-episode block ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for a in [a for a in ("none", "pgvector", "pgvector-diy+outcomes", "mem0", "senselab-nofeedback", "senselab-nopriors", "senselab") if a in rc.HEAD]:
        r = rc.sel(rows, arm=a, fleet=1)
        pre = [_rate([x for x in r if x["episode"] // 10 == k]) for k in (0, 1)]
        post = [_rate([x for x in r if x["episode"] // 10 == k and x["changed"]]) for k in (2, 3, 4, 5)]
        ax.plot([5, 15], pre, color=COLORS[a], lw=1, ls=":", marker="o", ms=3)
        ax.plot([25, 35, 45, 55], post, color=COLORS[a], lw=2.4 if a == "senselab" else 1.4, marker="o", label=rc.LABEL[a])
    ax.axvline(20, color="k", ls="--", lw=1)
    ax.text(20.6, 0.93, "rules change for half the classes", fontsize=8)
    ax.set_xlabel("episode (dotted: all tasks before the change; solid: changed-class tasks after it)")
    ax.set_ylabel("first-attempt success")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax.set_title("Recovery after the world changes (5 scenarios x 3 seeds, single agent)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig6_recovery.png", dpi=150)

    # ---- fig5: long horizon --------------------------------------------------------------
    long = pb.load(prefix, "long")
    if long:
        scens = sorted({x["scenario"] for x in long})
        fig, axes = plt.subplots(1, len(scens), figsize=(6 * len(scens), 4), sharey=True, squeeze=False)
        for ax, scen in zip(axes[0], scens):
            for a in [a for a in ("none", "pgvector", "pgvector-diy+outcomes", "mem0", "senselab") if a in pb.arms(long)]:
                r = pb.sel(long, arm=a, scenario=scen)
                if not r:
                    continue
                ys = [_rate([x for x in r if x["episode"] // 20 == k]) for k in range(6)]
                ax.plot([10, 30, 50, 70, 90, 110], ys, marker="o", color=COLORS[a], lw=2.4 if a == "senselab" else 1.4, label=rc.LABEL[a])
            ax.set_title(f"{scen}: 120 episodes, no regime change, 300 distractors", fontsize=10)
            ax.set_xlabel("episode")
            ax.set_ylim(0, 1)
        axes[0][0].set_ylabel("first-attempt success")
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig5_long_horizon.png", dpi=150)
    plt.close("all")
    print("figures written to", out)


if __name__ == "__main__":
    main()
