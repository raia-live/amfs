"""Generate publication-quality figures for the combined paper from the raw
result JSON (pro_results.json, gate_sweep.json). Vector PDF for LaTeX + PNG for preview.

Deliberately restrained styling: serif type to match the LaTeX body, a single
accent colour for AMFS against neutral greys, thin spines, error bars with caps,
and a light value-axis grid. No default matplotlib palette, no 3D, no gradients.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

results = json.loads((ROOT / "pro_results.json").read_text())
sweep = json.loads((ROOT / "gate_sweep.json").read_text())

# ---- house style -----------------------------------------------------------
ACCENT = "#1b3a6b"      # AMFS
NEUTRAL = "#9aa3ad"     # other systems
CEIL = "#c8ccd1"        # controls (in-context / no-memory)
LINE_A = "#1b3a6b"
LINE_B = "#b0453b"
INK = "#222222"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

LABELS = {
    "no-memory": "No memory",
    "in-context": "In-context (ceiling)",
    "vector": "Vector store",
    "mem0": "mem0",
    "supermemory": "Supermemory",
    "amfs-pro": "SenseLab",
}


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ---- Figure 1: recall with Wilson 95% CI -----------------------------------
def fig_recall():
    order = ["no-memory", "supermemory", "in-context", "mem0", "vector", "amfs-pro"]
    rec = results["recall"]
    order = sorted(order, key=lambda k: rec[k]["pct"])
    vals = [rec[k]["pct"] for k in order]
    los = [rec[k]["pct"] - rec[k]["ci"][0] for k in order]
    his = [rec[k]["ci"][1] - rec[k]["pct"] for k in order]
    colors = []
    for k in order:
        if k == "amfs-pro":
            colors.append(ACCENT)
        elif k in ("no-memory", "in-context"):
            colors.append(CEIL)
        else:
            colors.append(NEUTRAL)

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = range(len(order))
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    ax.errorbar(vals, y, xerr=[los, his], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=3, zorder=4)
    for yi, v, h in zip(y, vals, his):
        ax.text(v + h + 1.8, yi, f"{v:.1f}%", va="center", ha="left", fontsize=8.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([LABELS[k] for k in order])
    ax.set_xlim(0, 108)
    ax.set_xlabel("Cross-model recall (%)")
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig_recall")


# ---- Figure 2: gate tradeoff -----------------------------------------------
def fig_gate():
    rows = sweep["sweep"]
    x = [r["threshold"] for r in rows]
    recall = [r["recall_gate_pass_pct"] for r in rows]
    abstain = [r["miss_abstain_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(x, recall, "-o", color=LINE_A, lw=1.6, ms=4.5, label="Recall gate-pass")
    ax.plot(x, abstain, "--s", color=LINE_B, lw=1.6, ms=4.2, label="Abstain-on-miss")
    ax.axvline(0.55, color="#888", lw=0.9, ls=":", zorder=0)
    ax.text(0.552, 6, "operating\npoint (0.55)", fontsize=7.5, color="#555", va="bottom")
    ax.set_xlabel("Semantic-relevance floor (SEM_FLOOR)")
    ax.set_ylabel("Percent")
    ax.set_ylim(-4, 108)
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    save(fig, "fig_gate")


# ---- Figure 3: abstain-on-miss ---------------------------------------------
def fig_abstain():
    ab = results["abstain_on_miss"]
    order = ["no-memory", "supermemory", "amfs-pro", "vector", "mem0"]
    order = sorted(order, key=lambda k: ab[k]["decline_pct"])
    vals = [ab[k]["decline_pct"] for k in order]
    los = [ab[k]["decline_pct"] - ab[k]["ci"][0] for k in order]
    his = [ab[k]["ci"][1] - ab[k]["decline_pct"] for k in order]
    colors = [ACCENT if k == "amfs-pro" else (CEIL if k == "no-memory" else NEUTRAL) for k in order]

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    xpos = range(len(order))
    ax.bar(xpos, vals, color=colors, width=0.6, zorder=3)
    ax.errorbar(xpos, vals, yerr=[los, his], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=3, zorder=4)
    for xi, v, h in zip(xpos, vals, his):
        ax.text(xi, v + h + 1.6, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels([LABELS[k] for k in order], fontsize=8.5)
    ax.set_ylim(0, 75)
    ax.set_ylabel("Declines on a miss (%)")
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig_abstain")


if __name__ == "__main__":
    fig_recall()
    fig_gate()
    fig_abstain()
    print("figures ->", FIGDIR)
