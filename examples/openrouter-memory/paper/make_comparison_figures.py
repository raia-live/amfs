"""Extra figures for the performance comparison paper: query latency (p50/p95)
and ingestion latency (time-to-searchable). Same restrained house style as
make_figures.py. Reads pro_results.json and ingestion.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

results = json.loads((ROOT / "pro_results.json").read_text())
ingest = json.loads((ROOT / "ingestion.json").read_text())
precision = json.loads((ROOT / "precision_results.json").read_text())

ACCENT = "#1b3a6b"
NEUTRAL = "#9aa3ad"
CEIL = "#c8ccd1"
P95C = "#b0453b"
INK = "#222222"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

LABELS = {
    "no-memory": "No memory", "in-context": "In-context",
    "vector": "Vector store", "mem0": "mem0",
    "supermemory": "Supermemory", "amfs-pro": "SenseLab",
    "senselab": "SenseLab", "senselab-default": "SenseLab\n(default fusion)",
    "senselab-semantic": "SenseLab\n(semantic)",
    "senselab-xenc": "SenseLab\n(+rerank)",
    "senselab-llm": "SenseLab\n(+LLM rerank)",
}


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def fig_latency():
    order = ["no-memory", "in-context", "vector", "mem0", "supermemory", "amfs-pro"]
    ov = results["overhead"]
    p50 = [ov[k]["latency_p50_ms"] for k in order]
    p95 = [ov[k]["latency_p95_ms"] for k in order]
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.bar(x - w / 2, p50, w, color=ACCENT, label="p50", zorder=3)
    ax.bar(x + w / 2, p95, w, color=P95C, label="p95", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in order], fontsize=8.5)
    ax.set_ylabel("Reader latency (ms)")
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig_latency")


def fig_ingestion():
    order = ["amfs-pro", "vector", "mem0", "supermemory"]
    order = sorted(order, key=lambda k: ingest[k]["time_to_searchable_s_p50"])
    vals = [max(ingest[k]["time_to_searchable_s_p50"], 0.01) for k in order]
    colors = [ACCENT if k == "amfs-pro" else NEUTRAL for k in order]

    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    y = np.arange(len(order))
    ax.barh(y, vals, color=colors, height=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 120)
    for yi, k, v in zip(y, order, vals):
        timed_out = ingest[k]["time_to_searchable_s_max"] >= 40.0 and k == "supermemory"
        label = f"{v:.2f} s" if v < 1 else (f">{v:.0f} s (timeout)" if timed_out else f"{v:.1f} s")
        ax.text(v * 1.35, yi, label, va="center", ha="left", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[k] for k in order])
    ax.set_xlabel("Time-to-searchable after write (seconds, log scale)")
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig_ingestion")


def fig_precision():
    """Hit@1 under contention: direct vs adversarial (paraphrased) questions."""
    order = ["vector", "senselab-semantic", "senselab-xenc", "senselab-llm",
             "mem0", "supermemory"]
    order = [k for k in order if k in precision]
    direct = [precision[k]["direct"]["hit@1"] for k in order]
    adv = [precision[k]["adversarial"]["hit@1"] for k in order]
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    b1 = ax.bar(x - w / 2, direct, w, color=ACCENT, label="Direct question", zorder=3)
    b2 = ax.bar(x + w / 2, adv, w, color=NEUTRAL, label="Adversarial paraphrase", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                    f"{r.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in order], fontsize=8.5)
    ax.set_ylabel("Hit@1 (%)  — correct fact ranked first")
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.14))
    save(fig, "fig_precision")


if __name__ == "__main__":
    fig_latency()
    fig_ingestion()
    fig_precision()
    print("figures ->", FIGDIR)
