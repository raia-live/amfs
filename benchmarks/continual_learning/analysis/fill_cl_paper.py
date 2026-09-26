"""Fill ``paper/paper.tex`` from ``results/paper-report`` (the ``--paper`` analyses).

Each ``%%TAB_*%%`` marker in the template names one table of one report file; the markdown
table under that heading becomes a booktabs table, optionally with columns dropped. The
template is kept as ``paper/paper.tex.in`` on the first run and re-filled from there, so the
analyses can be re-run and the paper rebuilt as runs land:

    python -m benchmarks.continual_learning.analysis.regime_change --paper
    python -m benchmarks.continual_learning.analysis.transfer --paper
    python -m benchmarks.continual_learning.analysis.probes --paper
    python -m benchmarks.continual_learning.analysis.figures --paper
    python -m benchmarks.continual_learning.analysis.fill_cl_paper
    (cd benchmarks/continual_learning/paper && pdflatex paper.tex && pdflatex paper.tex)

A marker whose table is not in the report yet is left as a visible red TODO.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .fill_paper import md_table_to_tex, tex_escape

HERE = Path(__file__).resolve().parent.parent
REPORT_DIR = HERE / "results" / "paper-report"
TEX = HERE / "paper" / "paper.tex"

#: marker -> (report file, heading prefix, caption, label, columns to drop)
TABLES: dict[str, tuple[str, str, str, str, tuple[str, ...]]] = {
    "TAB_HEADLINE": ("REPORT.md", "1. After the world changes",
                     "After the world changes: changed-class tasks, episodes 20--59, single agent. First-attempt "
                     "success carries a cell-level bootstrap 95\\% interval; \\emph{stale-pick} is the share of "
                     "first attempts that applied the pre-change fix.", "tab:headline",
                     ("success excl. infra", "infra_err", "wall s", "tok/ep")),
    "TAB_PAIRED": ("REPORT.md", "1c. Paired cell deltas",
                   "SenseLab (outcome loop on) minus each arm, paired on (scenario, seed, model, fleet): "
                   "post-change changed-class first-attempt success, episodes 20--59 and 40--59.", "tab:paired", ()),
    "TAB_BYSCEN": ("REPORT.md", "1d. Post-change changed-class first-attempt success by scenario",
                   "Post-change changed-class first-attempt success by scenario, single agent.", "tab:byscen",
                   ("policy-qa", "dispatch")),
    "TAB_ESC_BYSCEN": ("REPORT.md", "1e. Post-change changed-class escalation rate by scenario",
                       "Post-change changed-class escalation rate by scenario, single agent.", "tab:escbyscen",
                       ("policy-qa", "dispatch")),
    "TAB_RECOVERY": ("REPORT.md", "2. Recovery curve",
                     "Recovery after the change: first-attempt success on changed classes by 10-episode block; "
                     "cells recovered = three consecutive first-attempt successes on a changed class.", "tab:recovery", ()),
    "TAB_STALE": ("REPORT.md", "2b. Stale-pick rate",
                  "Stale-pick rate after the change: share of first attempts applying the pre-change fix, by block.",
                  "tab:stale", ()),
    "TAB_PRECHANGE": ("REPORT.md", "3. Before the change",
                      "Before the change: first-attempt success on all tasks by 5-episode block, episodes 0--19.",
                      "tab:prechange", ("tok/ep",)),
    "TAB_HORIZON": ("REPORT.md", "4b. Whole horizon",
                    "Whole horizon, every task (episodes 0--59, single agent): outcomes per 1{,}000 tasks. "
                    "Memory time is the time spent in memory calls; SenseLab timings are those of the production "
                    "API under the grid's full load.", "tab:horizon",
                    ("success excl. infra", "infra_err", "tokens / 1000 tasks", "$ / 1000 tasks")),
    "TAB_FLEET6": ("REPORT.md", "6. Fleet of 6",
                   "Six agents sharing one store: post-change changed-class tasks. Each agent sees ten of the "
                   "sixty episodes; \\emph{first post-change exposure} is an agent's first task on a class after "
                   "the change.", "tab:fleet6", ("infra_err",)),
    "TAB_FLEET10": ("REPORT.md", "6b. Fleet of 10",
                    "Ten agents sharing one store, 100 episodes with the change at 30: post-change changed-class "
                    "tasks.", "tab:fleet10", ("infra_err",)),
    "TAB_ACTIONS": ("REPORT.md", "9. Actions tried per changed class",
                    "What the fleets tried on each changed class after the change (all fleet sizes).",
                    "tab:actions", ()),
    "TAB_XFER_PIONEER": ("TRANSFER.md", "A. Pioneer protocol",
                         "Pioneer protocol: one agent alone for 20 episodes, five peers join at 20. First exposure = "
                         "an agent's first episode on a task class it has never handled.", "tab:pioneer",
                         ("tok/ep pioneer", "tok/ep followers")),
    "TAB_XFER_PIONEER_BLOCKS": ("TRANSFER.md", "A2. Followers after joining",
                                "Followers after joining: first-attempt success on all their episodes by 10-episode "
                                "block.", "tab:pioneerblocks", ()),
    # B1 exists only while some arm has not completed every scenario; once all arms have, the
    # B table is itself the matched comparison and is used instead.
    "TAB_XFER_NEWCOMER": ("TRANSFER.md", ("B1. Newcomer protocol on the scenarios", "B. Newcomer protocol"),
                          "Newcomer after a regime change: five agents live through the change at 20; a sixth "
                          "arrives at 40 with no history and takes every episode from then on. The first three "
                          "rows cover the scenarios every arm completed (support and ci-fix; same cells, seeds "
                          "and models); the rows marked \\emph{all} cover the four scenarios of the protocol for "
                          "the arms that ran all of them.", "tab:newcomer",
                          ("newcomer all changed ep40+ 1st",)),
    "TAB_XFER_NEWCOMER_BLOCKS": ("TRANSFER.md", "B2. Newcomer on changed classes by 5-episode block",
                                 "The newcomer's first-attempt success on changed classes by 5-episode block after "
                                 "arrival, all four scenarios (arms that ran all of them).", "tab:newcomerblocks", ()),
    "TAB_XFER_COLD": ("TRANSFER.md", "C. Cold start onto a healthy store",
                      "Cold start onto a healthy store (no regime change): five agents for 40 episodes, a sixth "
                      "arrives at 40.", "tab:cold", ()),
    "TAB_XFER_XMODEL": ("TRANSFER.md", "D. Cross-model onboarding",
                        "Cross-model onboarding onto a healthy store: five veterans on one model for 40 episodes, "
                        "a newcomer on the other model arrives at 40. Same-model rows of the same arm are the "
                        "reference.", "tab:xmodel",
                        ("tok/ep newcomer", "median tokens to parity", "store", "median episodes to parity")),
    "TAB_PROBE_RUNBOOK": ("PROOF_POINTS.md", "A. runbook",
                          "Probe: two near-tie runbooks, one causes an outage.", "tab:runbook", ()),
    "TAB_PROBE_UNKNOWNS": ("PROOF_POINTS.md", "B. unknowns",
                           "Probe: 35\\% of tasks have no runbook; the right move is to escalate.", "tab:unknowns", ()),
    "TAB_PROBE_HANDOFF": ("PROOF_POINTS.md", "C. handoff",
                          "Probe: a new agent inherits eight findings from a predecessor, one of them wrong.",
                          "tab:handoff", ()),
    "TAB_PROBE_DRIFT_FACT": ("PROOF_POINTS.md", "D. drift-fact",
                             "Probe: an operational fact changes at episode 8.", "tab:driftfact", ()),
    "TAB_PROBE_DRIFT_TOOL": ("PROOF_POINTS.md", "D. drift-tool",
                             "Probe: a tool's parameter shape changes at episode 8.", "tab:drifttool", ()),
    "TAB_LONG_SUPPORT": ("PROOF_POINTS.md", "E.support",
                         "Long horizon, support: 120 episodes, no regime change, 300 distractors.", "tab:longsupport", ()),
    "TAB_LONG_DIAGNOSE": ("PROOF_POINTS.md", "E.diagnose",
                          "Long horizon, diagnose: 120 episodes, no regime change, 300 distractors.", "tab:longdiagnose", ()),
}


def split_headings(md: str) -> dict[str, list[str]]:
    """Heading text (any ``#`` level) -> the lines under it up to the next heading."""
    out: dict[str, list[str]] = {}
    cur = "_preamble"
    for ln in md.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", ln)
        if m:
            cur = m.group(1).strip()
            out[cur] = []
        else:
            out.setdefault(cur, []).append(ln)
    return out


# Arm labels as they appear in reports written before the paper's rename, mapped to the label the
# paper uses now. Applied to every table so an older report on disk still fills correctly.
ARM_RENAMES = {"SenseLab, continual learning": "SenseLab, outcome loop on"}


def rename_arms(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        for old, new in ARM_RENAMES.items():
            ln = ln.replace(old, new)
        out.append(ln)
    return out


# Per-marker row predicates over {column header: cell text}; rows failing the predicate are dropped.
ROW_FILTERS = {
    # The cross-model protocol is the healthy-store one; the regime-change rows restate the newcomer
    # table, and a '-' newcomer model is a run with no newcomer episodes yet.
    "TAB_XFER_XMODEL": lambda r: r.get("store") == "healthy" and r.get("newcomer model", "-") != "-",
    # Mem0 has not run every scenario of the newcomer protocol; its block row would mix a different
    # scenario set from the other two. Remove this filter once the Mem0 remainder is on file.
    "TAB_XFER_NEWCOMER_BLOCKS": lambda r: "Mem0" not in r.get("arm", ""),
}


def filter_rows(lines: list[str], pred) -> list[str]:
    rows = [ln for ln in lines if ln.strip().startswith("|")]
    if not rows or pred is None:
        return lines
    head = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    out = rows[:2]
    for ln in rows[2:]:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if pred(dict(zip(head, cells))):
            out.append(ln)
    return out


def drop_columns(lines: list[str], drop: tuple[str, ...]) -> list[str]:
    if not drop:
        return lines
    rows = [ln for ln in lines if ln.strip().startswith("|")]
    if not rows:
        return lines
    head = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    keep = [i for i, h in enumerate(head) if h not in drop]
    out = []
    for ln in rows:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            out.append("|" + "|".join("---" for _ in keep) + "|")
        else:
            cells = (cells + [""] * len(head))[:len(head)]
            out.append("| " + " | ".join(cells[i] for i in keep) + " |")
    return out


def fit_width(tex_table: str, max_cols: int = 5) -> str:
    """Wrap wide tabulars in ``\\resizebox`` so they fit the text width; narrow ones are left alone."""
    m = re.search(r"\\begin\{tabular\}\{([lr]+)\}", tex_table)
    if not m or len(m.group(1)) <= max_cols:
        return tex_table
    start = tex_table.index(r"\begin{tabular}")
    end = tex_table.index(r"\end{tabular}") + len(r"\end{tabular}")
    return tex_table[:start] + r"\resizebox{\linewidth}{!}{" + tex_table[start:end] + "}" + tex_table[end:]


def table_is_empty(lines: list[str]) -> bool:
    rows = [ln for ln in lines if ln.strip().startswith("|")]
    body = [r for r in rows[2:] if not all(set(c.strip()) <= set("-: ") for c in r.strip().strip("|").split("|"))]
    return len(body) == 0 or any("_(no rows)_" in ln for ln in lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    ap.add_argument("--tex", default=str(TEX))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tex_path = Path(args.tex)
    src_path = tex_path.with_suffix(".tex.in")
    if not src_path.exists():
        src_path.write_text(tex_path.read_text())
    tex = src_path.read_text()
    reports = {name: split_headings((Path(args.report_dir) / name).read_text())
               for name in ("REPORT.md", "TRANSFER.md", "PROOF_POINTS.md")
               if (Path(args.report_dir) / name).exists()}

    filled, missing = [], []
    for marker, (fname, prefix, caption, label, drop) in TABLES.items():
        if f"%%{marker}%%" not in tex:
            continue
        secs = reports.get(fname, {})
        prefixes = (prefix,) if isinstance(prefix, str) else prefix
        key = next((k for p in prefixes for k in secs if k.startswith(p)), None)
        lines = rename_arms(filter_rows(secs.get(key, []), ROW_FILTERS.get(marker))) if key else []
        if not key or table_is_empty(lines):
            tex = tex.replace(f"%%{marker}%%",
                              rf"\textcolor{{red}}{{TODO: table ``{tex_escape(prefixes[0])}'' not in {tex_escape(fname)} yet.}}")
            missing.append(marker)
            continue
        tex = tex.replace(f"%%{marker}%%", fit_width(md_table_to_tex(drop_columns(lines, drop), caption, label)))
        filled.append(marker)
    if r"\usepackage{float}" not in tex:
        tex = tex.replace(r"\usepackage{caption}", "\\usepackage{caption}\n\\usepackage{float}", 1)
    out = Path(args.out) if args.out else tex_path
    out.write_text(tex)
    left = re.findall(r"%%[A-Z_]+%%", tex)
    print(f"wrote {out}: {len(filled)} tables filled; missing from reports: {missing or 'none'}; "
          f"unknown markers: {left or 'none'}")


if __name__ == "__main__":
    main()
