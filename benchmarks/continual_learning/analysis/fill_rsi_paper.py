"""Fill ``paper/rsi.tex`` from ``results/rsi-report/REPORT.md`` (``analysis.rsi --paper``) and
the team-grows report.

Each ``%%TAB_*%%`` marker in ``paper/rsi.tex.in`` names one headed table of one report file;
the markdown table under that heading becomes a booktabs table. Harness arm names that reach
a table are renamed to system names here (amendment (o)): they never appear in the paper.

    python -m benchmarks.continual_learning.analysis.rsi --senselab-run rsi-c3 --paper
    python -m benchmarks.continual_learning.analysis.fill_rsi_paper
    (cd benchmarks/continual_learning/paper && pdflatex rsi.tex && pdflatex rsi.tex)

A marker whose table is not in the report yet is left as a visible red TODO.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .fill_cl_paper import drop_columns, filter_rows, fit_width, split_headings, table_is_empty
from .fill_paper import tex_escape

HERE = Path(__file__).resolve().parent.parent
REPORT_DIR = HERE / "results" / "rsi-report"
CONF_DIR = HERE / "results" / "rsi-report-confirmatory"
TEAM_GROWS = HERE / "results" / "rsi-teamgrows-dev-report" / "TEAM_GROWS.md"
POSTHOC = HERE / "results" / "rsi-report" / "POSTHOC_SHIPS.md"
TEX = HERE / "paper" / "rsi.tex"

CONF_NOTE = ("Confirmatory seeds 37 / 41 / 59 / 73, both agent models (GPT-5.5, Claude Opus~5), the two "
             "set-C families, run once on the \\code{post-canary-ni} cut after \\code{analysis-freeze-rsi} "
             "(amendment (w)).")

#: marker -> (report, heading prefix, caption, label, columns to drop)
TABLES: dict[str, tuple[str, str, str, str, tuple[str, ...]]] = {
    "TAB_HEADLINE": (
        "REPORT.md", "1. After the rule changes conditionally",
        "After the rule changes conditionally (set C, change at episode 30 of 100; GPT-5.5; four cells "
        "per system: \\code{support} and \\code{ci-fix} $\\times$ seeds 11 / 23). First-attempt success on "
        "every post-change episode. \\emph{Moved half}: changed-class instances whose rule changed; "
        "\\emph{kept half}: changed-class instances whose rule did not. Brackets: cell-level bootstrap 95\\% "
        "interval. Development seeds; SenseLab on the \\code{post-canary-ni} engine cut.",
        "tab:headline", ("cells", "infra err")),
    "TAB_PAIRED": (
        "REPORT.md", "1b. Paired cell deltas, moved half",
        "SenseLab minus each system on the moved half, per cell, in percentage points, paired on "
        "(family, seed).", "tab:paired", ()),
    "TAB_BYFAMILY": (
        "REPORT.md", "1c. By family",
        "By family: first-attempt success on the moved half, the kept half and the unchanged classes.",
        "tab:byfamily", ()),
    "TAB_CUTS": (
        "REPORT.md", "2. SenseLab by engine cut",
        "SenseLab on successive cuts of the engine, same protocol and seeds. Fix counts are over the four "
        "SenseLab cells of each run. \\code{pre-seam4}: amfs-internal \\#787/\\#788; \\code{post-seam5}: \\#793, \\#794; "
        "\\code{post-seam5b}: \\#795, \\#796; \\code{post-clean-success}: \\#797, \\#798; \\code{post-canary-ni}: "
        "\\#799 (\\S\\ref{sec:findings}).", "tab:cuts", ("cells", "what changed")),
    "TAB_ATTRIBUTION": (
        "REPORT.md", "3. Attribution",
        "Attribution. SenseLab against its own ablations on the same protocol: without the canary (policy "
        "\\code{auto\\_after\\_replay}, same run) and without the repair loop (policy \\code{review}: proposals "
        "never approved; outcome learning alone). This is the only table in which the ablations appear.",
        "tab:attribution", ("cells",)),
    "TAB_LIFECYCLE": (
        "REPORT.md", "4. Fix lifecycle",
        "Fix lifecycle by run and arm. Tier~1: stand-in replay on the fix branch; Tier~2: the customer's "
        "replay of the case set; a canary grades live exposure in the fix's situation. \\emph{open}: Tier~2 "
        "replays still awaited when the cell ended. No canary reached a decision inside a 100-episode cell; "
        "no fix was reverted; 96--100\\% of proposals were drafted by the repair agent.",
        "tab:lifecycle", ("reverted", "drafted by the repair agent")),
    "TAB_COST": (
        "REPORT.md", "5. Cost and latency",
        "Cost and latency per episode, by where the call is charged. \\emph{agent \\$}: the agent's own calls; "
        "\\emph{arm \\$}: reflection or consolidation the arm ran per episode; \\emph{judges \\$} and "
        "\\emph{replays \\$}: the loop's judge verdicts and the Tier~2 replays the harness re-ran on the fix "
        "branch, exact from the run records; \\emph{drafts}: fixes the repair agent drafted per 100 episodes, "
        "priced at the account ledger's remainder per draft (repair agent, Tier~1 stand-in replay, lever "
        "classification, discovery --- not persisted per fix, so an estimate; \\S\\ref{sec:cost}). "
        "\\emph{total} sums the five. \\emph{memory s}: time in memory calls against the production API.",
        "tab:cost", ()),
    "TAB_TEAMGROWS": (
        "TEAM_GROWS.md", "pioneer/A. Joiner first exposures after ep 60",
        "When the team grows (set A, change at 20, fleet of six sharing one store; \\emph{pioneer}: one agent "
        "works alone to episode 60, five join). First-attempt success of the joiners on their first exposure "
        "to each class after joining, with the veterans' rate before they joined; \\emph{stale-pick}: the "
        "joiner applied the pre-change action.", "tab:teamgrows",
        ("cells", "joiner all eps 1st", "joiner stale (changed, all eps)")),
    "TAB_POSTHOC": (
        "POSTHOC_SHIPS.md", "Windows",
        "Exploratory, not preregistered: traffic after the no-canary ablation shipped, against the "
        "SenseLab cell of the same family, seed and cut over the same episodes (nine cells with a ship; "
        "window from the first ship to the end of the cell). \\emph{Recurrence} is the canary's reading "
        "of a session: not a first-attempt success. DiD = (after $-$ before)$_{-\\text{canary}}$ $-$ "
        "(after $-$ before)$_{\\text{SenseLab}}$; negative means the ships were associated with less "
        "recurrence than shipping nothing.", "tab:posthoc", ()),
    # confirmatory read (results/rsi-report-confirmatory); pending until the seeds run
    "CONF_TAB_HEADLINE": (
        "CONF_REPORT.md", "1. After the rule changes conditionally",
        "Confirmatory: after the rule changes conditionally. Sixteen cells per system "
        "(\\code{support}, \\code{ci-fix} $\\times$ four seeds $\\times$ two models); first-attempt success on "
        "every post-change episode; brackets: cell-level bootstrap 95\\% interval. " + CONF_NOTE,
        "tab:conf-headline", ("cells", "infra err")),
    "CONF_TAB_PAIRED": (
        "CONF_REPORT.md", "1b. Paired cell deltas, moved half",
        "Confirmatory primary endpoint: SenseLab minus each system on the moved half, percentage points, "
        "paired on (family, seed, model); per family $\\times$ model means, the mean over the sixteen cells, "
        "cells won, the exact two-sided sign-flip $p$ over the cells and its Holm adjustment across the five "
        "systems (amendment (x)).", "tab:conf-paired", ()),
    "CONF_TAB_PAIRED_KEPT": (
        "CONF_REPORT.md", "1b-kept. Secondary endpoint, paired: kept half",
        "Confirmatory secondary endpoint, the kept half: SenseLab minus each system, paired as in "
        "Table~\\ref{tab:conf-paired}; Holm within this endpoint.", "tab:conf-paired-kept", ()),
    "CONF_TAB_PAIRED_UNCHANGED": (
        "CONF_REPORT.md", "1b-unchanged. Secondary endpoint, paired: unchanged classes",
        "Confirmatory secondary endpoint, the unchanged classes: SenseLab minus each system, paired as in "
        "Table~\\ref{tab:conf-paired}; Holm within this endpoint.", "tab:conf-paired-unchanged", ()),
    "CONF_TAB_BYFAMILY": (
        "CONF_REPORT.md", "1c. By family",
        "Confirmatory, by family: first-attempt success on the moved half, the kept half and the unchanged "
        "classes.", "tab:conf-byfamily", ()),
    "CONF_TAB_BYMODEL": (
        "CONF_REPORT.md", "1d. By model",
        "Confirmatory, by agent model: first-attempt success on the moved half, the kept half and the "
        "unchanged classes.", "tab:conf-bymodel", ()),
    "CONF_TAB_ATTRIBUTION": (
        "CONF_REPORT.md", "3. Attribution",
        "Confirmatory attribution: SenseLab against its own ablations on the same sixteen cells. The only "
        "confirmatory table in which the ablations appear.", "tab:conf-attribution", ("cells",)),
    "CONF_TAB_LIFECYCLE": (
        "CONF_REPORT.md", "4. Fix lifecycle",
        "Confirmatory fix lifecycle by arm over the sixteen fleet-1 cells (the fleet-10 cells are read in "
        "\\S\\ref{sec:confirmatory}).", "tab:conf-lifecycle", ("reverted", "drafted by the repair agent")),
    "CONF_TAB_COST": (
        "CONF_REPORT.md", "5. Cost and latency",
        "Confirmatory cost and latency per episode, columns as in Table~\\ref{tab:cost}; the drafts column is "
        "checked against the ledger difference the runs recorded.", "tab:conf-cost", ()),
}

ARM_RENAMES = {
    "SenseLab, outcome loop on": "SenseLab − repair loop",
    "senselab-rsi-nocanary": "SenseLab − canary",
    "senselab-rsi": "SenseLab",
    "pgvector+rerank": "pgvector + rerank",
    "pgvector-diy+outcomes": "pgvector + outcomes (DIY)",
    "raw-traces": "Context window",
    "none": "Frontier alone",
}
#: columns past which a table is shrunk to the text width (default 7)
WIDE = {"TAB_PAIRED": 8}
#: rows to keep per marker (the ablation row of the team-grows table belongs to the attribution table only)
ROW_FILTERS = {
    "TAB_TEAMGROWS": lambda row: row.get("arm") in ("senselab-rsi", "pgvector+rerank"),
}


def md_table_to_tex(lines: list[str], caption: str, label: str) -> str:
    """The report's markdown table as a booktabs table, every column kept (the shared helper drops
    constant columns, which in a paired-delta table is a data column)."""
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if ln.strip().startswith("|")]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    ncol = len(head)
    body = [(r + [""] * ncol)[:ncol] for r in body]
    spec = "l" + "r" * (ncol - 1)
    out = [r"\begin{table}[H]", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
           rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(rf"\textbf{{{tex_escape(h)}}}" for h in head) + r" \\", r"\midrule"]
    out += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in body]
    out += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(out)


def rename(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            cells[0] = ARM_RENAMES.get(cells[0], cells[0])
            ln = "| " + " | ".join(cells) + " |"
        out.append(ln)
    return out


def sections_by_part(md: str) -> dict[str, list[str]]:
    """``## part`` then ``### section`` -> ``part/section`` (the team-grows report), plus every heading alone."""
    out = split_headings(md)
    part = ""
    for ln in md.splitlines():
        if ln.startswith("## "):
            part = ln[3:].split(":")[0].strip()
        elif ln.startswith("### ") and part:
            out[f"{part}/{ln[4:].strip()}"] = out.get(ln[4:].strip(), [])
    # split_headings keys the *last* section of a repeated heading; rebuild per part explicitly.
    cur, part = None, ""
    per: dict[str, list[str]] = {}
    for ln in md.splitlines():
        if ln.startswith("## "):
            part = ln[3:].split(":")[0].strip()
            cur = None
        elif ln.startswith("### "):
            cur = f"{part}/{ln[4:].strip()}"
            per[cur] = []
        elif cur:
            per[cur].append(ln)
    out.update(per)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    ap.add_argument("--conf-report-dir", default=str(CONF_DIR),
                    help="the confirmatory read (analysis.rsi --seeds 37 41 59 73 ...); CONF_TAB_* markers")
    ap.add_argument("--tex", default=str(TEX))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tex_path = Path(args.tex)
    src_path = tex_path.with_suffix(".tex.in")
    tex = src_path.read_text()
    reports: dict[str, dict[str, list[str]]] = {}
    report = Path(args.report_dir) / "REPORT.md"
    if report.exists():
        reports["REPORT.md"] = split_headings(report.read_text())
    conf = Path(args.conf_report_dir) / "REPORT.md"
    if conf.exists():
        reports["CONF_REPORT.md"] = split_headings(conf.read_text())
    if TEAM_GROWS.exists():
        reports["TEAM_GROWS.md"] = sections_by_part(TEAM_GROWS.read_text())
    if POSTHOC.exists():
        reports["POSTHOC_SHIPS.md"] = split_headings(POSTHOC.read_text())

    filled, missing = [], []
    for marker, (fname, prefix, caption, label, drop) in TABLES.items():
        if f"%%{marker}%%" not in tex:
            continue
        secs = reports.get(fname, {})
        key = next((k for k in secs if k.startswith(prefix)), None)
        lines = rename(filter_rows(secs.get(key, []), ROW_FILTERS.get(marker))) if key else []
        if not key or table_is_empty(lines):
            note = (r"\textcolor{red}{[pending: the confirmatory seeds have not run; this table is filled from "
                    r"\code{results/rsi-report-confirmatory} --- ``" + tex_escape(prefix) + "''.]}"
                    if fname == "CONF_REPORT.md" else
                    rf"\textcolor{{red}}{{TODO: table ``{tex_escape(prefix)}'' not in {tex_escape(fname)} yet.}}")
            tex = tex.replace(f"%%{marker}%%", note)
            missing.append(marker)
            continue
        table = md_table_to_tex(drop_columns(lines, drop), caption, label)
        tex = tex.replace(f"%%{marker}%%", fit_width(table, max_cols=WIDE.get(marker, 6)))
        filled.append(marker)
    out = Path(args.out) if args.out else tex_path
    out.write_text(tex)
    left = re.findall(r"%%[A-Z_]+%%", tex)
    print(f"wrote {out}: {len(filled)} tables filled; missing from reports: {missing or 'none'}; "
          f"unknown markers: {left or 'none'}")


if __name__ == "__main__":
    main()
