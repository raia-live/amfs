"""Fill ``paper/distillation.tex`` from ``results/distill-report/REPORT.md``.

Markdown tables in the report become booktabs tables; the ``%%RESULTS%%`` and
``%%COST%%`` markers take them. Prose markers (``%%ABSTRACT%%``, ``%%CONCLUSION%%``,
``%%SCOPE_CHECK%%``) come from ``paper/distillation_prose.json`` when it exists and are
left as a visible TODO otherwise, so a build never silently ships an empty section.

    python -m benchmarks.continual_learning.analysis.fill_paper [--report PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPORT = HERE / "results" / "distill-report" / "REPORT.md"
TEX = HERE / "paper" / "distillation.tex"
PROSE = HERE / "paper" / "distillation_prose.json"


def tex_escape(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", r"\textasciitilde{}")  # serving-variant aliases: ci-sl-gpt55~k0
    s = s.replace("→", r"$\rightarrow$").replace("−", "$-$").replace("×", r"$\times$").replace("—", "---")
    s = s.replace("≥", r"$\geq$").replace("≤", r"$\leq$")
    return s


def md_table_to_tex(lines: list[str], caption: str, label: str) -> str:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if ln.strip().startswith("|")]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    ncol = len(head)
    body = [(r + [""] * ncol)[:ncol] for r in body]
    # The paper's conditions table names the models per condition, so the report's
    # (truncated) alias list adds nothing; columns that are empty or zero throughout
    # (fallback/ep, unparse/ep in most cells) are dropped as well.
    keep = [j for j, h in enumerate(head)
            if h != "models" and not all(re.fullmatch(r"(0(\.0+)?|—|-|)", r[j]) for r in body)
            and (j == 0 or len({r[j] for r in body} - {"—", "-", ""}) > 1)]  # constant columns say nothing
    head = [head[j] for j in keep]
    body = [[r[j] for j in keep] for r in body]
    ncol = len(head)
    spec = "l" + "r" * (ncol - 1)
    width = sum(max(len(h), *(len(r[j]) for r in body)) for j, h in enumerate(head))
    wide = width > 100  # past this the scriptsize table overruns the text width
    if len(body) > 30:
        # Too tall for one page: a longtable that breaks across pages (an [H] table this
        # tall leaves the previous page mostly blank).
        out = [r"{\scriptsize\setlength{\tabcolsep}{3pt}", rf"\begin{{longtable}}{{{spec}}}",
               rf"\caption{{{caption}}}\label{{{label}}} \\", r"\toprule",
               " & ".join(rf"\textbf{{{tex_escape(h)}}}" for h in head) + r" \\", r"\midrule", r"\endfirsthead",
               r"\toprule", " & ".join(rf"\textbf{{{tex_escape(h)}}}" for h in head) + r" \\", r"\midrule", r"\endhead",
               r"\bottomrule", r"\endfoot"]
        out += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in body]
        out += [r"\end{longtable}}", ""]
        return "\n".join(out)
    out = [r"\begin{table}[H]", r"\centering\scriptsize"]
    if wide:
        out.append(r"\resizebox{\textwidth}{!}{%")
    out += [rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
            " & ".join(rf"\textbf{{{tex_escape(h)}}}" for h in head) + r" \\", r"\midrule"]
    for r in body:
        out.append(" & ".join(tex_escape(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if wide:
        out.append("}")
    out += [rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(out)


def sections(md: str) -> dict[str, list[str]]:
    """Split the report at ``## `` headers; the value is the list of lines in each section.

    A ``# `` header after the first opens a new part; its sections are keyed
    ``<part>/<section>`` so the novel-change part can be routed to its own marker."""
    out: dict[str, list[str]] = {}
    cur, part = "_preamble", ""
    for ln in md.splitlines():
        if ln.startswith("# ") and out:
            part = ln[2:].strip() + "/"
            cur = part + "_intro"
            out[cur] = []
        elif ln.startswith("## "):
            cur = part + ln[3:].strip()
            out[cur] = []
        else:
            out.setdefault(cur, []).append(ln)
    return out


def latency_caption(name: str) -> tuple[str, str]:
    if "paired" in name:
        return ("Paired wall-clock deltas in seconds (a $-$ b) over the same seed, fleet and episode, so within one "
                "cell's time window; 95\\% bootstrap intervals over cells.", "tab:lat-paired")
    if "change set B" in name:
        return ("Wall-clock under change set B, post-change episodes of the changed classes.", "tab:lat-b")
    sc = name.split(",", 1)[1].strip() if "," in name else ""
    return (f"Latency, \\arm{{{tex_escape(sc)}}} scenario: wall-clock to the sealed conclusion (median and p90, before "
            "and after the change), time inside model calls, the harness's memory time, attempts, and wall per success. "
            "Absolute values are load-dependent (36 to 52 concurrent workers, shared memory API).", f"tab:lat-{tex_escape(sc)}")


def caption_for(name: str, i: int, part: str) -> tuple[str, str]:
    """Caption and label for a report section's table; labels are unique per part."""
    sfx = "-b" if part else ""
    if name.startswith("Hypotheses") and "first attempt" in name:
        return ("Paired first-attempt success deltas (condition A minus B), 95\\% bootstrap intervals over cells"
                + (" under change set B." if part else "."), f"tab:hyp-first{sfx}")
    if name.startswith("Hypotheses"):
        return ("Paired success-rate deltas (condition A minus B), 95\\% bootstrap intervals over cells"
                + (" under change set B." if part else "."), f"tab:hyp{sfx}")
    if name.startswith("Students individually"):
        return ("Each student: held-out task success and the product's own evaluation of the promoted version.",
                "tab:students")
    if name.startswith("Post-change, changed classes"):
        return ("Post-change episodes of the changed classes only: the change the students trained on (A) "
                "against one their weights never saw (B).", "tab:a-vs-b")
    if name.startswith("Memory at inference per student"):
        return ("What memory does for each student's first attempt, before and after the change: full serving "
                "minus no memory (S1$-$S0) and full serving minus context only (S1$-$S1p), paired.", "tab:per-student")
    if name.startswith("Parent vs child"):
        return ("Self-retrained students (v3) and their parents on seeds neither has seen, with the frontier "
                "teachers on the same seeds.", "tab:self")
    if name.startswith("Child"):
        return ("Child minus parent, paired by scenario, seed, fleet and episode; 95\\% bootstrap intervals "
                "over cells.", "tab:self-delta")
    if name.startswith("Skill-file student"):
        return ("The student distilled from the skill-file agent's traces (K0) against the memory-trace student "
                "and the frontier on the same seeds, paired; 95\\% bootstrap intervals over cells.", "tab:k0")
    sc = tex_escape(name.replace(" (B)", ""))
    return (f"Held-out results, \\arm{{{sc}}} scenario, fleet sizes pooled"
            + (", change set B." if part else "."), f"tab:res{sfx}{i}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--tex", default=str(TEX))
    ap.add_argument("--out", default=None, help="default: overwrite --tex in place, keeping a .in copy")
    args = ap.parse_args()

    tex_path = Path(args.tex)
    src_path = tex_path.with_suffix(".tex.in")
    if not src_path.exists():
        src_path.write_text(tex_path.read_text())  # first fill keeps the template
    tex = src_path.read_text()
    md = Path(args.report).read_text()
    secs = sections(md)

    results, cost, novel, selfr = [], [], [], []
    counters = {"": 0, "B": 0}
    for key, lines in secs.items():
        part, _, name = key.rpartition("/")
        if name in ("_preamble", "_intro"):
            continue
        if name.startswith("Cost per success"):
            cost.append(md_table_to_tex(lines, tex_escape(name), "tab:cost"))
            continue
        if name.startswith("Latency"):
            # The latency tables sit with cost. The first "Latency (...)" section is the
            # caveat paragraph with no table and yields nothing.
            cap, lab = latency_caption(name)
            t = md_table_to_tex(lines, cap, lab)
            if t:
                cost.append(t)
            continue
        bucket = selfr if part.startswith("Self") else (novel if part else results)
        tag = "B" if part else ""
        if not (name.startswith(("Hypotheses", "Students individually", "Post-change", "Parent", "Child", "Memory at",
                                 "Skill-file student"))):
            counters[tag] += 1
        cap, lab = caption_for(name, counters[tag], part)
        t = md_table_to_tex(lines, cap, lab)
        if t:
            bucket.append(t)

    prose = json.loads(PROSE.read_text()) if PROSE.exists() else {}

    def fill(marker: str, value: str) -> None:
        nonlocal tex
        tex = tex.replace(marker, value)

    fill("%%RESULTS%%", "\n".join(results) or r"\textcolor{red}{TODO: no result tables in the report.}")
    fill("%%NOVEL%%", "\n".join(novel) or r"\textcolor{red}{TODO: no change-set-B tables in the report.}")
    fill("%%SELF%%", "\n".join(selfr) or r"\textcolor{red}{TODO: no self-retrain tables in the report.}")
    fill("%%COST%%", "\n".join(cost) or r"\textcolor{red}{TODO: no cost table in the report.}")
    for key in sorted({m.strip("%").lower() for m in re.findall(r"%%[A-Z0-9_]+%%", tex)}):
        # The per-stage sentences of the path section are optional: one quantitative line
        # each, added once the numbers are final; until then the stage reads without it.
        missing = "" if key.startswith("path_stage") else rf"\textcolor{{red}}{{TODO: {tex_escape(key)} in distillation\_prose.json}}"
        fill(f"%%{key.upper()}%%", prose.get(key) or missing)
    if r"\usepackage{float}" not in tex:
        tex = tex.replace(r"\usepackage{caption}", "\\usepackage{caption}\n\\usepackage{float}", 1)

    out = Path(args.out) if args.out else tex_path
    out.write_text(tex)
    left = re.findall(r"%%[A-Z0-9_]+%%", tex)
    print(f"wrote {out}; {len(results)} result tables, {len(cost)} cost tables; unfilled markers: {left or 'none'}")


if __name__ == "__main__":
    main()
