"""The blind coding of regime-fix drafts for P11(a′) (HYPOTHESIS_RSI.md, amendment (v)).

P11(a′): of regime fixes whose proof set (the Tier 2 case set) contains at least one kept case,
the share whose drafted entry states a *condition* — names an input-dependent branch under
which the rule differs — is ≥ 0.5. Two coders who have not seen the change sets read the
drafted text alone and answer one question per item. Nothing else about the fix (arm, run,
verdict, situation) is on the sheet.

    make   --runs rsi-e2b-c6 rsi-c6 [--arms senselab-rsi ...]
           writes results/rsi-coding/sheet.csv (item_id, family, text — shuffled, seed 11) and
           results/rsi-coding/key.json (item_id -> fix, run, arm, cell, kept cases, Tier 2 status);
           coders get sheet.csv only.
    score  --answers coder1.csv coder2.csv
           each answer file: item_id, conditional (Y/N), condition_quoted (free text). Reports the
           per-coder share, raw agreement, Cohen's kappa, and the P11(a′) reading on the
           agreed items (an item counts as conditional only when both coders say Y; disagreements
           are listed for adjudication and reported, never silently resolved).

A "kept case" is a Tier 2 case whose task is a post-change changed-class episode on which the
rule did *not* change (``tags.change_cond`` false); cases join to episodes by task text within
the cell, which matched 115 of 115 cases on ``rsi-c5``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from . import rsi

OUT = rsi.RESULTS / "rsi-coding"
RSI_ARMS = ("senselab-rsi", "senselab-rsi-nocanary", "senselab-rsi-noreplay")


def regime_fixes_with_kept_case(run: str, arms: tuple[str, ...]) -> list[dict[str, Any]]:
    eps_by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    for e in rsi.episodes(run):
        eps_by_cell.setdefault(e.get("cell", ""), {})[e.get("task_prompt", "")] = e
    out = []
    for r in rsi.reports(run):
        cell = r.get("cell", "")
        arm = cell.split("|")[1] if "|" in cell else ""
        if arm not in arms:
            continue
        cell_eps = eps_by_cell.get(cell, {})
        for f in (r.get("report") or {}).get("fixes") or []:
            if f.get("judge_id") or f.get("behavior_id"):
                continue  # judge-raised fixes are not regime fixes
            cases = (f.get("replay") or {}).get("cases") or []
            kept = moved = unmatched = 0
            for c in cases:
                e = cell_eps.get(c.get("task_input") or "")
                if e is None:
                    unmatched += 1
                    continue
                t = e.get("tags") or {}
                if t.get("post_change") and t.get("change_class"):
                    if t.get("change_cond"):
                        moved += 1
                    else:
                        kept += 1
            if kept == 0:
                continue
            value = ((f.get("correction") or {}).get("value") or "").strip()
            if not value:
                continue
            out.append({"run": run, "arm": arm, "cell": cell, "fix_id": f.get("id"), "family": cell.split("|")[0],
                        "key": (f.get("correction") or {}).get("key"), "text": value,
                        "kept_cases": kept, "moved_cases": moved, "unmatched_cases": unmatched,
                        "tier2_status": (f.get("replay") or {}).get("status"), "status": f.get("status"),
                        "shipped": bool(f.get("shipped_at"))})
    return out


def make(runs: list[str], arms: tuple[str, ...], out: Path) -> None:
    items = [it for run in runs for it in regime_fixes_with_kept_case(run, arms)]
    for it in items:
        it["item_id"] = hashlib.sha256(f"{it['run']}|{it['fix_id']}".encode()).hexdigest()[:8]
    random.Random(11).shuffle(items)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "family", "text", "conditional (Y/N)", "condition_quoted"])
        for it in items:
            w.writerow([it["item_id"], it["family"], it["text"], "", ""])
    (out / "key.json").write_text(json.dumps({it["item_id"]: {k: v for k, v in it.items() if k != "text"}
                                              for it in items}, indent=1) + "\n")
    (out / "INSTRUCTIONS.md").write_text(
        "# Coding instructions\n\n"
        "Each row is a rule an automated system wrote for a support or CI agent after that agent kept "
        "failing a class of task. Read the text alone and answer one question:\n\n"
        "**Does the rule state a condition — an input-dependent branch — under which what the agent "
        "should do differs?** `Y` if the text names something about the case (a plan, a platform, a "
        "directory touched, a keyword, an amount, a customer attribute) that decides between two "
        "different actions; `N` if it prescribes one action regardless (an unconditional rule, or one "
        "whose only condition is the class itself, e.g. \"for Android notification tickets, do X\").\n\n"
        "Put `Y` or `N` in `conditional (Y/N)`. If `Y`, paste the phrase that states the condition into "
        "`condition_quoted`. Do not discuss items with the other coder. Do not look up the tasks.\n")
    print(f"{len(items)} items from {runs} -> {out / 'sheet.csv'} (coders), {out / 'key.json'} (private)")


def _read_answers(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            ans = (row.get("conditional (Y/N)") or row.get("conditional") or "").strip().upper()[:1]
            out[row["item_id"]] = (ans, (row.get("condition_quoted") or "").strip())
    return out


def score(answers: list[Path], out: Path) -> None:
    key = json.loads((out / "key.json").read_text())
    coders = [_read_answers(p) for p in answers]
    ids = [i for i in key if all(i in c and c[i][0] in ("Y", "N") for c in coders)]
    n = len(ids)
    if n == 0:
        print("no items answered by every coder")
        return
    shares = [sum(1 for i in ids if c[i][0] == "Y") / n for c in coders]
    if len(coders) == 2:
        a, b = coders
        agree = sum(1 for i in ids if a[i][0] == b[i][0]) / n
        pa = sum(1 for i in ids if a[i][0] == "Y") / n
        pb = sum(1 for i in ids if b[i][0] == "Y") / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
        both_y = [i for i in ids if a[i][0] == b[i][0] == "Y"]
        disagreements = [i for i in ids if a[i][0] != b[i][0]]
    else:
        agree = kappa = float("nan")
        both_y = [i for i in ids if all(c[i][0] == "Y" for c in coders)]
        disagreements = [i for i in ids if len({c[i][0] for c in coders}) > 1]
    by_arm: dict[str, list[str]] = {}
    for i in ids:
        by_arm.setdefault(key[i]["arm"], []).append(i)
    report = {
        "items": n, "coder_shares": shares, "raw_agreement": agree, "cohen_kappa": kappa,
        "conditional_both": len(both_y), "share_conditional_both": len(both_y) / n,
        "disagreements": disagreements,
        "p11a_prime": {"threshold": 0.5, "met": len(both_y) / n >= 0.5,
                       "rule": "an item counts as conditional only when both coders say Y; "
                               "disagreements are adjudicated openly and reported"},
        "by_arm": {arm: {"items": len(v), "conditional_both": sum(1 for i in v if i in both_y)} for arm, v in by_arm.items()},
    }
    (out / "score.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--runs", nargs="+", required=True)
    m.add_argument("--arms", nargs="+", default=list(RSI_ARMS))
    m.add_argument("--out", type=Path, default=OUT)
    s = sub.add_parser("score")
    s.add_argument("--answers", nargs="+", type=Path, required=True)
    s.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.cmd == "make":
        make(args.runs, tuple(args.arms), args.out)
    else:
        score(args.answers, args.out)


if __name__ == "__main__":
    main()
