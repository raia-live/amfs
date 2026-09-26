"""Distillation report: students, controls and teachers on the held-out seeds.

    python -m benchmarks.continual_learning.analysis.distill [--runs distill-] [--teacher-runs gridv4-core-]

Reads every ``results/distill-*/episodes.jsonl`` (plus the teacher grid's rows for the same
scenarios when a ``distill-teach-*`` run is absent), labels each row with its condition
(HYPOTHESIS_DISTILL.md: F1 F0 D1 D0 S1 S0 S1p S1k S1n B1), and writes
``results/distill-report/REPORT.md`` with the tables the paper quotes and the figures.
Pairing is by (scenario, seed, fleet, episode); intervals are bootstrap over cells.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .. import config
from .regime_change import boot_ci, is_infra, mean, table

RESULTS = config.RESULTS_DIR
OUT = RESULTS / "distill-report"

COND_ORDER = ["F1", "F0", "D1", "D0", "S1", "S1n", "S1p", "S1k", "S0", "B1", "K1", "K0", "K0n"]
COND_LABEL = {
    "F1": "frontier, full loop + SenseLab", "F0": "frontier, full loop, no memory",
    "D1": "frontier @decision (served prompt) + SenseLab", "D0": "frontier @decision, empty store",
    "S1": "student + SenseLab", "S1n": "student + SenseLab + note-writer",
    "S1p": "student + context only (no avoid/priors)", "S1k": "student, nothing served (k0)",
    "S0": "student, no memory", "B1": "untuned base + SenseLab",
    "K1": "frontier, full loop + skill file", "K0": "student from skill-file teacher + SenseLab",
    "K0n": "student from skill-file teacher, no memory",
}


def _registry() -> dict:
    try:
        return json.loads(config.STUDENT_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


REG = _registry()


def condition(x: dict) -> str | None:
    model, arm = x["model"], x["arm"]
    mem = arm.startswith("senselab")
    if arm == "skillfile":
        return "K1"
    if "@decision" in model:
        return "D1" if mem else "D0"
    base = model.split("+notes=", 1)[0]
    alias, _, variant = base.partition("~")
    reg = REG.get(alias)
    is_student = reg is not None or alias.startswith("senselab-")
    if not is_student:
        if arm in ("none", "senselab"):
            return "F1" if mem else "F0"
        return None
    if reg and reg.get("serve_base"):
        return "B1" if mem else None
    if reg and ((reg.get("teacher") or {}).get("arm") == "skillfile"):
        return "K0" if mem else "K0n"
    if not mem:
        return "S0"
    if "+notes=" in model:
        return "S1n"
    if variant == "plain":
        return "S1p"
    if variant == "k0":
        return "S1k"
    return "S1"


def load(prefix: str, teacher_prefix: str, scenarios: set[str]) -> list[dict]:
    rows: list[dict] = []
    # ``<prefix>train-*`` are the teachers' training runs on the training seeds — the
    # students' data, not an evaluation of anything.
    runs = sorted(p for p in RESULTS.glob(f"{prefix}*")
                  if (p / "episodes.jsonl").exists() and "report" not in p.name
                  and not p.name.startswith(f"{prefix}train"))
    have_teach = any(p.name.startswith(f"{prefix}teach") for p in runs)
    if not have_teach:
        runs += sorted(p for p in RESULTS.glob(f"{teacher_prefix}*") if (p / "episodes.jsonl").exists())
    for run in runs:
        for line in (run / "episodes.jsonl").read_text().splitlines():
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            if scenarios and x["scenario"] not in scenarios:
                continue
            if x.get("fleet_mode", "rr") != "rr":
                continue
            c = condition(x)
            if c is None:
                continue
            if run.name.startswith(teacher_prefix) and c not in ("F1", "F0", "K1"):
                continue
            x["cond"] = c
            x["run"] = run.name
            x["teacher_grid"] = run.name.startswith(teacher_prefix)
            u = x["usage"]
            r = x.get("reflection_usage") or {}
            # An arm's checkpoint work (the skill file's maintainer, proposer and validation
            # replays; pgvector-diy's consolidation) lands in memory.after_episode on the episode
            # that triggered it; it is the arm's spend and belongs in its cost per success.
            ae = x["memory"].get("after_episode") or {}
            x["tokens"] = (u["prompt_tokens"] + u["completion_tokens"] + r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
                           + (x["memory"].get("arm_extra_prompt_tokens") or 0) + (x["memory"].get("arm_extra_completion_tokens") or 0)
                           + (ae.get("arm_extra_prompt_tokens") or 0) + (ae.get("arm_extra_completion_tokens") or 0))
            x["cost"] = (u["cost_usd"] + r.get("cost_usd", 0.0) + (x["memory"].get("arm_extra_cost_usd") or 0.0)
                         + (ae.get("arm_extra_cost_usd") or 0.0))
            x["post"] = x["episode"] >= (x.get("change_at") or 20)
            # Latency, in seconds. wall_s is episode start to conclusion (sealed, success or
            # failure). model_s is the time inside model calls: for the frontier the provider
            # calls, for a student the gateway call (server-side retrieval + render + model);
            # memory_s is the harness's own memory operations (retrieve, commit). The arm's
            # checkpoint work (after_episode) is outside the episode and is not in wall_s.
            # first_action_s (episode start to the first domain action) exists only on records
            # written after 2026-09-20 16:20 JST; None before that.
            x["wall_s"] = (x.get("wall_ms") or 0.0) / 1000
            x["model_s"] = (u.get("latency_ms", 0.0) + r.get("latency_ms", 0.0)) / 1000
            x["memory_s"] = (x["memory"].get("memory_ms") or 0.0) / 1000
            fa = x.get("first_action_ms")
            x["first_action_s"] = fa / 1000 if fa is not None else None
            # "A" / "B": the regime change set; "self": the self-retrain comparison and the
            # skill-file student (K0), which run on a third set of seeds and are never pooled
            # with either.
            third = run.name.startswith(f"{prefix}self") or run.name.startswith(f"{prefix}k0")
            x["variant"] = "self" if third else (x.get("change_variant") or "A")
            x["changed"] = bool((x.get("tags") or {}).get("change_class"))
            x["infra"] = is_infra(x.get("error"))
            d = (x.get("flags") or {}).get("decision") or {}
            x["fallback_served"] = int(d.get("fallback_served") or 0)
            x["unparseable"] = int((x.get("flags") or {}).get("unparseable") or 0)
            x["non_terminal"] = int((x.get("flags") or {}).get("non_terminal_answers") or 0)
            rows.append(x)
    # One row per (cell, episode): a cell re-run from scratch, or run under two run names
    # (the main/main2 split), must never count twice. The later row wins.
    by_key = {(x["cell"], x["episode"]): x for x in rows}
    return list(by_key.values())


def _cells(rows, f):
    d = defaultdict(list)
    for x in rows:
        d[x["cell"]].append(float(f(x)))
    return d


def rate(rows, f=lambda x: x["success"]) -> str:
    rows = [x for x in rows if not x["infra"]]
    if not rows:
        return "—"
    lo, hi = boot_ci(_cells(rows, f))
    return f"{mean((f(x) for x in rows), 3):.3f} [{lo:.2f},{hi:.2f}]"


def per_success_cost(rows) -> str:
    rows = [x for x in rows if not x["infra"]]
    s = sum(x["success"] for x in rows)
    return f"${sum(x['cost'] for x in rows) / s:.4f}" if s else "—"


def paired_delta(rows, a: str, b: str, *, post=None, metric="success") -> str:
    """Mean of (a − b) *metric* over (scenario, seed, fleet, episode) pairs present in both.

    Several models sit in one condition (two students, their v1 and v2, both frontier
    teachers); each side is the mean over its models at that task, so a pair compares the
    condition and not whichever model happened to be loaded last."""
    key = lambda x: (x["scenario"], x["seed"], x["fleet"], x["episode"])  # noqa: E731
    A_, B_ = defaultdict(list), defaultdict(list)
    for x in rows:
        if x["infra"] or (post is not None and x["post"] != post):
            continue
        if x["cond"] == a:
            A_[key(x)].append(float(x[metric]))
        elif x["cond"] == b:
            B_[key(x)].append(float(x[metric]))
    A = {k: sum(v) / len(v) for k, v in A_.items()}
    B = {k: sum(v) / len(v) for k, v in B_.items()}
    common = sorted(set(A) & set(B))
    if not common:
        return "—"
    diffs = [A[k] - B[k] for k in common]
    cells = defaultdict(list)
    for k, dlt in zip(common, diffs):
        cells[(k[0], k[1], k[2])].append(float(dlt))
    lo, hi = boot_ci(cells)
    return f"{sum(diffs) / len(diffs):+.3f} [{lo:+.2f},{hi:+.2f}] n={len(common)}"


def recovery(rows) -> str:
    """Success on changed-class tasks in the 20 episodes after the change vs. all pre-change."""
    pre = [x for x in rows if not x["post"] and not x["infra"]]
    post = [x for x in rows if x["post"] and x["changed"] and not x["infra"]]
    if not pre or not post:
        return "—"
    return f"{mean((x['success'] for x in pre), 3):.3f} → {mean((x['success'] for x in post), 3):.3f}"


TESTS = [("H1 student recovers the teacher's gain", "S1", "F0"), ("   teacher's gain (reference)", "F1", "F0"),
         ("H2 memory at inference: S1 − S0", "S1", "S0"), ("H2 memory at inference: S1 − S1k", "S1", "S1k"),
         ("H3 outcome record: S1 − S1p", "S1", "S1p"), ("H4 frontier weights on the same prompt: D1 − S1", "D1", "S1"),
         ("H4 tool loop + own reflection: F1 − D1", "F1", "D1"), ("H5 note-writer: S1n − S1", "S1n", "S1"),
         ("H6 tuning over base: S1 − B1", "S1", "B1"), ("H7 skill file vs memory: K1 − F1", "K1", "F1")]
# H7's second half (K0 − S1) lives in the third-seeds section: K0 is evaluated on the self
# seeds, so it cannot be paired with grid A.


def headline(rows, sc: str, conds) -> str:
    """Per condition for one scenario (fleets pooled, models pooled per condition)."""
    lines = []
    for c in conds:
        r = [x for x in rows if x["scenario"] == sc and x["cond"] == c]
        if not r:
            continue
        models = sorted({x["model"].split("+notes=")[0] for x in r})
        lines.append([f"{c} {COND_LABEL[c]}", ", ".join(m[:28] for m in models)[:60], len(r),
                      rate(r), rate(r, lambda x: x["first_attempt_success"]), rate(r, lambda x: x["escalated"]),
                      f"{mean((x['tokens'] for x in r), 0):.0f}", per_success_cost(r), recovery(r),
                      f"{mean((x['fallback_served'] for x in r), 2)}", f"{mean((x['unparseable'] for x in r), 3)}"])
    return table(["condition", "models", "n", "success", "first-try", "escalated",
                  "tokens/ep", "$/success", "pre→post(changed)", "fallback/ep", "unparse/ep"], lines)


def hypotheses(rows, conds, metric: str = "success") -> str:
    lines = []
    for label, a, b in TESTS:
        if a in conds and b in conds:
            lines.append([label, paired_delta(rows, a, b, metric=metric), paired_delta(rows, a, b, post=False, metric=metric),
                          paired_delta(rows, a, b, post=True, metric=metric)])
    return table(["test", "all episodes", "pre-change", "post-change"], lines)


def _quant(vals, q: float) -> float:
    vals = sorted(vals)
    if not vals:
        return float("nan")
    k = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[k]


def _lat_row(label: str, r) -> list:
    r = [x for x in r if not x["infra"]]
    if not r:
        return [label, 0, "—", "—", "—", "—", "—", "—", "—"]
    pre = [x["wall_s"] for x in r if not x["post"]]
    post = [x["wall_s"] for x in r if x["post"]]
    s = sum(x["success"] for x in r)
    fa = [x["first_action_s"] for x in r if x["first_action_s"] is not None]
    return [label, len(r),
            f"{_quant(pre, .5):.1f} / {_quant(post, .5):.1f}" if pre and post else f"{_quant([x['wall_s'] for x in r], .5):.1f}",
            f"{_quant(pre, .9):.1f} / {_quant(post, .9):.1f}" if pre and post else f"{_quant([x['wall_s'] for x in r], .9):.1f}",
            f"{_quant([x['model_s'] for x in r], .5):.1f}", f"{_quant([x['memory_s'] for x in r], .5):.1f}",
            f"{mean((x['attempts'] for x in r), 2):.2f}",
            f"{sum(x['wall_s'] for x in r) / s:.1f}" if s >= 5 else "—",  # D0's single success is not a rate
            f"{_quant(fa, .5):.1f} (n={len(fa)})" if fa else "—"]


LAT_HEAD = ["condition", "n", "wall med pre/post", "wall p90 pre/post", "model med", "memory med", "attempts",
            "s/success", "first action med"]
LAT_PAIRS = [("S1 − F1", "S1", "F1"), ("S1 − F0", "S1", "F0"), ("S1 − S0", "S1", "S0"), ("S1 − S1p", "S1", "S1p"),
             ("S1 − S1k", "S1", "S1k"), ("D1 − S1", "D1", "S1"), ("F1 − F0", "F1", "F0"), ("K1 − F1", "K1", "F1")]


def latency_section(rows, conds, novel, selfr) -> str:
    md = ["\n## Latency (wall-clock seconds per episode; medians and p90)\n",
          "Wall is episode start to the sealed conclusion (success or failure), all attempts included. Model time is the "
          "time inside model calls: the provider calls for the frontier, the gateway call for a student (which includes the "
          "server-side retrieval and render). Memory time is the harness's own memory operations: for the frontier its "
          "searches, writes and the outcome commit; for a student only the outcome commit that seals the episode's trace, "
          "since its retrieval happens inside the gateway call. The arm's checkpoint work "
          "(skill-file maintainer, proposer, validation replays) runs between episodes and is not in wall. Time per success is "
          "the cell's total wall divided by its successes. Time to first action was not recorded when this grid ran (the field "
          "was added to the runner on 2026-09-20 16:20 JST and is carried by every run started after that); where a "
          "condition has it, the median is shown with its n. Absolute values are load-dependent: the grid ran 36 to 52 "
          "workers on one machine against a memory API shared with another study, so the paired deltas below (same seed, "
          "fleet and episode, so the same time window within a cell) are what to read.\n"]
    for sc in sorted({x["scenario"] for x in rows}):
        md.append(f"\n## Latency, {sc}\n")
        md.append(table(LAT_HEAD, [_lat_row(f"{c} {COND_LABEL[c]}", [x for x in rows if x["scenario"] == sc and x["cond"] == c])
                                   for c in conds if any(x["scenario"] == sc and x["cond"] == c for x in rows)]))
    md.append("\n## Latency, paired wall-clock deltas in seconds (a − b), 95% bootstrap intervals over cells\n")
    lines = []
    for label, a, b in LAT_PAIRS:
        if a in conds and b in conds:
            lines.append([label, paired_delta(rows, a, b, metric="wall_s"), paired_delta(rows, a, b, post=False, metric="wall_s"),
                          paired_delta(rows, a, b, post=True, metric="wall_s")])
    if selfr and any(x["cond"] == "K0" for x in selfr):
        tagged = [{**x, "cond": "K0" if x["cond"] == "K0" else ("S1" if x["cond"] == "S1" and not REG.get(x["model"], {}).get("parent") else x["cond"])}
                  for x in selfr]
        lines.append(["K0 − S1 (skill-file student vs memory-trace student, third seeds)", paired_delta(tagged, "K0", "S1", metric="wall_s"),
                      paired_delta(tagged, "K0", "S1", post=False, metric="wall_s"), paired_delta(tagged, "K0", "S1", post=True, metric="wall_s")])
    md.append(table(["pair", "all episodes", "pre-change", "post-change"], lines))
    if novel:
        md.append("\n## Latency under change set B, post-change episodes of the changed classes\n")
        lines = []
        for sc in sorted({x["scenario"] for x in novel}):
            for c in conds:
                r = [x for x in novel if x["scenario"] == sc and x["cond"] == c and x["post"] and x["changed"] and not x["infra"]]
                if r:
                    s = sum(x["success"] for x in r)
                    lines.append([sc, f"{c} {COND_LABEL[c]}", len(r), f"{_quant([x['wall_s'] for x in r], .5):.1f}",
                                  f"{_quant([x['wall_s'] for x in r], .9):.1f}", f"{mean((x['attempts'] for x in r), 2):.2f}",
                                  f"{sum(x['wall_s'] for x in r) / s:.1f}" if s else "—"])
        md.append(table(["scenario", "condition", "n", "wall med", "wall p90", "attempts", "s/success"], lines))
    return "\n".join(md)


def a_vs_b(rows_a, rows_b, conds) -> str:
    """Post-change, changed-class episodes only: the same condition under the change the
    students trained on (A) and under one their weights never saw (B)."""
    lines = []
    for sc in sorted({x["scenario"] for x in rows_b}):
        for c in conds:
            ra = [x for x in rows_a if x["scenario"] == sc and x["cond"] == c and x["post"] and x["changed"]]
            rb = [x for x in rows_b if x["scenario"] == sc and x["cond"] == c and x["post"] and x["changed"]]
            if not rb:
                continue
            lines.append([sc, f"{c} {COND_LABEL[c]}", len(ra), rate(ra), rate(ra, lambda x: x["first_attempt_success"]),
                          len(rb), rate(rb), rate(rb, lambda x: x["first_attempt_success"])])
    return table(["scenario", "condition", "n (A)", "success A", "first-try A", "n (B)", "success B", "first-try B"], lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="distill-")
    ap.add_argument("--teacher-runs", default="gridv4-core-")
    ap.add_argument("--scenarios", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scen = set(args.scenarios) if args.scenarios else {((e.get("teacher") or {}).get("scenario")) for e in REG.values()} - {None}
    all_rows = load(args.runs, args.teacher_runs, scen)
    # Change set A is the study proper; B (see Scenario.change_variant) is the novel-change
    # evaluation and gets its own section, never pooled with A.
    rows = [x for x in all_rows if x["variant"] == "A"]
    novel = [x for x in all_rows if x["variant"] == "B"]
    selfr = [x for x in all_rows if x["variant"] == "self"]
    if not rows:
        print("no rows")
        return
    conds = [c for c in COND_ORDER if any(x["cond"] == c for x in rows)]
    md = [f"# Distillation report\n\n{len(rows)} episodes, scenarios {sorted(scen)}, "
          f"conditions {conds}. Seeds: {sorted({x['seed'] for x in rows})}."
          + (f" Plus {len(novel)} episodes under change set B (section at the end).\n" if novel else "\n")]

    # 1. headline per scenario x condition (fleet pooled), models pooled per condition
    for sc in sorted(scen):
        md.append(f"\n## {sc}\n\n" + headline(rows, sc, conds))

    # 2. the hypotheses as paired deltas
    md.append("\n## Hypotheses (paired by scenario × seed × fleet × episode; success-rate deltas)\n")
    md.append(hypotheses(rows, conds))

    # 2b. the same deltas on the first attempt. An episode allows three attempts with the
    # harness's feedback in between, and a student that has both regimes' fixes in its
    # weights reaches the right one on the second try without any memory; eventual success
    # then saturates and what memory buys shows up as getting there first (and cheaper).
    md.append("\n## Hypotheses, first attempt only (same pairing; first-try success deltas)\n")
    md.append(hypotheses(rows, conds, metric="first_attempt_success"))

    # 3. per-model students, one row per (model, condition)
    md.append("\n## Students individually\n")
    lines = []
    student_conds = ("S1", "S0", "S1p", "S1k", "S1n", "B1", "K0")
    for m, c in sorted({(x["model"], x["cond"]) for x in rows if x["cond"] in student_conds},
                       key=lambda mc: (mc[0], COND_ORDER.index(mc[1]))):
        r = [x for x in rows if x["model"] == m and x["cond"] == c]
        reg = REG.get(m.split("+notes=")[0].split("~")[0], {})
        ev = reg.get("eval") or {}
        lines.append([m[:44], c, len(r), rate(r), rate(r, lambda x: x["first_attempt_success"]),
                      per_success_cost(r), f"{ev.get('tool_agreement', '—')}", f"{ev.get('decision_agreement', '—')}",
                      f"{ev.get('parse_rate', '—')}", "pass" if (reg.get("default_gate") or {}).get("passed") else ("fail" if reg else "—")])
    md.append(table(["model", "cond", "n", "success", "first-try", "$/success", "tool-agr", "dec-agr", "parse", "default gate"], lines))

    # 3b. memory's first-try effect per student, before and after the change. The served
    # outcome record helps after the change and can hurt before it; how much of each a
    # student shows is a property of the student (and, it turns out, of its teacher).
    md.append("\n## Memory at inference per student (paired first-try deltas; S1 − S0 and S1 − S1p)\n")
    lines = []
    for base_alias in sorted({x["model"].split("+notes=")[0].split("~")[0] for x in rows if x["cond"] in ("S1", "S0", "S1p")}):
        mine = [x for x in rows if x["model"].split("+notes=")[0].split("~")[0] == base_alias and x["cond"] in ("S1", "S0", "S1p")]
        teacher = ((REG.get(base_alias) or {}).get("teacher") or {}).get("model", "")
        lines.append([base_alias, teacher,
                      paired_delta(mine, "S1", "S0", post=False, metric="first_attempt_success"),
                      paired_delta(mine, "S1", "S0", post=True, metric="first_attempt_success"),
                      paired_delta(mine, "S1", "S1p", post=False, metric="first_attempt_success"),
                      paired_delta(mine, "S1", "S1p", post=True, metric="first_attempt_success")])
    md.append(table(["student", "teacher", "S1−S0 pre", "S1−S0 post", "S1−S1p pre", "S1−S1p post"], lines))

    # 4. cost table at list price and provider price
    md.append("\n## Cost per success (list price; provider price for students in parentheses)\n")
    lines = []
    for c in conds:
        r = [x for x in rows if x["cond"] == c and not x["infra"]]
        s = sum(x["success"] for x in r)
        if not s:
            continue
        list_cost = sum(x["cost"] for x in r) / s
        prov = ""
        if c.startswith("S"):
            fam_prices = {k: config.STUDENT_PROVIDER_PRICES.get(v.get("family", ""), None) for k, v in REG.items()}
            pc = 0.0
            for x in r:
                fam = (REG.get(x["model"].split("+notes=")[0].split("~")[0]) or {}).get("family", "senselab-s1")
                p = config.STUDENT_PROVIDER_PRICES.get(fam)
                if p:
                    pc += (x["usage"]["prompt_tokens"] * p.input + x["usage"]["completion_tokens"] * p.output) / 1e6
                pc += (x.get("reflection_usage") or {}).get("cost_usd", 0.0)
            prov = f" (${pc / s:.4f})"
        lines.append([f"{c} {COND_LABEL[c]}", len(r), f"${list_cost:.4f}{prov}", f"{mean((x['tokens'] for x in r), 0):.0f}",
                      f"{mean((x['wall_ms'] / 1000 for x in r), 1)}s"])
    md.append(table(["condition", "n", "$/success", "tokens/ep", "wall/ep"], lines))

    # Latency: what the episode records carry (wall to conclusion, time in model calls, the
    # harness's memory time, attempts), medians and p90, and paired within-cell deltas.
    md.append(latency_section(rows, conds, novel, selfr))

    # 5. the novel regime change: change set B, which no training trace contains
    if novel:
        nconds = [c for c in COND_ORDER if any(x["cond"] == c for x in novel)]
        md.append(f"\n# Novel regime change (change set B)\n\n{len(novel)} episodes. Under A the students' training traces "
                  "contain both regimes, so a student with no memory can reach the post-change fix on a retry; under B "
                  "the post-change (class, fix) pairs occur nowhere in training, so what a student recovers after the "
                  "change is what its memory gives it.\n")
        for sc in sorted({x["scenario"] for x in novel}):
            md.append(f"\n## {sc} (B)\n\n" + headline(novel, sc, nconds))
        md.append("\n## Post-change, changed classes: A (trained-on change) vs B (novel change)\n")
        md.append(a_vs_b(rows, novel, nconds))
        md.append("\n## Hypotheses under B (success-rate deltas)\n")
        md.append(hypotheses(novel, nconds))
        md.append("\n## Hypotheses under B, first attempt only\n")
        md.append(hypotheses(novel, nconds, metric="first_attempt_success"))

    # 6. self-retrained students against their parents, on a third set of seeds
    if selfr:
        md.append(f"\n# Self-retrained students\n\n{len(selfr)} episodes on seeds "
                  f"{sorted({x['seed'] for x in selfr})}. A v3 student is trained on its parent's own sealed traces from "
                  "the parent's evaluation cells (the student's decisions, the outcomes the harness committed), with no "
                  "frontier model in the loop; parent and child are compared on seeds neither has seen.\n")
        md.append("\n## Parent vs child\n")
        lines = []
        for sc in sorted({x["scenario"] for x in selfr}):
            for m in sorted({x["model"] for x in selfr if x["scenario"] == sc}):
                for c in COND_ORDER:
                    r = [x for x in selfr if x["scenario"] == sc and x["model"] == m and x["cond"] == c]
                    if not r:
                        continue
                    role = "child" if REG.get(m, {}).get("parent") else ("parent" if any(
                        REG.get(k, {}).get("parent") == m for k in REG) else "")
                    lines.append([sc, f"{m} {role}".strip(), c, len(r), rate(r), rate(r, lambda x: x["first_attempt_success"]),
                                  rate(r, lambda x: x["escalated"]), per_success_cost(r), recovery(r)])
        md.append(table(["scenario", "model", "cond", "n", "success", "first-try", "escalated", "$/success", "pre→post(changed)"], lines))
        md.append("\n## Child − parent (paired by scenario × seed × fleet × episode)\n")
        lines = []
        for child, e in sorted(REG.items()):
            parent = e.get("parent")
            if not parent:
                continue
            for c in ("S1", "S0"):
                # Tag rows so paired_delta can see child and parent as two "conditions".
                tagged = []
                for x in selfr:
                    if x["cond"] == c and x["model"] in (child, parent):
                        tagged.append({**x, "cond": "child" if x["model"] == child else "parent"})
                if tagged:
                    lines.append([f"{child} − {parent}, {c}", paired_delta(tagged, "child", "parent"),
                                  paired_delta(tagged, "child", "parent", post=True),
                                  paired_delta(tagged, "child", "parent", metric="first_attempt_success"),
                                  paired_delta(tagged, "child", "parent", post=True, metric="first_attempt_success")])
        md.append(table(["comparison", "success", "success post-change", "first-try", "first-try post-change"], lines))

        # H7, second half: the student distilled from the skill-file arm's traces, on the same
        # third seeds, against the parent v2 SenseLab student and the frontier there.
        k0 = [x for x in selfr if x["cond"] in ("K0", "K0n")]
        if k0:
            md.append("\n## Skill-file student (K0) on the same seeds\n\nK0 is a student distilled from the skill-file "
                      "arm's traces (both frontier teachers, seeds 101/113/127, 12 cells, the same trace count as a v2 "
                      "student); its training seeds are disjoint from these. Paired against the v2 SenseLab student "
                      "(the parent in the table above) under the same arm, and against the frontier.\n")
            lines = []
            for sc in sorted({x["scenario"] for x in k0}):
                parents = sorted({e.get("parent") for a, e in REG.items()
                                  if e.get("parent") and (e.get("teacher") or {}).get("scenario") == sc})
                for k0_alias in sorted({x["model"] for x in k0 if x["scenario"] == sc}):
                    for k_cond, s_cond, f_cond in (("K0", "S1", "F1"), ("K0n", "S0", "F0")):
                        for other_model, other_cond in [(p, s_cond) for p in parents] + [(None, f_cond)]:
                            tagged = []
                            for x in selfr:
                                if x["scenario"] != sc:
                                    continue
                                if x["model"] == k0_alias and x["cond"] == k_cond:
                                    tagged.append({**x, "cond": "k0"})
                                elif x["cond"] == other_cond and (other_model is None or x["model"] == other_model):
                                    tagged.append({**x, "cond": "other"})
                            if any(x["cond"] == "other" for x in tagged) and any(x["cond"] == "k0" for x in tagged):
                                label = f"{k0_alias} {k_cond} − {other_model or 'frontier'} {other_cond}"
                                lines.append([sc, label, paired_delta(tagged, "k0", "other"),
                                              paired_delta(tagged, "k0", "other", post=True),
                                              paired_delta(tagged, "k0", "other", metric="first_attempt_success"),
                                              paired_delta(tagged, "k0", "other", post=True, metric="first_attempt_success")])
            md.append(table(["scenario", "comparison", "success", "success post-change", "first-try", "first-try post-change"],
                            lines))

    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    _figures(rows, conds)
    if novel:
        _figures(novel, [c for c in COND_ORDER if any(x["cond"] == c for x in novel)], suffix="_novel")


def _figures(rows, conds, suffix: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"F1": "#1f77b4", "F0": "#7f7f7f", "D1": "#17becf", "D0": "#bcbd22", "S1": "#d62728", "S1n": "#ff7f0e",
              "S1p": "#e377c2", "S1k": "#8c564b", "S0": "#9467bd", "B1": "#2ca02c", "K1": "#000000", "K0": "#555555"}
    scen = sorted({x["scenario"] for x in rows})
    fig, axes = plt.subplots(1, len(scen), figsize=(6 * len(scen), 4), squeeze=False)
    for ax, sc in zip(axes[0], scen):
        for c in conds:
            r = [x for x in rows if x["scenario"] == sc and x["cond"] == c and not x["infra"]]
            if not r:
                continue
            by_ep = defaultdict(list)
            for x in r:
                by_ep[x["episode"] // 5 * 5].append(x["success"])
            xs = sorted(by_ep)
            ax.plot(xs, [sum(by_ep[e]) / len(by_ep[e]) for e in xs], label=c, color=colors.get(c), lw=2 if c in ("F1", "S1") else 1.2)
        ax.axvline(20, color="k", ls=":", lw=0.8)
        ax.set_title(sc); ax.set_xlabel("episode (5-episode bins)"); ax.set_ylabel("success rate"); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / f"fig_d1_learning_curves{suffix}.png", dpi=150); plt.close(fig)

    # cost vs success
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in conds:
        r = [x for x in rows if x["cond"] == c and not x["infra"]]
        s = sum(x["success"] for x in r)
        if not s:
            continue
        ax.scatter(sum(x["cost"] for x in r) / s, s / len(r), color=colors.get(c), s=60, label=c)
        ax.annotate(c, (sum(x["cost"] for x in r) / s, s / len(r)), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("$ per success (list price, log)"); ax.set_ylabel("success rate")
    ax.set_title("Cost of a solved task, by condition")
    fig.tight_layout(); fig.savefig(OUT / f"fig_d2_cost_vs_success{suffix}.png", dpi=150); plt.close(fig)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
