"""fleet-disjoint: partial discoveries spread across a fleet; held-out tasks need them combined.

The experiment that can kill the composition idea (recursive-learning plan, Phase 0). Three
agents share one workspace. Acme's payments desk runs on three *fictional* policies the base
model cannot know:

  A  ``TX-7``   a Vantix gateway timeout may still settle late: a ``hold`` must come
                immediately before the ``retry``, or the customer is charged twice.
  B  ``IK-15``  idempotency keys expire after 15 minutes: ``rotate_key`` before ``retry``.
  C  ``LR-3``   when the ledger disagrees with the gateway the ledger is authoritative:
                ``reconcile_ledger`` is the first step, and ``refund`` is never the answer.

Each policy is also *conditional*: applying it when its trigger is absent fails too
(``LR-0``, ``IK-DUP``, ``TX-HOLD-UNNEEDED``), so "always do everything" is not a strategy.

Discovery phase (first ``discovery_share`` of the episodes): agent 1 only ever sees A cases,
agent 2 only B, agent 3 only C. A failed attempt here is explained in full by the on-call
human — the policy text is the partial discovery that agent takes away.

Composition phase (the rest): cases that trigger two or all three policies, dealt round-robin.
Failure feedback names only the error code and one terse line; the policies are not restated.
One trap is deliberate: A and B look independent but a hold freezes the key, so the naive
concatenation of the two discoveries (``hold, rotate_key, retry``) fails with ``KEY-FROZEN``
and the fleet has to learn the interaction from that one observation.

What the report reads off the records: success on ``tags.phase == "composition"`` by rule
set, ``flags.trap_naive_concat``, attempts to first success per rule set, and — for the
memory arms — whether the lesson that carried the composition was written by a peer.
"""

from __future__ import annotations

import random
from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec, finish_params

STEPS = ["reconcile_ledger", "rotate_key", "hold", "retry", "refund", "mark_paid"]
RULES = ("A", "B", "C")
COMBOS = ("AB", "AC", "BC", "ABC")

POLICY = {
    "A": ("TX-7", "Vantix confirms timed-out captures late. Policy TX-7: on a gateway timeout, place a "
                  "settlement hold immediately before the retry; a retry without the hold charges the "
                  "customer twice."),
    "B": ("IK-15", "Vantix idempotency keys expire after 15 minutes. Policy IK-15: when the key is older "
                   "than 15 minutes, rotate_key before the retry; the old key is rejected as IK-EXPIRED."),
    "C": ("LR-3", "The ledger is authoritative over the gateway. Policy LR-3: when the two disagree, "
                  "reconcile_ledger must be the first step, before anything touches the gateway; a "
                  "refund is never the resolution for a ledger disagreement."),
}

MERCHANTS = ["Northwind Outfitters", "Bluefin Books", "Carrow & Sons", "Pikeway Fitness", "Lumen Decor",
             "Halcyon Travel", "Ridgeback Tools", "Saffron Table"]


def correct_steps(rules: str) -> list[str]:
    seq: list[str] = []
    if "C" in rules:
        seq.append("reconcile_ledger")
    if "B" in rules:
        seq.append("rotate_key")
    if "A" in rules:
        seq.append("hold")
    seq.append("retry")
    return seq


def violation(steps: list[str], rules: str) -> tuple[str, str, dict[str, Any]] | None:
    """First constraint the step order breaks: ``(code, terse_reason, flags)``; ``None`` if clean."""
    a, b, c = ("A" in rules), ("B" in rules), ("C" in rules)
    idx = {s: i for i, s in enumerate(steps)}
    if len(set(steps)) != len(steps) or any(s not in STEPS for s in steps):
        return "SEQ", "unknown or repeated steps in the resolution", {}
    if "refund" in idx:
        return "REFUND-BLOCKED", "a refund is not a resolution path for a stuck charge", {"naive_refund": True}
    if "mark_paid" in idx:
        return "NOT-SETTLED", "the charge was marked paid without a settled capture", {"naive_mark_paid": True}
    if "retry" not in idx or idx["retry"] != len(steps) - 1:
        return "NO-RETRY", "the case was left unresolved: retry must be the final step", {}
    if c and idx.get("reconcile_ledger") != 0:
        return "LR-3", "ledger disagreement was not reconciled before touching the gateway", {}
    if not c and "reconcile_ledger" in idx:
        return "LR-0", "nothing to reconcile: reconciliation on a consistent ledger opened a spurious dispute", {}
    if a and b and "hold" in idx and "rotate_key" in idx and idx["hold"] < idx["rotate_key"]:
        return "KEY-FROZEN", "a settlement hold freezes the idempotency key; the key rotated after the hold was rejected", \
            {"trap_naive_concat": True}
    if b and ("rotate_key" not in idx or idx["rotate_key"] > idx["retry"]):
        return "IK-EXPIRED", "retry rejected: idempotency key older than 15 minutes", {}
    if not b and "rotate_key" in idx:
        return "IK-DUP", "rotating a live key orphaned the original attempt: duplicate charge risk", {}
    if a and ("hold" not in idx or idx["hold"] != idx["retry"] - 1):
        return "TX-7", "retry after a timeout without a settlement hold immediately before it: double charge", {}
    if not a and "hold" in idx:
        return "TX-HOLD-UNNEEDED", "a settlement hold on a declined charge stalled the case for 24 hours", {}
    return None


class DisjointFleetScenario(Scenario):
    name = "fleet-disjoint"
    role = "You are a payments operations agent at Acme, resolving stuck customer charges on the Vantix gateway."
    agent_base = "payops-agent"
    fleet_size = 3
    discovery_share = 0.5
    procedural = True   # the repair arm proposes procedures, not facts
    judge_rubric = ("Acme payments policies TX-7 (timeout: hold immediately before retry), IK-15 (key older "
                    "than 15 min: rotate_key before retry, and before any hold) and LR-3 (ledger disagreement: "
                    "reconcile_ledger first, never refund). The correct resolution is the ordered step list "
                    "that satisfies every policy whose trigger is present and none whose trigger is absent.")

    tools = [
        ToolSpec("inspect_case", "Read the case: gateway status, idempotency key age, ledger vs gateway state.",
                 {"type": "object", "properties": {"case": {"type": "string"}}, "required": ["case"]}),
        ToolSpec("resolve_case", "Resolve the case by running the given ordered steps against Vantix and the ledger.",
                 finish_params({
                     "case": {"type": "string"},
                     "steps": {"type": "array", "items": {"type": "string", "enum": STEPS},
                               "description": "Ordered steps to run."},
                     "rationale": {"type": "string"},
                 }, ["case", "steps", "rationale"]), terminal=True),
        ToolSpec("escalate", "Hand the case to a human when you cannot resolve it safely.",
                 finish_params({"case": {"type": "string"}, "reason": {"type": "string"}},
                               ["case", "reason"]), terminal=True),
    ]

    def build(self) -> None:
        rng = self.rng
        n = self.episodes
        n_disc = max(self.fleet_size, int(round(n * self.discovery_share)))
        self.n_discovery = min(n_disc, n)
        # Composition cases: every combo about equally often, in a shuffled order that is
        # deterministic per seed.
        combos: list[str] = []
        while len(combos) < n - self.n_discovery:
            block = list(COMBOS)
            rng.shuffle(block)
            combos.extend(block)
        self.combos = combos[: n - self.n_discovery]
        self.cases: list[dict[str, Any]] = []
        for ep in range(n):
            if ep < self.n_discovery:
                rules = RULES[ep % self.fleet_size]      # agent k (round-robin) sees rule k only
                phase = "discovery"
            else:
                rules = self.combos[ep - self.n_discovery]
                phase = "composition"
            self.cases.append({
                "id": f"PAY-{4100 + ep * 7 + rng.randint(0, 5)}",
                "rules": rules, "phase": phase,
                "merchant": rng.choice(MERCHANTS),
                "amount": f"${rng.randint(18, 940)}.{rng.randint(0, 99):02d}",
                "key_age": rng.randint(16, 95) if "B" in rules else rng.randint(1, 9),
                "gateway": "timeout" if "A" in rules else "soft decline (code 05, retryable)",
                "ledger": ("CAPTURED (disagrees with the gateway)" if "C" in rules
                           else "PENDING (agrees with the gateway)"),
            })

    def seed_entries(self) -> list[tuple[str, str, float]]:
        return self.distractor_entries()   # nothing about the policies is in memory beforehand

    def agent_for(self, ep: int) -> str:
        # Discovery is by rule, not by transfer protocol: agent k gets rule k. The composition
        # phase is dealt round-robin, which for this scenario is the same formula.
        return f"{self.agent_base}-{ep % self.fleet_size + 1}"

    def task(self, ep: int) -> Task:
        c = self.cases[ep]
        prompt = (f"Case {c['id']}: a {c['amount']} charge for {c['merchant']} is stuck. Inspect the case, then "
                  f"resolve it with resolve_case using the correct ordered steps under Acme payments policy, "
                  f"or escalate if you cannot do so safely.")
        truth = {"case": c["id"], "rules": c["rules"], "good": correct_steps(c["rules"]), "phase": c["phase"]}
        tags = {"phase": c["phase"], "rules": c["rules"], "composition": c["phase"] == "composition",
                "n_rules": len(c["rules"])}
        return Task(ep, prompt, truth, self.agent_for(ep), tags)

    def _facts(self, c: dict[str, Any]) -> str:
        return (f"Case {c['id']} ({c['merchant']}, {c['amount']}). Gateway status: {c['gateway']}. "
                f"Idempotency key age: {c['key_age']} minutes. Ledger: {c['ledger']}.")

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        c = self.cases[task.episode]
        if name == "inspect_case":
            return StepResult(self._facts(c))
        if name == "escalate":
            return StepResult("Escalated, but the payments lead pushed back: this case is covered by written "
                              "policy and was expected to be resolved by the agent.", terminal=True,
                              success=False, severity="minor_failure",
                              flags={"unnecessary_escalation": True}, answer="escalate")
        if name != "resolve_case":
            return StepResult(f"Unknown tool {name}")
        steps = [str(x) for x in args.get("steps", [])]
        answer = ",".join(steps)
        if str(args.get("case", "")) and str(args.get("case")) != t["case"]:
            return StepResult(f"Resolution rejected: the request was for {t['case']}, you targeted {args.get('case')}.",
                              terminal=True, success=False, severity="failure", answer=answer)
        v = violation(steps, t["rules"])
        if v is None:
            return StepResult(f"Case {t['case']} resolved: retry settled, ledger and gateway agree, customer notified.",
                              terminal=True, success=True, answer=answer)
        code, reason, flags = v
        if t["phase"] == "discovery" and code in ("TX-7", "IK-EXPIRED", "LR-3", "LR-0", "IK-DUP", "TX-HOLD-UNNEEDED"):
            rule = t["rules"]
            pol_code, pol_text = POLICY[rule]
            text = (f"Resolution of {t['case']} FAILED ({code}): {reason}. The on-call payments lead explains: "
                    f"{pol_text} Step order used: {' -> '.join(steps) or '(none)'}.")
        else:
            text = (f"Resolution of {t['case']} FAILED ({code}): {reason}. Step order used: "
                    f"{' -> '.join(steps) or '(none)'}.")
        severity = "critical_failure" if code in ("TX-7", "KEY-FROZEN", "IK-DUP", "REFUND-BLOCKED") else "failure"
        return StepResult(text, terminal=True, success=False, severity=severity,
                          flags={"code": code, **flags}, answer=answer)
