"""triage: support tickets under a written policy with exceptions learnable only from outcomes."""

from __future__ import annotations

from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec, finish_params

ISSUES = ["duplicate_charge", "not_delivered", "damaged", "changed_mind", "subscription_dispute"]
TIERS = ["free", "pro", "enterprise"]


class TriageScenario(Scenario):
    name = "triage"
    agent_base = "triage-agent"
    role = "You are the billing-support triage agent at Acme."

    tools = [
        ToolSpec("decide", "Record your decision for the ticket.",
                 finish_params({
                     "ticket": {"type": "string"},
                     "action": {"type": "string", "enum": ["refund", "escalate", "deny"]},
                     "reason": {"type": "string"},
                 }, ["ticket", "action", "reason"]), terminal=True),
    ]

    def build(self) -> None:
        rng = self.rng
        self.tickets = []
        for ep in range(self.episodes):
            amount = rng.choice([12, 29, 45, 49, 60, 120, 199, 250, 480])
            tier = rng.choice(TIERS)
            issue = rng.choice(ISSUES)
            age = rng.choice([1, 3, 7, 14, 25, 35, 60])
            # ensure exceptions appear often enough to be learnable
            if ep % 3 == 1:
                tier = "enterprise"
            if ep % 4 == 2:
                age = rng.choice([35, 60])
            self.tickets.append({"id": f"T-{7000 + ep}", "amount": amount, "tier": tier, "issue": issue, "age": age})

    @staticmethod
    def true_action(t: dict[str, Any]) -> str:
        # hidden rules first
        if t["tier"] == "enterprise":
            return "escalate"
        if t["age"] > 30:
            return "deny"
        # written policy
        if t["amount"] >= 200:
            return "escalate"
        if t["amount"] < 50 and t["issue"] in ("duplicate_charge", "not_delivered"):
            return "refund"
        return "deny"

    @staticmethod
    def written_action(t: dict[str, Any]) -> str:
        if t["amount"] >= 200:
            return "escalate"
        if t["amount"] < 50 and t["issue"] in ("duplicate_charge", "not_delivered"):
            return "refund"
        return "deny"

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = [
            ("policy-refunds", "Refund policy: auto-refund when the amount is under $50 and the issue is a "
                               "duplicate charge or an item not delivered.", 0.9),
            ("policy-escalation", "Escalation policy: escalate any ticket with an amount of $200 or more.", 0.9),
            ("policy-default", "Default: deny with an explanation when no refund or escalation rule applies.", 0.9),
        ]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        t = self.tickets[ep]
        prompt = (f"Ticket {t['id']}: customer on the {t['tier']} plan reports '{t['issue'].replace('_', ' ')}' "
                  f"for a charge of ${t['amount']}; the charge is {t['age']} days old. Decide refund, "
                  f"escalate or deny.")
        return Task(ep, prompt, {"ticket": t, "expected": self.true_action(t),
                                 "written": self.written_action(t)}, self.agent_for(ep),
                    {"exception": self.true_action(t) != self.written_action(t)})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        if name != "decide":
            return StepResult(f"Unknown tool {name}")
        act = str(args.get("action", "")).lower()
        t = task.truth
        if act == t["expected"]:
            return StepResult("Decision accepted by the supervisor review.", terminal=True, success=True, answer=act)
        tk = t["ticket"]
        sev = "critical_failure" if (act == "refund" and t["expected"] != "refund") else "failure"
        hint = {"escalate": "this account type must always go to a human",
                "deny": "charges this old are outside the refund window",
                "refund": "this met the refund criteria"}[t["expected"]]
        return StepResult(f"Decision REVERSED by the supervisor: correct action for {tk['id']} was "
                          f"'{t['expected']}' ({hint}).", terminal=True, success=False, severity=sev,
                          flags={"followed_written_policy": act == t["written"] and t["expected"] != t["written"],
                                 "wrong_refund_usd": tk["amount"] if act == "refund" and t["expected"] != "refund" else 0},
                          answer=act)
