"""drift-fact and drift-tool: the world changes at episode k; seeded knowledge goes stale."""

from __future__ import annotations

from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec, finish_params

SERVICES = ["checkout", "inventory", "payments", "shipping", "notifications", "pricing"]
ATTRS = {
    "database": ["PostgreSQL 15", "PostgreSQL 16", "PostgreSQL 17", "MySQL 8.4"],
    "on-call day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "primary region": ["us-east-1", "us-west-2", "eu-central-1", "ap-northeast-1"],
    "API base path": ["/v1", "/v2", "/v3", "/api/v2"],
    "message broker": ["Kafka", "RabbitMQ", "Pub/Sub", "SQS"],
}


class DriftFactScenario(Scenario):
    name = "drift-fact"
    agent_base = "knowledge-agent"
    role = "You are the platform knowledge agent at Acme; engineers ask you operational questions."
    drift_at = 8

    tools = [
        ToolSpec("answer", "Submit your answer to the engineer's question.",
                 finish_params({"answer": {"type": "string", "description": "Short, specific answer."}},
                               ["answer"]), terminal=True),
    ]

    def build(self) -> None:
        rng = self.rng
        self.services = rng.sample(SERVICES, 4)
        self.attrs = list(ATTRS)
        self.truth: dict[tuple[str, str], str] = {}
        for s in self.services:
            for a in self.attrs:
                self.truth[(s, a)] = rng.choice(ATTRS[a])
        self.initial = dict(self.truth)
        pairs = [(s, a) for s in self.services for a in self.attrs]
        self.changed = rng.sample(pairs, 5)
        self.new_values: dict[tuple[str, str], str] = {}
        for s, a in self.changed:
            opts = [v for v in ATTRS[a] if v != self.initial[(s, a)]]
            self.new_values[(s, a)] = rng.choice(opts)
        # question schedule: changed pairs come up more often after the drift
        sched = []
        for ep in range(self.episodes):
            if ep >= self.drift_at and ep % 3 != 2:
                sched.append(self.changed[ep % len(self.changed)])
            else:
                sched.append(pairs[(ep * 7) % len(pairs)])
        self.sched = sched

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = []
        for (s, a), v in self.initial.items():
            out.append((f"fact-{s}-{a.replace(' ', '-')}", f"The {a} for the {s} service is {v}.", 0.8))
        return out + self.distractor_entries()

    def current(self, ep: int, key: tuple[str, str]) -> str:
        if ep >= self.drift_at and key in self.new_values:
            return self.new_values[key]
        return self.initial[key]

    def task(self, ep: int) -> Task:
        s, a = self.sched[ep]
        prompt = f"Question from an engineer (ticket ENG-{3100 + ep}): what is the {a} for the {s} service?"
        stale = self.initial[(s, a)] if (ep >= self.drift_at and (s, a) in self.new_values) else None
        return Task(ep, prompt, {"service": s, "attr": a, "expected": self.current(ep, (s, a)),
                                 "stale": stale}, self.agent_for(ep), {"post_drift": ep >= self.drift_at})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        if name != "answer":
            return StepResult(f"Unknown tool {name}")
        ans = str(args.get("answer", ""))
        t = task.truth
        exp = t["expected"]
        ok = exp.lower() in ans.lower()
        used_stale = bool(t["stale"]) and t["stale"].lower() in ans.lower() and not ok
        if ok:
            return StepResult("The engineer confirmed the answer matched the current configuration.",
                              terminal=True, success=True, answer=ans)
        fb = (f"The engineer reports this is wrong: the {t['attr']} for {t['service']} is currently "
              f"{exp}" + (" (it changed recently)." if t["stale"] else "."))
        return StepResult(fb, terminal=True, success=False, severity="failure",
                          flags={"stale": used_stale}, answer=ans)


class DriftToolScenario(Scenario):
    name = "drift-tool"
    agent_base = "paging-agent"
    role = "You are the incident-paging agent at Acme."
    drift_at = 8

    tools = [
        ToolSpec("page_oncall", "Page the on-call engineer for a service. Field names must match the current API.",
                 finish_params({
                     "service": {"type": "string"},
                     "severity": {"type": "string", "description": "legacy field; may be rejected"},
                     "priority": {"type": "integer", "description": "1 (highest) to 4 (lowest)"},
                     "channel": {"type": "string", "description": "e.g. pager, slack"},
                     "message": {"type": "string"},
                 }, ["service", "message"]), terminal=True),
    ]

    SEV = {"sev1": 1, "sev2": 2, "sev3": 3, "sev4": 4}

    def build(self) -> None:
        rng = self.rng
        self.incidents = []
        impacts = [("all customers cannot check out", "sev1"), ("one region degraded", "sev2"),
                   ("internal dashboard slow", "sev3"), ("cosmetic logging noise", "sev4"),
                   ("payments failing for 20% of users", "sev1"), ("nightly report delayed", "sev3")]
        for ep in range(self.episodes):
            impact, sev = impacts[(ep * 5) % len(impacts)]
            self.incidents.append((rng.choice(SERVICES), impact, sev))

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = [
            ("api-doc-page-oncall", "page_oncall API (v1): fields service, severity (sev1..sev4), message.", 0.8),
            ("policy-severity-mapping", "Severity policy: customer-facing outage = sev1; regional degradation = sev2; "
                                        "internal tooling = sev3; cosmetic = sev4.", 0.9),
        ]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        s, impact, sev = self.incidents[ep]
        prompt = (f"Incident INC-{5200 + ep} on {s}: {impact}. Page the on-call engineer with the right "
                  f"severity/priority using page_oncall.")
        return Task(ep, prompt, {"service": s, "sev": sev, "v2": ep >= self.drift_at},
                    self.agent_for(ep), {"post_drift": ep >= self.drift_at})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        if name != "page_oncall":
            return StepResult(f"Unknown tool {name}")
        t = task.truth
        want_p = self.SEV[t["sev"]]
        if t["v2"]:
            if "severity" in args or "priority" not in args or "channel" not in args:
                return StepResult("HTTP 400 from paging API: unknown field 'severity'; required fields are "
                                  "priority (integer 1-4) and channel. The API changed to v2.", terminal=True,
                                  success=False, severity="failure", flags={"tool_shape_miss": True},
                                  answer=str(args))
            ok = int(args.get("priority", 0)) == want_p
            if ok:
                return StepResult("Page delivered; on-call acknowledged.", terminal=True, success=True, answer=str(args))
            return StepResult(f"Page delivered but on-call flagged wrong priority: incident should be priority {want_p}.",
                              terminal=True, success=False, severity="minor_failure", answer=str(args))
        sev = str(args.get("severity", "")).lower()
        if not sev and "priority" in args:
            sev = f"sev{args['priority']}"
        if sev == t["sev"]:
            return StepResult("Page delivered; on-call acknowledged.", terminal=True, success=True, answer=str(args))
        return StepResult(f"Page delivered but on-call flagged wrong severity: incident should be {t['sev']}.",
                          terminal=True, success=False, severity="minor_failure", answer=str(args))
