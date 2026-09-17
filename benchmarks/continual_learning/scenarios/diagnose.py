"""diagnose: a never-repeating incident each episode, generated from a hidden causal model.

Nothing in memory ever contains the answer to the current incident. What memory can hold
is *heuristics* the agent wrote after earlier incidents: some right, some wrong. The
agent has diagnostic tools and one shot per attempt at a fix. From episode 16 the surface
vocabulary changes (metric names, service naming) so lexical recall of past incidents
cannot help — only generalized, validated rules can.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec, finish_params

FIXES = ["increase_pool_size", "rollback_deploy", "scale_out", "restart_pods", "set_cache_ttl",
         "enable_circuit_breaker", "rotate_logs", "rate_limit"]

# The hidden causal model. Each rule's evidence matches a textbook pattern, but in 4 of the
# 6 rules the remediation that works at Acme is NOT the textbook one ("textbook" below),
# because of infrastructure idiosyncrasies the agent can only discover from outcomes. A
# model with no experience will pick the textbook fix and fail; a model that has learned
# Acme's quirks — and trusts the right lessons — fixes it first time. Failure feedback is
# only "symptoms persist", so the agent cannot read the answer off the environment.
RULES: dict[str, dict[str, Any]] = {
    "pool": {"cause": "database connection pool starvation under load (the DB has a hard connection cap; "
                      "adding connections or pods makes contention worse, shedding load is what works)",
             "fix": "rate_limit", "textbook": "increase_pool_size", "wrong": ["increase_pool_size", "scale_out"],
             "metrics": {"p99_latency_ms": (1800, 4200), "error_rate": (0.002, 0.01), "cpu": (0.3, 0.5),
                         "db_pool_wait_ms": (400, 1500), "mem": (0.4, 0.6)},
             "logs": ["timeout acquiring connection from pool", "pool wait exceeded", "slow query queue"],
             "changes": ["traffic +40% after marketing email"]},
    "deploy": {"cause": "regression in the latest deploy", "fix": "rollback_deploy", "textbook": "rollback_deploy",
               "wrong": ["restart_pods", "scale_out"],
               "metrics": {"p99_latency_ms": (200, 400), "error_rate": (0.12, 0.35), "cpu": (0.3, 0.5),
                           "db_pool_wait_ms": (5, 20), "mem": (0.4, 0.6)},
               "logs": ["NullPointerException in OrderMapper", "500 on /v2/quote", "unhandled exception"],
               "changes": ["deploy v{v} 18 minutes ago"]},
    "cache": {"cause": "unbounded in-process cache growth", "fix": "set_cache_ttl", "textbook": "set_cache_ttl",
              "wrong": ["scale_out", "restart_pods"],
              "metrics": {"p99_latency_ms": (300, 600), "error_rate": (0.01, 0.03), "cpu": (0.4, 0.6),
                          "db_pool_wait_ms": (5, 20), "mem": (0.88, 0.97)},
              "logs": ["OOMKilled container restarted", "heap near limit", "cache entries: 9.8M"],
              "changes": ["no deploys in 6 days"]},
    "capacity": {"cause": "CPU saturation is pool starvation: Acme workers busy-wait for DB connections, so "
                          "the pool, not replica count, is the bottleneck (scaling out adds contention)",
                 "fix": "increase_pool_size", "textbook": "scale_out", "wrong": ["scale_out", "rate_limit"],
                 "metrics": {"p99_latency_ms": (900, 1600), "error_rate": (0.01, 0.04), "cpu": (0.9, 0.99),
                             "db_pool_wait_ms": (5, 30), "mem": (0.5, 0.7), "queue_depth": (800, 3000)},
                 "logs": ["request queue saturated", "worker busy"],
                 "changes": ["traffic 2.3x baseline (product launch)"]},
    "dependency": {"cause": "retry storm against a degraded dependency, introduced by the retry policy in the "
                            "recent deploy (a circuit breaker does not stop the new client's retries)",
                   "fix": "rollback_deploy", "textbook": "enable_circuit_breaker",
                   "wrong": ["enable_circuit_breaker", "scale_out"],
                   "metrics": {"p99_latency_ms": (2500, 6000), "error_rate": (0.08, 0.2), "cpu": (0.2, 0.4),
                               "db_pool_wait_ms": (5, 20), "mem": (0.4, 0.6), "upstream_latency_ms": (3000, 8000)},
                   "logs": ["upstream timeout calling {dep}", "circuit open? no", "retry storm to {dep}"],
                   "changes": ["deploy v{v} 2 days ago (new HTTP client retry policy)", "{dep} status page: degraded"]},
    "disk": {"cause": "disk filled by log volume; Acme's log shipper holds deleted file handles, so rotation "
                      "frees nothing until the pods restart",
             "fix": "restart_pods", "textbook": "rotate_logs", "wrong": ["rotate_logs", "scale_out"],
             "metrics": {"p99_latency_ms": (400, 900), "error_rate": (0.05, 0.15), "cpu": (0.3, 0.5),
                         "db_pool_wait_ms": (5, 20), "mem": (0.4, 0.6), "disk": (0.96, 0.995)},
             "logs": ["ENOSPC: no space left on device", "log volume 40x baseline", "write failed"],
             "changes": ["debug logging enabled yesterday"]},
}

# Regime change: infrastructure matured. The DB moved to a proxy with no hard connection cap
# (pool starvation is now fixed the textbook way — the pure unlearning case) and the log
# shipper became a per-pod sidecar with a fixed volume budget (disk fills when a pod is
# over-subscribed; spreading the load across more pods is what works — rotation and restarts
# both just buy minutes). A memory full of hard-won Acme quirks is now wrong on both rules,
# and on ``disk`` the textbook answer is still wrong too, so nobody wins by forgetting alone.
RULE_CHANGES: dict[str, dict[str, str]] = {
    "pool": {"fix": "increase_pool_size", "cause": "database connection pool starvation under load (the new DB proxy has no "
                                                    "hard cap; growing the pool is the fix, shedding load only hides it)"},
    "disk": {"fix": "scale_out", "cause": "disk filled by log volume; since the log shipper became a per-pod sidecar with a "
                                          "fixed volume budget, an over-subscribed pod fills its disk and only spreading the "
                                          "load across more pods clears it — rotation and restarts buy minutes"},
}

SERVICES = ["quote", "ledger", "search", "mailer", "catalog", "reports", "auth", "gateway", "wallet", "feed"]
DEPS = ["tax-service", "geo-lookup", "fraud-score", "fx-rates"]
DISTRACTORS = [("cpu", (0.65, 0.78)), ("gc_pause_ms", (120, 260)), ("open_files", (7000, 9000))]

RENAME = {"p99_latency_ms": "latency_p99", "error_rate": "http_5xx_ratio", "cpu": "cpu_util",
          "db_pool_wait_ms": "pool_acquire_wait", "mem": "heap_util", "queue_depth": "backlog",
          "upstream_latency_ms": "dep_rtt_p99", "disk": "volume_util", "gc_pause_ms": "gc_stw",
          "open_files": "fd_count"}


class DiagnoseScenario(Scenario):
    name = "diagnose"
    agent_base = "incident-agent"
    role = "You are the incident-response agent at Acme."
    uses_judge = True
    shift_at = 15
    judge_rubric = ("Grade the agent's explanation of the incident against the true root cause. "
                    "3 = names the true cause and the mechanism correctly; 2 = right cause, vague or "
                    "partly wrong mechanism; 1 = adjacent/partially right; 0 = wrong cause or no explanation.")

    tools = [
        ToolSpec("get_metrics", "Fetch the current key metrics for a service.",
                 {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
        ToolSpec("get_logs", "Fetch recent error-level log lines for a service.",
                 {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
        ToolSpec("get_recent_changes", "List recent deploys, config changes and traffic events for a service.",
                 {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
        ToolSpec("apply_fix", "Apply ONE remediation. You will be told whether the symptoms clear.",
                 finish_params({
                     "service": {"type": "string"},
                     "action": {"type": "string", "enum": FIXES},
                     "root_cause": {"type": "string", "description": "Your diagnosis in one sentence."},
                     "explanation": {"type": "string", "description": "Why this fix follows from the evidence."},
                 }, ["service", "action", "root_cause", "explanation"]), terminal=True),
    ]

    def build(self) -> None:
        rng = self.rng
        ids = list(RULES)
        sched = []
        for ep in range(self.episodes):
            sched.append(ids[(ep * 5 + self.seed) % len(ids)] if ep < len(ids) * 2 else rng.choice(ids))
        self.sched = sched
        self.instances = [self._instance(ep, sched[ep], 0) for ep in range(self.episodes)]

    def regenerate(self, ep: int, salt: int) -> None:
        rid = random.Random(f"diag-focus-{self.seed}-{ep}-{salt}").choice(list(RULE_CHANGES))
        self.sched[ep] = rid
        self.instances[ep] = self._instance(ep, rid, salt)

    def _instance(self, ep: int, rid: str, salt: int) -> dict[str, Any]:
        if True:
            r = RULES[rid]
            irng = random.Random(f"inst-{self.seed}-{ep}" + (f"-{salt}" if salt else ""))
            svc = irng.choice(SERVICES)
            svc_name = f"{svc}-api" if ep >= self.shift_at else f"svc-{svc}"
            metrics = {k: round(irng.uniform(*v), 3) for k, v in r["metrics"].items()}
            dk, dv = irng.choice([d for d in DISTRACTORS if d[0] not in metrics])
            metrics[dk] = round(irng.uniform(*dv), 3)
            dep = irng.choice(DEPS)
            logs = [l.format(dep=dep) for l in r["logs"]] + [irng.choice(["health probe ok", "cache warm complete"])]
            changes = [c.format(v=irng.randint(140, 199), dep=dep) for c in r["changes"]]
            if ep >= self.shift_at:
                metrics = {RENAME.get(k, k): v for k, v in metrics.items()}
            fix, cause, old, wrong = r["fix"], r["cause"], None, list(r["wrong"])
            if self.changed(ep) and rid in RULE_CHANGES:
                old, fix, cause = fix, RULE_CHANGES[rid]["fix"], RULE_CHANGES[rid]["cause"]
                wrong = [w for w in wrong if w != fix] + [old]
            return {"rule": rid, "service": svc_name, "metrics": metrics, "logs": logs,
                    "changes": changes, "cause": cause, "fix": fix, "wrong": wrong, "old_fix": old,
                    "change_class": rid in RULE_CHANGES}

    def seed_entries(self) -> list[tuple[str, str, float]]:
        # Generic, non-answer knowledge only: the fix catalogue and one misleading folk rule.
        out = [
            ("catalogue-remediations", "Available remediations: " + ", ".join(FIXES) + ".", 0.9),
            ("folk-rule-restart", "Team folklore: when in doubt, restart_pods first; it clears most incidents.", 0.7),
        ]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        inst = self.instances[ep]
        prompt = (f"PagerDuty alert INC-{6100 + ep}: {inst['service']} is unhealthy (SLO burn rate 14x). "
                  f"Investigate with the diagnostic tools and apply exactly one fix.")
        tags = {"rule": inst["rule"], "shifted": ep >= self.shift_at}
        if self.change_at:
            tags.update({"change_class": inst["change_class"], "post_change": self.changed(ep)})
        return Task(ep, prompt, dict(inst), self.agent_for(ep), tags)

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        if name == "get_metrics":
            return StepResult(json.dumps(t["metrics"]))
        if name == "get_logs":
            return StepResult("\n".join(t["logs"]))
        if name == "get_recent_changes":
            return StepResult("\n".join(t["changes"]))
        if name != "apply_fix":
            return StepResult(f"Unknown tool {name}")
        act = str(args.get("action", ""))
        ok = act == t["fix"]
        textbook = act == RULES[t["rule"]]["textbook"]
        stale = {"stale_pick": act == t["old_fix"]} if t.get("old_fix") else {}
        if ok:
            return StepResult(f"Applied {act}: error rate and latency returned to baseline within 3 minutes. "
                              f"Incident resolved.", terminal=True, success=True,
                              flags={"textbook_pick": textbook, **stale}, answer=act)
        return StepResult(f"Applied {act}: symptoms persist after 5 minutes; SLO burn continues. "
                          f"(Each failed remediation extends customer impact.)", terminal=True, success=False,
                          severity="failure", flags={"plausible_wrong": act in t["wrong"], "textbook_pick": textbook, **stale},
                          answer=act)


class SizingScenario(Scenario):
    """Numeric capacity planning under a hidden two-regime formula."""

    name = "sizing"
    agent_base = "capacity-agent"
    role = "You are the capacity-planning agent at Acme."

    tools = [
        ToolSpec("propose_sizing", "Propose replica count and DB connection pool size for the workload.",
                 finish_params({
                     "workload": {"type": "string"},
                     "replicas": {"type": "integer"},
                     "pool_size": {"type": "integer"},
                     "rationale": {"type": "string"},
                 }, ["workload", "replicas", "pool_size", "rationale"]), terminal=True),
    ]

    def build(self) -> None:
        rng = self.rng
        self.workloads = []
        for ep in range(self.episodes):
            rps = rng.choice([150, 300, 450, 700, 900, 1200, 1600, 2100, 2800, 3500])
            write_ratio = rng.choice([0.05, 0.1, 0.2, 0.45, 0.6, 0.75])
            p95_target = rng.choice([150, 250, 400])
            self.workloads.append({"name": f"wl-{rng.choice(SERVICES)}-{ep}", "rps": rps,
                                   "write_ratio": write_ratio, "p95_target_ms": p95_target})

    @staticmethod
    def optimum(w: dict[str, Any]) -> tuple[int, int]:
        read_heavy = w["write_ratio"] < 0.3
        cap = 150 if read_heavy else 70
        reps = -(-w["rps"] // cap)  # ceil
        pool = 16 if read_heavy else 32
        return reps, pool

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = [("sizing-guideline-generic", "Capacity guideline (2024): start with 2 replicas per 500 rps and a "
                                            "pool of 10 connections; adjust after load testing.", 0.7)]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        w = self.workloads[ep]
        prompt = (f"New workload {w['name']}: expected {w['rps']} rps, write ratio {w['write_ratio']}, "
                  f"p95 target {w['p95_target_ms']} ms. Propose replicas and DB pool size with propose_sizing.")
        reps, pool = self.optimum(w)
        return Task(ep, prompt, {"workload": w, "reps": reps, "pool": pool}, self.agent_for(ep),
                    {"regime": "read" if w["write_ratio"] < 0.3 else "write"})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        if name != "propose_sizing":
            return StepResult(f"Unknown tool {name}")
        t = task.truth
        try:
            reps = int(args.get("replicas", 0)); pool = int(args.get("pool_size", 0))
        except (TypeError, ValueError):
            return StepResult("Invalid numbers.", terminal=True, success=False, severity="failure", answer=str(args))
        r_ok = t["reps"] <= reps <= int(t["reps"] * 1.3) + 1
        p_ok = abs(pool - t["pool"]) <= 4
        err = abs(reps - t["reps"]) / max(t["reps"], 1)
        flags = {"replica_rel_error": round(err, 3), "pool_abs_error": abs(pool - t["pool"])}
        if r_ok and p_ok:
            return StepResult("Load test passed: p95 within target, cost within budget.", terminal=True,
                              success=True, flags=flags, answer=f"{reps}/{pool}")
        msgs = []
        if reps < t["reps"]:
            msgs.append("under-provisioned: CPU saturated and p95 breached")
        elif not r_ok:
            msgs.append("over-provisioned: cost alert, utilisation under 40%")
        if pool < t["pool"] - 4:
            msgs.append("DB pool exhausted under load")
        elif pool > t["pool"] + 4:
            msgs.append("DB connection limit warning: pool too large")
        return StepResult("Load test FAILED: " + "; ".join(msgs) + ".", terminal=True, success=False,
                          severity="failure", flags=flags, answer=f"{reps}/{pool}")
