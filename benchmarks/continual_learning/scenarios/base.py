"""Scenario protocol: a generator of tasks plus a deterministic environment.

A scenario owns
  * ``seed_entries()``   what is in memory before episode 1 (same for every arm)
  * ``task(ep)``         the user-facing request for episode ``ep`` and its hidden truth
  * ``tools``            the domain tools the agent may call (memory tools are added by
                         the harness); exactly one of them is *terminal*
  * ``step(task, call)`` the environment: executes a domain tool, returns the result and,
                         for a terminal tool, the verdict
  * ``feedback(...)``    what the agent is told after a failed attempt — the same text
                         every arm sees, so learning signal is identical
  * analysis helpers     contradiction detection over retrieved context, flags
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from ..arms.base import MemoryHit

ACTION_WORDS = re.compile(r"\b(increase_pool_size|rollback_deploy|scale_out|restart_pods|set_cache_ttl|"
                          r"enable_circuit_breaker|rotate_logs|rate_limit|refund|escalate|deny|"
                          r"migrate|roll|warm|flush|reindex|drain)\b", re.I)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    terminal: bool = False

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class Task:
    episode: int
    prompt: str
    truth: dict[str, Any]
    agent_id: str
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    content: str
    terminal: bool = False
    success: bool | None = None
    severity: str = "failure"          # for terminal failures: minor_failure | failure | critical_failure
    flags: dict[str, Any] = field(default_factory=dict)
    answer: str = ""


class Scenario:
    name: str = "base"
    role: str = "You are an SRE agent at Acme."
    tools: list[ToolSpec] = []
    uses_judge: bool = False
    judge_rubric: str = ""
    agent_base: str = "agent"
    fleet_size: int = 1  # >1: episodes are handled round-robin by N distinct agent identities on one store

    def __init__(self, seed: int, episodes: int, *, distractors: int = 0,
                 label_noise: float = 0.0, feedback_delay: int = 0, fleet: int = 0,
                 change_at: int = 0, change_focus: float = 0.0, fleet_mode: str = "rr",
                 join_at: int = 0, **_: Any) -> None:
        self.seed = seed
        self.episodes = episodes
        self.rng = random.Random(f"{self.name}-{seed}")
        self.distractors = distractors
        self.label_noise = label_noise
        self.feedback_delay = feedback_delay
        # Regime change: from episode ``change_at`` (0 = never) the world changes for a subset
        # of task classes — a policy, a system, an infrastructure quirk — and the previously
        # correct action stops working. Nobody tells the agent. Scenarios that support it
        # define ``CHANGES``; a memory that keeps serving the old lesson is now wrong.
        self.change_at = change_at
        # Share of post-change episodes drawn from the classes whose truth changed. A change
        # in the real world produces a spike of exactly the tickets it broke; uniform sampling
        # gives a 60-episode cell only ~15 changed-class tasks after the change, too few to
        # measure recovery. 0 = leave the scenario's natural schedule alone.
        self.change_focus = change_focus
        if fleet and fleet > 1:
            self.fleet_size = fleet
        # Knowledge-transfer protocols (only meaningful with a fleet):
        #   rr        every episode goes to the next agent round-robin (default)
        #   pioneer   agent 1 works alone until ``join_at``; then all N share the queue.
        #             Measures whether what one agent learned transfers to peers who never
        #             saw those tasks: the followers' FIRST exposure to each class.
        #   newcomer  agents 1..N-1 share the queue until ``join_at``; from then agent N, who has
        #             never worked here, takes every episode. With a regime change before
        #             ``join_at`` this measures whether the newcomer inherits the fleet's
        #             *unlearning* or repeats the stale fix its peers already burned on.
        self.fleet_mode = fleet_mode if self.fleet_size > 1 else "rr"
        self.join_at = join_at
        self._noise_rng = random.Random(f"noise-{self.name}-{seed}")
        self.build()
        self._apply_change_focus()

    def changed(self, ep: int) -> bool:
        return bool(self.change_at) and ep >= self.change_at

    def agent_for(self, ep: int) -> str:
        """Identity handling episode ``ep``. With a fleet, each episode goes to a different agent,
        so an agent only ever sees a fraction of the episodes directly; the rest it must learn
        from what its peers left in the shared store."""
        if self.fleet_size > 1:
            n = self.fleet_size
            if self.fleet_mode == "pioneer":
                if ep < self.join_at:
                    return f"{self.agent_base}-1"
                return f"{self.agent_base}-{(ep - self.join_at + 1) % n + 1}"  # ep == join_at -> agent 2
            if self.fleet_mode == "newcomer":
                if ep < self.join_at:
                    return f"{self.agent_base}-{ep % (n - 1) + 1}"
                return f"{self.agent_base}-{n}"
            return f"{self.agent_base}-{ep % n + 1}"
        return self.agent_base

    def regenerate(self, ep: int, salt: int) -> None:
        """Re-draw episode ``ep``'s instance with a different random salt. Scenarios that
        support ``change_focus`` implement it; the default leaves the schedule as built."""
        return None

    def _apply_change_focus(self) -> None:
        if not (self.change_at and self.change_focus):
            return
        post = list(range(self.change_at, self.episodes))
        if not post:
            return
        target = round(len(post) * self.change_focus)
        in_class = {ep for ep in post if self.task(ep).tags.get("change_class")}
        need = target - len(in_class)
        rng = random.Random(f"focus-{self.name}-{self.seed}")
        candidates = [ep for ep in post if ep not in in_class]
        rng.shuffle(candidates)
        for ep in candidates:
            if need <= 0:
                break
            for salt in range(1, 64):
                self.regenerate(ep, salt)
                if self.task(ep).tags.get("change_class"):
                    need -= 1
                    break

    # ---- to implement --------------------------------------------------------
    def build(self) -> None: ...

    def seed_entries(self) -> list[tuple[str, str, float]]:
        return self.distractor_entries()

    def task(self, ep: int) -> Task:
        raise NotImplementedError

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        raise NotImplementedError

    # ---- shared --------------------------------------------------------------
    def system_prompt(self, has_memory: bool) -> str:
        mem = ("\nYou have a long-term memory shared across runs. ALWAYS call memory_search before "
               "acting, and prefer what memory says when it is specific and consistent. If memory "
               "entries conflict, say so briefly in your final tool call. Some memory systems annotate "
               "entries with outcome evidence (validated / contested / discredited, wins and losses) or "
               "list entries to avoid: trust validated entries over untested ones, do not act on "
               "discredited ones, and if the only evidence you have is contested or discredited and the "
               "action is costly, escalate on the first attempt rather than guess. When you finish, list "
               "the memory keys you actually relied on in used_memory_keys." if has_memory else
               "\nYou have no long-term memory; use only the information in this conversation.")
        return (f"{self.role} Work through tools only: every turn must be a tool call. Be decisive and "
                f"concise. If an attempt fails you will be told why; then try again.{mem}")

    def noisy(self, success: bool) -> bool:
        """Apply outcome label noise (sensitivity sweep)."""
        if self.label_noise and self._noise_rng.random() < self.label_noise:
            return not success
        return success

    def distractor_entries(self) -> list[tuple[str, str, float]]:
        if not self.distractors:
            return []
        rng = random.Random(f"distractors-{self.seed}")
        svcs = ["billing", "search", "ledger", "mailer", "auth", "catalog", "reports", "gateway"]
        facts = ["runs on Kubernetes namespace {s}-prod", "owned by team {t}", "has SLO 99.9% availability",
                 "deploys via Argo CD app {s}", "uses Redis for session cache", "exposes gRPC on port {p}",
                 "alerts route to #{s}-alerts", "last load test peaked at {n} rps"]
        out = []
        for i in range(self.distractors):
            s = rng.choice(svcs); f = rng.choice(facts)
            out.append((f"note-{i:04d}", f"Service {s}-{i % 17}: " + f.format(
                s=s, t=rng.choice(["platform", "payments", "growth"]), p=rng.randint(7000, 9999),
                n=rng.randint(200, 5000)), 0.8))
        return out

    def contradictions(self, hits: list[MemoryHit]) -> int:
        """Pairs of retrieved entries that recommend different actions for overlapping subjects."""
        parsed = []
        for h in hits:
            acts = {a.lower() for a in ACTION_WORDS.findall(h.text)}
            subj = set(re.findall(r"\b[a-z][a-z0-9-]{3,}\b", h.text.lower())) - {"then", "with", "that", "this", "when", "from", "service"}
            parsed.append((acts, subj))
        n = 0
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                a1, s1 = parsed[i]; a2, s2 = parsed[j]
                if a1 and a2 and a1 != a2 and len(s1 & s2) >= 4:
                    n += 1
        return n

    def flags_for(self, task: Task, result: StepResult, hits: list[MemoryHit]) -> dict[str, Any]:
        return dict(result.flags)


def finish_params(extra: dict[str, Any], required: list[str]) -> dict[str, Any]:
    props = dict(extra)
    props["used_memory_keys"] = {"type": "array", "items": {"type": "string"},
                                 "description": "Keys of memory entries you relied on (empty if none)."}
    return {"type": "object", "properties": props, "required": required}
