"""The memory-arm protocol every system under test implements.

An *arm* is one memory layer wired the way its own documentation says to wire it. The
agent talks to an arm through exactly three verbs (``briefing``, ``search``, ``write``) and
the harness closes each episode with ``end``. Everything an arm does behind those verbs
(embedding, graph extraction, outcome propagation) is the thing being measured.

Every arm records its own accounting so the paper can report memory-side cost
separately from model-side cost:

* ``ops``            number of memory calls in the episode
* ``retrieved_bytes`` total characters handed to the model from memory
* ``ingest_wait_ms``  time spent waiting for a hosted store to make a write searchable
* ``extra_llm``       tokens/cost the arm itself spent on LLM calls (DIY consolidation)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

SEED_READY_TIMEOUT_S = 900.0


@dataclass
class MemoryHit:
    key: str
    text: str
    score: float = 0.0
    confidence: float | None = None
    # Outcome evidence, when the arm tracks it (SenseLab; the DIY+outcomes baseline). Rendered
    # to the agent so a validated lesson and a discredited one do not look alike in context.
    evidence_status: str | None = None   # untested | validated | contested | discredited
    success_count: int = 0
    failure_count: int = 0
    avoid: bool = False                  # served as "do not use", not as a candidate

    def evidence_label(self) -> str | None:
        if self.evidence_status is None:
            return None
        if self.evidence_status == "untested" and not (self.success_count or self.failure_count):
            return "untested"
        return f"{self.evidence_status}: {self.success_count} won / {self.failure_count} failed"

    def render(self) -> str:
        parts = []
        if self.confidence is not None:
            parts.append(f"confidence {self.confidence:.2f}")
        ev = self.evidence_label()
        if ev:
            parts.append(ev)
        meta = f" ({'; '.join(parts)})" if parts else ""
        return f"[{self.key}]{meta}: {self.text}"

    def as_log(self) -> dict[str, Any]:
        d = {"key": self.key, "score": round(self.score, 4)}
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 3)
        if self.evidence_status is not None:
            d.update(evidence_status=self.evidence_status, success_count=self.success_count,
                     failure_count=self.failure_count)
        if self.avoid:
            d["avoid"] = True
        return d


def render_hits(hits: list[MemoryHit]) -> str:
    """What the agent reads back from ``memory_search``. Identical formatting for every arm;
    arms without evidence simply have nothing in the parentheses beyond confidence."""
    keep = [h for h in hits if not h.avoid]
    avoid = [h for h in hits if h.avoid]
    if not keep and not avoid:
        return "No relevant memory entries."
    out = [h.render() for h in keep] or ["No relevant memory entries."]
    if avoid:
        out.append("\nDo NOT rely on these entries; they were discredited by recent outcomes:")
        out.extend(f"  - {h.render()}" for h in avoid)
    return "\n".join(out)


@dataclass
class Outcome:
    success: bool
    severity: str  # "success" | "minor_failure" | "failure" | "critical_failure"
    summary: str


@dataclass
class ArmAccounting:
    ops: int = 0
    retrieved_bytes: int = 0
    ingest_wait_ms: float = 0.0
    memory_ms: float = 0.0
    extra_prompt_tokens: int = 0
    extra_completion_tokens: int = 0
    extra_cost_usd: float = 0.0
    trace_verifiable: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_ops": self.ops,
            "retrieved_bytes": self.retrieved_bytes,
            "ingest_wait_ms": round(self.ingest_wait_ms, 1),
            "memory_ms": round(self.memory_ms, 1),
            "arm_extra_prompt_tokens": self.extra_prompt_tokens,
            "arm_extra_completion_tokens": self.extra_completion_tokens,
            "arm_extra_cost_usd": round(self.extra_cost_usd, 6),
            "trace_verifiable": self.trace_verifiable,
            **({"arm_notes": self.notes} if self.notes else {}),
        }


class EpisodeSession:
    """One agent identity's view of the store for one episode."""

    def __init__(self, arm: "MemoryArm", agent_id: str, episode: int) -> None:
        self.arm = arm
        self.agent_id = agent_id
        self.episode = episode
        self.acct = ArmAccounting()
        self.read_keys: list[str] = []
        # The actions the terminal tool accepts, as ``<tool>:<action>`` keys, set by the
        # harness from the scenario's tool schema before the first search. Arms that keep
        # an action record use it to name what has *not* been tried here.
        self.candidate_actions: list[str] | None = None
        # What the arm recommended on the last search, for the analysis cross-tab
        # (recommendation followed / outcome). ``None`` for arms without one.
        self.last_recommendation: dict[str, Any] | None = None

    # -- verbs the agent can use -------------------------------------------------
    def briefing(self) -> str | None:
        return None

    def search(self, query: str, top_k: int) -> list[MemoryHit]:
        raise NotImplementedError

    def search_footer(self) -> str | None:
        """Text appended to the last search's result, after the hits: what was tried on
        similar tasks in this scope and how it went, and a recommendation. Only arms that
        keep an action record return anything; the harness counts it as retrieved bytes."""
        return None

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        raise NotImplementedError

    def record_action(self, tool: str, arguments: dict[str, Any], result: str, success: bool) -> None:
        return None

    # -- harness hooks -----------------------------------------------------------
    def end(self, outcome: Outcome, *, task_input: str, response_text: str,
            cited_keys: list[str]) -> None:
        return None

    def attempt_failed(self, attempt: int, severity: str, cited_keys: list[str], answer: str) -> None:
        """A non-final attempt failed. Arms that support per-attempt credit assignment mark
        the boundary so the memories that attempt relied on carry its failure; the default
        protocol (one outcome per task) ignores it."""
        return None

    def close(self) -> None:
        return None

    # -- helpers -----------------------------------------------------------------
    def _timed(self):
        return _Timer(self.acct)


class _Timer:
    def __init__(self, acct: ArmAccounting) -> None:
        self.acct = acct

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.acct.memory_ms += (time.perf_counter() - self.t) * 1000
        self.acct.ops += 1


class MemoryArm:
    """A memory layer under test. Subclasses set ``name`` and ``learns_from_outcomes``."""

    name: str = "base"
    learns_from_outcomes: bool = False
    has_memory: bool = True
    # Arms without memory tools that still hand the agent a briefing (the raw-traces control).
    briefs: bool = False

    def open(self, scope: str) -> None:
        """Prepare an isolated namespace for one (scenario, seed)."""
        self.scope = scope
        # (due_episode, callable) pairs for the delayed-feedback sweep; drained by the runner
        self.deferred: list[tuple[int, Any]] = []

    def configure(self, scenario) -> None:
        """Called once per cell after ``open`` with the scenario about to run, for arms whose
        side machinery needs to know the domain (the repair arm's judge rubric and lever).
        The default ignores it; arms must not read task truths from it."""
        return None

    def drain_deferred(self, before_episode: int) -> None:
        """Deliver outcomes whose delay has elapsed (due <= before_episode)."""
        due = [d for d in self.deferred if d[0] <= before_episode]
        self.deferred = [d for d in self.deferred if d[0] > before_episode]
        for _, fire in due:
            fire()

    def seed(self, entries: list[tuple[str, str, float]], *, agent_id: str = "seed-agent") -> None:
        """Write initial knowledge. ``entries`` are (key, text, confidence)."""
        s = self.session(agent_id, episode=0)
        try:
            for key, text, conf in entries:
                s.write(key, text, confidence=conf, kind="fact")
            if hasattr(s, "wait_ready"):
                # bulk seeding is a one-off before episode 1: long deadline, sampled verification
                s.wait_ready(len(entries), timeout=SEED_READY_TIMEOUT_S, sample=3)
        finally:
            s.close()

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        raise NotImplementedError

    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        """Arm-side maintenance between episodes (DIY consolidation). Returns accounting."""
        return None

    def confidence_history(self, keys: list[str]) -> list[dict[str, Any]] | None:
        """Version-by-version confidence and evidence for ``keys`` at the end of a cell, for
        arms whose store keeps that history. The analysis plots how fast a stale lesson fell
        and a replacement rose. ``None`` when the arm has no such record."""
        return None

    def close(self) -> None:
        return None
