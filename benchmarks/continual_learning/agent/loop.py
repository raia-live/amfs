"""The tool-calling agent loop. Identical for every arm and every model.

One episode:
  1. open a session on the arm; fetch a briefing (arms that have one)
  2. system prompt + task prompt (+ briefing)
  3. up to MAX_STEPS turns; every turn is a tool call. Memory tools go to the arm, domain
     tools go to the scenario environment (and are recorded on the arm). A terminal domain
     tool ends an *attempt*; on failure the agent gets the environment's feedback and may
     retry until the retry budget is spent (then "escalated").
  4. reflection turn: the agent writes ONE note for future runs (arms with memory)
  5. ``session.end(outcome)`` — arms that learn from outcomes get the verdict
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .. import config
from ..arms.base import EpisodeSession, MemoryArm, MemoryHit, Outcome, render_hits
from ..scenarios.base import Scenario, StepResult, Task
from .clients import LLM, Usage

MAX_STEPS = 16
TOP_K = 5

MEMORY_TOOLS = [
    {"name": "memory_search", "description": "Search long-term memory. Returns the most relevant entries.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "memory_write",
     "description": "Save a note for future runs. Prefer general, reusable lessons over episode details. "
                    "kind: 'fact' (stable knowledge), 'belief' (a hypothesis you are not sure of), "
                    "'experience' (what you did and what happened).",
     "parameters": {"type": "object", "properties": {
         "key": {"type": "string", "description": "kebab-case, stable across runs for the same topic"},
         "note": {"type": "string"},
         "kind": {"type": "string", "enum": ["fact", "belief", "experience"]},
         "confidence": {"type": "number", "minimum": 0.1, "maximum": 1.0}},
         "required": ["key", "note", "kind"]}},
]

REFLECT_PROMPT = (
    "The run is over. Outcome: {outcome}.\n"
    "Write exactly ONE memory note that would most help a future agent facing a similar (not "
    "identical) situation. State the lesson as a general rule with the conditions under which it "
    "applies. If a memory entry you relied on turned out to be wrong, say which key and what was "
    "wrong with it. Call memory_write once."
)


@dataclass
class EpisodeRecord:
    scenario: str
    arm: str
    model: str
    seed: int
    episode: int
    agent_id: str
    task_prompt: str
    success: bool
    severity: str
    attempts: int
    escalated: bool
    first_attempt_success: bool
    answers: list[str]
    final_answer: str
    domain_tool_calls: int
    diagnostic_calls: int
    memory_searches: int
    memory_writes: int
    hits_total: int
    contradictions_in_context: int
    # every memory_search this episode: query + what came back (key, confidence, evidence,
    # avoid flag). This is what lets the analysis show *which* entry was read before a stale
    # pick, and how its confidence and evidence moved over the run.
    searches_log: list[dict[str, Any]]
    stale_in_context: int          # hits (non-avoid) whose key is a known stale seed/lesson for this task
    discredited_served: int        # hits served in the avoid list
    cited_keys: list[str]
    flags: dict[str, Any]
    tags: dict[str, Any]
    usage: dict[str, Any]
    reflection_usage: dict[str, Any]
    memory: dict[str, Any]
    wall_ms: float
    steps: int
    explanation: str = ""
    judge_score: int | None = None
    error: str | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tool_result_text(hits: list[MemoryHit]) -> str:
    return render_hits(hits)


def _mentions_stale(hit: MemoryHit, task: Task) -> bool:
    """Does a served (non-avoid) entry recommend the action that *used to* be right for this
    task? Only defined after a regime change (``truth.old_fix``)."""
    old = task.truth.get("old_fix") if isinstance(task.truth, dict) else None
    if not old:
        return False
    return str(old).lower() in hit.text.lower()


def run_episode(scenario: Scenario, arm: MemoryArm, llm: LLM, task: Task, *,
                seed: int, retry_budget: int, keep_transcript: bool = False) -> EpisodeRecord:
    t0 = time.perf_counter()
    session: EpisodeSession = arm.session(task.agent_id, task.episode)
    usage, r_usage = Usage(), Usage()
    all_hits: list[MemoryHit] = []
    searches_log: list[dict[str, Any]] = []
    answers: list[str] = []
    cited: list[str] = []
    flags: dict[str, Any] = {}
    attempts = 0
    success = False
    severity = "failure"
    escalated = False
    final_answer = ""
    explanation = ""
    domain_calls = diag_calls = searches = writes = 0
    steps = 0
    error = None
    deferred = False
    transcript: list[dict[str, Any]] = []
    terminal_names = {t.name for t in scenario.tools if t.terminal}
    tools = [t.schema() for t in scenario.tools] + (MEMORY_TOOLS if arm.has_memory else [])

    messages: list[dict[str, Any]] = [{"role": "system", "content": scenario.system_prompt(arm.has_memory)}]
    try:
        brief = session.briefing() if arm.has_memory else None
        user = task.prompt
        if brief:
            user += "\n\nBriefing from your knowledge base (compiled before this task):\n" + brief
        messages.append({"role": "user", "content": user})

        while steps < MAX_STEPS:
            steps += 1
            turn = llm.chat(messages, tools, tool_choice="required", max_tokens=config.STUDY.max_tokens_per_turn)
            usage.add(turn.usage)
            messages.append(turn.assistant_message)
            if keep_transcript:
                transcript.append({"assistant": turn.text, "tool_calls": [asdict(c) for c in turn.tool_calls]})
            if not turn.tool_calls:
                messages.append({"role": "user", "content": "You must respond with a tool call."})
                continue
            stop = False
            for call in turn.tool_calls:
                name, args = call.name, call.arguments
                if name == "memory_search":
                    searches += 1
                    q = str(args.get("query", ""))
                    hits = session.search(q, TOP_K)
                    all_hits.extend(hits)
                    searches_log.append({"attempt": attempts + 1, "query": q[:200], "hits": [h.as_log() for h in hits]})
                    content = _tool_result_text(hits)
                elif name == "memory_write":
                    writes += 1
                    session.write(str(args.get("key", f"note-ep{task.episode}"))[:80], str(args.get("note", ""))[:1200],
                                  confidence=float(args.get("confidence", 0.7) or 0.7), kind=str(args.get("kind", "experience")))
                    content = "Saved."
                else:
                    domain_calls += 1
                    res: StepResult = scenario.step(task, name, args)
                    content = res.content
                    if res.terminal:
                        attempts += 1
                        answers.append(res.answer)
                        ok = scenario.noisy(bool(res.success))
                        flags.update(res.flags)
                        cited = [str(k) for k in (args.get("used_memory_keys") or [])][:10]
                        final_answer = res.answer
                        explanation = " — ".join(str(args[k]) for k in ("root_cause", "explanation", "rationale", "reason")
                                                 if args.get(k))
                        session.record_action(name, {k: v for k, v in args.items() if k != "used_memory_keys"},
                                              content, ok)
                        if ok:
                            success, severity, stop = True, "success", True
                        elif attempts >= retry_budget:
                            severity, escalated, stop = res.severity, True, True
                            content += "\n\nRetry budget exhausted; the task is being escalated to a human."
                        else:
                            severity = res.severity
                            content += f"\n\nAttempt {attempts} of {retry_budget} failed. Try a different approach."
                            session.attempt_failed(attempts, res.severity, cited, res.answer)
                    else:
                        diag_calls += 1
                        session.record_action(name, args, content[:300], True)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": content})
                if keep_transcript:
                    transcript.append({"tool": name, "result": content[:500]})
                if stop:
                    break
            if stop:
                break
        else:
            escalated = True
            severity = "failure"

        outcome = Outcome(success=success, severity=severity,
                          summary=f"{'success' if success else severity} after {attempts} attempt(s); answer={final_answer}")

        # reflection: one note for the future (all arms with memory, identical prompt)
        if arm.has_memory:
            messages.append({"role": "user", "content": REFLECT_PROMPT.format(outcome=outcome.summary)})
            rt = llm.chat(messages, MEMORY_TOOLS[1:], tool_choice="memory_write", max_tokens=400)
            r_usage.add(rt.usage)
            for call in rt.tool_calls:
                if call.name == "memory_write":
                    writes += 1
                    a = call.arguments
                    session.write(str(a.get("key", f"lesson-ep{task.episode}"))[:80], str(a.get("note", ""))[:1200],
                                  confidence=float(a.get("confidence", 0.7) or 0.7), kind=str(a.get("kind", "experience")))
                    if keep_transcript:
                        transcript.append({"reflection": a})
        if scenario.feedback_delay > 0:
            # delayed-feedback sweep: the verdict reaches memory only `feedback_delay` episodes later
            # (the runner drains ``arm.deferred`` before each episode). Reads/writes are already done.
            def _fire(s=session, o=outcome, ti=task.prompt, rt=explanation or final_answer, ck=list(cited)):
                try:
                    s.end(o, task_input=ti, response_text=rt, cited_keys=ck)
                finally:
                    s.close()
            arm.deferred.append((task.episode + scenario.feedback_delay, _fire))
            deferred = True
        else:
            session.end(outcome, task_input=task.prompt, response_text=explanation or final_answer, cited_keys=cited)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"[:400]
        if not attempts:
            escalated = True
    finally:
        if not deferred:
            session.close()

    return EpisodeRecord(
        scenario=scenario.name, arm=arm.name, model=llm.model, seed=seed, episode=task.episode,
        agent_id=task.agent_id, task_prompt=task.prompt, success=success, severity=severity,
        attempts=attempts, escalated=escalated, first_attempt_success=bool(answers) and success and attempts == 1,
        answers=answers, final_answer=final_answer, domain_tool_calls=domain_calls, diagnostic_calls=diag_calls,
        memory_searches=searches, memory_writes=writes, hits_total=len(all_hits),
        contradictions_in_context=scenario.contradictions(all_hits), searches_log=searches_log,
        stale_in_context=sum(1 for h in all_hits if not h.avoid and _mentions_stale(h, task)),
        discredited_served=sum(1 for h in all_hits if h.avoid), cited_keys=cited, flags=flags,
        tags=task.tags, usage=asdict(usage), reflection_usage=asdict(r_usage), memory=session.acct.as_dict(),
        wall_ms=(time.perf_counter() - t0) * 1000, steps=steps, explanation=explanation, error=error,
        transcript=transcript,
    )
