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

SITUATION_PARAM_DESCRIPTION = (
    "A short, stable label for the KIND of task — two to four lowercase words naming what "
    "decides the right action (the issue, the failing check, the request type), never the "
    "customer's wording, ids or numbers. Use the exact same label for every task of the same "
    "kind, and a different label when the deciding detail differs: two tasks that would need "
    "different fixes must get different labels, so include the detail from the task that "
    "separates them (which check, which kind of change or account), e.g. 'snapshot diff on ui pr' "
    "vs 'snapshot diff on backend pr', not 'snapshot diff' for both."
)
SITUATION_LABELS_SHOWN = 16


def memory_tools(arm: MemoryArm) -> list[dict[str, Any]]:
    """The memory tools for *arm*: the shared schema, plus the ``situation`` parameter on
    ``memory_search`` for arms that keep a record per kind of task (``declares_situation``).
    The labels already used in the scope are listed in the parameter's description so the
    agent reuses them — the same taxonomy a triage step would keep."""
    if not getattr(arm, "declares_situation", False):
        return MEMORY_TOOLS
    search = json.loads(json.dumps(MEMORY_TOOLS[0]))
    desc = SITUATION_PARAM_DESCRIPTION
    known = list(arm.situation_labels() or [])[:SITUATION_LABELS_SHOWN]
    if known:
        desc += " Labels already in use here (reuse one when it fits): " + "; ".join(known) + "."
    coarse = list(getattr(arm, "pooled_situation_labels", list)() or [])[:6]
    if coarse:
        desc += (" Labels found too coarse — two kinds of task with different fixes shared them; "
                 "do not reuse, add the detail that separates them: " + "; ".join(coarse) + ".")
    search["parameters"]["properties"]["situation"] = {"type": "string", "description": desc}
    search["parameters"]["required"] = ["query", "situation"]
    return [search] + MEMORY_TOOLS[1:]


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
    written_keys: list[str]        # keys this agent wrote this episode (authorship for transfer analysis)
    flags: dict[str, Any]
    tags: dict[str, Any]
    usage: dict[str, Any]
    reflection_usage: dict[str, Any]
    memory: dict[str, Any]
    wall_ms: float
    steps: int
    #: Terminal calls to the scenario's ``edit_tools`` that did not resolve the task: each is a
    #: change the agent made to the world that it then had to undo or supersede. The narrow
    #: intervention claim (Phase 0.A) is measured on this, not only on success.
    unnecessary_edits: int = 0
    # Wall-clock from episode start to the agent's first domain action (the first decision it
    # commits to, terminal or not): its time to first response. None for records written
    # before this field existed and for episodes that never acted.
    first_action_ms: float | None = None
    explanation: str = ""
    judge_score: int | None = None
    error: str | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tool_result_text(hits: list[MemoryHit]) -> str:
    return render_hits(hits)


def candidate_actions(scenario: Scenario) -> list[str] | None:
    """The terminal tool's ``action`` enum as ``<tool>:<action>`` keys — the same key the
    SDK derives from a recorded tool call — or ``None`` when the scenario's terminal tool
    has no such enum (free-form answers, multi-step runbooks)."""
    for t in scenario.tools:
        if not t.terminal:
            continue
        props = (t.parameters or {}).get("properties") or {}
        enum = (props.get("action") or {}).get("enum")
        if isinstance(enum, list) and enum:
            return [f"{t.name}:{a}" for a in enum]
    return None


def _mentions_stale(hit: MemoryHit, task: Task) -> bool:
    """Does a served (non-avoid) entry recommend the action that *used to* be right for this
    task? Only defined after a regime change (``truth.old_fix``)."""
    old = task.truth.get("old_fix") if isinstance(task.truth, dict) else None
    if not old:
        return False
    return str(old).lower() in hit.text.lower()


def _system_prompt(scenario: Scenario, arm: MemoryArm, task: Task, session: EpisodeSession) -> str:
    text = scenario.system_prompt(arm.has_memory)
    addendum = arm.prompt_addendum(task.agent_id) if hasattr(arm, "prompt_addendum") else None
    if addendum:
        text += "\n\n" + addendum
        session.acct.retrieved_bytes += len(addendum)
    return text


def run_episode(scenario: Scenario, arm: MemoryArm, llm: LLM, task: Task, *,
                seed: int, retry_budget: int, keep_transcript: bool = False) -> EpisodeRecord:
    if getattr(llm, "decision_mode", False):
        return _run_decision_episode(scenario, arm, llm, task, seed=seed, retry_budget=retry_budget,
                                     keep_transcript=keep_transcript)
    t0 = time.perf_counter()
    session: EpisodeSession = arm.session(task.agent_id, task.episode)
    usage, r_usage = Usage(), Usage()
    all_hits: list[MemoryHit] = []
    searches_log: list[dict[str, Any]] = []
    answers: list[str] = []
    cited: list[str] = []
    written: list[str] = []
    flags: dict[str, Any] = {}
    attempts = 0
    first_action_ms: float | None = None
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
    edit_tools = set(getattr(scenario, "edit_tools", ()) or ())
    edit_calls = 0
    tools = [t.schema() for t in scenario.tools] + (memory_tools(arm) if arm.has_memory else [])
    session.candidate_actions = candidate_actions(scenario)

    messages: list[dict[str, Any]] = [{"role": "system", "content": _system_prompt(scenario, arm, task, session)}]
    try:
        # ``briefs`` lets a control arm without memory tools (raw-traces) still put text in
        # front of the agent before the task.
        brief = session.briefing() if (arm.has_memory or getattr(arm, "briefs", False)) else None
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
                    sit = args.get("situation")
                    sit = str(sit).strip()[:120] if isinstance(sit, str) and sit.strip() else None
                    # Only arms that asked for a label get the keyword: the other arms'
                    # sessions keep the two-argument signature.
                    hits = session.search(q, TOP_K, situation=sit) if sit is not None else session.search(q, TOP_K)
                    all_hits.extend(hits)
                    entry = {"attempt": attempts + 1, "query": q[:200], "hits": [h.as_log() for h in hits]}
                    if sit is not None:
                        entry["situation"] = sit
                    content = _tool_result_text(hits)
                    footer = session.search_footer()
                    if footer:
                        content += "\n\n" + footer
                        session.acct.retrieved_bytes += len(footer)
                        # The text the agent read, kept with the search: what the
                        # recommendation *said* is what "followed / ignored" is judged
                        # against, and pilots 5a-5d could only reconstruct it after the fact.
                        entry["footer"] = footer[:1500]
                    if session.last_recommendation:
                        entry["recommendation"] = session.last_recommendation
                    searches_log.append(entry)
                elif name == "memory_write":
                    writes += 1
                    wkey = str(args.get("key", f"note-ep{task.episode}"))[:80]
                    written.append(wkey)
                    session.write(wkey, str(args.get("note", ""))[:1200],
                                  confidence=float(args.get("confidence", 0.7) or 0.7), kind=str(args.get("kind", "experience")))
                    content = "Saved."
                else:
                    domain_calls += 1
                    if first_action_ms is None:
                        first_action_ms = (time.perf_counter() - t0) * 1000
                    res: StepResult = scenario.step(task, name, args)
                    content = res.content
                    if res.terminal:
                        attempts += 1
                        answers.append(res.answer)
                        # What this attempt did, on the searches that preceded it: the
                        # analysis reads recommendation-followed / outcome from here.
                        if isinstance(args.get("action"), str):
                            for e_ in reversed(searches_log):
                                if e_.get("attempt") != attempts:
                                    break
                                e_["taken"] = f"{name}:{args['action']}"
                        if name in edit_tools:
                            edit_calls += 1
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
                          summary=f"{'success' if success else severity} after {attempts} attempt(s); answer={final_answer}",
                          escalated=escalated)

        # reflection: one note for the future (all arms with memory, identical prompt)
        if arm.has_memory:
            # Guarded on its own: the verdict below must reach the store even if
            # this extra LLM turn or its write fails. Letting the exception
            # escape here would skip ``session.end`` while the record still
            # said ``success`` — a cell that looks learned-from when the memory
            # never received the outcome, which silently breaks H1–H5 and the
            # ``senselab`` vs ``senselab-nofeedback`` comparison.
            try:
                messages.append({"role": "user", "content": REFLECT_PROMPT.format(outcome=outcome.summary)})
                rt = llm.chat(messages, MEMORY_TOOLS[1:], tool_choice="memory_write", max_tokens=400)
                r_usage.add(rt.usage)
                for call in rt.tool_calls:
                    if call.name == "memory_write":
                        writes += 1
                        a = call.arguments
                        wkey = str(a.get("key", f"lesson-ep{task.episode}"))[:80]
                        written.append(wkey)
                        session.write(wkey, str(a.get("note", ""))[:1200],
                                      confidence=float(a.get("confidence", 0.7) or 0.7), kind=str(a.get("kind", "experience")))
                        if keep_transcript:
                            transcript.append({"reflection": a})
            except Exception as e:  # noqa: BLE001
                error = f"reflection: {type(e).__name__}: {e}"[:400]
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
        discredited_served=sum(1 for h in all_hits if h.avoid), cited_keys=cited, written_keys=written, flags=flags,
        tags=task.tags, usage=asdict(usage), reflection_usage=asdict(r_usage), memory=session.acct.as_dict(),
        wall_ms=(time.perf_counter() - t0) * 1000, steps=steps, first_action_ms=first_action_ms,
        unnecessary_edits=max(0, edit_calls - (1 if success else 0)),
        explanation=explanation, error=error,
        transcript=transcript,
    )


# ---------------------------------------------------------------------------------------------
# Decision mode: one served decision per attempt.
#
# The Managed Models student (and the frontier ``@decision`` control that answers the identical
# rendered prompt) does not run the tool loop. The gateway retrieves from the cell's scope,
# renders the prompt the student was trained on and returns one action. The harness executes
# it; a failed attempt goes back as a failed tool result, which the gateway renders as the
# "Tried in this task" block — the same shape a fail-then-succeed teacher trace trained on.
# Diagnostic tools are not available: the student was trained on the last consequential call,
# and a diagnostic answer here is a failed attempt (it shows up as tried, so the next answer
# moves on). The student writes no reflection of its own; ``llm.note_writer`` names a cheap
# model that writes the one note the full-loop agent would have, charged to the condition.
# ---------------------------------------------------------------------------------------------

NOTE_WRITER_PROMPT = (
    "You are the reflection step for a decision agent that has just finished a task. Below is the "
    "task, the memory it was shown, what it tried and how each attempt went.\n\n{body}\n\n"
)


def _hits_from_amfs(block: dict[str, Any]) -> list[MemoryHit]:
    hits: list[MemoryHit] = []
    for e in block.get("context") or []:
        if not isinstance(e, dict) or not e.get("key"):
            continue
        hits.append(MemoryHit(key=str(e["key"]), text=str(e.get("value") or ""),
                              confidence=float(e.get("confidence") or 0.0),
                              evidence_status=str(e.get("evidence_status") or "untested"),
                              success_count=int(e.get("success_count") or 0),
                              failure_count=int(e.get("failure_count") or 0)))
    for a in block.get("avoid") or []:
        if not isinstance(a, dict) or not a.get("key"):
            continue
        hits.append(MemoryHit(key=str(a["key"]), text=str(a.get("value") or ""), evidence_status="discredited",
                              success_count=int(a.get("success_count") or 0),
                              failure_count=int(a.get("failure_count") or 0), avoid=True))
    return hits


def _run_decision_episode(scenario: Scenario, arm: MemoryArm, llm: LLM, task: Task, *,
                          seed: int, retry_budget: int, keep_transcript: bool = False) -> EpisodeRecord:
    from .clients import get_llm

    t0 = time.perf_counter()
    session: EpisodeSession = arm.session(task.agent_id, task.episode)
    usage, r_usage = Usage(), Usage()
    all_hits: list[MemoryHit] = []
    searches_log: list[dict[str, Any]] = []
    answers: list[str] = []
    cited: list[str] = []
    written: list[str] = []
    flags: dict[str, Any] = {}
    attempts = 0
    first_action_ms: float | None = None
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
    served_by: list[str] = []
    trace_ids: list[str] = []
    attempt_log: list[tuple[str, str]] = []   # (answer, environment result) per attempt, for the note-writer
    terminal_names = {t.name for t in scenario.tools if t.terminal}
    session.candidate_actions = candidate_actions(scenario)
    # The system message is for the transcript and the note-writer; the gateway substitutes the
    # instruction the student was trained on, and the control sees that same instruction.
    messages: list[dict[str, Any]] = [{"role": "system", "content": _system_prompt(scenario, arm, task, session)},
                                      {"role": "user", "content": task.prompt}]
    meta = {"entity_path": arm.scope, "agent_id": task.agent_id,
            "outcome_ref": f"{task.agent_id}-ep{task.episode}", "candidate_actions": session.candidate_actions}
    try:
        while attempts < retry_budget:
            steps += 1
            turn = llm.chat(messages, None, max_tokens=config.STUDY.max_tokens_per_turn, amfs=meta)
            usage.add(turn.usage)
            messages.append(turn.assistant_message)
            block = turn.amfs or {}
            hits = _hits_from_amfs(block)
            all_hits.extend(hits)
            searches += 1
            session.acct.ops += 1
            prompt_user = str((block.get("prompt") or {}).get("user") or "")
            session.acct.retrieved_bytes += max(0, len(prompt_user) - len(task.prompt))
            served_by.append(str(block.get("served_by") or "?"))
            if block.get("trace_id"):
                trace_ids.append(str(block["trace_id"]))
            entry: dict[str, Any] = {"attempt": attempts + 1, "query": task.prompt[:200],
                                     "hits": [h.as_log() for h in hits], "retrieval": block.get("retrieval"),
                                     "served_by": block.get("served_by"), "trace_id": block.get("trace_id")}
            rec = (block.get("priors") or {}).get("recommendation") if isinstance(block.get("priors"), dict) else None
            if isinstance(rec, dict) and rec.get("mode"):
                session.last_recommendation = {"mode": rec.get("mode"), "suggested_action": rec.get("suggested_action"),
                                               "regime_shift": bool((block.get("priors") or {}).get("regime_shift"))}
                entry["recommendation"] = session.last_recommendation
                session.acct.notes[f"rec_{rec['mode']}"] = session.acct.notes.get(f"rec_{rec['mode']}", 0) + 1
            searches_log.append(entry)
            kept = [h for h in hits if not h.avoid]
            session.read_keys.extend(h.key for h in kept[:1])
            if any(h.avoid for h in hits):
                session.acct.notes["avoid_served"] = session.acct.notes.get("avoid_served", 0) + sum(h.avoid for h in hits)
            if keep_transcript:
                transcript.append({"assistant": turn.text, "tool_calls": [asdict(c) for c in turn.tool_calls],
                                   "prompt": block.get("prompt"), "served_by": block.get("served_by")})

            attempts += 1
            if not turn.tool_calls:
                # Not an action. At temperature 0 the same prompt yields the same prose, so a
                # retry would be a duplicate call; the attempt is a failure and the task escalates.
                answers.append("")
                attempt_log.append(("", f"not a tool call: {turn.text[:200]}"))
                flags["unparseable"] = flags.get("unparseable", 0) + 1
                severity, escalated = "failure", True
                break

            call = turn.tool_calls[0]
            name, args = call.name, call.arguments
            if name not in terminal_names:
                # The action written where the tool goes (``refund`` for ``resolve(action=refund)``)
                # names a decision as unambiguously as the tool call does; the candidate keys are
                # the scenario's own enum. Same leniency for every condition through this branch.
                for key in candidate_actions(scenario) or []:
                    tool, _, action = key.partition(":")
                    if name == action and "action" not in args:
                        name, args = tool, {"action": action, **args}
                        call.name, call.arguments = name, args
                        break
            stop = False
            if name not in terminal_names:
                diag_calls += 1
                answers.append(f"<{name}>")
                flags["non_terminal_answers"] = flags.get("non_terminal_answers", 0) + 1
                content = json.dumps({"success": False, "error": f"{name} is not available in single-shot mode; "
                                                                  f"answer with the terminal tool ({', '.join(sorted(terminal_names))})."})
                session.record_action(name, args, "not executed: diagnostic call in decision mode", False)
                if attempts >= retry_budget:
                    severity, escalated, stop = "failure", True, True
                else:
                    severity = "failure"
                    session.attempt_failed(attempts, "failure", [], f"<{name}>")
            else:
                domain_calls += 1
                if first_action_ms is None:
                    first_action_ms = (time.perf_counter() - t0) * 1000
                res: StepResult = scenario.step(task, name, args)
                answers.append(res.answer)
                ok = scenario.noisy(bool(res.success))
                flags.update(res.flags)
                final_answer = res.answer
                explanation = " — ".join(str(args[k]) for k in ("root_cause", "explanation", "rationale", "reason")
                                         if args.get(k))
                session.record_action(name, {k: v for k, v in args.items() if k != "used_memory_keys"}, res.content, ok)
                content = json.dumps({"success": ok, ("result" if ok else "error"): res.content})
                if ok:
                    success, severity, stop = True, "success", True
                elif attempts >= retry_budget:
                    severity, escalated, stop = res.severity, True, True
                else:
                    severity = res.severity
                    session.attempt_failed(attempts, res.severity, [], res.answer)
            messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": content})
            attempt_log.append((answers[-1], content))
            if keep_transcript:
                transcript.append({"tool": name, "result": content[:500]})
            if stop:
                break
        else:
            escalated = True

        outcome = Outcome(success=success, severity=severity,
                          summary=f"{'success' if success else severity} after {attempts} attempt(s); answer={final_answer}",
                          escalated=escalated)

        if arm.has_memory and getattr(llm, "note_writer", None):
            try:
                writer = get_llm(llm.note_writer)
                lines = [f"TASK:\n{task.prompt}", "MEMORY SHOWN:\n" + render_hits(all_hits)]
                for i, (a, result) in enumerate(attempt_log, 1):
                    lines.append(f"ATTEMPT {i}: chose {a or '(no action)'} -> {result[:300]}")
                body = "\n\n".join(lines)
                wmsgs = [{"role": "system", "content": scenario.system_prompt(True)},
                         {"role": "user", "content": NOTE_WRITER_PROMPT.format(body=body) + REFLECT_PROMPT.format(outcome=outcome.summary)}]
                rt = writer.chat(wmsgs, MEMORY_TOOLS[1:], tool_choice="memory_write", max_tokens=400)
                r_usage.add(rt.usage)
                for c in rt.tool_calls:
                    if c.name == "memory_write":
                        writes += 1
                        a = c.arguments
                        wkey = str(a.get("key", f"lesson-ep{task.episode}"))[:80]
                        written.append(wkey)
                        session.write(wkey, str(a.get("note", ""))[:1200],
                                      confidence=float(a.get("confidence", 0.7) or 0.7), kind=str(a.get("kind", "experience")))
                        if keep_transcript:
                            transcript.append({"reflection": a, "writer": llm.note_writer})
            except Exception as e:  # noqa: BLE001
                error = f"note-writer: {type(e).__name__}: {e}"[:400]

        if scenario.feedback_delay > 0:
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

    flags["decision"] = {"served_by": served_by, "trace_ids": trace_ids,
                         "fallback_served": sum(1 for s in served_by if s == "fallback")}
    return EpisodeRecord(
        scenario=scenario.name, arm=arm.name, model=llm.model, seed=seed, episode=task.episode,
        agent_id=task.agent_id, task_prompt=task.prompt, success=success, severity=severity,
        attempts=attempts, escalated=escalated, first_attempt_success=bool(answers) and success and attempts == 1,
        answers=answers, final_answer=final_answer, domain_tool_calls=domain_calls, diagnostic_calls=diag_calls,
        memory_searches=searches, memory_writes=writes, hits_total=len(all_hits),
        contradictions_in_context=scenario.contradictions(all_hits), searches_log=searches_log,
        stale_in_context=sum(1 for h in all_hits if not h.avoid and _mentions_stale(h, task)),
        discredited_served=sum(1 for h in all_hits if h.avoid), cited_keys=cited, written_keys=written, flags=flags,
        tags=task.tags, usage=asdict(usage), reflection_usage=asdict(r_usage), memory=session.acct.as_dict(),
        wall_ms=(time.perf_counter() - t0) * 1000, steps=steps, first_action_ms=first_action_ms,
        explanation=explanation, error=error,
        transcript=transcript,
    )
