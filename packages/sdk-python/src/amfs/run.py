"""The three seams a customer wires SenseLab into an agent they control.

SenseLab never runs the customer's agent. What it can do is stand at three
points of a run and be useful at each:

* ``begin(task)`` — before the agent acts: hand it :class:`~amfs.guidance.Guidance`
  built from the briefing and a retrieve, scoped to this run's environment,
  and stamp *which* guidance was served on the session.
* ``on_tool_result(...)`` — after each consequential tool call: record what was
  done (the half of a trace SenseLab cannot see on its own) and, when the call
  failed, look up what worked for that failure here and hand it back.
* ``complete(...)`` — when the run ends: commit the outcome with its provenance
  (who decided it was a success) so the trace can be judged and the memory it
  used can be reinforced or discredited.

Everything here composes calls ``AgentMemory`` already has; there is no new
server surface. A framework hook (CrewAI, LangGraph, Strands, a plain loop)
calls the three methods and the loop closes: guidance in, actions out, outcome
in, and the next ``begin`` on the same entity reads what this run taught.

Example::

    run = Run(mem)
    guidance = run.begin("fix the failing CI on PR 42", entity_path="acme/ci",
                         agent_version="ci-bot@1.4.0", runtime="python3.12")
    if guidance.should_inject():
        prompt = guidance.text + "\\n\\n" + prompt
    ...
    hint = run.on_tool_result("shell", {"cmd": "pip install -r requirements.txt"},
                              result=stderr, success=False)
    if hint is not None and hint.should_inject():
        prompt += hint.text
    ...
    run.complete(True, verified_by="ci", evidence={"run_id": "98765"},
                 response_text=final_answer,
                 causal_entry_keys=cited or [guidance.top_key])

Name what the agent acted on. A ``begin`` serves several entries and the
agent follows one; ``causal_entry_keys`` on ``attempt_failed`` and
``complete`` is how the outcome reaches that one and not the others. Ask the
model for the keys it used (they are in the rendered text) and fall back to
``guidance.top_key``. Left unnamed, every entry read is charged for every
failure, and a correct lesson that merely shared the context with a wrong
one is discredited alongside it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from amfs_core.models import ENVIRONMENT_KEYS, MemoryEntry, OutcomeType

from amfs.guidance import Guidance
from amfs.memory import GUIDANCE_COUNT_ATTRIBUTE, GUIDANCE_ID_ATTRIBUTE, AgentMemory

logger = logging.getLogger(__name__)

#: Hits fetched for the opening guidance and for a failure hint.
BEGIN_LIMIT = 8
HINT_LIMIT = 3
#: How much of a failed tool result becomes the failure query.
HINT_RESULT_CHARS = 300


class Run:
    """One task of a customer's agent, seen from SenseLab's side.

    *assign_branch* is an optional hook ``(agent_id, unit) -> branch | None``
    for a canary assignment (the Pro API offers one); when it names a branch
    the run reads memory from it and writes stay on ``main``. Without the hook,
    or when it fails, the run reads ``main``.
    """

    def __init__(
        self,
        memory: AgentMemory,
        *,
        assign_branch: Callable[[str, str | None], str | None] | None = None,
    ) -> None:
        self.memory = memory
        self._assign_branch = assign_branch
        self.entity_path: str | None = None
        self.task_input: str | None = None
        self.guidances: list[Guidance] = []
        self._candidate_actions: list[str] | None = None
        self._situation: str | None = None
        self._completed = False

    # ------------------------------------------------------------------
    # begin
    # ------------------------------------------------------------------

    def begin(
        self,
        task: str | Mapping[str, Any],
        *,
        entity_path: str,
        environment: Mapping[str, Any] | None = None,
        model: str | None = None,
        agent_version: str | None = None,
        runtime: str | None = None,
        unit: str | None = None,
        candidate_actions: list[str] | None = None,
        situation: str | None = None,
        limit: int = BEGIN_LIMIT,
    ) -> Guidance:
        """Open the run and return the guidance to inject.

        *task* is the request as it arrived (a string, or a dict that is
        JSON-encoded for the query and kept as ``task_input``). *model*,
        *agent_version* and *runtime* — or an *environment* dict of the same
        keys — describe this run so procedures are scoped to it and a later
        comparison can hold them stable. *unit* is the assignment unit for a
        canary (a customer id, a repo), passed to the hook if one was given.
        *candidate_actions* are the ``tool:action`` keys the agent could take,
        so the priors can name the untried ones.
        """
        self.entity_path = entity_path
        self.task_input = task if isinstance(task, str) else json.dumps(task, default=str)
        self._candidate_actions = list(candidate_actions) if candidate_actions else None
        self._situation = situation

        # The explicit keywords win over the mapping, but only when given:
        # an unset keyword must not erase what ``environment`` carried.
        merged: dict[str, Any] = dict(environment or {})
        for key, value in (("model", model), ("agent_version", agent_version), ("runtime", runtime)):
            if value is not None:
                merged[key] = value
        env: dict[str, str] = {}
        for key, value in merged.items():
            if key in ENVIRONMENT_KEYS and isinstance(value, str) and value.strip():
                env[key] = value.strip()
        if env:
            # Attributes, not identity metadata: they reach the sealed trace on
            # every path and ``AgentMemory.environment()`` reads them back.
            self.memory.set_session_attributes(env)

        branch = self._checkout(unit)
        try:
            digests = self.memory.briefing(
                entity_path=entity_path, compact=True, credit_reuse=True
            )
        except Exception:  # noqa: BLE001 - a briefing is a bonus, the retrieve is the floor
            logger.debug("briefing unavailable for %s", entity_path, exc_info=True)
            digests = []
        hits = self.memory.retrieve(
            self.task_input,
            entity_path=entity_path,
            include_priors=True,
            candidate_actions=self._candidate_actions,
            situation=situation,
            compact=True,
            limit=limit,
            abstain=True,
        )
        guidance = Guidance.build(
            digests=digests, hits=hits, meta=self.memory.last_priors,
            branch=branch, entity_path=entity_path,
        )
        self._stamp(guidance)
        return guidance

    def _checkout(self, unit: str | None) -> str:
        if self._assign_branch is None:
            return self.memory.branch
        try:
            branch = self._assign_branch(self.memory.agent_id, unit)
        except Exception:  # noqa: BLE001 - assignment is best-effort
            branch = None
        if branch and branch != self.memory.branch:
            self.memory.checkout(branch)
        return self.memory.branch

    def _stamp(self, guidance: Guidance) -> None:
        self.guidances.append(guidance)
        try:
            self.memory.set_session_attributes({
                GUIDANCE_ID_ATTRIBUTE: guidance.guidance_id,
                GUIDANCE_COUNT_ATTRIBUTE: len(self.guidances),
            })
        except (TypeError, ValueError):
            # A bag at the key cap: the guidance still applies, it just is not
            # named on the trace.
            pass

    # ------------------------------------------------------------------
    # on_tool_result
    # ------------------------------------------------------------------

    def on_tool_result(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        result: Any = "",
        *,
        success: bool = True,
        action_key: str | None = None,
        duration_ms: int = 0,
        candidate_actions: list[str] | None = None,
        guide_on_failure: bool = True,
    ) -> Guidance | None:
        """Record a tool call; on a failure, return guidance for it.

        The record is what adherence and step-level attribution key on, so pass
        *action_key* when the tool's choice is one of a fixed set
        (``"resolve:refund"``). On ``success=False`` with *guide_on_failure*
        the run asks memory what worked for this failure on this entity —
        priors and procedures included — and returns it as guidance, or
        ``None`` when there is nothing worth showing.
        """
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        self.memory.record_action(
            tool_name,
            dict(arguments or {}),
            result=text,
            duration_ms=duration_ms,
            success=success,
            action_key=action_key,
        )
        if success or not guide_on_failure or not self.entity_path:
            return None
        query = f"{tool_name} failed: {text[:HINT_RESULT_CHARS]}"
        hits = self.memory.retrieve(
            query,
            entity_path=self.entity_path,
            include_priors=True,
            candidate_actions=candidate_actions or self._candidate_actions,
            situation=self._situation,
            compact=True,
            limit=HINT_LIMIT,
            abstain=True,
        )
        guidance = Guidance.build(
            hits=hits, meta=self.memory.last_priors,
            branch=self.memory.branch, entity_path=self.entity_path,
        )
        if guidance.is_empty:
            return None
        self._stamp(guidance)
        return guidance

    # ------------------------------------------------------------------
    # attempt_failed
    # ------------------------------------------------------------------

    def attempt_failed(
        self,
        summary: str | None = None,
        *,
        outcome_type: OutcomeType | str = OutcomeType.MINOR_FAILURE,
        causal_entry_keys: list[str | None] | None = None,
    ) -> None:
        """Close the approach just tried as failed before trying another.

        *causal_entry_keys* names the entries the attempt acted on — what the
        agent cited, or :attr:`Guidance.top_key` when it cited nothing. Pass
        them: a ``begin`` serves several entries and the agent follows one,
        and without the names the whole read window is charged for the
        failure, so correct lessons that merely shared the context lose
        confidence with the one that was wrong. Bare keys (no ``/``) are
        qualified with the run's ``entity_path``. Without them the memory read
        since the previous boundary receives the failure at ``complete``;
        what is read afterwards is credited with the outcome.
        """
        self.memory.record_attempt(
            outcome_type=outcome_type,
            summary=summary,
            causal_entry_keys=self._qualify(causal_entry_keys),
        )

    def _qualify(self, keys: list[str | None] | None) -> list[str] | None:
        """``entity_path/key`` for every key, qualifying bare ones with the
        run's entity. ``None`` stays ``None`` (the caller declined to name
        anything, and ``AgentMemory`` falls back to the read window)."""
        if keys is None:
            return None
        out: list[str] = []
        for key in keys:
            # ``cited or [guidance.top_key]`` yields ``[None]`` on empty
            # guidance; a None is "nothing to name", not an entry called None.
            if key is None:
                continue
            key = str(key).strip()
            if not key:
                continue
            if "/" not in key and self.entity_path:
                key = f"{self.entity_path}/{key}"
            if key not in out:
                out.append(key)
        return out

    # ------------------------------------------------------------------
    # complete
    # ------------------------------------------------------------------

    def complete(
        self,
        outcome: bool | OutcomeType | str,
        *,
        outcome_ref: str | None = None,
        verified_by: str | None = None,
        evidence: Mapping[str, Any] | None = None,
        task_input: str | None = None,
        response_text: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        decision_summary: str | None = None,
        causal_entry_keys: list[str | None] | None = None,
    ) -> list[MemoryEntry]:
        """Commit the run's outcome and seal its trace.

        *outcome* is ``True`` / ``False`` or an ``OutcomeType``. *verified_by*
        names who decided it when the agent did not (``"ci"``, ``"human"``,
        ``"verifier"``); without it the outcome is the agent's own declaration.
        *task_input* defaults to what ``begin`` was given.

        *causal_entry_keys* names the entries the final decision acted on and
        is credited with the outcome (see :meth:`attempt_failed`); ``None``
        credits everything read since the last attempt boundary, ``[]``
        credits nothing. Bare keys are qualified with the run's entity.
        """
        if isinstance(outcome, bool):
            otype = OutcomeType.SUCCESS if outcome else OutcomeType.FAILURE
        elif isinstance(outcome, OutcomeType):
            otype = outcome
        else:
            otype = OutcomeType(str(outcome).lower())
        ref = outcome_ref or f"run-{self.memory.session_id[:8]}"
        entries = self.memory.commit_outcome(
            ref,
            otype,
            decision_summary=decision_summary,
            task_input=task_input if task_input is not None else self.task_input,
            response_text=response_text,
            attributes=dict(attributes or {}),
            entity_path=self.entity_path,
            situation=self._situation,
            verified_by=verified_by,
            evidence=evidence,
            causal_entry_keys=self._qualify(causal_entry_keys),
        )
        self._completed = True
        return entries

    @property
    def completed(self) -> bool:
        return self._completed


__all__ = ["Run", "BEGIN_LIMIT", "HINT_LIMIT"]
