"""Render memory into the text an agent or a model reads.

One implementation of "what does served memory look like", shared by the SDK's
``Guidance`` (what a customer's agent is handed at ``begin()``), the MCP
briefing, and — through a re-export — the managed-models prompt renderer that
trains and serves tuned models. The bodies here were lifted verbatim from that
renderer so its ``RENDERER_VERSION`` golden tests stay byte-identical; anything
that changes the rendered text is a train/serve skew and must bump the version
on the consuming side.

Everything is pure. Entries come in as :class:`ContextEntry`; procedures render
as a block of numbered steps ahead of the memory context; the task's own record
of tried actions renders as a third block.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .actions import action_key as _action_key
from .evidence import is_synthetic_key, is_training_excluded_key

#: Confidence buckets, in the order they render. Coarse on purpose: a prompt keyed
#: on the second decimal of an evidence posterior would make near-identical
#: decisions look like distinct examples. The thresholds match how retrieval
#: treats the number: 0.5 is the default recall gate, 0.8 the point at which
#: briefings call it strong.
CONFIDENCE_BUCKETS: tuple[tuple[float, str], ...] = ((0.8, "high"), (0.5, "medium"), (0.0, "low"))

#: The evidence vocabulary as the model sees it; anything else renders as
#: ``untested``, which is what a trace sealed before the evidence model means.
EVIDENCE_STATUSES = frozenset({"untested", "validated", "contested", "discredited"})

#: Ceiling on injected memory. Tuned models are trained to act on a small,
#: ranked context; pasting an unbounded briefing in at serve time is both a
#: cost regression and a distribution shift away from what was trained.
MAX_CONTEXT_ENTRIES = 12
MAX_CONTEXT_CHARS = 4_000

#: Ceilings on the procedure block, for the same reason. Few and long rather
#: than many and short: a procedure is followed, not ranked, and three is
#: already more ways of doing one task than a prompt can act on.
MAX_PROCEDURES = 3
MAX_PROCEDURE_CHARS = 2_000
MAX_PROCEDURE_STEPS = 12

#: The memory type a procedure carries (``amfs_core.models.MemoryType.PROCEDURE``).
PROCEDURE_TYPE = "procedure"

#: Version of the rendering below. The managed-models renderer keeps its own
#: ``RENDERER_VERSION`` and re-exports these functions; this one lets a
#: ``guidance_id`` name the shape of the text it hashes.
RENDER_VERSION = "v4"


def confidence_bucket(confidence: float) -> str:
    for floor, label in CONFIDENCE_BUCKETS:
        if confidence >= floor:
            return label
    return CONFIDENCE_BUCKETS[-1][1]


@dataclass(frozen=True)
class ContextEntry:
    """One memory entry as the model sees it."""

    entity_path: str
    key: str
    value: Any
    confidence: float = 1.0
    evidence_status: str = "untested"
    success_count: int = 0
    failure_count: int = 0
    #: ``fact`` (also beliefs and experiences, which render the same way) or
    #: ``procedure``, which renders in its own block. Traces sealed before the
    #: type was recorded read as facts, which is how they rendered then.
    memory_type: str = "fact"
    #: The entry's version, when known. Not rendered; carried so a
    #: ``guidance_id`` can name exactly which claims the text was built from.
    version: int | None = None

    @property
    def is_procedure(self) -> bool:
        return self.memory_type == PROCEDURE_TYPE

    @property
    def is_synthetic(self) -> bool:
        """System-written (``lesson-contrast-*``): derived from outcomes, not
        authored. Agents see these in retrieve and briefing; a model prompt
        must not, at train or serve time, so the two stay the same shape."""
        return is_synthetic_key(self.key)

    @property
    def is_training_excluded(self) -> bool:
        """Out of the model's context at train and serve time: synthetic
        lessons, and the repair loop's corrective writes (``risk-*``,
        ``correction-*``). A corrective entry is knowledge an *agent* must
        read — it names the rule that stopped working — but a model trained
        on the decisions made with it would learn the failure's shadow rather
        than the behaviour, and the same entry rendered at serve time would be
        a prompt the training rows never contained."""
        return is_training_excluded_key(self.key)

    def evidence_label(self) -> str:
        status = self.evidence_status if self.evidence_status in EVIDENCE_STATUSES else "untested"
        if self.success_count or self.failure_count:
            return f"{status} {self.success_count}/{self.failure_count}"
        return status

    def _heading(self) -> str:
        return (
            f"{self.entity_path}/{self.key} "
            f"({confidence_bucket(self.confidence)}, {self.evidence_label()})"
        )

    def render(self) -> str:
        return f"- {self._heading()}: {_stringify(self.value)}"

    def render_procedure(self) -> str:
        """The procedure as steps to follow: heading and goal on the first
        line, then the steps numbered, then what to check and what to do
        when it fails, each on its own indented line when the procedure has
        one. A procedure written as a list of lines renders those lines as
        the steps; anything else renders as a single stringified step, so a
        malformed procedure still reaches the model rather than vanishing."""
        goal, steps, extras = _procedure_parts(self.value)
        lines = [f"- {self._heading()}: {goal}" if goal else f"- {self._heading()}:"]
        for index, step in enumerate(steps[:MAX_PROCEDURE_STEPS], start=1):
            lines.append(f"  {index}. {step}")
        for label, text in extras:
            lines.append(f"  {label}: {text}")
        return "\n".join(lines)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _one_line(value: Any) -> str:
    return " ".join(_stringify(value).split())


def _procedure_parts(value: Any) -> tuple[str, list[str], list[tuple[str, str]]]:
    """``(goal, steps, extras)`` of a procedure value in either shape the OSS
    core accepts: a dict with ``goal``/``steps``/``preconditions``/``verify``/
    ``on_failure``, or a string whose lines are the steps."""
    if isinstance(value, dict):
        goal = _one_line(value.get("goal") or "") if value.get("goal") else ""
        raw_steps = value.get("steps")
        steps: list[str] = []
        for step in raw_steps if isinstance(raw_steps, list) else []:
            text = step.get("action") if isinstance(step, dict) else step
            if isinstance(text, str) and text.strip():
                steps.append(_one_line(text))
            elif step is not None:
                steps.append(_one_line(step))
        extras: list[tuple[str, str]] = []
        for label in ("preconditions", "verify", "on_failure"):
            if value.get(label):
                extras.append((label, _one_line(value[label])))
        if not goal and not steps and not extras:
            steps = [_one_line(value)]
        return goal, steps, extras
    if isinstance(value, str):
        lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return "", [_strip_marker(ln) for ln in lines], []
        return "", [value.strip()], []
    return "", [_one_line(value)], []


def _strip_marker(line: str) -> str:
    """``- step``, ``* step``, ``1. step``, ``1) step`` → ``step``; the block
    numbers the steps itself."""
    text = line.lstrip("-*• ").strip()
    head, _, rest = text.partition(" ")
    if head[:-1].isdigit() and head[-1:] in ".)" and rest:
        return rest.strip()
    return text


def render_procedures(entries: list[ContextEntry]) -> str:
    """The procedure block: the procedures the agent had, highest confidence
    first, or ``""`` when it had none. Ordered and capped like the memory
    context and for the same reasons."""
    ordered = sorted(
        (e for e in entries if e.is_procedure and not e.is_training_excluded),
        key=lambda e: (-e.confidence, e.entity_path, e.key),
    )[:MAX_PROCEDURES]
    blocks: list[str] = []
    used = 0
    for entry in ordered:
        rendered = entry.render_procedure()
        if used + len(rendered) > MAX_PROCEDURE_CHARS:
            continue
        blocks.append(rendered)
        used += len(rendered)
    return "\n".join(blocks)


def render_context(entries: list[ContextEntry]) -> str:
    """Render ranked memory context, highest confidence first.

    Deterministic ordering matters beyond tidiness: the same entries in a
    different order are a different training example, which would let identical
    decisions appear as distinct rows and inflate the effective dataset size
    with duplicates the dedup pass cannot see.

    Procedures are not here: they have a block of their own
    (:func:`render_procedures`).
    """
    ordered = sorted(
        (e for e in entries if not e.is_training_excluded and not e.is_procedure),
        key=lambda e: (-e.confidence, e.entity_path, e.key),
    )[:MAX_CONTEXT_ENTRIES]

    lines: list[str] = []
    used = 0
    for entry in ordered:
        rendered = entry.render()
        if used + len(rendered) > MAX_CONTEXT_CHARS:
            # Skip, not stop: one oversized entry at the top used to render the
            # whole context to nothing, and the example was then labelled as
            # made without memory.
            continue
        lines.append(rendered)
        used += len(rendered)
    return "\n".join(lines)


@dataclass(frozen=True)
class TriedAction:
    """One action already taken in this task, as the model sees it."""

    action_key: str
    success: bool = False

    def render(self) -> str:
        return f"- {self.action_key} — {'succeeded' if self.success else 'failed'}"


#: Ceiling on the tried-actions block, for the same reason as the context
#: ceilings: a runaway session must not render a prompt shape no training row
#: ever had.
MAX_TRIED_ACTIONS = 8


def tried_actions(
    tool_calls: Any,
    attempts: Any = None,
    *,
    before_index: int | None = None,
) -> list[TriedAction]:
    """The task's own action record, in the order the actions were taken.

    An action counts as tried when a failed attempt ended on it (the last of
    the attempt's ``action_indices``) or when it was recorded with
    ``success=False``; each key appears once, at its first failure. Reads and
    diagnostics are not decisions and never appear. ``before_index`` bounds the
    record to the calls preceding the target action, so a training row shows
    what the agent knew when it chose, not what it learned afterwards.

    Shared by the dataset builder (trace rows) and the gateway (live session),
    so a served prompt has the block a training row had.
    """
    calls = _as_list(tool_calls)
    boundary = len(calls) if before_index is None else max(0, min(before_index, len(calls)))
    failed_indices: set[int] = set()
    for attempt in _as_list(attempts):
        if not isinstance(attempt, dict):
            attempt = getattr(attempt, "model_dump", lambda **_: {})(mode="json")
        indices = attempt.get("action_indices") or []
        if indices:
            last = indices[-1]
            if isinstance(last, int) and not isinstance(last, bool):
                failed_indices.add(last)
    out: list[TriedAction] = []
    seen: set[str] = set()
    for index, call in enumerate(calls[:boundary]):
        if not isinstance(call, dict):
            call = getattr(call, "model_dump", lambda **_: {})(mode="json")
        tool_name = call.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        failed = index in failed_indices or call.get("success") is False
        if not failed:
            continue
        key = _action_key(
            tool_name, call.get("arguments") or {}, explicit=call.get("action_key") or None
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(TriedAction(action_key=key, success=False))
        if len(out) >= MAX_TRIED_ACTIONS:
            break
    return out


def _as_list(raw: Any) -> list[Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except ValueError:
            return []
    return raw if isinstance(raw, list) else []


def render_tried(tried: list[TriedAction] | None) -> str:
    """The tried-actions block, or ``""`` when nothing has been tried."""
    if not tried:
        return ""
    return "\n".join(t.render() for t in tried[:MAX_TRIED_ACTIONS])


def guidance_id(entries: list[ContextEntry], *, branch: str = "main") -> str:
    """A short stable name for *what was served*: the branch, the exact
    ``entity/key@version`` set, and the render version. Two sessions handed the
    same claims get the same id; a new version of one entry changes it. Stamped
    on the session so a trace can say which guidance the agent saw without
    storing the rendered text twice."""
    parts = sorted(
        f"{e.entity_path}/{e.key}@{e.version if e.version is not None else '?'}" for e in entries
    )
    raw = "\n".join([branch or "main", RENDER_VERSION, *parts])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "CONFIDENCE_BUCKETS",
    "EVIDENCE_STATUSES",
    "MAX_CONTEXT_CHARS",
    "MAX_CONTEXT_ENTRIES",
    "MAX_PROCEDURES",
    "MAX_PROCEDURE_CHARS",
    "MAX_PROCEDURE_STEPS",
    "MAX_TRIED_ACTIONS",
    "PROCEDURE_TYPE",
    "RENDER_VERSION",
    "ContextEntry",
    "TriedAction",
    "confidence_bucket",
    "guidance_id",
    "render_context",
    "render_procedures",
    "render_tried",
    "tried_actions",
]
