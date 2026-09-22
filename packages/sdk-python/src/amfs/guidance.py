"""What a customer's agent is handed before it acts, as one object.

A briefing and a retrieve return ranked memory; an agent then has to decide
how much of it to trust and how to put it in front of its model. ``Guidance``
does both once: it carries the procedures that apply to this run (and the ones
that do not, with the reason), the facts worth citing, the action priors and
the recommendation, a ``strength`` label — ``strong`` / ``thin`` / ``none`` —
and a ``text`` rendering from the shared renderer (``amfs_core.render``), so
what the agent sees is the same shape a tuned model was trained on.

``guidance_id`` names exactly what was served: the branch, the entries and
their versions, the render version. The :class:`amfs.run.Run` façade stamps it
on the session so the sealed trace can say which guidance the agent saw
without storing the rendered text twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

from amfs_core.actions import (
    guidance_strength,
    plan_actions,
    pooled_classes,
    priors_local,
    render_priors,
)
from amfs_core.lessons import lesson_of
from amfs_core.render import (
    ContextEntry,
    render_context,
    render_procedures,
)
from amfs_core.render import (
    guidance_id as _guidance_id,
)

#: Strength labels, in order of how much an agent should lean on the text.
STRENGTHS = ("strong", "thin", "none")


@dataclass
class Guidance:
    """Served memory for one task, ready to inject."""

    #: The rendered blocks, procedures first, then context, then priors and the
    #: recommendation. Empty when there is nothing to show.
    text: str
    #: How much to lean on it: ``strong`` (validated knowledge or a winning
    #: action here), ``thin`` (evidence, none of it confirmed), ``none``
    #: (nothing, or only untested notes — treat the text as hints).
    strength: str
    #: Names what was served; stamped on the session by :class:`amfs.run.Run`.
    guidance_id: str
    #: The memory branch the guidance was read from (``main`` or a canary).
    branch: str = "main"
    #: Every entry the text was built from, as the renderer saw it.
    entries: list[ContextEntry] = field(default_factory=list)
    #: Procedures that apply to this run: ``{key, entity_path, goal, steps,
    #: applicability, ...}`` rows from the briefing's ``procedures`` section.
    procedures: list[dict[str, Any]] = field(default_factory=list)
    #: Procedures that exist but whose environment preconditions this run
    #: contradicts, each with ``applicability_detail`` naming which.
    not_applicable: list[dict[str, Any]] = field(default_factory=list)
    #: The action priors block from ``retrieve(include_priors=True)``, if any.
    priors: dict[str, Any] | None = None
    #: ``{"mode": "act"|"explore"|"escalate"|"abstain", "suggested_action", "why"}``.
    recommendation: dict[str, Any] | None = None
    #: Whether the scope shows a regime shift (long-validated rules failing).
    regime_shift: bool = False
    #: Why the guidance is empty when memory could not be reached: the
    #: transport error, as text. ``None`` when the read succeeded (an empty
    #: guidance with no error means memory had nothing to say).
    error: str | None = None
    #: The actions the run said it could take, when it said so. A plan computed
    #: on the client (:attr:`plan`) is drawn from these only: a retry that asks
    #: for guidance over the actions it has not tried must not be handed the
    #: one that just failed because the situation's record has it as a winner.
    candidate_actions: list[str] | None = None

    @property
    def mode(self) -> str | None:
        """The recommendation's mode, or ``None``."""
        return (self.recommendation or {}).get("mode")

    @property
    def suggested_action(self) -> str | None:
        return (self.recommendation or {}).get("suggested_action")

    @property
    def plan(self) -> list[str]:
        """The order to try actions in, for the whole budget — the
        recommendation's action first, then what has a winning record here,
        then the untried candidates, never what failed here (see
        :func:`amfs_core.actions.plan_actions`). Computed from the priors
        when the server did not send one. Empty when there is nothing to order."""
        if self.mode not in ("act", "explore"):
            # An abstain, or no recommendation at all: the record here is not
            # about this task, and an order drawn from it would be advice with
            # nothing behind it.
            return []
        sent = (self.recommendation or {}).get("plan")
        if isinstance(sent, list) and sent:
            plan = [str(a) for a in sent]
        elif self.priors:
            plan = plan_actions(
                self.priors, self.recommendation, candidate_actions=self.candidate_actions,
            )
        else:
            return []
        if self.candidate_actions:
            allowed = set(self.candidate_actions)
            plan = [a for a in plan if a in allowed]
        return plan

    @property
    def next_action(self) -> str | None:
        """The first action of the plan, or the suggested action, or ``None``."""
        plan = self.plan
        return plan[0] if plan else self.suggested_action

    @property
    def lessons(self) -> list[dict[str, Any]]:
        """The structured lessons among the shown entries, each as ``{"key":
        "entity_path/key", "situation", "action", "worked", "text",
        "evidence_status"}`` in rendered order. What :meth:`amfs.run.Run.learn`
        wrote and this run was shown."""
        out: list[dict[str, Any]] = []
        for e in self.shown:
            lesson = lesson_of(e.value)
            if lesson is None:
                continue
            out.append({
                "key": f"{e.entity_path}/{e.key}",
                "situation": lesson["situation"],
                "action": lesson["action"],
                "worked": lesson["worked"],
                "text": lesson.get("text"),
                "evidence_status": e.evidence_status,
            })
        return out

    def lessons_claiming(self, action: str, *, worked: bool = True) -> list[str]:
        """Keys of the shown lessons that claim *action* worked (or, with
        ``worked=False``, did not). The causal keys for an outcome of taking
        that action: a run that took ``fix:fix_code`` because a lesson said
        so credits — or charges — that lesson and not the others it was
        shown. Pass the result as *causal_entry_keys* to
        :meth:`amfs.run.Run.complete` or :meth:`amfs.run.Run.attempt_failed`."""
        return [
            lesson["key"] for lesson in self.lessons
            if lesson["action"] == action and bool(lesson["worked"]) is worked
        ]

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def shown(self) -> list[ContextEntry]:
        """The entries the rendered text actually shows, in its order: the
        renderer omits training-excluded keys (``lesson-contrast-*``,
        ``risk-*``, ``correction-*``), so those are not something the agent
        could have acted on and must not be blamed for what it did."""
        return [e for e in self.entries if not e.is_training_excluded]

    @property
    def entry_keys(self) -> list[str]:
        """Every shown entry as ``entity_path/key``, in the order rendered
        (highest confidence first). The form :meth:`amfs.run.Run.complete`
        and :meth:`amfs.run.Run.attempt_failed` take as *causal_entry_keys*."""
        return [f"{e.entity_path}/{e.key}" for e in self.shown]

    @property
    def top_key(self) -> str | None:
        """The entry the agent most plausibly acted on when it cited nothing:
        the highest-ranked shown non-procedure entry, or the first shown
        procedure when that is all there was. ``None`` for empty guidance."""
        shown = self.shown
        for e in shown:
            if not e.is_procedure:
                return f"{e.entity_path}/{e.key}"
        return f"{shown[0].entity_path}/{shown[0].key}" if shown else None

    def should_inject(self) -> bool:
        """The default policy for a customer's agent: inject when there is text
        and the strength is not ``none``. A caller who wants to show hints too
        can inject on ``not is_empty`` instead."""
        return not self.is_empty and self.strength != "none"

    @classmethod
    def unavailable(cls, error: str, *, branch: str = "main") -> Guidance:
        """The guidance for a run whose memory read failed: empty, strength
        ``none``, the error kept. The agent runs on its own for this task,
        which is what it did before memory; the alternative — an exception
        out of ``begin`` — took the whole worker thread down on the demo when
        one ``/search`` timed out."""
        return cls(text="", strength="none", guidance_id=_guidance_id([], branch=branch),
                   branch=branch, error=error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "guidance_id": self.guidance_id,
            "strength": self.strength,
            "branch": self.branch,
            "mode": self.mode,
            "suggested_action": self.suggested_action,
            "procedures": [p.get("key") for p in self.procedures],
            "not_applicable": [p.get("key") for p in self.not_applicable],
            "entries": [f"{e.entity_path}/{e.key}" for e in self.entries],
            "regime_shift": self.regime_shift,
            "plan": self.plan,
            "error": self.error,
        }

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        digests: list[Any] | None = None,
        hits: list[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
        branch: str = "main",
        entity_path: str | None = None,
        candidate_actions: Sequence[str] | None = None,
    ) -> Guidance:
        """Assemble guidance from what the SDK already returns.

        *digests* are ``AgentMemory.briefing(...)`` results (the lead entity
        digest's sections are read); *hits* are ``AgentMemory.retrieve(...)``
        results (``ScoredEntry``); *meta* is ``AgentMemory.last_priors`` — the
        trailing ``_meta`` element with ``priors``, ``recommendation``,
        ``regime_shift``, ``guidance_strength`` and ``not_applicable``.
        Any of them may be missing; the guidance is built from what is there.
        """
        meta = dict(meta or {})
        lead = _lead_summary(digests or [], entity_path)
        entries: dict[str, ContextEntry] = {}

        for hit in hits or []:
            entry = getattr(hit, "entry", hit)
            if getattr(hit, "breakdown", None) and hit.breakdown.get("_avoid"):
                continue
            ce = _context_entry(entry)
            entries.setdefault(f"{ce.entity_path}/{ce.key}", ce)

        procedures = [dict(p) for p in (lead.get("procedures") or []) if isinstance(p, dict)]
        not_applicable = [
            dict(p) for p in (lead.get("procedures_not_applicable") or []) if isinstance(p, dict)
        ]
        for row in meta.get("not_applicable") or []:
            if isinstance(row, dict):
                not_applicable.append({
                    "key": row.get("key"), "entity_path": row.get("entity_path"),
                    "applicability": "not_applicable",
                    "applicability_detail": row.get("why") or [],
                })
        dropped = {(p.get("entity_path"), p.get("key")) for p in not_applicable}
        entries = {
            k: e for k, e in entries.items() if (e.entity_path, e.key) not in dropped
        }

        priors = meta.get("priors")
        recommendation = meta.get("recommendation")
        regime_shift = bool(meta.get("regime_shift") or lead.get("regime_shift"))
        strength = _strength(lead, meta, priors, list(entries.values()), regime_shift)

        blocks: list[str] = []
        ordered = list(entries.values())
        proc_block = render_procedures(ordered)
        if proc_block:
            blocks.append("Procedures:\n" + proc_block)
        ctx_block = render_context(ordered)
        if ctx_block:
            blocks.append("Memory context:\n" + ctx_block)
        if not_applicable:
            lines = []
            for p in not_applicable[:3]:
                why = "; ".join(str(d) for d in (p.get("applicability_detail") or []))
                lines.append(f"- {p.get('entity_path')}/{p.get('key')}" + (f" ({why})" if why else ""))
            blocks.append("Not for this run:\n" + "\n".join(lines))
        priors_block = render_priors(priors, recommendation)
        if priors_block:
            blocks.append(priors_block)
        if regime_shift:
            blocks.append(
                "Regime shift suspected: long-validated entries here started failing; "
                "verify before reusing them."
            )
        text = "\n\n".join(blocks)
        if text and strength == "none":
            text = "(Guidance strength: none — nothing here is validated; treat as hints.)\n\n" + text

        return cls(
            text=text,
            strength=strength,
            guidance_id=_guidance_id(ordered, branch=branch),
            branch=branch,
            entries=ordered,
            procedures=procedures,
            not_applicable=not_applicable,
            priors=priors if isinstance(priors, dict) else None,
            recommendation=recommendation if isinstance(recommendation, dict) else None,
            regime_shift=regime_shift,
            candidate_actions=[str(a) for a in candidate_actions] if candidate_actions else None,
        )


def _lead_summary(digests: list[Any], entity_path: str | None) -> dict[str, Any]:
    for d in digests:
        dtype = getattr(d, "digest_type", None)
        dtype = str(getattr(dtype, "value", dtype) or "")
        if dtype == "entity" and (entity_path is None or getattr(d, "scope", None) == entity_path):
            summary = getattr(d, "summary", None)
            return dict(summary) if isinstance(summary, Mapping) else {}
    for d in digests:
        summary = getattr(d, "summary", None)
        if isinstance(summary, Mapping):
            return dict(summary)
    return {}


def _context_entry(entry: Any) -> ContextEntry:
    mt = getattr(entry, "memory_type", None)
    return ContextEntry(
        entity_path=str(getattr(entry, "entity_path", "")),
        key=str(getattr(entry, "key", "")),
        value=getattr(entry, "value", None),
        confidence=float(getattr(entry, "confidence", 1.0) or 0.0),
        evidence_status=str(getattr(entry, "evidence_status", "untested") or "untested"),
        success_count=int(getattr(entry, "success_count", 0) or 0),
        failure_count=int(getattr(entry, "failure_count", 0) or 0),
        memory_type=str(getattr(mt, "value", mt) or "fact"),
        version=getattr(entry, "version", None),
    )


def _strength(
    lead: Mapping[str, Any],
    meta: Mapping[str, Any],
    priors: Any,
    entries: list[ContextEntry],
    regime_shift: bool,
) -> str:
    """The server's label when it sent one, else computed from what we have.
    ``strong`` from either source wins; otherwise the weaker of the two is
    taken, because a scope the briefing rated ``none`` is not made ``thin`` by
    hits the retrieve happened to find.

    One exception: when the retrieve's own neighbourhood is pooled — its
    priors say near-identical tasks here were resolved by contradicting
    actions and ``recommend`` abstained — the briefing's ``strong`` does not
    stand. That label is rated over the entity's whole record, and the winner
    it saw is the very action that won on the other class half the time; the
    retrieve is the reading scoped to this task, and it says ``thin``."""
    briefing_label = lead.get("guidance_strength")
    retrieve_label = meta.get("guidance_strength")
    if _neighbourhood_pooled(meta, priors) and retrieve_label in STRENGTHS:
        return retrieve_label
    labels = [s for s in (briefing_label, retrieve_label) if s in STRENGTHS]
    if not labels:
        statuses = [e.evidence_status for e in entries]
        return guidance_strength(
            priors if isinstance(priors, Mapping) else None, statuses, regime_shift=regime_shift
        )
    if "strong" in labels:
        return "strong"
    return "none" if "none" in labels else "thin"


def _neighbourhood_pooled(meta: Mapping[str, Any], priors: Any) -> bool:
    """Whether the retrieve's local priors describe two kinds of task pooled
    under one description: the recommendation carries ``pooled`` when the
    caller asked to be told, and the contrasts say so themselves when it did
    not. Entity-wide priors are never read for it (see ``priors_local``)."""
    rec = meta.get("recommendation")
    if isinstance(rec, Mapping) and rec.get("pooled"):
        return True
    if not isinstance(priors, Mapping) or not priors_local(priors):
        return False
    return pooled_classes(list(priors.get("contrasts") or [])) is not None


__all__ = ["Guidance", "STRENGTHS"]
