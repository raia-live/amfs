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
from typing import Any

from amfs_core.actions import guidance_strength, render_priors
from amfs_core.render import (
    ContextEntry,
    guidance_id as _guidance_id,
    render_context,
    render_procedures,
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

    @property
    def mode(self) -> str | None:
        """The recommendation's mode, or ``None``."""
        return (self.recommendation or {}).get("mode")

    @property
    def suggested_action(self) -> str | None:
        return (self.recommendation or {}).get("suggested_action")

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def entry_keys(self) -> list[str]:
        """Every served entry as ``entity_path/key``, in the order rendered
        (highest confidence first). The form :meth:`amfs.run.Run.complete`
        and :meth:`amfs.run.Run.attempt_failed` take as *causal_entry_keys*."""
        return [f"{e.entity_path}/{e.key}" for e in self.entries]

    @property
    def top_key(self) -> str | None:
        """The entry the agent most plausibly acted on when it cited nothing:
        the highest-ranked non-procedure entry, or the first procedure when
        that is all there was. ``None`` for empty guidance."""
        for e in self.entries:
            if not e.is_procedure:
                return f"{e.entity_path}/{e.key}"
        return self.entry_keys[0] if self.entries else None

    def should_inject(self) -> bool:
        """The default policy for a customer's agent: inject when there is text
        and the strength is not ``none``. A caller who wants to show hints too
        can inject on ``not is_empty`` instead."""
        return not self.is_empty and self.strength != "none"

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
    hits the retrieve happened to find."""
    labels = [
        s for s in (lead.get("guidance_strength"), meta.get("guidance_strength"))
        if s in STRENGTHS
    ]
    if not labels:
        statuses = [e.evidence_status for e in entries]
        return guidance_strength(
            priors if isinstance(priors, Mapping) else None, statuses, regime_shift=regime_shift
        )
    if "strong" in labels:
        return "strong"
    return "none" if "none" in labels else "thin"


__all__ = ["Guidance", "STRENGTHS"]
