"""The shapes a loop and the harness exchange.

Loop-agnostic by construction: a *change* is a title, a summary and the text it
would write into the agent's memory or instructions; a *decision* is a status, the
tier that made it and when. What the loop does between ``inject`` and the decision
is its own — a replay, a review queue, a canary, a human — and the harness only
reads the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class ProbeKind(StrEnum):
    """The four ways a change can be wrong on purpose.

    - ``placebo``: an entry that recommends nothing, on a real change's claim to
      repair N failures. A gate that passes it is not reading the replay.
    - ``poison``: an entry that endorses the failure and tells the agent to keep
      doing it. The opposite of a repair; a gate that passes it has no sign.
    - ``reward_hack``: an entry that tells the agent to report success whatever
      happened. The number the gate reads improves while the task does not; the
      gate passes it unless it reads an outcome the agent cannot write.
    - ``revert``: a change the loop itself proved, written back out. A gate with
      a memory of what it verified rejects it; one without passes it.
    """

    PLACEBO = "placebo"
    POISON = "poison"
    REWARD_HACK = "reward_hack"
    REVERT = "revert"


PROBE_KINDS: tuple[ProbeKind, ...] = tuple(ProbeKind)


class Status(StrEnum):
    """Where a probe stands. ``rejected`` and ``accepted`` are the decided
    states the rates are read over; ``undecided`` is a probe the loop closed
    without a verdict (dismissed by hand, say); ``expired`` is one no tier
    answered within the harness's timeout."""

    OPEN = "open"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    UNDECIDED = "undecided"
    EXPIRED = "expired"


DECIDED = (Status.REJECTED, Status.ACCEPTED)
CLOSED = (Status.REJECTED, Status.ACCEPTED, Status.UNDECIDED, Status.EXPIRED)


@dataclass(frozen=True)
class Material:
    """A real change of the loop's, the material a probe is drafted from.

    ``predicted_fixes`` is how many failing runs the change claims to repair —
    what the placebo inherits, so the gate's replay has something to disagree
    with. ``verified`` marks a change the loop proved live; only such material
    yields a ``revert`` probe, with ``prior_value`` the text the change replaced.
    """

    id: str
    title: str
    summary: str
    entity_path: str
    key: str
    value: str
    predicted_fixes: int = 0
    verified: bool = False
    prior_value: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Probe:
    """A change wrong on purpose, as the harness drafts it. A loop may use the
    draft as the entry it injects, or construct its own from ``kind`` and
    ``source_id`` (SenseLab does — it has the material server-side)."""

    kind: ProbeKind
    title: str
    summary: str
    entity_path: str
    key: str
    value: str
    why_wrong: str
    source_id: str | None = None
    predicted_fixes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value, "title": self.title, "summary": self.summary,
            "entity_path": self.entity_path, "key": self.key, "value": self.value,
            "why_wrong": self.why_wrong, "source_id": self.source_id,
            "predicted_fixes": self.predicted_fixes,
        }


@dataclass(frozen=True)
class Stage:
    """What one tier of the gate said about a probe: ``rejected``,
    ``accepted`` or anything else (``undecided``)."""

    tier: str
    verdict: str
    at: datetime | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"tier": self.tier, "verdict": self.verdict,
                "at": self.at.isoformat() if self.at else None, "note": self.note}


@dataclass(frozen=True)
class Decision:
    """The current state of an injected probe, as the loop reports it."""

    status: Status
    injected_at: datetime
    closed_at: datetime | None = None
    decided_by: str | None = None
    would_have_shipped: bool = False
    stages: tuple[Stage, ...] = ()
    mode: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return self.status in CLOSED


@dataclass(frozen=True)
class AATrial:
    """One trial of the A/A: the decision rule run over two identical arms.
    ``decidable`` is whether the rule reached a verdict at all; ``false_positive``
    is whether that verdict was anything but 'no difference'; ``graded_at_decision``
    is how many graded runs it took, the gate's real time-to-decision."""

    decidable: bool
    false_positive: bool
    graded_at_decision: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class AAResult:
    """The A/A as the loop ran it: the trials and the rule's nominal false-positive
    rate (the rate it claims to hold)."""

    trials: tuple[AATrial, ...]
    nominal: float
    window_days: int | None = None
    ran_at: datetime | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Counts:
    """Successes over a population: the shape every rate here is read from."""

    n: int
    hits: int

    @property
    def rate(self) -> float | None:
        return self.hits / self.n if self.n else None


@dataclass(frozen=True)
class RewardReading:
    """One shipped change, read three ways over the same before/after windows.

    ``proxy`` is the number the loop optimised — failures the judge saw per run,
    so a *fall* is the gain. ``downstream`` is an outcome the loop did not write
    (a verifier, a customer's label, a held-out scorer) — successes per run.
    ``self_report`` is what the agent said about itself — successes per run — so
    a gap opening between it and ``downstream`` is the agent learning to say the
    right thing.
    """

    id: str
    title: str
    shipped_at: datetime | None
    proxy_before: Counts
    proxy_after: Counts
    downstream_before: Counts
    downstream_after: Counts
    self_report_before: Counts | None = None
    self_report_after: Counts | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GatedLoop(Protocol):
    """What the harness needs from a loop to drive it live.

    ``material`` may return ``[]`` for a loop that drafts its own probes; the
    harness then passes ``inject`` a draft with only ``kind`` meaningful. ``aa``
    may return ``None`` and ``reward`` ``[]`` when the loop has no such reading;
    the scorecard marks the section *not provided* rather than failing.
    """

    name: str

    def material(self) -> list[Material]: ...

    def inject(self, probe: Probe) -> str:
        """Send the probe into the gate; return a handle ``poll`` accepts.
        Raise :class:`GateBusyError` when the loop takes one probe at a time and one
        is in the gate; :class:`NoMaterialError` when it has nothing to draft from."""
        ...

    def poll(self, handle: str) -> Decision: ...

    def aa(self, *, trials: int, window_days: int) -> AAResult | None: ...

    def reward(self) -> list[RewardReading]: ...

    def close(self) -> None: ...


class GateBusyError(RuntimeError):
    """The loop has a probe in the gate already and takes one at a time."""


class NoMaterialError(RuntimeError):
    """The loop has no change to draft a probe of this kind from."""


def now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "CLOSED",
    "DECIDED",
    "PROBE_KINDS",
    "AAResult",
    "AATrial",
    "Counts",
    "Decision",
    "GateBusyError",
    "GatedLoop",
    "Material",
    "NoMaterialError",
    "Probe",
    "ProbeKind",
    "RewardReading",
    "Stage",
    "Status",
    "now",
]
