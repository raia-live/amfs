"""Evidence labels: the one word an agent sees next to an entry.

Kept free of model imports so both ``models.MemoryEntry`` and the evidence
module can use it. Two rules, selected by ``AMFS_EVIDENCE_LABELS``:

``posterior`` (default)
    Judge the record, not the last event. An entry with four successes and one
    failure is still a rule worth acting on; calling it ``contested`` — the same
    word a 1/1 entry gets — made agents in the benchmark abandon working rules
    after a single noisy failure. ``validated`` therefore means *the record
    supports acting on this*: a clean run, or a posterior of at least
    :data:`VALIDATED_MIN_P` over at least :data:`VALIDATED_MIN_N` outcomes, or at
    least :data:`VALIDATED_MIN_WINS` successes with at most one failure.

``strict``
    The previous rule: ``validated`` only while every outcome succeeded. Kept
    for callers whose downstream logic depends on that reading.
"""

from __future__ import annotations

import os

LABELS_ENV = "AMFS_EVIDENCE_LABELS"
#: Beta prior pseudo-counts shared with ``amfs_core.evidence.PRIOR_STRENGTH``.
_PRIOR_STRENGTH = 2.0
VALIDATED_MIN_P = 0.7
VALIDATED_MIN_N = 2
VALIDATED_MIN_WINS = 4
VALIDATED_MAX_LOSSES_WITH_WINS = 1


def label_rule() -> str:
    value = os.environ.get(LABELS_ENV, "posterior").strip().lower()
    return "strict" if value == "strict" else "posterior"


def posterior_mean(evidence_success: float, evidence_failure: float, prior: float = 0.5) -> float:
    """Posterior probability of success from decayed evidence masses.

    Same form as the confidence posterior in ``amfs_core.evidence`` but with a
    neutral 0.5 prior, so it measures the *record* rather than the author's
    claim: an untested entry is 0.5 whatever confidence it was written with.
    """
    e_s = max(0.0, float(evidence_success or 0.0))
    e_f = max(0.0, float(evidence_failure or 0.0))
    return (_PRIOR_STRENGTH * prior + e_s) / (_PRIOR_STRENGTH + e_s + e_f)


def evidence_label(
    *,
    success_count: int,
    failure_count: int,
    evidence_success: float,
    evidence_failure: float,
    discredited: bool,
    outcome_count: int = 0,
    confidence: float = 1.0,
    rule: str | None = None,
) -> str:
    """One of ``untested``, ``validated``, ``contested``, ``discredited``."""
    if discredited:
        return "discredited"
    n = int(success_count) + int(failure_count)
    if n == 0:
        if outcome_count == 0:
            return "untested"
        # Outcomes committed before the split counts existed: direction is
        # unknown, so read it off the confidence they left behind.
        return "validated" if confidence >= 0.5 else "contested"
    if failure_count == 0:
        return "validated"
    if (rule or label_rule()) == "strict":
        return "contested"
    p = posterior_mean(evidence_success, evidence_failure)
    if p >= VALIDATED_MIN_P and n >= VALIDATED_MIN_N:
        return "validated"
    if success_count >= VALIDATED_MIN_WINS and failure_count <= VALIDATED_MAX_LOSSES_WITH_WINS:
        return "validated"
    return "contested"


__all__ = [
    "LABELS_ENV",
    "VALIDATED_MIN_N",
    "VALIDATED_MIN_P",
    "VALIDATED_MIN_WINS",
    "evidence_label",
    "label_rule",
    "posterior_mean",
]
