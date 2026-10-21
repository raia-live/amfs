"""Evidence-based confidence: how an outcome changes what an entry is worth.

The original model multiplied confidence by a constant per outcome type
(``OUTCOME_MULTIPLIERS``): ``*1.03`` on success, ``*0.90`` on failure. Two
properties of that model made it nearly invisible in practice:

* A stale entry at 0.9 needs six straight failures to fall under a 0.5 gate,
  and an agent that retries around the failure commits ``success`` for the
  episode anyway, so the failures were never even counted.
* Every entry cited in a successful session gets the full ``*1.03``, so a
  generic entry read in every session climbs to 1.0 and stays there, and
  nothing distinguishes "cited a hundred times" from "cited once".

This module replaces the multipliers with a recency-weighted Beta posterior:

    confidence = (PRIOR_STRENGTH * prior + E_s) / (PRIOR_STRENGTH + E_s + E_f)

where ``prior`` is the confidence the author wrote and ``E_s`` / ``E_f`` are
evidence masses that decay by ``EVIDENCE_DECAY`` on every update, so the last
few outcomes dominate. Each outcome adds a weight

    w = severity(outcome_type) * causal_confidence * credit / n_causal
        * (1 + |target - confidence|) * verification

to one of the masses. ``(1 + |target - confidence|)`` is the *surprise*: a
failure on an entry the agent trusted at 0.95 counts nearly twice as much as
one on an entry at 0.5, and a success on an already-trusted entry counts for
little. Dividing by ``n_causal`` is the *credit split*: an outcome that cited
eight entries cannot hand each of them a full unit of evidence.

``credit`` is the entry's share of that split, by *uplift*: over the entity's
recent sessions, how much more often the ones that read this entry succeeded
than the ones that did not (:func:`uplift_credit`). A uniform split hands the
one entry that mattered and the eleven that did not the same evidence. With
uplift an entry whose presence tracks success keeps its full share of a
success and is charged less for a failure; an entry whose presence tracks
failure is charged in full and credited less. The measurement is shrunk
toward uniform while the sample is small and floored at ``UPLIFT_FLOOR``, so
it reorders credit and never silences an entry. An adapter that keeps
outcomes maintains ``MemoryEntry.uplift`` (:func:`uplift_from_counts`); one
that does not leaves it unset and the split stays uniform.

``verification`` is :data:`VERIFIED_EVIDENCE_WEIGHT` when the outcome was
decided outside the agent — ``verified_by`` set to CI, a monitor, a reviewer,
a customer — and 1 when the agent declared it itself. The agent's opinion of
its own run is evidence, but an engine whose reward is only that opinion
grades its own homework; a verified outcome is the stronger signal and moves
the evidence more.

One exception to the surprise term: the *first* failure of an entry that has
at least ``FIRST_STRIKE_MIN_WINS`` successes and no failure yet is weighed
without surprise. A single failure on a four-times-validated rule is more often
the agent's slip or a noisy environment than a change in the world, and with
full surprise it discredited the rule outright (0.89 -> 0.49) and pulled it out
of retrieval. Without it the same failure leaves the rule contested at ~0.62; a
second failure still discredits it (~0.40), so a regime change is unlearned in
two strikes exactly as before.

With the defaults below a fresh 0.7 entry drops to ~0.26 on its first failure
and lifts to ~0.82 on its first success; a long-validated entry survives one
failure (contested, ~0.62) and is discredited on the second. Those are the
timescales a regime change in a live system plays out on.

The same arithmetic is implemented in PL/pgSQL in ``amfs_postgres`` migrations
008 and 009 and must be kept identical; ``tests/unit/test_evidence.py`` pins the
numbers both implementations have to produce.

``AMFS_OUTCOME_MODEL=multiplicative`` restores the constant-multiplier model
for deployments that need the old numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Iterable

from amfs_core.models import (
    OUTCOME_MULTIPLIERS,
    MemoryEntry,
    OutcomeRecord,
    OutcomeType,
    clamp_confidence,
)

#: Pseudo-count behind the author's prior. Two units: one confident outcome
#: moves a fresh entry a lot, and a handful settle it.
PRIOR_STRENGTH = 2.0
#: Multiplied into both evidence masses before each update. 0.8 means an
#: outcome five updates ago carries a third of its original weight.
EVIDENCE_DECAY = 0.8
#: Posterior below which a failing entry is marked discredited.
DISCREDIT_THRESHOLD = 0.5
#: Successes an entry needs, with no failure yet, for its first failure to be
#: weighed without the surprise term (see the module docstring).
FIRST_STRIKE_MIN_WINS = 3
#: A regime shift is a rule that was validated repeatedly and whose record no
#: longer supports acting on it. ``REGIME_MIN_SUCCESSES`` is how validated it
#: must have been; "no longer" is read from the evidence label, so the
#: first-strike case (one failure against a long run, still ``validated``) is
#: not a shift and the second failure in a row is. The briefing's
#: ``regime_shift`` section and retrieve's ``regime_shift`` flag both read this.
#: A shift is an event, not a state: the flag holds for ``REGIME_WINDOW_DAYS``
#: after the rule's latest failure and then clears. Without the bound a rule
#: discredited weeks ago would keep steering every retrieve on its entity to
#: ``explore`` long after a replacement had been found and validated.
REGIME_MIN_SUCCESSES = 3
REGIME_WINDOW_DAYS = 7
#: With the rule's recent outcomes in hand (``regime_shifted(recent=...)``),
#: a shift needs this many failures among the newest ``REGIME_RECENT_OUTCOMES``
#: of them, from distinct sessions. The aggregate record cannot tell one
#: mislabelled failure from a change in the world: on one deployment a rule
#: validated ten times read as shifted on a single failure the monitor's
#: pattern had mislabelled, and a fix was drafted for a policy that had not
#: changed. Two sessions failing where the rule used to hold is the floor at
#: which "it stopped working" is a claim about the rule rather than the run.
REGIME_MIN_FAILURES = 2
REGIME_RECENT_OUTCOMES = 5
#: Failures weigh more than successes: trust is easy to lose, slow to rebuild.
SEVERITY: dict[str, float] = {
    OutcomeType.SUCCESS.value: 1.0,
    OutcomeType.CLEAN_DEPLOY.value: 1.0,
    OutcomeType.MINOR_FAILURE.value: 1.5,
    OutcomeType.REGRESSION.value: 1.5,
    OutcomeType.FAILURE.value: 2.0,
    OutcomeType.P2_INCIDENT.value: 2.0,
    OutcomeType.CRITICAL_FAILURE.value: 3.0,
    OutcomeType.P1_INCIDENT.value: 3.0,
}
SUCCESS_TYPES = frozenset({OutcomeType.SUCCESS.value, OutcomeType.CLEAN_DEPLOY.value})
#: How much more an outcome decided outside the agent weighs than the agent's
#: own declaration (``verified_by`` on the outcome; see the module docstring).
#: 1.5: a CI failure moves an entry half again as far as "the agent says it
#: failed". Not an environment variable: the SQL rendering carries the same
#: constant and the two must agree.
VERIFIED_EVIDENCE_WEIGHT = 1.5
#: Credit by uplift. ``UPLIFT_WINDOW`` recent outcomes on the entity are the
#: sample; ``UPLIFT_SHRINK`` is the sample size at which a measured uplift
#: counts half (``n / (n + UPLIFT_SHRINK)``), so a handful of sessions barely
#: move the split; ``UPLIFT_FLOOR`` is the least share an entry keeps, so a
#: disfavoured entry is still charged and credited, just less.
UPLIFT_WINDOW = 500
UPLIFT_WINDOW_DAYS = 90
UPLIFT_SHRINK = 10
UPLIFT_FLOOR = 0.25

OUTCOME_MODEL_ENV = "AMFS_OUTCOME_MODEL"


def outcome_model() -> str:
    """``"evidence"`` (default) or ``"multiplicative"``, from the environment."""
    value = os.environ.get(OUTCOME_MODEL_ENV, "evidence").strip().lower()
    return "multiplicative" if value == "multiplicative" else "evidence"


def is_success(outcome_type: OutcomeType | str) -> bool:
    return _name(outcome_type) in SUCCESS_TYPES


def is_known(outcome_type: OutcomeType | str) -> bool:
    """Whether the model has a verdict for this type. Unknown types are not evidence."""
    return _name(outcome_type) in SEVERITY


def recent_failures_confirm_shift(
    recent: Iterable[Any],
    *,
    min_failures: int = REGIME_MIN_FAILURES,
    window: int = REGIME_RECENT_OUTCOMES,
) -> bool:
    """Whether a rule's recent outcomes, newest first, carry a shift.

    *recent* is the rule's outcomes as the caller has them — objects or
    mappings with ``outcome_type``, a ``session_id`` (``outcome_ref`` or
    ``trace_id`` will do) and, where the caller knows it, ``verified_by`` or
    ``verified``. Among the newest *window* of them at least *min_failures*
    must be failures from distinct sessions; and when any outcome in that
    window was verified by someone other than the agent, at least one of the
    counted failures must be — an agent's own reports of failure do not
    carry a shift past a monitor that is watching and did not fail the rule.
    A window with no verified outcome at all is read on the reports alone.
    """
    newest = list(recent)[: max(1, int(window))]
    if not newest:
        return False

    def _get(item: Any, *names: str) -> Any:
        for name in names:
            value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
            if value not in (None, ""):
                return value
        return None

    def _verified(item: Any) -> bool:
        flag = _get(item, "verified")
        if isinstance(flag, bool):
            return flag
        return bool(_get(item, "verified_by"))

    any_verified = any(_verified(item) for item in newest)
    sessions: set[str] = set()
    verified_failure = False
    for position, item in enumerate(newest):
        outcome = _get(item, "outcome_type")
        if outcome is None or not is_known(outcome) or is_success(outcome):
            continue
        session = _get(item, "session_id", "outcome_ref", "trace_id", "id")
        sessions.add(str(session) if session is not None else f"#{position}")
        verified_failure = verified_failure or _verified(item)
    if len(sessions) < max(1, int(min_failures)):
        return False
    return verified_failure or not any_verified


def regime_shifted(
    entry: Any,
    *,
    now: datetime | None = None,
    window_days: int | None = REGIME_WINDOW_DAYS,
    recent: Iterable[Any] | None = None,
    min_failures: int = REGIME_MIN_FAILURES,
) -> bool:
    """Whether *entry* looks like a rule that used to work and has just stopped.

    Validated at least ``REGIME_MIN_SUCCESSES`` times, the latest outcome a
    failure within the last *window_days*, and the evidence label no longer
    ``validated`` — ``contested`` or ``discredited``. Reading the label rather
    than a failure ratio is what keeps this consistent with first-strike
    tolerance: the same failure that the label forgives (one against a long
    run) is not a shift, and the second failure in a row, which the label does
    not forgive, is. A ratio over the evidence masses cannot draw that line —
    one failure carries severity 2 against a decayed run of successes and
    already reads as half the mass.

    The label forgives one failure only on a record with none before it. A
    rule with one failure in its history reads ``contested`` on the next, so
    a single mislabelled run flips a ten-times-validated rule to "shifted"
    and a fix is drafted for a policy that did not change. A caller that has
    the rule's recent outcomes passes them as *recent* (newest first; see
    :func:`recent_failures_confirm_shift`), and the shift then also needs
    *min_failures* failures among the newest ``REGIME_RECENT_OUTCOMES`` of
    them, from distinct sessions, verified where verification exists. Without
    *recent* the aggregate rule stands alone, as before.

    The window is what makes this an event rather than a permanent mark: a
    rule discredited last month is history the ``discredited`` section covers,
    not a reason to keep skipping the action that has been winning since. Pass
    ``window_days=None`` to read the state without the bound.

    Works on anything with the evidence fields of ``MemoryEntry``, including the
    discredited entries retrieve keeps aside: a long-validated rule that was
    discredited by its recent failures is the strongest form of the signal.
    """
    successes = int(getattr(entry, "success_count", 0) or 0)
    failures = int(getattr(entry, "failure_count", 0) or 0)
    last = getattr(entry, "last_outcome", None)
    if successes < REGIME_MIN_SUCCESSES or failures < 1 or last is None or is_success(last):
        return False
    if recent is not None and not recent_failures_confirm_shift(
        recent, min_failures=min_failures
    ):
        return False
    if window_days is not None:
        at = getattr(entry, "last_outcome_at", None)
        if not isinstance(at, datetime):
            # No timestamp, no way to call it recent.
            return False
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        moment = now or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        if (moment - at).total_seconds() > window_days * 86400:
            return False
    status = getattr(entry, "evidence_status", None)
    if not isinstance(status, str):
        from .labels import evidence_label

        status = evidence_label(
            success_count=successes,
            failure_count=failures,
            evidence_success=float(getattr(entry, "evidence_success", 0.0) or 0.0),
            evidence_failure=float(getattr(entry, "evidence_failure", 0.0) or 0.0),
            discredited=getattr(entry, "discredited_at", None) is not None,
            outcome_count=int(getattr(entry, "outcome_count", 0) or 0),
            confidence=float(getattr(entry, "confidence", 1.0) or 0.0),
        )
    return status != "validated"


def severity(outcome_type: OutcomeType | str) -> float:
    return SEVERITY.get(_name(outcome_type), 1.0)


def _name(outcome_type: OutcomeType | str) -> str:
    return outcome_type.value if isinstance(outcome_type, OutcomeType) else str(outcome_type)


@dataclass(frozen=True)
class EvidenceUpdate:
    """The fields a single outcome changes on an entry."""

    confidence: float
    evidence_success: float
    evidence_failure: float
    success_count: int
    failure_count: int
    prior_confidence: float
    last_outcome: str
    discredited: bool
    #: How much the outcome moved the posterior, signed. Exposed so callers
    #: can report "this failure cost the entry 0.31" instead of just the result.
    delta: float

    def as_entry_update(self, now: datetime) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "evidence_success": self.evidence_success,
            "evidence_failure": self.evidence_failure,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "prior_confidence": self.prior_confidence,
            "last_outcome": self.last_outcome,
            "last_outcome_at": now,
            "discredited_at": now if self.discredited else None,
        }


def first_strike(outcome_type: OutcomeType | str, success_count: int, failure_count: int) -> bool:
    """Is this the first failure of an entry with a clean, sufficiently long record?"""
    return (
        not is_success(outcome_type)
        and int(failure_count) == 0
        and int(success_count) >= FIRST_STRIKE_MIN_WINS
    )


def verification_weight(verified: bool) -> float:
    """:data:`VERIFIED_EVIDENCE_WEIGHT` for an outcome decided outside the
    agent, 1 for the agent's own declaration."""
    return VERIFIED_EVIDENCE_WEIGHT if verified else 1.0


def outcome_verified_by(record: Any) -> str | None:
    """Who decided the outcome, when it was not the agent: the ``verified_by``
    the SDK carries as a session attribute (``session_metadata.attributes``)
    or, on an older record, at the top of the metadata. ``None`` for an
    agent-declared outcome."""
    meta = getattr(record, "session_metadata", None)
    if not isinstance(meta, dict):
        return None
    value = meta.get("verified_by")
    if not value:
        attrs = meta.get("attributes")
        value = attrs.get("verified_by") if isinstance(attrs, dict) else None
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def uplift_from_counts(
    read_successes: int, read_total: int, unread_successes: int, unread_total: int
) -> tuple[float | None, int]:
    """An entry's uplift from the entity's recent sessions: the success rate of
    the sessions that read it minus that of the sessions that did not, and the
    sample size the credit is shrunk by (the smaller side). ``(None, 0)`` when
    either side is empty — one-sided data is not a comparison."""
    if read_total <= 0 or unread_total <= 0:
        return None, 0
    uplift = read_successes / read_total - unread_successes / unread_total
    return max(-1.0, min(1.0, uplift)), min(int(read_total), int(unread_total))


def outcome_read_keys(record: Any) -> set[str]:
    """Every entry spec a session read: the terminal outcome's causal keys and
    the failed attempts'. An episode that read a rule, failed on it and
    resolved without it still had the rule in front of it."""
    keys = {str(k) for k in (getattr(record, "causal_entry_keys", None) or ())}
    for attempt in getattr(record, "attempts", None) or ():
        keys.update(str(k) for k in (getattr(attempt, "causal_entry_keys", None) or ()))
    return keys


def uplift_counts(
    records: Iterable[Any],
    entity_path: str,
    keys: Iterable[str],
    *,
    now: datetime | None = None,
) -> dict[str, tuple[float | None, int]]:
    """Each key's ``(uplift, sessions)`` over the entity's recent outcomes —
    the reference definition an adapter that keeps outcomes maintains in
    ``MemoryEntry.uplift`` (the Postgres adapter renders it to SQL).

    The sample is the newest ``UPLIFT_WINDOW`` known outcomes about the entity
    (``entity_paths`` names it, or a cited key is under it) from the last
    ``UPLIFT_WINDOW_DAYS``. A session *read* a key when any of its steps
    cited it (:func:`outcome_read_keys`); success is :func:`is_success`.
    """
    now = now or datetime.now(UTC)
    since = now - timedelta(days=UPLIFT_WINDOW_DAYS)
    prefix = entity_path.rstrip("/") + "/"
    sample: list[tuple[datetime, bool, set[str]]] = []
    for record in records:
        outcome_type = getattr(record, "outcome_type", None)
        outcome_type = getattr(outcome_type, "value", outcome_type)
        if not is_known(outcome_type):
            continue
        at = getattr(record, "committed_at", None)
        if at is None or at < since:
            continue
        read = outcome_read_keys(record)
        paths = set(getattr(record, "entity_paths", None) or ())
        if entity_path not in paths and not any(k.startswith(prefix) for k in read):
            continue
        sample.append((at, is_success(outcome_type), read))
    sample.sort(key=lambda row: row[0], reverse=True)
    sample = sample[:UPLIFT_WINDOW]
    out: dict[str, tuple[float | None, int]] = {}
    for key in keys:
        spec = prefix + key
        read_n = read_s = unread_n = unread_s = 0
        for _at, ok, read in sample:
            if spec in read:
                read_n += 1
                read_s += int(ok)
            else:
                unread_n += 1
                unread_s += int(ok)
        out[key] = uplift_from_counts(read_s, read_n, unread_s, unread_n)
    return out


def uplift_credit(uplift: float | None, sessions: int, *, success: bool) -> float:
    """The entry's share of a split, in ``[UPLIFT_FLOOR, 1]``.

    Full share for a success on an entry with non-negative uplift and for a
    failure on one with non-positive uplift — the outcome went the way the
    entry's record says it goes. Otherwise the share falls with the uplift's
    size, shrunk toward 1 while the sample is small and never below the
    floor. No measurement (``uplift`` unset or no sessions) is the uniform
    split the model always had.
    """
    if uplift is None or sessions <= 0:
        return 1.0
    signed = uplift if success else -uplift
    if signed >= 0:
        return 1.0
    shrink = sessions / (sessions + UPLIFT_SHRINK)
    return max(UPLIFT_FLOOR, min(1.0, 1.0 + shrink * signed))


def evidence_weight(
    outcome_type: OutcomeType | str,
    *,
    current_confidence: float,
    causal_confidence: float = 1.0,
    n_causal: int = 1,
    success_count: int = 0,
    failure_count: int = 0,
    credit: float = 1.0,
    verified: bool = False,
) -> float:
    """The evidence mass one outcome adds to one of its causal entries.

    *credit* is the entry's uplift share (:func:`uplift_credit`); *verified*
    says the outcome was decided outside the agent (:func:`verification_weight`).
    Both default to the uniform, agent-declared weight the model always had.
    """
    target = 1.0 if is_success(outcome_type) else 0.0
    surprise = 1.0 + abs(target - clamp_confidence(current_confidence))
    if first_strike(outcome_type, success_count, failure_count):
        surprise = 1.0
    share = 1.0 / max(1, n_causal)
    return (
        severity(outcome_type) * max(0.0, causal_confidence)
        * max(UPLIFT_FLOOR, min(1.0, credit)) * share * surprise
        * verification_weight(verified)
    )


def posterior(prior: float, evidence_success: float, evidence_failure: float) -> float:
    return clamp_confidence(
        (PRIOR_STRENGTH * prior + evidence_success)
        / (PRIOR_STRENGTH + evidence_success + evidence_failure)
    )


def apply_outcome(
    entry: MemoryEntry,
    outcome_type: OutcomeType | str,
    *,
    causal_confidence: float = 1.0,
    n_causal: int = 1,
    was_discredited: bool | None = None,
    verified: bool = False,
) -> EvidenceUpdate:
    """Compute the evidence update one outcome makes to ``entry``.

    Pure: nothing is written. ``entry.prior_confidence`` is used as the prior
    when set, otherwise the entry's current confidence becomes the prior (the
    first outcome an entry ever sees). ``was_discredited`` defaults to
    ``entry.discredited_at is not None``; an entry stays discredited until
    evidence lifts the posterior back over the threshold. The entry's share
    of the split is read off its ``uplift`` (:func:`uplift_credit`);
    *verified* is whether the outcome was decided outside the agent.
    """
    prior = entry.prior_confidence if entry.prior_confidence is not None else entry.confidence
    prior = clamp_confidence(prior)
    success = is_success(outcome_type)
    w = evidence_weight(
        outcome_type,
        current_confidence=entry.confidence,
        causal_confidence=causal_confidence,
        n_causal=n_causal,
        success_count=entry.success_count,
        failure_count=entry.failure_count,
        credit=uplift_credit(
            getattr(entry, "uplift", None), int(getattr(entry, "uplift_sessions", 0) or 0),
            success=success,
        ),
        verified=verified,
    )
    e_s = entry.evidence_success * EVIDENCE_DECAY + (w if success else 0.0)
    e_f = entry.evidence_failure * EVIDENCE_DECAY + (0.0 if success else w)
    new_conf = posterior(prior, e_s, e_f)
    discredited_before = (
        was_discredited if was_discredited is not None else entry.discredited_at is not None
    )
    if new_conf < DISCREDIT_THRESHOLD and not success:
        discredited = True
    elif new_conf >= DISCREDIT_THRESHOLD:
        discredited = False
    else:
        # A success that did not clear the threshold leaves the flag as it was.
        discredited = discredited_before
    return EvidenceUpdate(
        confidence=new_conf,
        evidence_success=e_s,
        evidence_failure=e_f,
        success_count=entry.success_count + (1 if success else 0),
        failure_count=entry.failure_count + (0 if success else 1),
        prior_confidence=prior,
        last_outcome=_name(outcome_type),
        discredited=discredited,
        delta=new_conf - entry.confidence,
    )


def apply_outcome_multiplicative(
    entry: MemoryEntry,
    outcome_type: OutcomeType | str,
    *,
    causal_confidence: float = 1.0,
) -> EvidenceUpdate:
    """The legacy constant-multiplier update, in the same shape.

    Kept for ``AMFS_OUTCOME_MODEL=multiplicative``. Counts and evidence masses
    are still maintained so the status vocabulary works under either model.
    """
    key: OutcomeType | str
    try:
        key = OutcomeType(_name(outcome_type))
    except ValueError:
        key = _name(outcome_type)
    multiplier = OUTCOME_MULTIPLIERS.get(key, 1.0)
    new_conf = clamp_confidence(entry.confidence * multiplier * causal_confidence)
    success = is_success(outcome_type)
    prior = entry.prior_confidence if entry.prior_confidence is not None else entry.confidence
    # Same hysteresis as the evidence model and the SQL step: a success that
    # does not clear the threshold leaves the flag where it was.
    if new_conf < DISCREDIT_THRESHOLD and not success:
        discredited = True
    elif new_conf >= DISCREDIT_THRESHOLD:
        discredited = False
    else:
        discredited = entry.discredited_at is not None
    return EvidenceUpdate(
        confidence=new_conf,
        evidence_success=entry.evidence_success + (1.0 if success else 0.0),
        evidence_failure=entry.evidence_failure + (0.0 if success else 1.0),
        success_count=entry.success_count + (1 if success else 0),
        failure_count=entry.failure_count + (0 if success else 1),
        prior_confidence=clamp_confidence(prior),
        last_outcome=_name(outcome_type),
        discredited=discredited,
        delta=new_conf - entry.confidence,
    )


def evidence_signal(entry: MemoryEntry) -> float:
    """A ranking term in ``[-1, 1]`` from the entry's outcome evidence.

    ``0`` for an untested entry; positive when successes outweigh failures,
    negative otherwise, approaching ±1 as evidence accumulates. Confidence
    alone cannot tell an author's untested 0.9 from a 0.9 earned over a dozen
    outcomes; this term can, which is why retrieval blends it in separately.
    Discredited entries are pinned to ``-1`` so any caller that lets them
    through ranks them last.
    """
    if entry.discredited_at is not None:
        return -1.0
    e_s = max(0.0, float(entry.evidence_success))
    e_f = max(0.0, float(entry.evidence_failure))
    if e_s == 0.0 and e_f == 0.0:
        return 0.0
    return (e_s - e_f) / (1.0 + e_s + e_f)


#: Weighted outcomes on tasks like the query an entry needs before its local
#: record can *rescue* it (``locally_valid``). Below this the pooled record
#: stands alone for that decision.
LOCAL_EVIDENCE_MIN_N = 2
#: Nearby outcomes an entry needs before its local record enters the ranking
#: blend at all. One: a single outcome on a task like this one is worth half
#: the pooled record's say in the evidence term (``n / (n + 1)``), which is
#: what lets the first failure of a long-validated rule on a new class of
#: task move it *before* the second failure on the same class arrives. Grid
#: v4 measured the cost of waiting: the same quirk failed 4-7 times per store
#: while the pooled record stayed ``validated``.
LOCAL_EVIDENCE_BLEND_MIN_N = 1
#: Local success rate at or above which a globally discredited entry is kept
#: in the ranked list (labelled ``contested``) for the query it still works
#: for, instead of going to the avoid list.
LOCAL_RESCUE_MIN_P = 0.6
#: Consecutive most-recent failures on tasks like the query after which a
#: globally validated entry is treated as discredited *for this query*
#: (``locally_discredited``): out of the ranked list and into the avoid list,
#: with what replaced it. Two, not one — a single failure may be the agent's,
#: not the rule's; two in a row on the same kind of task with no success
#: between them is the rule.
LOCAL_DISCREDIT_MIN_N = 2


def evidence_signal_from_counts(success: float, failure: float) -> float:
    """:func:`evidence_signal` over explicit masses, for a record that is not
    the entry's pooled one — the query-conditioned counts ``evidence_near``
    returns, weighted by task similarity."""
    e_s = max(0.0, float(success))
    e_f = max(0.0, float(failure))
    if e_s == 0.0 and e_f == 0.0:
        return 0.0
    return (e_s - e_f) / (1.0 + e_s + e_f)


def blend_local_evidence(pooled: float, local: dict[str, Any] | None) -> tuple[float, float]:
    """The evidence term for ranking, given the pooled signal and the local record.

    Returns ``(evidence, local_weight)``. With no local record, or none on a
    task near enough to count, the pooled signal stands and the weight is
    ``0``. Otherwise the local signal is blended in with weight
    ``n / (n + LOCAL_EVIDENCE_MIN_N)`` — a third at one nearby outcome, half
    at two, three quarters at six — so a rule's record on *this kind of task*
    takes over from its record everywhere as the local evidence accumulates.
    The curve is the one the rescue threshold was calibrated on; only the
    gate moved, from two nearby outcomes to one. A single nearby outcome
    moves the ranking but cannot on its own flip a long pooled record's
    sign: ``locally_valid`` and ``locally_discredited`` — the decisions that
    move an entry between the ranked list and the avoid list — keep their
    own, stricter minimums.

    Grid v3's diagnose scenario is the case: a rule validated on one class of
    incident and discredited on another read ``contested`` to every query, so
    the class it still worked for lost it. Pooled evidence answers "has this
    entry been right"; this answers "has it been right for tasks like this".
    """
    if not local:
        return pooled, 0.0
    n = float(local.get("n", 0) or 0)
    if n < LOCAL_EVIDENCE_BLEND_MIN_N:
        return pooled, 0.0
    sig = evidence_signal_from_counts(local.get("success", 0.0), local.get("failure", 0.0))
    w = n / (n + LOCAL_EVIDENCE_MIN_N)
    return (1.0 - w) * pooled + w * sig, w


def locally_valid(local: dict[str, Any] | None) -> bool:
    """Whether the local record alone says the entry works for tasks like this:
    at least ``LOCAL_EVIDENCE_MIN_N`` weighted outcomes and a success share at
    or above ``LOCAL_RESCUE_MIN_P``."""
    if not local:
        return False
    n = float(local.get("n", 0) or 0)
    if n < LOCAL_EVIDENCE_MIN_N:
        return False
    s = max(0.0, float(local.get("success", 0.0) or 0.0))
    f = max(0.0, float(local.get("failure", 0.0) or 0.0))
    if s + f <= 0.0:
        return False
    return s / (s + f) >= LOCAL_RESCUE_MIN_P


def locally_discredited(local: dict[str, Any] | None) -> bool:
    """Whether the local record alone says the entry has stopped working for
    tasks like this: its ``LOCAL_DISCREDIT_MIN_N`` most recent nearby outcomes
    are all failures.

    The mirror of :func:`locally_valid`. That one reads the local record to
    keep a globally discredited entry for the class of task it still works
    on; this reads it to set aside a globally *validated* entry for the class
    it has stopped working on. Pooled evidence cannot see the difference — a
    rule validated twenty times on one class and failed twice on another is
    still ``validated`` — and grid v4 put the cost at 4-7 repeated failures
    per store before the pooled label moved.

    Reads ``local["recent"]``: the nearby outcomes newest first, each
    ``{"success": bool, ...}``, as ``evidence_near`` returns them. Any success
    among the newest ``LOCAL_DISCREDIT_MIN_N`` clears it — the rule worked
    again, or the failures were the agent's. A record without ``recent`` (an
    adapter predating it) never discredits: the sums cannot say which came
    last.
    """
    if not local:
        return False
    recent = local.get("recent")
    if not isinstance(recent, list) or len(recent) < LOCAL_DISCREDIT_MIN_N:
        return False
    newest = recent[:LOCAL_DISCREDIT_MIN_N]
    return all(isinstance(r, dict) and not bool(r.get("success")) for r in newest)


def _split_spec(spec: str) -> tuple[str, str] | None:
    parts = spec.rsplit("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def outcome_steps(record: OutcomeRecord) -> list[tuple[OutcomeType, list[str]]]:
    """The ordered ``(outcome_type, causal_entry_keys)`` steps a record applies.

    Failed attempts first, oldest to newest, then the terminal outcome. Each
    step's keys are deduplicated, and each step is credit-split over its own
    keys, not over the union — an attempt that read one entry hands it a full
    unit of failure even if the whole task read twenty.
    """
    return [(t, keys) for t, keys, _ in outcome_steps_with_versions(record)]


def outcome_steps_with_versions(
    record: OutcomeRecord,
) -> list[tuple[OutcomeType, list[str], dict[str, int]]]:
    """``outcome_steps`` plus, per step, the ``entry_key -> version`` map read."""
    steps: list[tuple[OutcomeType, list[str], dict[str, int]]] = []
    for attempt in sorted(record.attempts, key=lambda a: a.attempt):
        keys = list(dict.fromkeys(attempt.causal_entry_keys))
        if keys:
            steps.append((attempt.outcome_type, keys, dict(attempt.causal_entry_versions)))
    steps.append((
        record.outcome_type,
        list(dict.fromkeys(record.causal_entry_keys)),
        dict(record.causal_entry_versions),
    ))
    return steps


#: ``(entity_path, key, version) -> MemoryEntry | None``: how an adapter finds
#: the version an agent read, so a step can tell whether the claim changed.
VersionLookup = Callable[[str, str, int], "MemoryEntry | None"]


def claim_still_held(
    entry: MemoryEntry,
    read_version: int | None,
    lookup: VersionLookup | None,
) -> bool:
    """Does the live ``entry`` still say what version ``read_version`` said?

    ``True`` when no version was recorded, when the live version *is* the one
    read, when the read version cannot be found, or when the two values are
    the same claim (outcome propagation and identical restatements both open
    new versions without changing the claim). ``False`` only when the key was
    rewritten with something else in between — then the outcome is about the
    old claim and must not land on the new one.
    """
    if read_version is None or lookup is None or read_version == entry.version:
        return True
    was = lookup(entry.entity_path, entry.key, int(read_version))
    if was is None:
        return True
    return same_claim(was.value, entry.value)


def apply_record_to_entry(
    entry: MemoryEntry,
    record: OutcomeRecord,
    *,
    now: datetime | None = None,
    model: str | None = None,
    version_lookup: VersionLookup | None = None,
) -> tuple[MemoryEntry, list[EvidenceUpdate]]:
    """Apply every step of ``record`` that cites ``entry`` and return the new entry.

    This is what the filesystem and S3 adapters (and the Postgres trigger, in
    SQL) do per causal entry. ``version`` is left alone: adapters assign it on
    write. With *version_lookup* a step whose recorded read version no longer
    matches the live claim is skipped (see ``claim_still_held``).
    """
    now = now or datetime.now(UTC)
    model = model or outcome_model()
    updates: list[EvidenceUpdate] = []
    current = entry
    verified = outcome_verified_by(record) is not None
    for outcome_type, keys, versions in outcome_steps_with_versions(record):
        if entry.entry_key not in keys:
            continue
        if not claim_still_held(entry, versions.get(entry.entry_key), version_lookup):
            continue
        if not is_known(outcome_type):
            continue
        if model == "multiplicative":
            upd = apply_outcome_multiplicative(
                current, outcome_type, causal_confidence=record.causal_confidence
            )
        else:
            upd = apply_outcome(
                current,
                outcome_type,
                causal_confidence=record.causal_confidence,
                n_causal=len(keys),
                verified=verified,
            )
        fields = upd.as_entry_update(now)
        if upd.discredited and current.discredited_at is not None:
            # Still discredited: keep the moment it happened, not the latest hit.
            fields["discredited_at"] = current.discredited_at
        fields["validators"] = validators_after(
            current.validators, record.agent_id, is_success(outcome_type)
        )
        current = current.model_copy(update={**fields, "outcome_count": current.outcome_count + 1})
        updates.append(upd)
    return current, updates


#: Most distinct validators kept on an entry; the list is a signal, not a log.
MAX_VALIDATORS = 10


def validators_after(current: list[str] | None, agent_id: str | None, success: bool) -> list[str]:
    """The entry's validators once ``agent_id`` committed this outcome.

    A success adds the agent (moved to the end if already present); a failure
    leaves the list alone — the record of who stood behind the claim is still
    true, and the counts say what happened since. Capped at ``MAX_VALIDATORS``,
    dropping the oldest.
    """
    out = [v for v in (current or []) if v]
    if not success or not agent_id:
        return out
    out = [v for v in out if v != agent_id] + [agent_id]
    return out[-MAX_VALIDATORS:]


def cited_entries(record: OutcomeRecord) -> list[tuple[str, str]]:
    """Every distinct ``(entity_path, key)`` any step of ``record`` cites."""
    seen: dict[tuple[str, str], None] = {}
    for _, keys in outcome_steps(record):
        for spec in keys:
            split = _split_spec(spec)
            if split is not None:
                seen.setdefault(split, None)
    return list(seen)


#: Session attributes that mark an outcome as a **replay**: a run the repair
#: loop made to test a fix, not a session anyone served. A Tier 1 replay
#: carries ``attributes.origin`` in :data:`REPLAY_ORIGINS`; a Tier 2 customer
#: replay carries ``attributes.replay_delivery_id`` (the SDK's
#: ``ReplayReceiver`` stamps it). The hosted loop applies the same two marks
#: when it grades: a replay is graded by the tier that asked for it and by
#: nothing else — it does not vote in a canary, fill its window or count as
#: exposure. This is the memory-side half of that rule (see
#: :func:`is_replay_outcome`).
REPLAY_ORIGINS: tuple[str, ...] = ("repair",)
REPLAY_DELIVERY_ATTRIBUTE = "replay_delivery_id"


def is_replay_outcome(record: OutcomeRecord) -> bool:
    """Whether *record* is a replay's outcome (``REPLAY_ORIGINS`` /
    ``REPLAY_DELIVERY_ATTRIBUTE`` on ``session_metadata["attributes"]``).

    A replay's outcome is the loop's verdict on the fix, not evidence about the
    entries the run read — least of all about the fix itself. Measured on
    clbench (2026-09-30, fleet 10, six canaries): a correction written at 0.78
    was discredited 65 seconds later, before its canary had started, by the
    failed attempts inside two Tier 2 replays that both *passed* — each
    attempt handed the correction a failure (0.78 → 0.41 after the first), the
    posterior crossed ``DISCREDIT_THRESHOLD`` and the canary arm could no
    longer be served the entry it existed to test. Five of the six corrections
    went that way, two before their canary's first session; the treated arm
    read its fix in 0 of 74 sessions. The same runs had written 425
    ``lesson-contrast-replay-*`` entries onto ``main``, each naming a
    correction under test as the thing to avoid. A record with no attributes,
    or attributes that are not a mapping, is not a replay.
    """
    meta = getattr(record, "session_metadata", None)
    if not isinstance(meta, dict):
        return False
    attrs = meta.get("attributes")
    if not isinstance(attrs, dict):
        return False
    if str(attrs.get("origin") or "") in REPLAY_ORIGINS:
        return True
    return REPLAY_DELIVERY_ATTRIBUTE in attrs


def contrast_lesson(record: OutcomeRecord) -> dict[str, Any] | None:
    """The auto-written lesson for a fail-then-succeed record, or ``None``.

    Only when at least one failed attempt cited an entry and the terminal
    outcome is a success: "these entries led to a failed attempt on this task;
    the task was resolved without them". Callers write it under the agent's
    identity with a key that starts with ``SYNTHETIC_KEY_PREFIX`` so training
    and eval pipelines can exclude it.
    """
    if not is_success(record.outcome_type) or not record.attempts:
        return None
    failed_keys: list[str] = []
    summaries: list[str] = []
    for attempt in sorted(record.attempts, key=lambda a: a.attempt):
        if is_success(attempt.outcome_type):
            continue
        failed_keys.extend(k for k in attempt.causal_entry_keys if k not in failed_keys)
        if attempt.summary:
            summaries.append(attempt.summary)
    resolved_with = [k for k in record.causal_entry_keys if k not in failed_keys]
    if not failed_keys:
        return None
    # The actions, when the record labelled them (``actions_taken``): what the
    # failed attempts *did* and what the terminal success did instead. The
    # entry keys above say which memory misled; these say which action to take
    # instead, which is what the next agent on a near-identical task needs.
    # Optional and additive — a lesson written without them (a repair-loop
    # pointer carries only ``avoid``/``resolved_with``) reads the same.
    failed_actions: list[str] = []
    resolved_action: str | None = None
    for act in record.actions_taken or ():
        key = str(act.get("action_key") or "")
        if not key:
            continue
        if act.get("attempt") is not None and not act.get("success"):
            if key not in failed_actions:
                failed_actions.append(key)
        elif act.get("attempt") is None and act.get("success"):
            resolved_action = key
    excerpt = (record.situation or record.task_input or "").strip()
    return {
        "kind": "contrast",
        "outcome_ref": record.outcome_ref,
        "failed_attempts": len([a for a in record.attempts if not is_success(a.outcome_type)]),
        "avoid": failed_keys,
        "resolved_with": resolved_with,
        "attempt_summaries": summaries,
        "failed_actions": failed_actions,
        "resolved_action": resolved_action,
        "task_excerpt": excerpt[:200] if excerpt else None,
        "lesson": (
            f"{len(failed_keys)} remembered entr{'y' if len(failed_keys) == 1 else 'ies'} "
            f"led to a failed attempt before this task was resolved"
            + (
                f" using {len(resolved_with)} other "
                f"entr{'y' if len(resolved_with) == 1 else 'ies'}"
                if resolved_with
                else ""
            )
            + "."
        ),
    }


#: Keys with these prefixes are written by the system, not the agent: derived
#: lessons that should inform retrieval and briefings but never be trained on
#: or graded as if the agent had authored them.
#: Fields that make up an entry's outcome record. They describe the *claim*, not
#: the row, so an identical restatement of the claim carries them forward.
EVIDENCE_FIELDS: tuple[str, ...] = (
    "outcome_count",
    "success_count",
    "failure_count",
    "evidence_success",
    "evidence_failure",
    "last_outcome",
    "last_outcome_at",
    "discredited_at",
    "prior_confidence",
)


def same_claim(a: Any, b: Any) -> bool:
    """Two values that say the same thing, allowing for the JSON round trip
    (dict key order, int/float) that a stored value has been through.

    Two structured lessons (:mod:`amfs_core.lessons`) make the same claim when
    their ``(situation, action, worked)`` agree, whatever their text: the
    text is the agent's words for the claim, and a rewrite of the words must
    not open a new record."""
    if a == b:
        return True
    from .lessons import lesson_claim

    claim_a = lesson_claim(a)
    if claim_a is not None:
        return claim_a == lesson_claim(b)
    try:
        import json

        return json.dumps(a, sort_keys=True, default=str) == json.dumps(
            b, sort_keys=True, default=str
        )
    except Exception:  # noqa: BLE001
        return False


def inherit_evidence(new: MemoryEntry, current: MemoryEntry | None) -> MemoryEntry:
    """Carry the outcome record across a rewrite that does not change the claim.

    Agents restate their lessons: the reflection step at the end of every task
    writes the same key again, usually with the same text and the same
    declared confidence. Before this rule every such write opened a new
    version with an empty record, so a lesson validated twelve times looked
    untested the moment its author repeated it, and a lesson discredited
    yesterday came back clean today. The record is about what the entry
    claims; an unchanged claim keeps it, and the evidence-derived posterior
    outranks the writer's restated prior. A changed claim is a new hypothesis
    and starts untested, as before.
    """
    if current is None or not same_claim(new.value, current.value):
        return new
    if not (current.success_count or current.failure_count or current.outcome_count):
        return new
    update: dict[str, Any] = {f: getattr(current, f) for f in EVIDENCE_FIELDS}
    update["confidence"] = current.confidence
    update["validators"] = list(current.validators or [])
    return new.model_copy(update=update)


#: Keys the system writes for itself. Read paths hide them from agents (the
#: briefing folds them into ``discredited.replaced_by``) and training and eval
#: pipelines exclude them.
SYNTHETIC_KEY_PREFIXES: tuple[str, ...] = ("lesson-contrast-",)

#: Keys the repair loop writes *for* agents: a risk note or corrected value
#: proposed by SenseLab eval and approved by a person. Unlike synthetic keys
#: these are knowledge an agent should read — they are what stops the next
#: failure — so read paths keep them. Training pipelines exclude them: a
#: model is meant to learn the corrected *behaviour* from the trajectories the
#: correction produced, not to memorise the note that produced it, and a note
#: that names a past failure would otherwise be the strongest lexical signal
#: in every prompt it appears in.
CORRECTIVE_KEY_PREFIXES: tuple[str, ...] = ("risk-", "correction-")


def is_synthetic_key(key: str) -> bool:
    return key.startswith(SYNTHETIC_KEY_PREFIXES)


def is_corrective_key(key: str) -> bool:
    """Whether *key* is a repair-loop correction (see ``CORRECTIVE_KEY_PREFIXES``)."""
    return key.startswith(CORRECTIVE_KEY_PREFIXES)


def is_training_excluded_key(key: str) -> bool:
    """Whether a training or fine-tuning pipeline should leave *key* out of the
    prompt it renders: synthetic lessons and repair-loop corrections both."""
    return is_synthetic_key(key) or is_corrective_key(key)


def contrast_lesson_key(outcome_ref: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in outcome_ref)[:80]
    return f"{SYNTHETIC_KEY_PREFIXES[0]}{safe}"


def replacements_from_lessons(entries: Iterable[MemoryEntry]) -> dict[str, list[str]]:
    """``entry_key -> [entry_key, ...]``: for each entry a contrast lesson
    names under ``avoid``, the entries the same lesson says the task was
    resolved with, in first-seen order.

    Reads only the ``{avoid, resolved_with}`` shape every lesson writer
    shares — :func:`contrast_lesson` here and the repair loop's pointer — so
    a lesson with or without the optional action fields resolves the same.
    Entries that are not lessons are skipped, so callers can pass a mixed
    list. Shared by the briefing's ``discredited[].replaced_by`` and the
    retrieve avoid list.
    """
    out: dict[str, list[str]] = {}
    for e in entries:
        if not is_synthetic_key(e.key) or not isinstance(e.value, dict):
            continue
        avoid = e.value.get("avoid") or []
        resolved = e.value.get("resolved_with") or []
        if not isinstance(avoid, list) or not isinstance(resolved, list):
            continue
        for spec in avoid:
            bucket = out.setdefault(str(spec), [])
            for r in resolved:
                if str(r) not in bucket:
                    bucket.append(str(r))
    return out


__all__ = [
    "CORRECTIVE_KEY_PREFIXES",
    "DISCREDIT_THRESHOLD",
    "EVIDENCE_DECAY",
    "PRIOR_STRENGTH",
    "UPLIFT_FLOOR",
    "UPLIFT_SHRINK",
    "UPLIFT_WINDOW",
    "UPLIFT_WINDOW_DAYS",
    "VERIFIED_EVIDENCE_WEIGHT",
    "outcome_read_keys",
    "outcome_verified_by",
    "uplift_counts",
    "uplift_credit",
    "uplift_from_counts",
    "verification_weight",
    "REPLAY_DELIVERY_ATTRIBUTE",
    "REPLAY_ORIGINS",
    "SEVERITY",
    "SUCCESS_TYPES",
    "SYNTHETIC_KEY_PREFIXES",
    "EvidenceUpdate",
    "apply_outcome",
    "is_known",
    "is_replay_outcome",
    "apply_outcome_multiplicative",
    "apply_record_to_entry",
    "cited_entries",
    "contrast_lesson",
    "contrast_lesson_key",
    "evidence_signal",
    "evidence_weight",
    "is_corrective_key",
    "is_success",
    "is_synthetic_key",
    "is_training_excluded_key",
    "LOCAL_DISCREDIT_MIN_N",
    "LOCAL_EVIDENCE_BLEND_MIN_N",
    "LOCAL_EVIDENCE_MIN_N",
    "LOCAL_RESCUE_MIN_P",
    "blend_local_evidence",
    "locally_discredited",
    "locally_valid",
    "outcome_model",
    "outcome_steps",
    "posterior",
    "REGIME_MIN_FAILURES",
    "REGIME_RECENT_OUTCOMES",
    "recent_failures_confirm_shift",
    "regime_shifted",
    "replacements_from_lessons",
    "severity",
]
