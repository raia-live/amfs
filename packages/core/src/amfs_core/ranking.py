"""The composite retrieval score, shared by the server and the SDK's local path.

Retrieval blends four things: how well an entry answers the query
(*relevance*: bi-encoder similarity, a lexical match, or the cross-encoder's
standing once a reranker has spoken), how recent it is, how much the store
trusts it (*confidence*) and what the outcome record says about it
(*evidence*, ``amfs_core.evidence.evidence_signal``).

Until 2026-09-18 the blend was a weighted sum. That let trust outrank topic:
on the grid v3 runbook probe the server surfaced the seeded runbook for the
task's own service in 47 of 100 first searches, against 84 for plain cosine
ranking. Once the agent held a few validated, confidence-1.0 notes about
*other* services in the same domain, their confidence and evidence terms
(0.35 of score between them) exceeded the similarity gap to the relevant but
untested 0.7-confidence runbook (0.5 x 0.15), and the agent read no runbook
and escalated. Every production store shares a domain vocabulary across its
notes, so this was not a benchmark artefact.

The fix is to let trust and recency *modulate* relevance rather than add to
it: an entry earns at most :data:`TRUST_MODULATION` more score for a perfect
record, and only in proportion to how relevant it is. Two entries of equal
relevance are still ordered by their records, as strongly as before; an
entry the query is not about cannot climb over one it is about on record
alone. The tie-break survives, the crowding does not.

The additive form is kept behind ``anchored=False`` (and the server's
``AMFS_RANK_ADDITIVE_TRUST`` environment switch) so the change can be rolled
back without a deploy of code.
"""

from __future__ import annotations

#: The most a perfect record (confidence 1.0, evidence +1) can scale relevance
#: by, above an entry with no confidence and no record. 0.5 keeps trust
#: decisive between entries of similar relevance (a validated 1.0 outranks an
#: untested 0.7 by about 20% of score) while a relevance gap of a third or
#: more — one service's runbook against another's — cannot be closed by
#: record alone.
TRUST_MODULATION = 0.5


def composite_score(
    *,
    relevance: float,
    recency: float,
    confidence: float,
    evidence: float = 0.0,
    keyword: float = 0.0,
    semantic_weight: float = 0.5,
    recency_weight: float = 0.3,
    confidence_weight: float = 0.2,
    keyword_weight: float = 0.15,
    evidence_weight: float = 0.15,
    anchored: bool = True,
) -> float:
    """The composite score for one candidate.

    *relevance* is in ``[0, 1]``; *keyword* is ``1.0`` for a lexical match,
    else ``0``; *recency* and *confidence* are in ``[0, 1]``; *evidence* is
    in ``[-1, 1]``.

    Anchored (default)::

        rel   = semantic_weight * relevance + keyword_weight * keyword
        trust = (confidence_weight * confidence + evidence_weight * evidence)
                / (confidence_weight + evidence_weight)        # in [-1, 1]
        score = rel * (1 + TRUST_MODULATION * trust + recency_weight * recency)

    Recency modulates too, for the same reason trust does: a note written
    this morning about the wrong service must not outrank last year's runbook
    for the right one. A temporal query raises ``recency_weight`` (see
    ``amfs_core.query_norm``) and with it how far recency can carry an entry
    — still in proportion to relevance.

    Additive (``anchored=False``), the pre-2026-09-18 form::

        score = semantic_weight * relevance + recency_weight * recency
                + confidence_weight * confidence + keyword_weight * keyword
                + evidence_weight * evidence
    """
    if not anchored:
        return (
            semantic_weight * relevance
            + recency_weight * recency
            + confidence_weight * confidence
            + keyword_weight * keyword
            + evidence_weight * evidence
        )
    rel = semantic_weight * relevance + keyword_weight * keyword
    if rel <= 0.0:
        return 0.0
    scale = confidence_weight + evidence_weight
    trust = (
        (confidence_weight * confidence + evidence_weight * evidence) / scale
        if scale > 0.0 else 0.0
    )
    trust = max(-1.0, min(1.0, trust))
    return rel * (1.0 + TRUST_MODULATION * trust + recency_weight * recency)
