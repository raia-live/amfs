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

import math
import re
from typing import Any

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


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
#: Characters of a value the lexical term reads. Enough for any note; a bound
#: so a pasted log does not make this pass the cost of the request.
KEYWORD_DOC_CHARS = 4_000


def _terms(text: str) -> set[str]:
    """Lower-cased word tokens with a plural-stripping stem: ``returns`` and
    ``return``, ``migrations`` and ``migration`` are the same term. Keys are
    split on ``-`` and ``_`` by the same regex, so ``runbook-returns-a``
    contributes ``runbook`` and ``return``."""
    out: set[str] = set()
    for w in _WORD_RE.findall(text.lower()):
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.add(w)
    return out


def keyword_coverage(
    query: str,
    docs: dict[str, str],
    background: list[str] | None = None,
) -> dict[str, float]:
    """The lexical relevance of each document to *query*, in ``[0, 1]``.

    The share of the query's terms a document contains, each term weighted by
    how much it says about *which* document the query is about. Terms no
    candidate contains are left out of the denominator, since they cannot
    discriminate either.

    A term's weight comes from one of two corpora:

    * **The entity's past tasks** (*background*), when the term occurs in
      them: ``log((M+1)/(df_tasks+1))``. An agent phrases its query like the
      task in front of it, so the words it shares with an entry are of two
      kinds — the template every task on this entity shares ("standard deploy
      … correct step order") and the words that vary from task to task ("the
      *returns* service"). Only the second kind says what this task is about,
      and only the task history can tell them apart: a word in every past task
      identifies nothing; a word in few of them is the subject. Statistics over
      the candidate pool cannot make the distinction — early in an entity's
      life a template word is as rare among the entries as the subject is,
      and a pool of thirty notes has no signal at all — which is what left the
      seeded runbook for the task's own service ranked below notes about other
      services written in the query's phrasing.
    * **The candidate pool** otherwise: ``log((N+1)/(df+1))`` over *docs*, for
      a term the tasks never used (the agent's own vocabulary — "runbook",
      "validated"), and for every term when there is no background at all.

    Each weight is then divided by the term's *redundancy*, one plus the sum
    of its Jaccard overlaps with the other query terms' document sets. Four
    words that always appear together across the pool are one piece of
    evidence, not four, and without this a doc matching the phrasing
    outweighed one matching the subject on count alone.

    Replaces the binary "was a lexical hit" flag the retrieve blend used until
    2026-09-18. With an OR-combined full-text query almost every candidate is
    a lexical hit, so the flag was 1.0 across the board and the one word that
    separated *returns service standard deploy* from the same request for
    *checkout* carried no weight at all — the bi-encoder does not separate
    service names either.

    Computed over the candidate pool and a bounded task corpus: a few hundred
    short documents per request, so it is a Python pass, not a query. When the
    weights vanish (every query term in every document, or in none) the result
    falls back to presence — the old flag.
    """
    q = _terms(query or "")
    if not docs:
        return {}
    doc_terms = {k: _terms(t[:KEYWORD_DOC_CHARS]) for k, t in docs.items()}
    if not q:
        return {k: 0.0 for k in doc_terms}
    n = len(doc_terms)
    docsets = {t: {k for k, ts in doc_terms.items() if t in ts} for t in q}
    docsets = {t: s for t, s in docsets.items() if s}
    if not docsets:
        return {k: 0.0 for k in doc_terms}
    task_terms = [_terms(b[:KEYWORD_DOC_CHARS]) for b in (background or []) if b]
    m = len(task_terms)
    weights: dict[str, float] = {}
    for t, s in docsets.items():
        df_tasks = sum(1 for ts in task_terms if t in ts)
        if df_tasks:
            w = math.log((m + 1) / (df_tasks + 1))
        else:
            w = math.log((n + 1) / (len(s) + 1))
        redundancy = 1.0 + sum(
            len(s & o) / len(s | o) for u, o in docsets.items() if u != t
        )
        weights[t] = w / redundancy
    total = sum(weights.values())
    if total <= 0.0:
        return {k: (1.0 if any(t in ts for t in docsets) else 0.0) for k, ts in doc_terms.items()}
    return {
        k: sum(w for t, w in weights.items() if t in ts) / total for k, ts in doc_terms.items()
    }


def entry_text(key: str, value: Any) -> str:
    """The text the lexical term reads for an entry: its key and its value."""
    return f"{key} {value if isinstance(value, str) else str(value)}"
