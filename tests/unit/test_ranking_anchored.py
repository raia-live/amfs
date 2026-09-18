"""Trust modulates relevance; it no longer outranks it.

The numbers are the grid v3 runbook probe's (2026-09-18). The agent held a
few validated, confidence-1.0 notes about other services in the same domain
(bi-encoder similarity to the query about 0.47) and the seeded runbook for
the service the task was about (similarity about 0.62, author confidence
0.7, never tested). The additive blend put the notes first — 0.5 x 0.15 of
similarity gap against 0.35 of confidence and evidence — and the agent,
seeing no runbook, escalated. 47 of 100 first searches surfaced the runbook,
against 84 for plain cosine ranking.
"""

from __future__ import annotations

import os

import pytest
from amfs_core.ranking import TRUST_MODULATION, composite_score


def _runbook(anchored: bool) -> float:
    return composite_score(
        relevance=0.62, recency=0.9, confidence=0.7, evidence=0.0, anchored=anchored,
    )


def _other_service_note(anchored: bool) -> float:
    return composite_score(
        relevance=0.47, recency=0.95, confidence=1.0, evidence=0.67, anchored=anchored,
    )


def test_the_additive_blend_reproduces_the_probe_failure():
    assert _other_service_note(anchored=False) > _runbook(anchored=False)


def test_the_anchored_blend_surfaces_the_relevant_runbook():
    assert _runbook(anchored=True) > _other_service_note(anchored=True)


def test_trust_still_orders_entries_of_equal_relevance():
    validated = composite_score(relevance=0.6, recency=0.5, confidence=1.0, evidence=0.8)
    untested = composite_score(relevance=0.6, recency=0.5, confidence=0.7, evidence=0.0)
    contested = composite_score(relevance=0.6, recency=0.5, confidence=0.5, evidence=-0.5)
    assert validated > untested > contested
    # Decisive, not cosmetic: a validated 1.0 clears an untested 0.7 by well
    # over the rounding the response applies.
    assert validated / untested > 1.15


def test_a_perfect_record_is_worth_at_most_the_modulation():
    bare = composite_score(relevance=0.5, recency=0.0, confidence=0.0, evidence=0.0)
    perfect = composite_score(relevance=0.5, recency=0.0, confidence=1.0, evidence=1.0)
    assert perfect == pytest.approx(bare * (1.0 + TRUST_MODULATION))


def test_recency_cannot_carry_an_off_topic_note_either():
    fresh_wrong = composite_score(relevance=0.47, recency=1.0, confidence=0.7)
    old_right = composite_score(relevance=0.62, recency=0.0, confidence=0.7)
    assert old_right > fresh_wrong


def test_a_temporal_query_lets_recency_decide_between_equals():
    # normalize_temporal raises recency_weight for "yesterday"-style queries.
    fresh = composite_score(relevance=0.6, recency=1.0, confidence=0.7, recency_weight=0.6)
    stale = composite_score(relevance=0.6, recency=0.1, confidence=0.7, recency_weight=0.6)
    assert fresh > stale


def test_keyword_only_hits_are_still_ordered_by_record():
    validated = composite_score(relevance=0.0, keyword=1.0, recency=0.5, confidence=0.9, evidence=0.7)
    untested = composite_score(relevance=0.0, keyword=1.0, recency=0.5, confidence=0.8)
    assert validated > untested > 0.0


def test_no_relevance_means_no_score_under_the_anchor():
    assert composite_score(relevance=0.0, keyword=0.0, recency=1.0, confidence=1.0, evidence=1.0) == 0.0


def test_server_switch_restores_the_additive_form(monkeypatch):
    from amfs_http import server

    monkeypatch.delenv("AMFS_RANK_ADDITIVE_TRUST", raising=False)
    assert server._rank_anchored() is True
    monkeypatch.setenv("AMFS_RANK_ADDITIVE_TRUST", "1")
    assert server._rank_anchored() is False
    assert "AMFS_RANK_ADDITIVE_TRUST" in os.environ


def test_keyword_coverage_weights_the_rare_term_not_the_shared_phrasing() -> None:
    """Every candidate says 'standard deploy step order'; only one says
    'returns'. The shared words carry no weight, the service name carries it
    all, so the returns runbook scores highest for the returns request and a
    note about checkout does not — and plurals stem, so 'returns' finds
    'return'."""
    from amfs_core.ranking import entry_text, keyword_coverage

    docs = {
        "runbook-returns-a": "returns service standard deploy: migrate, warm, roll. step order matters",
        "runbook-checkout-a": "checkout service standard deploy: warm, migrate, roll. step order matters",
        "checkout-note": "for checkout service standard deploys the step order is migrate warm roll",
        "fraud-note": "fraud service standard deploy order: warm flush migrate",
    }
    cov = keyword_coverage(
        "return service standard deploy correct step order runbook",
        {k: entry_text(k, v) for k, v in docs.items()},
    )
    assert cov["runbook-returns-a"] > cov["runbook-checkout-a"] > cov["fraud-note"]
    assert cov["runbook-returns-a"] > cov["checkout-note"]
    # The key counts as text: "runbook-checkout-a" carries "runbook", the note does not.
    assert cov["runbook-checkout-a"] > cov["checkout-note"]
    assert 0.0 <= min(cov.values()) and max(cov.values()) <= 1.0
    # Every query term in every document: no term discriminates, presence is
    # all that is left — the old flag.
    flat = keyword_coverage("deploy", {"a": "a deploy", "b": "b deploy"})
    assert flat == {"a": 1.0, "b": 1.0}
    # A term no document has does not deflate everyone.
    assert keyword_coverage("zzz", {"a": "deploy"}) == {"a": 0.0}
    assert keyword_coverage("", {"a": "deploy"}) == {"a": 0.0}
    assert keyword_coverage("deploy", {}) == {}
