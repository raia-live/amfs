"""The abstain floor must not delete what the reranker rescued.

Step 8 used to *replace* the composite with the rerank score, which pinned the
cross-encoder's favourite at rank one by construction — and step 9 always keeps
rank one. Once the rerank became only the relevance term (amfs#399), confidence
and recency can push that favourite to second place, where step 9 judged it on
``semantic`` alone and dropped it.

That is the worst case for the pipeline, because an entry the reranker rescued is
by definition one the bi-encoder scored low: rescuing those is what a reranker is
for. So the floor was deleting precisely the judgement the reranker had been added
to make, and the entry it deleted could be one reinforcement had promoted.
"""

from __future__ import annotations

from pathlib import Path

_SERVER_SRC = (
    Path(__file__).resolve().parents[2]
    / "packages/http-server/src/amfs_http/server.py"
).read_text()


def _row(key, *, score, semantic, rerank_normalised=None, keyword=0.0):
    """A (entry, score, breakdown) triple in the shape step 9 consumes."""
    bd = {"semantic": semantic, "keyword": keyword}
    if rerank_normalised is not None:
        bd["rerank_normalised"] = rerank_normalised
    return (key, score, bd)


def _apply_floor(scored, floor=0.15):
    """Step 9 exactly as the handler runs it."""
    if floor > 0 and len(scored) > 1:
        kept = [scored[0]]
        for entry, score, bd in scored[1:]:
            relevance = max(
                float(bd.get("semantic") or 0.0),
                float(bd.get("rerank_normalised") or 0.0),
            )
            if relevance < floor and not bd.get("keyword"):
                continue
            kept.append((entry, score, bd))
        scored = kept
    return [e for e, _, _ in scored]


class TestWhatTheRerankerRescued:
    def test_a_rescued_entry_in_second_place_survives(self) -> None:
        """The regression. Low bi-encoder score, high rerank, not rank one."""
        kept = _apply_floor([
            _row("incumbent", score=1.10, semantic=0.40, rerank_normalised=0.55),
            _row("rescued", score=1.02, semantic=0.04, rerank_normalised=0.91),
        ])
        assert "rescued" in kept

    def test_it_survives_even_when_ranked_last(self) -> None:
        kept = _apply_floor([
            _row("first", score=1.20, semantic=0.50, rerank_normalised=0.60),
            _row("second", score=1.10, semantic=0.30, rerank_normalised=0.40),
            _row("rescued", score=0.98, semantic=0.01, rerank_normalised=0.88),
        ])
        assert kept == ["first", "second", "rescued"]


class TestAbstentionStillWorks:
    """The floor exists to answer 'nothing here is relevant'. Widening the test
    must not cost that, or retrieval starts answering with junk."""

    def test_a_batch_the_reranker_dislikes_is_still_trimmed(self) -> None:
        """``_normalise_rerank`` anchors on the logistic of the batch median, so
        a wholly irrelevant candidate set normalises low and still falls through."""
        kept = _apply_floor([
            _row("best-of-a-bad-lot", score=0.20, semantic=0.10, rerank_normalised=0.03),
            _row("junk-one", score=0.11, semantic=0.02, rerank_normalised=0.02),
            _row("junk-two", score=0.10, semantic=0.01, rerank_normalised=0.01),
        ])
        assert kept == ["best-of-a-bad-lot"], "only the always-kept top result"

    def test_an_unreranked_tail_entry_is_judged_as_before(self) -> None:
        """The reranker scores only the top 30, so most entries carry no rerank
        value at all and must behave exactly as they did."""
        kept = _apply_floor([
            _row("head", score=1.00, semantic=0.60, rerank_normalised=0.70),
            _row("tail-irrelevant", score=0.09, semantic=0.02),
            _row("tail-relevant", score=0.40, semantic=0.55),
        ])
        assert kept == ["head", "tail-relevant"]

    def test_a_keyword_match_still_exempts_a_low_scorer(self) -> None:
        kept = _apply_floor([
            _row("head", score=1.00, semantic=0.60, rerank_normalised=0.70),
            _row("literal-hit", score=0.30, semantic=0.01, keyword=1.0),
        ])
        assert "literal-hit" in kept

    def test_the_single_best_result_is_never_dropped(self) -> None:
        kept = _apply_floor([_row("only-hope", score=0.05, semantic=0.001)])
        assert kept == ["only-hope"]


class TestTheHandlerUsesThisRule:
    """Read from disk rather than imported: step 9 is inline in an async handler
    that needs an adapter, an embedder and a reranker to reach, and importing
    ``amfs_http`` needs fastapi, which the convention here is to *skip* on — a
    guard that silently skips is not a guard.
    """

    def test_the_handler_judges_the_floor_on_the_rerank_too(self) -> None:
        # "# 9." and not "Abstain floor", whose first occurrence in the file is
        # the docstring of ``_retrieve_min_semantic`` — a section that would pass
        # this assertion for reasons unconnected to the code being guarded.
        floor_section = _SERVER_SRC.split("# 9. Abstain floor")[1].split("Reuse accounting")[0]
        assert "rerank_normalised" in floor_section, (
            "the abstain floor is judging bi-encoder semantic alone again"
        )

    def test_the_floor_default_is_unchanged(self) -> None:
        """This widens what passes the floor, not where the floor sits."""
        assert 'os.environ.get("AMFS_RETRIEVE_MIN_SEMANTIC", "0.15")' in _SERVER_SRC
