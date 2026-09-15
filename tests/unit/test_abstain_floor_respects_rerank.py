"""The abstain floor must keep what the reranker endorsed and no more.

Step 8 used to *replace* the composite with the rerank score, which put the
cross-encoder's favourite at rank one by construction — and step 9 keeps rank one
unconditionally. Once the rerank became only the relevance term (amfs#399),
confidence and recency can push that favourite to second place, where step 9
judged it on the bi-encoder ``semantic`` score alone and dropped it. That is the
worst case to drop: an entry the reranker rescued is by definition one the
bi-encoder scored low, because rescuing those is what a reranker is for.

The correction has to be the cross-encoder's *absolute* opinion. Exempting on the
normalised score instead keeps junk, because normalisation reports standing within
the batch: its best member scores high however poor the batch is. Every batch
below is fed through the real ``_normalise_rerank`` and ``_rerank_absolute`` rather
than asserted against values by hand, because the first version of this file did
the latter and let exactly that mistake through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SERVER_SRC = (
    Path(__file__).resolve().parents[2]
    / "packages/http-server/src/amfs_http/server.py"
).read_text()

server = pytest.importorskip("amfs_http.server", reason="fastapi not installed")

FLOOR = 0.15


def _rows(logits, semantics, keywords=None):
    """Candidates as step 9 receives them, scored by the real functions.

    ``logits`` are what the cross-encoder emitted, so a test states the model's
    output and nothing downstream of it.
    """
    keywords = keywords or [0.0] * len(logits)
    normalised = server._normalise_rerank(list(logits))
    absolute = server._rerank_absolute(list(logits))
    return [
        (
            f"logit{lg:+.1f}",
            0.0,
            {"semantic": sem, "keyword": kw,
             "rerank": lg, "rerank_normalised": nm, "rerank_absolute": ab},
        )
        for lg, sem, kw, nm, ab in zip(logits, semantics, keywords, normalised, absolute)
    ]


def _apply_floor(scored, floor=FLOOR):
    """Step 9 exactly as the handler runs it."""
    if floor > 0 and len(scored) > 1:
        kept = [scored[0]]
        for entry, score, bd in scored[1:]:
            endorsed = float(bd.get("rerank_absolute") or 0.0) >= server.RERANK_ENDORSED
            if (
                float(bd.get("semantic") or 0.0) < floor
                and not bd.get("keyword")
                and not endorsed
            ):
                continue
            kept.append((entry, score, bd))
        scored = kept
    return [e for e, _, _ in scored]


class TestWhatTheRerankerEndorsed:
    """A high logit with a low bi-encoder score is the rescue case."""

    def test_a_rescued_entry_below_rank_one_survives(self) -> None:
        kept = _apply_floor(_rows(
            logits=[6.0, 7.5],
            semantics=[0.40, 0.03],  # the rescued one, invisible to the bi-encoder
        ))
        assert "logit+7.5" in kept

    def test_it_survives_from_last_place(self) -> None:
        kept = _apply_floor(_rows(
            logits=[6.0, 5.0, 7.0],
            semantics=[0.50, 0.30, 0.01],
        ))
        assert kept == ["logit+6.0", "logit+5.0", "logit+7.0"]


class TestAbstentionSurvivesTheChange:
    """The floor exists to answer "nothing here is relevant". Bugbot's finding on
    the first version of this fix: exempting on the normalised score breaks that,
    because the normalised score is batch-relative."""

    def test_a_batch_the_reranker_rejects_is_still_trimmed(self) -> None:
        kept = _apply_floor(_rows(
            logits=[-6.0, -7.0, -9.0],
            semantics=[0.10, 0.02, 0.01],
        ))
        assert kept == ["logit-6.0"], "only the always-kept top result"

    def test_the_case_that_broke_the_first_attempt(self) -> None:
        """A candidate at logit -3.0 among peers at -9 normalises to the top of
        its batch while the model gives it a 4.7% chance of being relevant. The
        measured numbers, asserted so a future change to either function has to
        confront them."""
        logits = [-11.0, -9.0, -6.0, -3.0]
        normalised = server._normalise_rerank(logits)
        absolute = server._rerank_absolute(logits)

        assert normalised[-1] > 0.97, "batch-relative standing is high"
        assert absolute[-1] < 0.05, "the model's own opinion is not"
        assert normalised[-1] >= FLOOR, "so a floor on normalised would keep it"
        assert absolute[-1] < server.RERANK_ENDORSED, "and endorsement does not"

        kept = _apply_floor(_rows(logits=logits, semantics=[0.01] * 4))
        assert kept == ["logit-11.0"], "nothing here deserved to be kept"

    def test_an_unreranked_tail_entry_is_judged_as_before(self) -> None:
        """The reranker scores only the top 30, so most entries carry no rerank
        fields at all and must behave exactly as they did."""
        kept = _apply_floor([
            ("head", 1.0, {"semantic": 0.60, "keyword": 0.0}),
            ("tail-irrelevant", 0.09, {"semantic": 0.02, "keyword": 0.0}),
            ("tail-relevant", 0.40, {"semantic": 0.55, "keyword": 0.0}),
        ])
        assert kept == ["head", "tail-relevant"]

    def test_a_keyword_match_still_exempts_a_low_scorer(self) -> None:
        kept = _apply_floor(_rows(
            logits=[6.0, -8.0],
            semantics=[0.60, 0.01],
            keywords=[0.0, 1.0],
        ))
        assert "logit-8.0" in kept

    def test_the_single_best_result_is_never_dropped(self) -> None:
        kept = _apply_floor([("only-hope", 0.05, {"semantic": 0.001})])
        assert kept == ["only-hope"]


class TestTheAbsoluteScore:
    def test_it_is_batch_independent(self) -> None:
        """The property the fix rests on: the same logit gets the same answer
        whatever it is scored alongside."""
        alone = server._rerank_absolute([2.0, 2.0])[0]
        among_strong = server._rerank_absolute([2.0, 9.0, 8.5])[0]
        among_weak = server._rerank_absolute([2.0, -9.0, -8.5])[0]
        assert alone == pytest.approx(among_strong) == pytest.approx(among_weak)

    def test_the_boundary_is_the_models_own(self) -> None:
        """Zero logit is even odds, which is where endorsement begins."""
        assert server._rerank_absolute([0.0, 7.0])[0] == pytest.approx(0.5)
        assert server.RERANK_ENDORSED == 0.5

    def test_logits_that_all_land_inside_zero_to_one_read_as_probabilities(self) -> None:
        """The known limit of the shared heuristic, pinned rather than hidden.

        A batch of logits that happens to fall entirely inside 0..1 cannot be
        told from probabilities, and ``_normalise_rerank`` has always had the same
        ambiguity. What matters is which way it errs: read as probabilities these
        come out *lower*, so the floor declines to exempt an entry it might have.
        That withholds a rescue; it never keeps junk, which is the direction that
        would matter.
        """
        assert server._rerank_absolute([0.2, 0.8]) == [0.2, 0.8]
        assert server._rerank_absolute([0.2, 0.8])[0] < server.RERANK_ENDORSED

    def test_a_calibrated_reranker_is_passed_through(self) -> None:
        """Some implementations return probabilities. Squashing those again would
        push a confident 0.99 down to 0.73 and endorse a rejected 0.01 at 0.5."""
        already = [0.99, 0.5, 0.01]
        assert server._rerank_absolute(already) == already

    def test_both_functions_agree_on_which_kind_they_were_given(self) -> None:
        """They must share the test, or one reads a probability as a logit."""
        assert server._rerank_is_calibrated([0.99, 0.01]) is True
        assert server._rerank_is_calibrated([7.2, -9.0]) is False

    def test_it_does_not_overflow_on_extreme_logits(self) -> None:
        assert server._rerank_absolute([-800.0, 800.0]) == [pytest.approx(0.0),
                                                            pytest.approx(1.0)]

    def test_an_empty_batch_is_empty(self) -> None:
        assert server._rerank_absolute([]) == []


class TestTheHandlerUsesThisRule:
    """Read from disk as well as imported: step 9 is inline in an async handler
    that needs an adapter, an embedder and a reranker to reach, and the convention
    for ``amfs_http`` is ``importorskip`` on fastapi — a guard that silently skips
    is not a guard."""

    def test_the_floor_exempts_on_the_absolute_score(self) -> None:
        floor_section = _SERVER_SRC.split("# 9. Abstain floor")[1].split("Reuse accounting")[0]
        assert "rerank_absolute" in floor_section, (
            "the abstain floor is judging the bi-encoder alone again"
        )

    def test_the_floor_does_not_exempt_on_batch_standing(self) -> None:
        """The regression Bugbot caught on the first attempt at this fix."""
        floor_section = _SERVER_SRC.split("# 9. Abstain floor")[1].split("Reuse accounting")[0]
        assert "rerank_normalised" not in floor_section, (
            "batch-relative standing is not evidence of absolute relevance"
        )

    def test_step_eight_records_the_absolute_score(self) -> None:
        assert '"rerank_absolute": absolute_score' in _SERVER_SRC

    def test_the_floor_default_is_unchanged(self) -> None:
        """This changes what is exempt from the floor, not where the floor sits."""
        assert 'os.environ.get("AMFS_RETRIEVE_MIN_SEMANTIC", "0.15")' in _SERVER_SRC
