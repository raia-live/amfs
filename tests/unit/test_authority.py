"""Unit tests for the shared authority ranking.

One scorer feeds the Who Knows What page, the briefing's who_to_ask block,
retrieval and discovery, so what is pinned here is the behaviour every surface
inherits: that validated outcomes beat raw volume, that a dormant author is
demoted rather than erased, and that the explanation only claims what was
actually measured.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("amfs_core", reason="amfs_core not installed")

from amfs_core.authority import (
    AuthorRank,
    humanise_age,
    is_siloed,
    rank_authors,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
ENTITY = "myapp/auth"


def _row(
    agent_id: str,
    *,
    entity_path: str = ENTITY,
    entry_count: int = 1,
    recalls: int = 0,
    validated: int = 0,
    days_ago: float = 1.0,
    confidence: float = 0.8,
) -> dict:
    return {
        "agent_id": agent_id,
        "entity_path": entity_path,
        "entry_count": entry_count,
        "avg_confidence": confidence,
        "last_written": NOW - timedelta(days=days_ago),
        "first_written": NOW - timedelta(days=days_ago + 10),
        "total_recalls": recalls,
        "outcome_linked_count": validated,
    }


class TestRanking:

    def test_no_authors_for_unknown_entity(self):
        assert rank_authors("myapp/nothing", stats=[_row("a")], now=NOW) == []

    def test_ignores_rows_for_other_entities(self):
        stats = [_row("a"), _row("b", entity_path="other/thing", entry_count=99)]
        ranked = rank_authors(ENTITY, stats=stats, now=NOW)
        assert [r.agent_id for r in ranked] == ["a"]

    def test_validated_outcomes_beat_raw_volume(self):
        """The whole point of outcome-weighting: a chatty agent is not the authority."""
        stats = [
            _row("chatty", entry_count=40),
            _row("proven", entry_count=12, validated=8, recalls=20),
        ]
        ranked = rank_authors(ENTITY, stats=stats, now=NOW)
        assert ranked[0].agent_id == "proven"

    def test_recency_demotes_but_does_not_erase(self):
        stats = [
            _row("dormant", entry_count=30, days_ago=800),
            _row("active", entry_count=30, days_ago=1),
        ]
        ranked = rank_authors(ENTITY, stats=stats, now=NOW)
        assert [r.agent_id for r in ranked] == ["active", "dormant"]
        # Demoted, still present and still scoring — on a quiet topic the only
        # author may be dormant and is still the right person to ask.
        assert ranked[1].score > 0

    def test_sole_author_of_a_quiet_topic_still_ranks(self):
        ranked = rank_authors(ENTITY, stats=[_row("solo", days_ago=400)], now=NOW)
        assert len(ranked) == 1
        assert ranked[0].score > 0
        assert ranked[0].strength == "primary"

    def test_limit_applies_and_none_returns_all(self):
        stats = [_row(f"a{i}", entry_count=10 - i) for i in range(5)]
        assert len(rank_authors(ENTITY, stats=stats, now=NOW)) == 3
        assert len(rank_authors(ENTITY, stats=stats, limit=None, now=NOW)) == 5

    def test_order_is_stable_for_identical_authors(self):
        stats = [_row("b-agent"), _row("a-agent")]
        first = [r.agent_id for r in rank_authors(ENTITY, stats=stats, now=NOW)]
        second = [r.agent_id for r in rank_authors(ENTITY, stats=stats, now=NOW)]
        assert first == second == ["a-agent", "b-agent"]


class TestShareAndStrength:

    def test_share_sums_to_one_across_authors(self):
        stats = [_row("a", entry_count=3), _row("b", entry_count=1)]
        ranked = rank_authors(ENTITY, stats=stats, limit=None, now=NOW)
        assert sum(r.share for r in ranked) == pytest.approx(1.0)

    def test_strength_labels_follow_share(self):
        # Shares of 0.67 / 0.28 / 0.06 straddle the 0.6 and 0.2 thresholds.
        stats = [
            _row("dominant", entry_count=60),
            _row("helper", entry_count=25),
            _row("passer-by", entry_count=5),
        ]
        by_id = {
            r.agent_id: r
            for r in rank_authors(ENTITY, stats=stats, limit=None, now=NOW)
        }
        assert by_id["dominant"].strength == "primary"
        assert by_id["helper"].strength == "contributor"
        assert by_id["passer-by"].strength == "touched"

    def test_share_is_relative_to_the_topic_not_the_account(self):
        """A quiet topic's only author is primary, however small the counts."""
        ranked = rank_authors(ENTITY, stats=[_row("solo", entry_count=2)], now=NOW)
        assert ranked[0].share == pytest.approx(1.0)
        assert ranked[0].strength == "primary"


class TestReason:

    def test_reason_states_volume_validation_and_recency(self):
        stats = [_row("api-agent", entry_count=34, validated=3, days_ago=2)]
        reason = rank_authors(ENTITY, stats=stats, now=NOW)[0].reason
        assert "34 memories" in reason
        assert "3 with validated outcomes" in reason
        assert "2 days ago" in reason

    def test_reason_omits_claims_it_cannot_make(self):
        """No validated outcomes and no recalls means the sentence says neither."""
        reason = rank_authors(ENTITY, stats=[_row("a", entry_count=1)], now=NOW)[0].reason
        assert "validated" not in reason
        assert "recalled" not in reason
        assert "1 memory" in reason

    def test_singular_and_plural_agree(self):
        one = rank_authors(ENTITY, stats=[_row("a", entry_count=1, validated=1, recalls=1)], now=NOW)[0]
        assert "1 memory here" in one.reason
        assert "1 with validated outcome," in one.reason
        assert "recalled 1 time by" in one.reason


class TestHumaniseAge:

    @pytest.mark.parametrize(
        "delta,expected",
        [
            (timedelta(minutes=5), "in the last hour"),
            (timedelta(hours=3), "3 hours ago"),
            (timedelta(days=1), "1 day ago"),
            (timedelta(days=2), "2 days ago"),
            (timedelta(days=90), "3 months ago"),
            (timedelta(days=800), "2 years ago"),
        ],
    )
    def test_phrasing(self, delta, expected):
        assert humanise_age(NOW - delta, NOW) == expected

    def test_unknown_time_is_stated_not_guessed(self):
        assert humanise_age(None, NOW) == "at an unknown time"

    def test_naive_timestamps_are_treated_as_utc(self):
        """Postgres can hand back naive datetimes; subtracting one would raise."""
        stats = [_row("a")]
        stats[0]["last_written"] = datetime(2026, 8, 9, 12, 0)
        ranked = rank_authors(ENTITY, stats=stats, now=NOW)
        assert ranked[0].last_written is not None
        assert "1 day ago" in ranked[0].reason


class TestSiloDetection:

    def test_single_author_is_siloed(self):
        assert is_siloed(rank_authors(ENTITY, stats=[_row("a")], now=NOW))

    def test_two_authors_are_not(self):
        stats = [_row("a"), _row("b")]
        assert not is_siloed(rank_authors(ENTITY, stats=stats, now=NOW))


class TestSerialisation:

    def test_to_dict_carries_the_explanation(self):
        ranked = rank_authors(ENTITY, stats=[_row("a", entry_count=4)], now=NOW)
        payload = ranked[0].to_dict()
        assert payload["agent_id"] == "a"
        assert payload["entity_path"] == ENTITY
        assert payload["strength"] == "primary"
        assert payload["reason"]
        assert isinstance(ranked[0], AuthorRank)

    def test_top_keys_are_attached_when_supplied(self):
        ranked = rank_authors(
            ENTITY,
            stats=[_row("a")],
            now=NOW,
            top_keys={"a": ["decision-token-rotation"]},
        )
        assert ranked[0].top_keys == ["decision-token-rotation"]

class TestDescendantRollup:
    """``agent_entity_stats`` scopes by prefix, so the ranker has to decide what
    to do with the child rows it gets back. Off, each path is its own topic —
    what the expertise grid wants, since it draws them as separate columns. On,
    the subtree is one area of ownership — what "who should I ask about this"
    wants, and the reason the query fetched the children at all."""

    def test_child_only_authors_are_ignored_by_default(self):
        ranked = rank_authors(
            ENTITY,
            stats=[_row("child-only", entity_path=f"{ENTITY}/tokens")],
            now=NOW,
        )
        assert ranked == []

    def test_child_only_authors_rank_when_rolled_up(self):
        ranked = rank_authors(
            ENTITY,
            stats=[_row("child-only", entity_path=f"{ENTITY}/tokens")],
            now=NOW,
            include_descendants=True,
        )
        assert [a.agent_id for a in ranked] == ["child-only"]

    def test_an_agents_child_rows_are_summed_not_ranked_separately(self):
        """Three child paths must not become three copies of one agent, each
        holding a third of the evidence it actually has."""
        ranked = rank_authors(
            ENTITY,
            stats=[
                _row("spread", entity_path=f"{ENTITY}/tokens", entry_count=2,
                     validated=1, recalls=3, days_ago=9),
                _row("spread", entity_path=f"{ENTITY}/sessions", entry_count=3,
                     validated=2, recalls=1, days_ago=2),
                _row("spread", entity_path=f"{ENTITY}/mfa", entry_count=1,
                     days_ago=40),
            ],
            now=NOW,
            include_descendants=True,
        )
        assert len(ranked) == 1
        author = ranked[0]
        assert author.entry_count == 6
        assert author.validated_outcomes == 3
        assert author.total_recalls == 4
        # The most recent of its rows, so a live author is not aged out by an
        # old memory it happens to also own.
        assert author.last_written == NOW - timedelta(days=2)

    def test_share_is_computed_over_the_whole_subtree(self):
        ranked = rank_authors(
            ENTITY,
            stats=[
                _row("owner", entity_path=f"{ENTITY}/tokens", entry_count=3),
                _row("owner", entity_path=ENTITY, entry_count=6),
                _row("other", entity_path=f"{ENTITY}/mfa", entry_count=1),
            ],
            now=NOW,
            include_descendants=True,
        )
        by_id = {a.agent_id: a for a in ranked}
        assert by_id["owner"].share == pytest.approx(0.9)
        assert by_id["other"].share == pytest.approx(0.1)

    def test_a_sibling_prefix_is_not_a_descendant(self):
        """`myapp/authz` starts with `myapp/auth` as a string but is a
        different topic; only a `/` boundary counts."""
        ranked = rank_authors(
            ENTITY,
            stats=[_row("sibling", entity_path="myapp/authz")],
            now=NOW,
            include_descendants=True,
        )
        assert ranked == []
