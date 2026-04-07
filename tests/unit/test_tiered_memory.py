"""Unit tests for tiered memory and progressive retrieval (HMO Track 2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from amfs_core.models import MemoryEntry, MemoryTier, Provenance, TierConfig
from amfs_core.tiering import PriorityScorer, TierAssigner


def _entry(
    key: str,
    *,
    confidence: float = 1.0,
    recall_count: int = 0,
    age_days: float = 0.0,
) -> MemoryEntry:
    written_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return MemoryEntry(
        entity_path="svc",
        key=key,
        value="v",
        provenance=Provenance(agent_id="a", session_id="s", written_at=written_at),
        confidence=confidence,
        recall_count=recall_count,
    )


class TestMemoryTierEnum:
    def test_values(self) -> None:
        assert MemoryTier.HOT == 1
        assert MemoryTier.WARM == 2
        assert MemoryTier.ARCHIVE == 3

    def test_default_tier(self) -> None:
        e = _entry("k")
        assert e.tier == 3


class TestPriorityScorer:
    def test_fresh_high_confidence_scores_high(self) -> None:
        scorer = PriorityScorer()
        e = _entry("k1", confidence=1.0, age_days=0)
        score = scorer.score(e)
        assert score > 0

    def test_old_entry_scores_lower(self) -> None:
        scorer = PriorityScorer()
        fresh = _entry("k1", confidence=1.0, age_days=0)
        old = _entry("k2", confidence=1.0, age_days=90)
        assert scorer.score(fresh) > scorer.score(old)

    def test_high_recall_count_boosts_score(self) -> None:
        scorer = PriorityScorer()
        cold = _entry("k1", confidence=0.8, recall_count=0, age_days=10)
        hot = _entry("k2", confidence=0.8, recall_count=50, age_days=10)
        assert scorer.score(hot) > scorer.score(cold)

    def test_low_confidence_penalized(self) -> None:
        scorer = PriorityScorer()
        high = _entry("k1", confidence=1.0, age_days=5)
        low = _entry("k2", confidence=0.2, age_days=5)
        assert scorer.score(high) > scorer.score(low)

    def test_score_batch_returns_all_entries(self) -> None:
        scorer = PriorityScorer()
        entries = [_entry(f"k{i}") for i in range(5)]
        scores = scorer.score_batch(entries)
        assert len(scores) == 5
        assert all(isinstance(v, float) for v in scores.values())

    def test_custom_config_weights(self) -> None:
        scorer_alpha = PriorityScorer(TierConfig(alpha=2.0, beta=0.0, decay_lambda=0.0))
        scorer_beta = PriorityScorer(TierConfig(alpha=0.0, beta=2.0, decay_lambda=0.0))
        e = _entry("k1", confidence=0.8, age_days=0)
        score_a = scorer_alpha.score(e)
        score_b = scorer_beta.score(e)
        assert score_a != score_b


class TestTierAssigner:
    def test_small_set_all_hot(self) -> None:
        config = TierConfig(hot_capacity=10, warm_capacity=20)
        assigner = TierAssigner(config)
        entries = [_entry(f"k{i}") for i in range(5)]
        scores = {e.entry_key: float(5 - i) for i, e in enumerate(entries)}
        result = assigner.assign(entries, scores)
        assert all(t == MemoryTier.HOT for t in result.values())

    def test_overflow_to_warm(self) -> None:
        config = TierConfig(hot_capacity=2, warm_capacity=3)
        assigner = TierAssigner(config)
        entries = [_entry(f"k{i}") for i in range(5)]
        scores = {e.entry_key: float(5 - i) for i, e in enumerate(entries)}
        result = assigner.assign(entries, scores)
        hot_count = sum(1 for t in result.values() if t == MemoryTier.HOT)
        warm_count = sum(1 for t in result.values() if t == MemoryTier.WARM)
        assert hot_count == 2
        assert warm_count == 3

    def test_overflow_to_archive(self) -> None:
        config = TierConfig(hot_capacity=1, warm_capacity=1)
        assigner = TierAssigner(config)
        entries = [_entry(f"k{i}") for i in range(5)]
        scores = {e.entry_key: float(5 - i) for i, e in enumerate(entries)}
        result = assigner.assign(entries, scores)
        hot_count = sum(1 for t in result.values() if t == MemoryTier.HOT)
        warm_count = sum(1 for t in result.values() if t == MemoryTier.WARM)
        archive_count = sum(1 for t in result.values() if t == MemoryTier.ARCHIVE)
        assert hot_count == 1
        assert warm_count == 1
        assert archive_count == 3

    def test_assign_with_scores(self) -> None:
        config = TierConfig(hot_capacity=2, warm_capacity=3)
        assigner = TierAssigner(config)
        scorer = PriorityScorer(config)
        entries = [
            _entry("fresh", confidence=1.0, recall_count=10, age_days=0),
            _entry("old", confidence=0.5, recall_count=0, age_days=90),
            _entry("mid", confidence=0.8, recall_count=5, age_days=10),
        ]
        tiers, scores = assigner.assign_with_scores(entries, scorer)
        assert len(tiers) == 3
        assert len(scores) == 3
        assert tiers["svc/fresh"] == MemoryTier.HOT


class TestProgressiveSearch:
    """Verify that search(depth=...) filters by tier."""

    def test_depth_filters_entries(self, tmp_path) -> None:
        from amfs_core.engine import CausalTagger, CoWEngine
        from amfs_filesystem.adapter import FilesystemAdapter

        adapter = FilesystemAdapter(root=tmp_path / "store")
        tagger = CausalTagger(agent_id="test-agent", session_id="s1")
        engine = CoWEngine(adapter=adapter, tagger=tagger)

        from amfs.memory import AgentMemory

        mem = AgentMemory(adapter=adapter, agent_id="test-agent", session_id="s1")

        mem.write("svc", "hot-key", "val1", confidence=1.0)
        mem.write("svc", "warm-key", "val2", confidence=0.8)
        mem.write("svc", "cold-key", "val3", confidence=0.5)

        all_results = mem.search(entity_path="svc", depth=3)
        assert len(all_results) == 3

    def test_search_query_depth_default(self) -> None:
        from amfs_core.models import SearchQuery
        sq = SearchQuery(entity_path="svc")
        assert sq.depth == 3
