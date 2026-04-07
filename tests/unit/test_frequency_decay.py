"""Unit tests for frequency-modulated decay (HMO Track 1)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from amfs_core.models import MemoryEntry, MemoryType, Provenance


def _entry(
    *,
    recall_count: int = 0,
    outcome_count: int = 0,
    memory_type: MemoryType = MemoryType.FACT,
    age_days: float = 30.0,
    confidence: float = 1.0,
) -> MemoryEntry:
    written_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return MemoryEntry(
        entity_path="svc",
        key="k",
        value="v",
        provenance=Provenance(agent_id="a", session_id="s", written_at=written_at),
        confidence=confidence,
        recall_count=recall_count,
        outcome_count=outcome_count,
        memory_type=memory_type,
    )


class TestRecallCountField:
    def test_default_zero(self) -> None:
        e = _entry()
        assert e.recall_count == 0

    def test_serialization_roundtrip(self) -> None:
        e = _entry(recall_count=42)
        data = e.model_dump(mode="json")
        restored = MemoryEntry.model_validate(data)
        assert restored.recall_count == 42


class TestFrequencyModulatedDecay:
    """Verify that higher recall_count slows decay via log1p modulation."""

    def test_no_decay_when_disabled(self) -> None:
        e = _entry(recall_count=100, age_days=365)
        assert e.effective_confidence() == e.confidence

    def test_zero_recall_count_baseline(self) -> None:
        """With recall_count=0, log1p(0)=0 so effective_half_life = base * 1."""
        e = _entry(recall_count=0, age_days=30)
        eff = e.effective_confidence(decay_half_life_days=30)
        expected = 1.0 * math.pow(0.5, 30 / 30)
        assert abs(eff - expected) < 1e-9

    def test_high_recall_count_decays_slower(self) -> None:
        """Entries accessed 100 times should retain more confidence than cold ones."""
        cold = _entry(recall_count=0, age_days=60)
        hot = _entry(recall_count=100, age_days=60)

        cold_eff = cold.effective_confidence(decay_half_life_days=30)
        hot_eff = hot.effective_confidence(decay_half_life_days=30)

        assert hot_eff > cold_eff, "Frequently recalled entry should decay slower"

    def test_moderate_recall_count(self) -> None:
        e = _entry(recall_count=10, age_days=30)
        half_life = 30.0
        freq_factor = 1 + math.log1p(10)
        effective_hl = half_life * freq_factor
        expected = math.pow(0.5, 30.0 / effective_hl)
        eff = e.effective_confidence(decay_half_life_days=30)
        assert abs(eff - expected) < 1e-9

    def test_combined_with_outcome_boost(self) -> None:
        """outcome_count > 0 should still double the effective half-life on top of frequency."""
        e = _entry(recall_count=5, outcome_count=2, age_days=30)
        half_life = 30.0
        freq_hl = half_life * (1 + math.log1p(5))
        outcome_hl = freq_hl * 2
        expected = math.pow(0.5, 30.0 / outcome_hl)
        eff = e.effective_confidence(decay_half_life_days=30)
        assert abs(eff - expected) < 1e-9

    def test_belief_decays_faster(self) -> None:
        fact = _entry(recall_count=5, age_days=30, memory_type=MemoryType.FACT)
        belief = _entry(recall_count=5, age_days=30, memory_type=MemoryType.BELIEF)
        assert belief.effective_confidence(decay_half_life_days=30) < \
               fact.effective_confidence(decay_half_life_days=30)

    def test_experience_decays_slower(self) -> None:
        fact = _entry(recall_count=5, age_days=30, memory_type=MemoryType.FACT)
        exp = _entry(recall_count=5, age_days=30, memory_type=MemoryType.EXPERIENCE)
        assert exp.effective_confidence(decay_half_life_days=30) > \
               fact.effective_confidence(decay_half_life_days=30)

    def test_four_signal_ordering(self) -> None:
        """Most decayed -> least: cold belief < cold fact < hot fact < hot+outcome fact."""
        cold_belief = _entry(recall_count=0, outcome_count=0, memory_type=MemoryType.BELIEF, age_days=60)
        cold_fact = _entry(recall_count=0, outcome_count=0, memory_type=MemoryType.FACT, age_days=60)
        hot_fact = _entry(recall_count=50, outcome_count=0, memory_type=MemoryType.FACT, age_days=60)
        hot_outcome = _entry(recall_count=50, outcome_count=3, memory_type=MemoryType.FACT, age_days=60)

        hl = 30.0
        scores = [
            cold_belief.effective_confidence(decay_half_life_days=hl),
            cold_fact.effective_confidence(decay_half_life_days=hl),
            hot_fact.effective_confidence(decay_half_life_days=hl),
            hot_outcome.effective_confidence(decay_half_life_days=hl),
        ]
        assert scores == sorted(scores), f"Expected ascending order, got {scores}"


class TestRecallCountInEngine:
    """Verify recall_count increments on read and carries forward on write."""

    def test_recall_count_increments_on_read(self, tmp_path) -> None:
        from amfs_core.engine import CausalTagger, CoWEngine
        from amfs_filesystem.adapter import FilesystemAdapter

        adapter = FilesystemAdapter(root=tmp_path / "store")
        tagger = CausalTagger(agent_id="test-agent", session_id="s1")
        engine = CoWEngine(adapter=adapter, tagger=tagger)

        engine.write("svc", "key1", "value1")
        entry = engine.read("svc", "key1")
        assert entry is not None
        assert entry.version == 1

        raw = adapter.read("svc", "key1")
        assert raw.recall_count == 1

        engine.read("svc", "key1")
        engine.read("svc", "key1")
        raw2 = adapter.read("svc", "key1")
        assert raw2.recall_count == 3

    def test_recall_count_carried_on_write(self, tmp_path) -> None:
        from amfs_core.engine import CausalTagger, CoWEngine
        from amfs_filesystem.adapter import FilesystemAdapter

        adapter = FilesystemAdapter(root=tmp_path / "store")
        tagger = CausalTagger(agent_id="test-agent", session_id="s1")
        engine = CoWEngine(adapter=adapter, tagger=tagger)

        engine.write("svc", "key1", "v1")
        engine.read("svc", "key1")
        engine.read("svc", "key1")

        engine.write("svc", "key1", "v2")
        entry = adapter.read("svc", "key1")
        assert entry.version == 2
        assert entry.recall_count >= 2, "recall_count should carry forward on write"

    def test_read_does_not_bump_version(self, tmp_path) -> None:
        from amfs_core.engine import CausalTagger, CoWEngine
        from amfs_filesystem.adapter import FilesystemAdapter

        adapter = FilesystemAdapter(root=tmp_path / "store")
        tagger = CausalTagger(agent_id="test-agent", session_id="s1")
        engine = CoWEngine(adapter=adapter, tagger=tagger)

        engine.write("svc", "key1", "v1")
        for _ in range(10):
            engine.read("svc", "key1")

        entry = adapter.read("svc", "key1")
        assert entry.version == 1, "Reads must not create new CoW versions"
        assert entry.recall_count == 10
