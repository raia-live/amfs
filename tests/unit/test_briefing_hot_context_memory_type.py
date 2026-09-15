"""A briefing's causal snapshot must say what kind of memory it surfaced.

``hot_context`` gained ``entity_path`` and ``version`` so a briefing could be
booked as a real read, but not ``memory_type`` — and ``record_surfaced`` defaults a
missing one to "fact". So every belief and experience an agent got from a briefing
entered the causal chain claiming to be a fact.

It matters for two reasons beyond tidiness: the distinction drives decay (beliefs
0.5x, experiences 1.5x), and the snapshot is the record a trace preserves and a
tuned model learns from. A briefing-sourced causal entry has to be
indistinguishable from a directly-read one, which was the whole point of carrying
``version``.
"""

from __future__ import annotations

from pathlib import Path

from amfs_core.engine import ReadTracker
from amfs_core.models import MemoryType

_BRIEFING_SRC = (
    Path(__file__).resolve().parents[2]
    / "packages/cortex/src/amfs_cortex/briefing.py"
).read_text()


class TestTheSnapshotRecordsTheKind:
    def _snapshot(self, memory_type):
        tracker = ReadTracker()
        tracker.record_surfaced(
            "myapp/auth", "risk-token-replay",
            version=3, value="tokens are replayable for 60s",
            confidence=0.7, memory_type=memory_type, written_by="api-agent",
        )
        return tracker.entry_snapshot("myapp/auth/risk-token-replay")

    def test_a_belief_is_recorded_as_a_belief(self) -> None:
        assert self._snapshot(MemoryType.BELIEF.value)["memory_type"] == "belief"

    def test_an_experience_is_recorded_as_an_experience(self) -> None:
        assert self._snapshot(MemoryType.EXPERIENCE.value)["memory_type"] == "experience"

    def test_a_missing_kind_still_falls_back_to_fact(self) -> None:
        """The fallback is not the bug — supplying nothing to it was. A digest
        compiled before this field existed must keep working."""
        assert self._snapshot(None)["memory_type"] == "fact"


class TestTheBuildersCarryIt:
    """Both hot-context builders, because there are two and only one being fixed
    is how the field came to be missing from one of them in the first place."""

    def test_every_hot_entry_builder_includes_memory_type(self) -> None:
        builders = _BRIEFING_SRC.count('"recall_count": e.recall_count')
        carried = _BRIEFING_SRC.count('"memory_type": e.memory_type.value')
        assert builders >= 2, "expected both hot-context builders in this module"
        assert carried == builders, (
            f"{builders} hot-context builders but {carried} carry memory_type"
        )

    def test_it_is_a_plain_string_not_an_enum_repr(self) -> None:
        """The dict is serialised into a digest summary, so an enum would arrive
        as its repr and never match the values ``record_surfaced`` expects."""
        assert '"memory_type": e.memory_type,' not in _BRIEFING_SRC
