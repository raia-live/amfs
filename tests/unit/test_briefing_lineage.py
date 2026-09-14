"""A briefing must enter the causal chain, or reinforcement has nothing to act on.

The loop is: an agent is briefed, works, commits an outcome, and the entries that
informed the work gain confidence. The last step reads the session's causal
entries. ``retrieve`` put its top hit there; ``briefing`` did not — amfs#400 gave
it a ``recall_count`` bump, which records *usage*, while reinforcement runs on
*lineage*. So an agent that followed the documented brief-first workflow committed
outcomes that reinforced nothing, and the more faithfully it followed the docs the
less its memory improved.

These tests are about that one edge, so they drive ``AgentMemory.briefing``
against a stub adapter rather than a live Cortex.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/core/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/sdk-python/src"))

from amfs.memory import AgentMemory  # noqa: E402
from amfs_core.aggregates import REUSE_CREDIT_K  # noqa: E402


def _digest(hot):
    return SimpleNamespace(summary={"narrative": "compiled synthesis", "hot_context": hot})


def _hot(entity_path, key, *, version=3, confidence=0.8, agent="peer-agent"):
    return {
        "entity_path": entity_path,
        "key": key,
        "version": version,
        "value": f"the stored text of {key}",
        "confidence": confidence,
        "agent": agent,
        "outcome_count": 2,
        "recall_count": 5,
    }


class _StubAdapter:
    """Just enough adapter for ``briefing`` to take its native path."""

    def __init__(self, digests):
        self._digests = digests
        self.calls: list[dict] = []

    def briefing(self, **kwargs):
        self.calls.append(kwargs)
        return self._digests


def _memory(digests):
    mem = AgentMemory.__new__(AgentMemory)
    from amfs_core.engine import ReadTracker

    mem._adapter = _StubAdapter(digests)
    mem._read_tracker = ReadTracker()
    # agent_id and session_id are read-only properties off the tagger.
    mem._tagger = SimpleNamespace(agent_id="test-agent", session_id="test-session")
    mem._config = SimpleNamespace(namespace="default")
    mem._branch = "main"
    return mem


class TestABriefingEntersTheCausalChain:
    def test_a_credited_briefing_books_the_entry_it_surfaced(self):
        mem = _memory([_digest([_hot("myapp/auth", "decision-session-store")])])

        mem.briefing(entity_path="myapp/auth", credit_reuse=True)

        assert "myapp/auth/decision-session-store" in mem._read_tracker._reads

    def test_the_snapshot_is_what_reinforcement_and_later_readers_need(self):
        """Confidence at the moment of use, not at commit time."""
        mem = _memory([_digest([_hot("myapp/auth", "risk-token-replay",
                                     version=7, confidence=0.55)])])

        mem.briefing(entity_path="myapp/auth", credit_reuse=True)

        snap = mem._read_tracker._entries["myapp/auth/risk-token-replay"]
        assert snap["version"] == 7
        assert snap["confidence"] == 0.55
        assert snap["written_by"] == "peer-agent"
        assert mem._read_tracker._versions["myapp/auth/risk-token-replay"] == 7

    def test_an_uncredited_briefing_books_nothing(self):
        """The dashboard renders briefings on every page view.

        Booking those would attribute a human's page load to an agent's causal
        chain, and amfs#257 is the precedent for what that does to the numbers.
        """
        mem = _memory([_digest([_hot("myapp/auth", "decision-session-store")])])

        mem.briefing(entity_path="myapp/auth")

        assert mem._read_tracker._reads == {}

    def test_the_narrative_is_never_credited(self):
        """It is a synthesis *about* entries; the agent did not read them."""
        mem = _memory([SimpleNamespace(summary={"narrative": "a synthesis", "key_facts": ["x"]})])

        mem.briefing(entity_path="myapp/auth", credit_reuse=True)

        assert mem._read_tracker._reads == {}

    def test_crediting_is_capped_the_same_way_the_recall_bump_is(self):
        """Same cap and same order, so lineage names the entry whose count moved."""
        mem = _memory([
            _digest([_hot("myapp/auth", "first"), _hot("myapp/auth", "second")]),
            _digest([_hot("myapp/billing", "third")]),
        ])

        mem.briefing(entity_path="myapp/auth", credit_reuse=True)

        assert len(mem._read_tracker._reads) == REUSE_CREDIT_K
        assert "myapp/auth/first" in mem._read_tracker._reads

    def test_a_digest_predating_the_version_field_still_books(self):
        """Dropping it would silently reopen the loop for older digests."""
        item = _hot("myapp/auth", "decision-old")
        del item["version"]
        mem = _memory([_digest([item])])

        mem.briefing(entity_path="myapp/auth", credit_reuse=True)

        assert mem._read_tracker._versions["myapp/auth/decision-old"] == 1

    def test_a_malformed_hot_context_never_breaks_the_briefing(self):
        """Lineage is bookkeeping; it must not cost the caller their answer."""
        mem = _memory([_digest(["not a dict", {"key": "no path"}, {"entity_path": "p"}])])

        assert mem.briefing(entity_path="myapp/auth", credit_reuse=True) is not None
        assert mem._read_tracker._reads == {}

    def test_the_digests_are_returned_unchanged(self):
        digests = [_digest([_hot("myapp/auth", "decision-session-store")])]
        mem = _memory(digests)

        assert mem.briefing(entity_path="myapp/auth", credit_reuse=True) is digests


class TestRecordSurfacedMatchesRecord:
    """A briefing-sourced causal entry must be indistinguishable downstream."""

    def test_the_same_fields_are_written_as_a_direct_read(self):
        from datetime import datetime, timezone

        from amfs_core.engine import ReadTracker
        from amfs_core.models import MemoryEntry, MemoryType, Provenance

        direct = ReadTracker()
        entry = MemoryEntry(
            entity_path="myapp/auth",
            key="decision-x",
            value="text",
            version=4,
            confidence=0.9,
            memory_type=MemoryType.BELIEF,
            provenance=Provenance(agent_id="peer", session_id="s",
                                  written_at=datetime.now(timezone.utc)),
        )
        direct.record(entry)

        surfaced = ReadTracker()
        surfaced.record_surfaced(
            "myapp/auth", "decision-x", version=4, value="text",
            confidence=0.9, memory_type="belief", written_by="peer",
        )

        assert direct._entries.keys() == surfaced._entries.keys()
        assert direct._entries["myapp/auth/decision-x"] == surfaced._entries["myapp/auth/decision-x"]
        assert direct._versions == surfaced._versions


class TestTheSnapshotIsTheValueTheEntryHeld:
    """``MemoryEntry.value`` is ``Any``, and coercing it rewrote the record.

    The snapshot is what a trace carries and what a tuned model learns from, so a
    value that differs from the one the entry held is a wrong training example
    rather than a cosmetic difference.
    """

    @pytest.mark.parametrize("value", [
        {"threshold": 0.8, "region": "us-east"},   # structured: str() gave a repr
        ["step-one", "step-two"],
        0,        # falsy: `or ""` collapsed each of these
        False,
        [],
        {},
        "",
    ])
    def test_a_briefing_books_the_value_unchanged(self, value):
        mem = _memory([_digest([{
            "entity_path": "myapp/auth", "key": "decision-x",
            "version": 2, "value": value, "confidence": 0.7,
        }])])

        mem.briefing(entity_path="myapp/auth", credit_reuse=True)

        snapshot = mem._read_tracker.entry_snapshot("myapp/auth/decision-x")
        assert snapshot["value"] == value
        assert type(snapshot["value"]) is type(value)

    def test_a_structured_value_matches_what_a_direct_read_stores(self):
        from datetime import UTC, datetime

        from amfs_core.engine import ReadTracker
        from amfs_core.models import MemoryEntry, MemoryType, Provenance

        value = {"runbook": ["drain", "deploy"], "auto": False}

        direct = ReadTracker()
        direct.record(MemoryEntry(
            entity_path="myapp/auth", key="decision-x", value=value, version=2,
            confidence=0.7, memory_type=MemoryType.FACT,
            provenance=Provenance(agent_id="peer", session_id="s",
                                  written_at=datetime.now(UTC)),
        ))

        mem = _memory([_digest([{
            "entity_path": "myapp/auth", "key": "decision-x",
            "version": 2, "value": value, "confidence": 0.7,
            "memory_type": "fact", "agent": "peer",
        }])])
        mem.briefing(entity_path="myapp/auth", credit_reuse=True)

        key = "myapp/auth/decision-x"
        assert mem._read_tracker.entry_snapshot(key) == direct.entry_snapshot(key)

    def test_memory_type_defaults_rather_than_crashing(self):
        from amfs_core.engine import ReadTracker

        t = ReadTracker()
        t.record_surfaced("p", "k", version=1, value="v", confidence=0.5)
        assert t._entries["p/k"]["memory_type"] == "fact"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
