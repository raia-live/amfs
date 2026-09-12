"""Benchmark and system rows must not be counted as somebody's memory.

Benchmarks write to the same tables real memory lives in — deliberately, since a
benchmark writing somewhere else would not be exercising the thing it measures —
so every aggregate over those tables counts them unless it is told not to. On the
account totals that is simply a wrong number: a user reading "1,204 memories"
should be reading how many memories they have.

The two things worth testing beyond "the filter filters" are the near misses and
the scoped exception. The rules are prefix rules on purpose, so an agent someone
named ``benchmarking-agent`` keeps appearing in its owner's own totals; and a
caller that names an entity_path has opted into it, which is what lets a briefing
be built on a benchmark's own path.
"""

from __future__ import annotations

from datetime import UTC, datetime

from amfs_core.aggregates import extended_stats_from_entries
from amfs_core.exclusions import (
    is_excluded_agent,
    is_excluded_entity,
    is_excluded_entry,
)
from amfs_core.models import MemoryEntry, Provenance

NOW = datetime.now(UTC)


def _entry(
    entity_path: str = "repo/module",
    agent_id: str = "agent-a",
    key: str = "k",
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value={"v": 1},
        provenance=Provenance(agent_id=agent_id, session_id="s1", written_at=NOW),
        confidence=0.8,
    )


class TestWhatCounts:
    def test_the_system_scratch_paths_are_excluded(self) -> None:
        assert is_excluded_entity("_system")
        assert is_excluded_entity("_system/embeddings")
        assert is_excluded_entity("bench-retrieval")
        assert is_excluded_entity("bench/retrieval")
        assert is_excluded_entity("_bench-retrieval")
        assert is_excluded_entity("BENCH-retrieval")  # the rule is case-blind

    def test_a_users_own_paths_are_not(self) -> None:
        """The near misses, which are the whole reason these are prefix rules.

        Someone is entitled to call their own path ``benchmarking`` and still see
        it in their own totals.
        """
        assert not is_excluded_entity("benchmarking-agent")
        assert not is_excluded_entity("benchmark/results")
        assert not is_excluded_entity("_systemic/x")
        assert not is_excluded_entity("_systems/x")
        assert not is_excluded_entity("amfs/core-engine")
        assert not is_excluded_entity("user/preferences")
        assert not is_excluded_entity("x/_system")  # matched at the front only
        assert not is_excluded_entity("")
        assert not is_excluded_entity(None)

    def test_bench_harnesses_and_the_server_are_excluded_as_agents(self) -> None:
        assert is_excluded_agent("bench-runner")
        assert is_excluded_agent("_bench/runner")
        assert is_excluded_agent("amfs-server")
        assert is_excluded_agent("system")
        assert not is_excluded_agent("benchmarking-agent")
        assert not is_excluded_agent("dashboard-agent")
        assert not is_excluded_agent(None)

    def test_either_rule_is_enough(self) -> None:
        """A benchmark writing to a real-looking path is still the benchmark's row."""
        assert is_excluded_entry(_entry(entity_path="bench-x", agent_id="dashboard-agent"))
        assert is_excluded_entry(_entry(entity_path="repo/module", agent_id="bench-runner"))
        assert not is_excluded_entry(_entry())


class TestTheTotals:
    def test_bench_rows_do_not_inflate_the_totals(self) -> None:
        entries = [
            _entry(key="real-1"),
            _entry(key="real-2", entity_path="repo/other"),
            _entry(key="b1", entity_path="bench-retrieval", agent_id="bench-runner"),
            _entry(key="b2", entity_path="_system/embeddings", agent_id="amfs-server"),
        ]
        stats = extended_stats_from_entries(entries)

        assert stats["total_entries"] == 2
        assert stats["total_entities"] == 2
        assert stats["total_agents"] == 1
        assert set(stats["entities"]) == {"repo/module", "repo/other"}
        assert set(stats["agents"]) == {"agent-a"}

    def test_an_account_that_is_only_bench_rows_reads_as_empty_not_as_busy(self) -> None:
        stats = extended_stats_from_entries(
            [_entry(key="b1", entity_path="bench-x"), _entry(key="b2", entity_path="_system")]
        )
        assert stats["total_entries"] == 0
        assert stats["total_entities"] == 0
        # Not a crash on the empty case, and no confidence invented for it.
        assert stats["confidence_avg"] == 0.0
        assert stats["oldest_entry_at"] is None

    def test_the_recall_credit_is_not_earned_by_bench_reuse(self) -> None:
        """``recalled_tokens_saved`` is shown to users as value delivered.

        A benchmark reading its own rows a thousand times must not turn into a
        thousand reuses on somebody's dashboard.
        """
        e = _entry(key="b", entity_path="bench-x", agent_id="bench-runner")
        e.recall_count = 1000
        stats = extended_stats_from_entries([_entry(key="real"), e])
        assert stats["total_recalls"] == 0
        assert stats["recalled_tokens_saved"] == 0

    def test_the_adapter_default_agrees_with_the_pure_helper(self) -> None:
        """``AdapterABC.stats`` iterates ``list()``, so it needs the same filter.

        The SQL aggregate, this default and the room-scoped HTTP path are three
        routes to one number, and a filter on some of them would make that untrue
        in a way nobody would notice. Called on a stand-in rather than a real
        adapter because ``stats`` reads nothing but ``self.list()``.
        """
        from amfs_core.abc import AdapterABC

        class _Listing:
            def __init__(self, entries: list[MemoryEntry]) -> None:
                self._entries = entries

            def list(self) -> list[MemoryEntry]:
                return self._entries

        stats = AdapterABC.stats(
            _Listing(
                [
                    _entry(key="real"),
                    _entry(key="b", entity_path="bench-x", agent_id="bench-runner"),
                ]
            )
        )
        assert stats.total_entries == 1
        assert stats.total_agents == 1
        assert set(stats.entities) == {"repo/module"}
