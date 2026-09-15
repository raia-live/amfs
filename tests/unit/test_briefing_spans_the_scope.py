"""A briefing on a scope must reach the topics under it, not just the scope.

Every caller that opens a session can name the repository it is in and nothing
finer. Nobody writes memory at a repository root, though — it goes to
``<repo>/deploy``, ``<repo>/ci-cd``, ``<repo>/auth``. Matched by equality, a
briefing on ``amfs-internal`` therefore reads back nothing at all, and reports
that truthfully, which is how a store holding exactly the runbook the session
needed can still hand back an empty briefing.

"Who to ask" already spanned descendants — ``_inject_who_to_ask`` passes
``include_descendants=True`` to ``rank_authors``. Hot context did not, so the
same briefing would name the right person and none of what they knew. These
tests pin the inconsistency closed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

pytest.importorskip("amfs_cortex", reason="amfs_cortex not installed")

from amfs_core.models import (
    Digest,
    DigestType,
    MemoryEntry,
    Provenance,
)
from amfs_cortex.briefing import BriefingService


def _now() -> datetime:
    return datetime.now(UTC)


def _entry(key: str, *, entity_path: str) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=f"value-{key}",
        provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
        confidence=1.0,
    )


def _entity_digest(scope: str) -> Digest:
    return Digest(
        digest_type=DigestType.ENTITY,
        scope=scope,
        summary={"narrative": f"Summary of {scope}", "agents": ["a1"]},
        entry_count=5,
        source_agents=["a1"],
        compiled_at=_now() - timedelta(hours=1),
        namespace="default",
    )


def _adapter(digests: list[Digest], entries: list[MemoryEntry]) -> MagicMock:
    adapter = MagicMock()
    adapter.list_digests.return_value = digests
    adapter.search.return_value = entries
    adapter.list_branches.return_value = []
    return adapter


def _search_queries(adapter: MagicMock) -> list:
    return [call.args[0] for call in adapter.search.call_args_list if call.args]


class TestTheScopeIsAPrefix:
    def test_a_compiled_scope_asks_about_everything_beneath_it(self) -> None:
        """The fix, at the only layer that can express it: the query."""
        adapter = _adapter(
            [_entity_digest("amfs-internal")],
            [_entry("runbook", entity_path="amfs-internal/deploy")],
        )

        BriefingService(adapter).briefing(entity_path="amfs-internal")

        queries = _search_queries(adapter)
        assert queries, "the briefing issued no search at all"
        assert all(q.include_descendants for q in queries), (
            "hot context asked about the scope exactly, so it can only ever find "
            "entries written at the repository root — where nobody writes"
        )

    def test_an_uncompiled_scope_asks_the_same_way(self) -> None:
        """The standalone path exists precisely for a scope nothing has compiled.

        That is the likeliest state of a repo root, so it must not be the one
        path that still matches exactly.
        """
        adapter = _adapter([], [_entry("runbook", entity_path="amfs-internal/deploy")])

        BriefingService(adapter).briefing(entity_path="amfs-internal")

        queries = _search_queries(adapter)
        assert queries
        assert all(q.include_descendants for q in queries)

    def test_the_scope_is_still_the_one_that_was_asked_for(self) -> None:
        """Widening how a path matches must not change which path is matched."""
        adapter = _adapter([_entity_digest("amfs-internal")], [])

        BriefingService(adapter).briefing(entity_path="amfs-internal")

        assert all(q.entity_path == "amfs-internal" for q in _search_queries(adapter))


class TestEachEntrySaysWhereItCameFrom:
    def test_hot_context_names_the_path_of_every_entry(self) -> None:
        """Once entries span a scope, the key alone stops identifying them.

        Two topics under one repo can both hold ``task-summary-release``, and a
        reader given only keys cannot tell which is about deploys.
        """
        adapter = _adapter(
            [_entity_digest("amfs-internal")],
            [
                _entry("hotfix-needs-backmerge", entity_path="amfs-internal/ci-cd"),
                _entry("pattern-sync-by-merge", entity_path="amfs-internal/deploy"),
            ],
        )

        result = BriefingService(adapter).briefing(entity_path="amfs-internal")

        hot = result[0].summary["hot_context"]
        assert [h["entity_path"] for h in hot] == [
            "amfs-internal/ci-cd",
            "amfs-internal/deploy",
        ]

    def test_the_uncompiled_path_labels_entries_too(self) -> None:
        adapter = _adapter([], [_entry("runbook", entity_path="amfs-internal/deploy")])

        result = BriefingService(adapter).briefing(entity_path="amfs-internal")

        hot = result[0].summary["hot_context"]
        assert hot[0]["entity_path"] == "amfs-internal/deploy"
