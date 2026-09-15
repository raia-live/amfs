"""The filesystem store must be able to enumerate the paths it can write.

``entity_dir`` maps an entity path straight onto nested directories, so ``a/b``
is stored at ``<ns>/a/b``. ``all_entity_paths`` walked a single level of the
namespace and reported directory *names*, so it answered ``a`` and never
``a/b`` — and every caller that enumerates the store starts there. An entry
written to a nested path was therefore invisible to ``list(None)`` and to the
base ``search`` built on it, while a direct ``list("a/b")`` returned it
perfectly well. Writes were never the problem; enumeration was.

Nested paths are the normal case rather than an exotic one: entity paths are
conventionally ``{scope}/{topic}``, so almost every real entry is nested and
the bug hid behind the flat paths that tests happened to use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from amfs_core.models import MemoryEntry, Provenance, SearchQuery
from amfs_filesystem.adapter import FilesystemAdapter


def _adapter(tmp_path: Path) -> FilesystemAdapter:
    return FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")


def _write(adapter: FilesystemAdapter, entity_path: str, key: str) -> None:
    adapter.write(
        MemoryEntry(
            entity_path=entity_path,
            key=key,
            value=f"value of {entity_path}/{key}",
            provenance=Provenance(
                agent_id="ag", session_id="s", written_at=datetime.now(UTC)
            ),
            confidence=1.0,
        )
    )


def _pairs(entries: list[MemoryEntry]) -> list[tuple[str, str]]:
    return sorted((e.entity_path, e.key) for e in entries)


class TestEnumeratingTheStore:
    def test_a_nested_path_is_listed(self, tmp_path: Path) -> None:
        """The bug, at its source."""
        adapter = _adapter(tmp_path)
        _write(adapter, "amfs-internal/deploy", "runbook")

        assert adapter._layout.all_entity_paths() == ["amfs-internal/deploy"]

    def test_a_nested_entry_is_reachable_without_naming_its_path(
        self, tmp_path: Path
    ) -> None:
        """What the bug cost: the entry existed and could not be found."""
        adapter = _adapter(tmp_path)
        _write(adapter, "amfs-internal/deploy", "runbook")

        assert _pairs(adapter.list(None)) == [("amfs-internal/deploy", "runbook")]
        assert _pairs(adapter.list("amfs-internal/deploy")) == [
            ("amfs-internal/deploy", "runbook")
        ]

    def test_paths_are_found_at_any_depth(self, tmp_path: Path) -> None:
        adapter = _adapter(tmp_path)
        _write(adapter, "a/b/c/d", "deep")

        assert adapter._layout.all_entity_paths() == ["a/b/c/d"]
        assert _pairs(adapter.list(None)) == [("a/b/c/d", "deep")]

    def test_a_path_that_holds_entries_and_child_paths_is_both(
        self, tmp_path: Path
    ) -> None:
        """A repo root usually has a note of its own as well as topics under it."""
        adapter = _adapter(tmp_path)
        _write(adapter, "amfs-internal", "root-note")
        _write(adapter, "amfs-internal/deploy", "runbook")

        assert adapter._layout.all_entity_paths() == [
            "amfs-internal",
            "amfs-internal/deploy",
        ]
        assert _pairs(adapter.list(None)) == [
            ("amfs-internal", "root-note"),
            ("amfs-internal/deploy", "runbook"),
        ]

    def test_one_directory_serving_as_key_and_as_path_is_not_a_leaf(
        self, tmp_path: Path
    ) -> None:
        """The structural collision, which is why the walk enters key directories.

        Entity ``x`` holding key ``y`` and entity ``x/y`` holding key ``z`` both
        live under ``<ns>/x/y``: it is a key directory of ``x`` and an entity
        directory of ``x/y`` at the same time. Stopping at key directories would
        lose ``x/y`` entirely.
        """
        adapter = _adapter(tmp_path)
        _write(adapter, "x", "y")
        _write(adapter, "x/y", "z")

        assert adapter._layout.all_entity_paths() == ["x", "x/y"]
        assert _pairs(adapter.list(None)) == [("x", "y"), ("x/y", "z")]

    def test_a_flat_path_is_unaffected(self, tmp_path: Path) -> None:
        adapter = _adapter(tmp_path)
        _write(adapter, "flat", "k")

        assert adapter._layout.all_entity_paths() == ["flat"]

    def test_an_empty_store_lists_nothing(self, tmp_path: Path) -> None:
        assert _adapter(tmp_path)._layout.all_entity_paths() == []


class TestDescendantSearchOnTheLocalAdapter:
    def test_a_scope_reaches_the_topics_beneath_it(self, tmp_path: Path) -> None:
        """The end the enumeration fix serves.

        The base ``search`` gathers descendants through ``list(None)``, so this
        returned nothing at all until the walk went deeper — silently, which is
        the worst way for a read to be wrong.
        """
        adapter = _adapter(tmp_path)
        _write(adapter, "amfs-internal", "root-note")
        _write(adapter, "amfs-internal/deploy", "runbook")
        _write(adapter, "amfs-internal/ci-cd", "backmerge")

        found = adapter.search(
            SearchQuery(entity_path="amfs-internal", include_descendants=True)
        )

        assert _pairs(found) == [
            ("amfs-internal", "root-note"),
            ("amfs-internal/ci-cd", "backmerge"),
            ("amfs-internal/deploy", "runbook"),
        ]

    def test_a_scope_does_not_reach_a_repo_that_merely_starts_the_same(
        self, tmp_path: Path
    ) -> None:
        """``amfs`` and ``amfs-internal`` are two repositories."""
        adapter = _adapter(tmp_path)
        _write(adapter, "amfs", "own-note")
        _write(adapter, "amfs-internal/deploy", "runbook")

        found = adapter.search(SearchQuery(entity_path="amfs", include_descendants=True))

        assert _pairs(found) == [("amfs", "own-note")]

    def test_the_default_still_matches_one_path_exactly(self, tmp_path: Path) -> None:
        adapter = _adapter(tmp_path)
        _write(adapter, "amfs-internal", "root-note")
        _write(adapter, "amfs-internal/deploy", "runbook")

        found = adapter.search(SearchQuery(entity_path="amfs-internal"))

        assert _pairs(found) == [("amfs-internal", "root-note")]
