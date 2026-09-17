"""Entry-returning SELECTs name their columns; they never fetch `embedding` or
`search_tsv`.

Neither column is used to build a MemoryEntry, and together they are most of
a row's bytes: on 5,000 rows with 384-dim vectors, `SELECT *` returned 20.5 MB
where the projection returns 3.4 MB. With `SELECT *`, an unscoped list() over
a 20k-row namespace moved ~80 MB per call, and the dashboard page that fans out
to several such calls took 20-35 s while the database sat at 1% CPU
(2026-09-17). The failure is silent and grows with the table, so these are
shape tests: the projection and the row decoder must agree, and the two hot
queries must use the projection.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from amfs_postgres.adapter import (
    ENTRY_SELECT,
    ENTRY_SELECT_NO_ARTIFACT_COL,
    _ENTRY_COLUMNS,
    PostgresAdapter,
    entry_select,
)

SRC = Path(__file__).resolve().parents[2] / "packages/adapters/postgres/src/amfs_postgres"
SYNC = SRC / "adapter.py"
ASYNC = SRC / "async_adapter.py"

HEAVY = {"embedding", "search_tsv"}


def _method_source(path: Path, name: str) -> str:
    src = path.read_text()
    marker = f"    def {name}(" if f"    def {name}(" in src else f"    async def {name}("
    start = src.index(marker)
    rest = src[start + 10 :]
    end = re.search(r"\n    (?:async )?def ", rest)
    return src[start : start + 10 + (end.start() if end else len(rest))]


def _row_columns_read_by_row_to_entry() -> set[str]:
    """Every `row["x"]` / `row.get("x", ...)` key inside _row_to_entry."""
    tree = ast.parse(SYNC.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_row_to_entry"
    )
    keys: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == "row"
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "row"
            and node.args and isinstance(node.args[0], ast.Constant)
        ):
            keys.add(node.args[0].value)
    return keys


class TestProjectionMatchesDecoder:
    def test_decoder_reads_only_projected_columns(self) -> None:
        """A column the decoder reads but the projection omits would be a
        KeyError on every list() — or, for .get(), a silent default."""
        read = _row_columns_read_by_row_to_entry()
        assert read, "could not find any row[...] reads in _row_to_entry"
        missing = read - set(_ENTRY_COLUMNS)
        assert not missing, f"_row_to_entry reads {sorted(missing)} but the projection omits them"

    def test_projection_excludes_the_heavy_columns(self) -> None:
        assert not HEAVY & set(_ENTRY_COLUMNS)
        for heavy in HEAVY:
            assert heavy not in ENTRY_SELECT
            assert heavy not in ENTRY_SELECT_NO_ARTIFACT_COL

    def test_no_duplicate_columns(self) -> None:
        assert len(_ENTRY_COLUMNS) == len(set(_ENTRY_COLUMNS))

    def test_is_artifact_toggles_with_the_migration_flag(self) -> None:
        assert "is_artifact" in entry_select(True).split(", ")
        assert "is_artifact" not in entry_select(False).split(", ")
        assert entry_select(True) == ENTRY_SELECT
        assert entry_select(False) == ENTRY_SELECT_NO_ARTIFACT_COL

    def test_decoder_survives_a_projected_row(self) -> None:
        """A row shaped exactly like the projection decodes; nothing in the
        decoder secretly needs a column outside it."""
        from datetime import datetime, timezone

        row = {c: None for c in _ENTRY_COLUMNS}
        row.update(
            entity_path="repo/mod", key="k", version=1, value={"a": 1},
            agent_id="a", session_id="s", written_at=datetime.now(timezone.utc),
            confidence=0.9, outcome_count=0, recall_count=0, tier=3,
            memory_type="fact", shared=True, artifact_refs=[], branch="main",
            is_artifact=False, pattern_refs=[],
        )
        entry = PostgresAdapter._row_to_entry(row)
        assert entry.entity_path == "repo/mod" and entry.embedding is None


class TestHotQueriesUseTheProjection:
    """list() and search() are the two entry-returning queries that run
    unscoped and at scale. They must not regress to SELECT *."""

    def _assert_uses_projection(self, path: Path, method: str) -> None:
        src = _method_source(path, method)
        assert "entry_select(" in src, f"{path.name}:{method} does not use entry_select()"
        assert not re.search(r"SELECT\s+\*\s+FROM\s+amfs_memory_entries", src), (
            f"{path.name}:{method} still selects * from amfs_memory_entries"
        )

    def test_sync_list(self) -> None:
        self._assert_uses_projection(SYNC, "list")

    def test_sync_search(self) -> None:
        self._assert_uses_projection(SYNC, "search")

    def test_async_list(self) -> None:
        self._assert_uses_projection(ASYNC, "list")

    def test_async_search(self) -> None:
        self._assert_uses_projection(ASYNC, "search")
