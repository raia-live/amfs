"""Backfilling embeddings onto rows that were written without one.

Two properties, neither of which held before, and both of which fail quietly
rather than loudly:

* It runs to completion. Reading a single batch and returning looks identical
  to success from the caller's side — a number comes back — while most of the
  store still has no vector and so is invisible to semantic search.
* A row the embedder cannot handle does not block the rows behind it. Ordering
  by written_at and skipping failures with `continue` left those rows NULL, so
  they were re-read at the head of every subsequent batch, forever. One bad
  value is then enough to stall the backfill permanently.

Both come down to advancing a keyset cursor by id regardless of whether the
row succeeded, which is what backfill_is_artifact already did.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from amfs_postgres.adapter import PostgresAdapter

_ZERO_UUID = "00000000-0000-0000-0000-000000000000"


class _FakeCursor:
    """Understands just the two statements backfill_embeddings issues."""

    def __init__(self, db: _FakeDB) -> None:
        self._db = db

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def execute(self, sql: str, params: tuple = ()) -> None:
        if sql.startswith("SELECT id, value"):
            _namespace, last_id, limit = params
            self._db.selects.append((last_id, limit))
            pending = sorted(
                row_id
                for row_id, vector in self._db.embeddings.items()
                if vector is None and row_id > last_id
            )
            self._rows = [
                {"id": row_id, "value": json.dumps({"row": row_id})}
                for row_id in pending[:limit]
            ]
        elif sql.startswith("UPDATE amfs_memory_entries SET embedding"):
            vector, row_id = params
            self._db.embeddings[row_id] = vector
        else:  # pragma: no cover - would mean the query changed shape
            raise AssertionError(f"unexpected statement: {sql}")

    def fetchall(self) -> list[dict]:
        return self._rows


class _FakeDB:
    def __init__(self, row_ids: list[str]) -> None:
        self.embeddings: dict[str, str | None] = {r: None for r in row_ids}
        self.selects: list[tuple[str, int]] = []

    def connection(self) -> Any:
        cursor = _FakeCursor(self)
        pool_ctx = MagicMock()
        pool_ctx.__enter__ = MagicMock(return_value=_conn_with(cursor))
        pool_ctx.__exit__ = MagicMock(return_value=None)
        return pool_ctx

    @property
    def embedded(self) -> set[str]:
        return {r for r, v in self.embeddings.items() if v is not None}

    @property
    def missing(self) -> set[str]:
        return {r for r, v in self.embeddings.items() if v is None}


def _conn_with(cursor: _FakeCursor) -> MagicMock:
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    return conn


def _adapter(db: _FakeDB, *, failing: set[str] | None = None) -> PostgresAdapter:
    """A PostgresAdapter with a fake pool, built without touching a database."""
    failing = failing or set()

    def embed_value(value: Any) -> list[float]:
        row_id = value["row"] if isinstance(value, dict) else value
        if row_id in failing:
            raise ValueError(f"cannot embed {row_id}")
        return [0.1, 0.2, 0.3]

    adapter = object.__new__(PostgresAdapter)
    adapter._pool = db
    adapter._namespace = "default"
    adapter._has_embedding_col = True
    embedder = MagicMock()
    embedder.embed_value = MagicMock(side_effect=embed_value)
    adapter._embedder = embedder
    return adapter


def _ids(count: int) -> list[str]:
    return [f"row-{n:03d}" for n in range(count)]


class TestItRunsToCompletion:
    def test_more_rows_than_one_batch_are_all_embedded(self):
        db = _FakeDB(_ids(25))

        updated = _adapter(db).backfill_embeddings(batch_size=10)

        assert updated == 25
        assert db.missing == set()

    def test_the_cursor_advances_instead_of_re_reading_from_the_start(self):
        db = _FakeDB(_ids(25))

        _adapter(db).backfill_embeddings(batch_size=10)

        # First read starts at the zero UUID, each later one resumes after the
        # last id seen. Re-reading from the start is how the old version looped
        # over the same rows.
        starts = [start for start, _limit in db.selects]
        assert starts[0] == _ZERO_UUID
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)

    def test_an_empty_store_reads_once_and_stops(self):
        db = _FakeDB([])

        assert _adapter(db).backfill_embeddings(batch_size=10) == 0
        assert len(db.selects) == 1


class TestAFailingRowCannotStarveTheOthers:
    def test_rows_after_an_unembeddable_one_are_still_reached(self):
        """The old version left the failure NULL and ordered by written_at, so
        this row came back at the head of every batch and nothing behind it was
        ever reached."""
        db = _FakeDB(_ids(25))

        updated = _adapter(db, failing={"row-000"}).backfill_embeddings(batch_size=10)

        assert updated == 24
        assert db.missing == {"row-000"}

    def test_a_whole_first_batch_of_failures_does_not_stall_it(self):
        db = _FakeDB(_ids(25))
        doomed = set(_ids(25)[:10])

        updated = _adapter(db, failing=doomed).backfill_embeddings(batch_size=10)

        assert updated == 15
        assert db.missing == doomed

    def test_it_terminates_when_nothing_can_be_embedded(self):
        # Every row fails, so no row's state ever changes. Only the cursor
        # advancing ends this; without it the loop never finishes.
        db = _FakeDB(_ids(12))
        everything = set(_ids(12))

        assert _adapter(db, failing=everything).backfill_embeddings(batch_size=5) == 0
        assert db.missing == everything


class TestBoundingTheWork:
    def test_max_rows_stops_early(self):
        db = _FakeDB(_ids(25))

        updated = _adapter(db).backfill_embeddings(batch_size=10, max_rows=12)

        assert updated == 12
        assert len(db.embedded) == 12

    def test_max_rows_never_reads_more_than_it_needs(self):
        db = _FakeDB(_ids(25))

        _adapter(db).backfill_embeddings(batch_size=10, max_rows=12)

        # The final batch asks for 2, not 10.
        assert [limit for _start, limit in db.selects] == [10, 2]

    def test_max_rows_above_the_row_count_is_harmless(self):
        db = _FakeDB(_ids(5))

        assert _adapter(db).backfill_embeddings(batch_size=10, max_rows=99) == 5


class TestWhenItCannotRunAtAll:
    @pytest.mark.parametrize(
        ("attribute", "value"),
        [("_embedder", None), ("_has_embedding_col", False)],
    )
    def test_it_does_nothing_without_an_embedder_or_a_column(self, attribute, value):
        db = _FakeDB(_ids(5))
        adapter = _adapter(db)
        setattr(adapter, attribute, value)

        assert adapter.backfill_embeddings() == 0
        assert db.selects == [], "must not query at all"
