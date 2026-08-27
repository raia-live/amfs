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
                {"id": row_id, "value": self._db.stored_value(row_id)}
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
    def __init__(
        self, row_ids: list[str], *, values: dict[str, Any] | None = None
    ) -> None:
        self.embeddings: dict[str, str | None] = {r: None for r in row_ids}
        self.selects: list[tuple[str, int]] = []
        self.scopes: list[bool] = []
        #: Overrides for what a row's ``value`` column holds. The default is a
        #: JSON object, which is the common case; a bare string is the case that
        #: used to be misread as an unembeddable row.
        self.values: dict[str, Any] = values or {}

    def stored_value(self, row_id: str) -> Any:
        if row_id in self.values:
            return self.values[row_id]
        return json.dumps({"row": row_id})

    def connection(self, *, transactional: bool = True) -> Any:
        # The backfill checks out with transactional=False, because it calls the
        # embedder between statements and would otherwise hold a transaction
        # open across a batch of network round trips. Recorded rather than
        # ignored so that the scope stays part of what these tests describe.
        self.scopes.append(transactional)
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

    def test_the_bound_counts_rows_examined_not_rows_embedded(self):
        """A bound a bad row can lift is not a bound.

        Counting successes meant failures were free: the loop kept fetching
        batches until it found `max_rows` that worked, so a caller asking for 10
        rows of work against 10 unembeddable rows walked the entire store
        looking for successes that were never coming. The caller that asks for a
        bound is the one on a time slice, which is exactly the one that cannot
        afford that.
        """
        db = _FakeDB(_ids(25))
        doomed = set(_ids(25)[:10])

        updated = _adapter(db, failing=doomed).backfill_embeddings(
            batch_size=10, max_rows=10
        )

        assert updated == 0, "none of the first ten can be embedded"
        # One batch of ten, and then it stops. Not 25 rows read in pursuit of
        # ten successes.
        assert db.selects == [(_ZERO_UUID, 10)]
        assert db.embedded == set()

    def test_a_partly_failing_slice_still_stops_at_the_bound(self):
        db = _FakeDB(_ids(40))
        doomed = {"row-000", "row-003", "row-007"}

        updated = _adapter(db, failing=doomed).backfill_embeddings(
            batch_size=5, max_rows=10
        )

        assert updated == 7, "ten examined, three of them unembeddable"
        assert sum(limit for _start, limit in db.selects) == 10


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


class TestValuesThatAreNotJson:
    """A value can be a bare string, and a bare string is not JSON.

    The decode and the embed used to share one ``try``, so ``json.loads`` on a
    plain string raised JSONDecodeError and the row was counted as one the
    embedder could not handle. It kept its NULL embedding, stayed invisible to
    semantic search, and was re-read and re-failed by every later backfill —
    reported only as a warning, so the store looked backfilled.
    """

    def test_a_plain_string_value_is_embedded_not_skipped(self):
        db = _FakeDB(_ids(3), values={"row-001": "analytics runs on clickhouse"})

        updated = _adapter(db).backfill_embeddings()

        assert updated == 3, "the bare-string row must be embedded like the rest"
        assert db.missing == set()

    def test_it_is_the_raw_string_that_reaches_the_embedder(self):
        """Not a JSON-quoted version of it, and not a repr."""
        db = _FakeDB(["row-000"], values={"row-000": "analytics on clickhouse"})
        adapter = _adapter(db)

        adapter.backfill_embeddings()

        adapter._embedder.embed_value.assert_called_once_with(
            "analytics on clickhouse"
        )

    def test_a_json_value_is_still_decoded(self):
        """The fallback must not stop real JSON being parsed into an object."""
        db = _FakeDB(["row-000"], values={"row-000": json.dumps({"row": "row-000"})})
        adapter = _adapter(db)

        adapter.backfill_embeddings()

        adapter._embedder.embed_value.assert_called_once_with({"row": "row-000"})

    def test_a_genuine_embedder_failure_is_still_skipped(self):
        """The fallback widens what counts as decodable, not what counts as
        embeddable — a row the embedder rejects must still be left alone."""
        db = _FakeDB(_ids(3), values={"row-001": "unembeddable"})

        updated = _adapter(db, failing={"unembeddable"}).backfill_embeddings()

        assert updated == 2
        assert db.missing == {"row-001"}


class TestItDoesNotHoldATransactionAcrossTheEmbedder:
    """Why this checks out non-transactionally, and why that is not an oversight.

    Ordinary checkouts wrap the block in a transaction so the tenant settings
    can be scoped to it, which is what makes the adapter safe behind a
    transaction-mode pooler. This loop cannot take that: it calls the embedder
    once per row inside the block, so the transaction would stay open across a
    whole batch of network round trips — holding a transaction id, blocking
    vacuum, and stretching to however long the embedding service takes.

    Per-statement commit is also load-bearing for the behaviour the tests above
    describe: one row's failure leaves the rest of the batch committed.
    """

    def test_every_batch_checks_out_without_a_transaction(self):
        db = _FakeDB(_ids(25))

        _adapter(db).backfill_embeddings(batch_size=10)

        assert db.scopes, "expected at least one checkout"
        assert all(scope is False for scope in db.scopes), (
            f"a transactional checkout would hold a transaction open across "
            f"the embedder calls: {db.scopes}"
        )

    def test_the_batches_are_separate_checkouts(self):
        """Each batch its own checkout, so the connection is handed back — and
        the tenant settings blanked — between them rather than held for the
        whole run."""
        db = _FakeDB(_ids(25))

        _adapter(db).backfill_embeddings(batch_size=10)

        assert len(db.scopes) == len(db.selects) > 1
