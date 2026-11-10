"""An entry past its ``ttl_at`` is not listed, on any adapter.

``read`` has refused an expired entry since #429 and ``search`` since the
first half of this change, but ``list()`` went on returning it until the
lifecycle sweep archived it — up to a sweep interval later, and never on a
store nothing sweeps. Everything that reads memory by listing saw it in the
meantime: the entries page, the stats and digests built on a listing, the
prompt the OpenRouter proxy injects memory into.

These tests pin the rule the adapters now share:

- ``list()`` returns live entries — current and not expired;
- ``list(include_superseded=True)`` is the history and hides nothing;
- ``list_expired()`` is how the sweep still finds what ``list()`` leaves out;
- once archived, an entry is back with confidence 0.0, as the lifecycle
  documents ("archived, never deleted").

The filesystem and S3 adapters run here; the Postgres ones are covered by
``tests/integration`` (the contract suite and ``test_postgres_adapter.py``),
which need a database, and by the SQL-shape tests in ``test_sql_read_path.py``.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.engine import CausalTagger, CoWEngine
from amfs_core.lifecycle import LifecycleManager
from amfs_core.models import MemoryEntry, OutcomeRecord, Provenance, SearchQuery
from amfs_filesystem.adapter import FilesystemAdapter

ENTITY = "svc/checkout"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _past() -> datetime:
    return _now() - timedelta(hours=1)


def _future() -> datetime:
    return _now() + timedelta(hours=1)


def _entry(
    key: str,
    ttl_at: datetime | None = None,
    *,
    entity_path: str = ENTITY,
    value: Any = "v",
    agent_id: str = "agent-a",
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=value,
        provenance=Provenance(agent_id=agent_id, session_id="s", written_at=_now()),
        ttl_at=ttl_at,
    )


# ---------------------------------------------------------------------------
# The two adapters that can run without a service
# ---------------------------------------------------------------------------


class _FakeBucket:
    """Just enough of a boto3 S3 client for S3Adapter (as in test_s3_adapter_ttl)."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_paginator(self, name: str) -> Any:
        objects = self.objects

        class _Paginator:
            def paginate(self, *, Bucket: str, Prefix: str) -> Any:  # noqa: N803
                keys = sorted(k for k in objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]} if keys else {}

        return _Paginator()

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: Any) -> None:  # noqa: N803
        self.objects[Key] = Body

    def copy_object(self, *, Bucket: str, CopySource: dict[str, str], Key: str) -> None:  # noqa: N803
        self.objects[Key] = self.objects[CopySource["Key"]]

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        self.objects.pop(Key, None)


@pytest.fixture(params=["filesystem", "s3"])
def adapter(
    request: pytest.FixtureRequest, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> AdapterABC:
    if request.param == "filesystem":
        return FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    s3_adapter = pytest.importorskip("amfs_s3.adapter", reason="amfs-adapter-s3 not installed")
    bucket = _FakeBucket()
    monkeypatch.setattr(s3_adapter.boto3, "client", lambda *a, **kw: bucket)
    return s3_adapter.S3Adapter("amfs-test", namespace="test")


def _seed(adapter: AdapterABC) -> None:
    adapter.write(_entry("expired", _past()))
    adapter.write(_entry("live", _future()))
    adapter.write(_entry("forever", None))


def _keys(entries: list[MemoryEntry]) -> list[str]:
    return sorted(e.key for e in entries)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


def test_an_expired_entry_is_not_listed(adapter: AdapterABC) -> None:
    _seed(adapter)

    assert _keys(adapter.list(ENTITY)) == ["forever", "live"]
    # The namespace-wide listing too: it is the one stats and digests read.
    assert _keys(adapter.list()) == ["forever", "live"]


def test_an_entry_whose_ttl_has_not_passed_is_listed_with_its_ttl(adapter: AdapterABC) -> None:
    ttl = _future()
    adapter.write(_entry("live", ttl))

    (listed,) = adapter.list(ENTITY)
    assert listed.key == "live" and listed.ttl_at == ttl


def test_a_naive_ttl_is_read_as_utc(adapter: AdapterABC) -> None:
    naive_now = _now().replace(tzinfo=None)
    adapter.write(_entry("naive-expired", naive_now - timedelta(hours=1)))
    adapter.write(_entry("naive-live", naive_now + timedelta(hours=1)))

    assert _keys(adapter.list(ENTITY)) == ["naive-live"]
    assert _keys(adapter.list_expired()) == ["naive-expired"]


def test_the_history_still_shows_an_expired_version(adapter: AdapterABC) -> None:
    """Expiry hides an entry from retrieval. It does not rewrite the record:
    the history, a read at a version and the integrity check see everything."""
    adapter.write(_entry("rotated", None, value="first"))
    adapter.write(_entry("rotated", _past(), value="second"))

    assert adapter.list(ENTITY) == []
    history = adapter.list(ENTITY, include_superseded=True)
    assert sorted(e.version for e in history) == [1, 2]
    assert adapter.read_at_version(ENTITY, "rotated", 2).value == "second"
    assert adapter.verify_integrity(ENTITY)["total_checked"] == 2


def test_a_key_renewed_after_it_expired_is_listed_again(adapter: AdapterABC) -> None:
    adapter.write(_entry("token", _past(), value="old"))
    assert adapter.list(ENTITY) == []

    written = adapter.write(_entry("token", _future(), value="new"))

    assert written.version == 2
    (listed,) = adapter.list(ENTITY)
    assert listed.value == "new" and listed.version == 2


def test_what_is_built_on_the_listing_follows_it(adapter: AdapterABC) -> None:
    """Search, the agent listing and the stats are ``AdapterABC`` code over
    ``list()`` on these adapters. They must agree with it and with each other
    about which entries exist."""
    _seed(adapter)

    assert _keys(adapter.search(SearchQuery(entity_path=ENTITY))) == ["forever", "live"]
    assert _keys(adapter.list_entries_for_agent("agent-a")) == ["forever", "live"]
    assert adapter.stats().total_entries == 2
    scope = adapter.scope_counts(ENTITY, agent_id="agent-a")
    assert (scope["existing_entries"], scope["keys"]) == (2, ["forever", "live"])
    assert adapter.read(ENTITY, "expired") is None


# ---------------------------------------------------------------------------
# list_expired() and the sweep
# ---------------------------------------------------------------------------


def test_list_expired_finds_exactly_what_list_leaves_out(adapter: AdapterABC) -> None:
    now = _now()
    adapter.write(_entry("old", now - timedelta(hours=1)))
    adapter.write(_entry("older", now - timedelta(hours=2), entity_path="svc/billing"))
    adapter.write(_entry("live", now + timedelta(hours=1)))
    adapter.write(_entry("forever", None))
    # Superseded while expired: the current version is what counts.
    adapter.write(_entry("renewed", now - timedelta(hours=1)))
    adapter.write(_entry("renewed", now + timedelta(hours=1)))

    assert [e.key for e in adapter.list_expired(now=now)] == ["older", "old"]
    assert [e.key for e in adapter.list_expired(now=now, limit=1)] == ["older"]
    # ``now`` is the clock the caller sweeps by, not the wall clock.
    assert adapter.list_expired(now=now - timedelta(hours=3)) == []
    assert _keys(adapter.list()) == ["forever", "live", "renewed"]


def test_the_sweep_archives_what_list_leaves_out(adapter: AdapterABC) -> None:
    """``list()`` no longer shows the sweep anything to archive, so the sweep
    must find expired entries another way — and an entry that had been
    rewritten before it expired (version 2 and up) must be archived like any
    other. On ``main`` that one raised ``VersionConflictError`` out of the
    sweep, which also left every entry after it unarchived."""
    adapter.write(_entry("rewritten", None, value="first"))
    adapter.write(_entry("rewritten", _past(), value="second"))
    adapter.write(_entry("once", _past()))
    adapter.write(_entry("live", _future()))

    archived = LifecycleManager(adapter).sweep()

    assert sorted((e.key, e.version) for e in archived) == [("once", 2), ("rewritten", 3)]
    assert all(e.confidence == 0.0 and e.ttl_at is None for e in archived)
    assert adapter.list_expired() == []
    # Archived, not deleted: the entry is back, at confidence 0, with its value.
    assert _keys(adapter.list(ENTITY)) == ["live", "once", "rewritten"]
    assert adapter.read(ENTITY, "rewritten").confidence == 0.0
    assert adapter.read(ENTITY, "rewritten").value == "second"
    # And a second sweep finds nothing to do.
    assert LifecycleManager(adapter).sweep() == []


def test_the_sweep_does_not_archive_over_a_newer_write(adapter: AdapterABC) -> None:
    """Between the sweep's read and its write, the key can be written again —
    by an agent renewing it, or by a second sweeper. The archived copy names
    the version it expects to become, the adapter refuses it, and the sweep
    moves on to the next entry instead of failing."""
    adapter.write(_entry("renewed", _past(), value="stale"))
    adapter.write(_entry("other", _past()))
    stale_listing = adapter.list_expired()
    adapter.write(_entry("renewed", _future(), value="fresh"))

    class _RacedAdapter:
        """The same store, with the sweep's read frozen before the renewal."""

        def list_expired(self, *, now: datetime | None = None) -> list[MemoryEntry]:
            return stale_listing

        def write(self, entry: MemoryEntry) -> MemoryEntry:
            return adapter.write(entry)

    archived = LifecycleManager(_RacedAdapter()).sweep()  # type: ignore[arg-type]

    assert [e.key for e in archived] == ["other"]
    current = adapter.read(ENTITY, "renewed")
    assert current.value == "fresh" and current.confidence == 1.0 and current.version == 2


# ---------------------------------------------------------------------------
# The engine: a backstop for an adapter that still returns expired entries
# ---------------------------------------------------------------------------


class _UnfilteredAdapter(AdapterABC):
    """A third-party adapter written before any of this: no TTL handling."""

    def __init__(self) -> None:
        self._versions: dict[tuple[str, str], list[MemoryEntry]] = {}
        self.recalls: list[tuple[str, str]] = []

    def read(self, entity_path, key, *, min_confidence=0.0, branch="main"):
        versions = self._versions.get((entity_path, key))
        return versions[-1] if versions else None

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        versions = self._versions.setdefault((entry.entity_path, entry.key), [])
        entry = entry.model_copy(update={"version": len(versions) + 1})
        versions.append(entry)
        return entry

    def list(self, entity_path=None, *, include_superseded=False, branch="main"):
        out: list[MemoryEntry] = []
        for (ep, _key), versions in self._versions.items():
            if entity_path is not None and ep != entity_path:
                continue
            out.extend(versions if include_superseded else versions[-1:])
        return out

    def watch(self, entity_path, callback) -> WatchHandle:
        return WatchHandle(lambda: None)

    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        return []

    def increment_recall_count(self, entity_path, key, *, branch="main") -> None:
        self.recalls.append((entity_path, key))


def test_the_engine_withholds_an_expired_entry_the_adapter_returns() -> None:
    adapter = _UnfilteredAdapter()
    adapter.write(_entry("expired", _past()))
    adapter.write(_entry("live", _future()))
    engine = CoWEngine(adapter, CausalTagger("agent-b"))

    assert engine.read(ENTITY, "expired") is None
    # A read that returns nothing is not a reuse.
    assert adapter.recalls == []
    assert engine.read(ENTITY, "live").key == "live"
    assert adapter.recalls == [(ENTITY, "live")]

    assert _keys(engine.list(ENTITY)) == ["live"]
    assert _keys(engine.list()) == ["live"]
    # The history is untouched by the backstop, as by the adapters.
    assert _keys(engine.list(ENTITY, include_superseded=True)) == ["expired", "live"]
    assert [e.key for e in engine.history(ENTITY, "expired")] == ["expired"]


def test_the_sweep_still_lists_an_adapter_that_has_no_list_expired() -> None:
    """The fallback the sweep keeps: an adapter that lists expired entries and
    offers no ``list_expired`` is swept from its listing, as before."""
    adapter = _UnfilteredAdapter()
    adapter.write(_entry("expired", _past()))
    adapter.write(_entry("live", _future()))

    archived = LifecycleManager(adapter).sweep()

    assert [(e.key, e.version, e.confidence) for e in archived] == [("expired", 2, 0.0)]


def test_writing_over_an_expired_entry_continues_its_version_line(tmp_path) -> None:
    """The engine reads the current entry to pick the next version. An expired
    one reads as missing, so it asks for version 1 — and the adapter, which
    knows what is stored, still continues the line instead of clobbering it."""
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    engine = CoWEngine(adapter, CausalTagger("agent-a"))
    engine.write(ENTITY, "token", "old", ttl_at=_past())

    written = engine.write(ENTITY, "token", "new")

    assert written.version == 2
    assert engine.read(ENTITY, "token").value == "new"
    assert [e.version for e in engine.history(ENTITY, "token")] == [1, 2]


# ---------------------------------------------------------------------------
# The async Postgres read, driven against a stand-in pool
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = list(rows)
        self.executed: list[str] = []

    async def __aenter__(self) -> _Cursor:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append(sql)

    async def fetchone(self) -> Any:
        return self._rows.pop(0) if self._rows else None


class _Conn:
    def __init__(self, cur: _Cursor) -> None:
        self._cur = cur

    async def __aenter__(self) -> _Conn:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def cursor(self) -> _Cursor:
        return self._cur


class _Pool:
    def __init__(self, cur: _Cursor) -> None:
        self._cur = cur

    def connection(self) -> _Conn:
        return _Conn(self._cur)


@pytest.mark.parametrize(
    ("ttl_at", "served"),
    [(None, True), ("future", True), ("past", False)],
)
def test_the_async_postgres_read_withholds_an_expired_entry(
    monkeypatch: pytest.MonkeyPatch, ttl_at: str | None, served: bool
) -> None:
    """``AsyncPostgresAdapter.read`` is what ``GET /api/v1/entries/{path}/{key}``
    asks first. It had no TTL check, so the route served an expired entry that
    the sync adapter behind it would have withheld."""
    pg = pytest.importorskip(
        "amfs_postgres.async_adapter", reason="amfs-adapter-postgres not installed"
    )
    ttl = {"future": _future(), "past": _past()}.get(ttl_at) if ttl_at else None
    stored = _entry("token", ttl)
    monkeypatch.setattr(
        pg.AsyncPostgresAdapter, "_row_to_entry", staticmethod(lambda row: row["entry"])
    )
    a = object.__new__(pg.AsyncPostgresAdapter)
    a._pool = _Pool(_Cursor([{"entry": stored}]))
    a._namespace = "test"

    got = asyncio.run(a.read(ENTITY, "token"))

    assert (got is stored) if served else (got is None)
