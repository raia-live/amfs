"""The immutable trace store survives losing its connection.

The store holds one psycopg connection for the life of the process. On
2026-09-18 Cloud SQL closed it under load and every seal on that instance
failed with ``the connection is closed`` from then on, because nothing ever
looked at the connection again: ``_get_immutable_store`` returned the cached
store, ``save`` raised, the warning was logged, the outcome's immutable copy
was lost. For hours.

Two behaviours are pinned here. A store whose connection reports itself
closed is rebuilt before it is used. A store whose connection is broken but
does not know it yet — the first call after the server closed the socket —
fails once, is rebuilt, and the seal is retried on the new connection.

The Pro trace package is absent in OSS, so the seal path is stubbed at the
module boundary as in ``test_seal_once_per_outcome``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest
from amfs_core.models import DecisionTrace
from amfs_http import server as http_server


class _Conn:
    def __init__(self, *, closed: bool = False) -> None:
        self.closed = closed
        self.broken = False
        self.close_calls = 0

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


class _Store:
    """A store on one connection; ``fail_first`` makes the first save raise
    the error psycopg raises on a connection the server has already dropped."""

    def __init__(self, conn: _Conn, *, fail_first: bool = False) -> None:
        self._conn = conn
        self.saved: list[Any] = []
        self._fail_first = fail_first

    def get_latest_hash(self, session_id):
        return None

    def save(self, trace):
        if self._fail_first:
            self._fail_first = False
            self._conn.closed = True
            raise psycopg.OperationalError("the connection is closed")
        self.saved.append(trace)
        return SimpleNamespace(id="00000000-0000-0000-0000-000000000001")


@pytest.fixture
def seal_path(monkeypatch):
    monkeypatch.setattr(http_server, "_HAS_PRO_TRACES", True, raising=False)
    monkeypatch.setattr(
        http_server, "_pro_immutable_from_oss_trace",
        lambda oss, **kw: SimpleNamespace(outcome_ref=oss.outcome_ref, **kw),
        raising=False,
    )
    monkeypatch.setattr(http_server, "_pro_finalize_spans", lambda imm: imm, raising=False)
    monkeypatch.setattr(http_server, "seal", lambda imm, *a, **kw: imm, raising=False)
    monkeypatch.setattr(http_server, "get_signing_key", lambda: "key", raising=False)
    monkeypatch.setattr(http_server, "get_signing_key_id", lambda: "kid", raising=False)
    monkeypatch.setenv("AMFS_POSTGRES_DSN", "postgresql://stub")
    monkeypatch.setattr(http_server, "_immutable_trace_store", None)
    monkeypatch.setattr(http_server, "_seal_sequence", {})


def _handle() -> Any:
    trace = DecisionTrace(
        agent_id="sre-agent", session_id="s-1", outcome_ref="deploy-1",
        outcome_type="success",
        session_started_at=datetime(2026, 1, 1, tzinfo=UTC),
        session_ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
    )
    return SimpleNamespace(
        _last_trace=trace, session_id="server", agent_id="amfs-server",
    )


def _install_factory(monkeypatch, stores: list[_Store]) -> list[dict]:
    """``PostgresImmutableTraceStore(conn, auto_schema=...)`` hands out
    ``stores`` in order and records how it was called."""
    calls: list[dict] = []

    def _factory(conn, **kw):
        calls.append({"conn": conn, **kw})
        return stores.pop(0)

    monkeypatch.setattr(http_server, "PostgresImmutableTraceStore", _factory, raising=False)
    monkeypatch.setattr(
        psycopg, "connect", lambda dsn, **kw: _Conn(), raising=True
    )
    return calls


def test_closed_connection_is_rebuilt_before_use(seal_path, monkeypatch):
    dead = _Store(_Conn(closed=True))
    live = _Store(_Conn())
    calls = _install_factory(monkeypatch, [live])
    monkeypatch.setattr(http_server, "_immutable_trace_store", dead)

    trace_id = http_server._auto_seal_trace(_handle())

    assert trace_id is not None
    assert dead.saved == [] and len(live.saved) == 1
    # Rebuilt without re-running the DDL: the schema was applied at first open.
    assert calls == [{"conn": calls[0]["conn"], "auto_schema": False}]
    assert http_server._immutable_trace_store is live


def test_connection_that_dies_mid_seal_is_retried_once(seal_path, monkeypatch):
    first = _Store(_Conn(), fail_first=True)
    second = _Store(_Conn())
    _install_factory(monkeypatch, [first, second])

    trace_id = http_server._auto_seal_trace(_handle())

    assert trace_id is not None
    assert first.saved == [] and len(second.saved) == 1
    assert first._conn.close_calls == 1, "the dead connection is closed, not leaked"
    assert http_server._immutable_trace_store is second


def test_a_non_connection_error_fails_once_without_reconnect(seal_path, monkeypatch):
    class _Boom(_Store):
        def save(self, trace):
            raise ValueError("bad trace")

    store = _Boom(_Conn())
    calls = _install_factory(monkeypatch, [store])

    assert http_server._auto_seal_trace(_handle()) is None
    assert len(calls) == 1, "no reconnect for an error that is not the connection's"
    assert http_server._immutable_trace_store is store
