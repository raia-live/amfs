"""Regression test for the SaaS double-write timeline event.

A single write must produce exactly one WRITE timeline event. On the SaaS path
the write goes SDK -> HttpAdapter -> server, and the server's write handler logs
the WRITE event; the SDK must therefore NOT also log it. This is signalled by the
adapter's ``server_side_write_events`` flag.
"""

from __future__ import annotations

from amfs import AgentMemory
from amfs.memory import _get_sdk_executor
from amfs_core.models import Event, EventType
from amfs_filesystem.adapter import FilesystemAdapter


class _RecordingLocalAdapter(FilesystemAdapter):
    """Direct adapter (SDK is the only event logger)."""

    server_side_write_events = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[Event] = []

    def log_event(self, event: Event) -> Event:
        self.events.append(event)
        return event


class _RecordingRemoteAdapter(_RecordingLocalAdapter):
    """Simulates HttpAdapter: the remote server owns WRITE event logging."""

    server_side_write_events = True


def _flush_bg() -> None:
    # Executor is max_workers=1 FIFO, so a sentinel submitted after the write's
    # background task guarantees that task has completed once the sentinel does.
    _get_sdk_executor().submit(lambda: None).result(timeout=5)


def _write_events(adapter: _RecordingLocalAdapter) -> list[Event]:
    return [e for e in adapter.events if e.event_type == EventType.WRITE]


def test_direct_adapter_logs_one_write_event(tmp_path) -> None:
    adapter = _RecordingLocalAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.write("svc/mod", "k", "v")
    _flush_bg()

    assert len(_write_events(adapter)) == 1


def test_server_side_adapter_does_not_double_log(tmp_path) -> None:
    adapter = _RecordingRemoteAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.write("svc/mod", "k", "v")
    _flush_bg()

    # Server owns the WRITE event; the SDK must not emit its own.
    assert _write_events(adapter) == []
