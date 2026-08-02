"""Captured prompt/response text: scanning, and surviving a round trip.

``DecisionTrace.task_input`` and ``response_text`` hold the request an agent was
given and the answer it produced. Three things went wrong when the feature was
added, all of them silent, and each test here pins one of them:

1. The scan lived in the HTTP server only, so an agent committing an outcome
   through the SDK or the MCP tool stored raw text — while the MCP tool's own
   documentation said secrets were redacted.
2. The scan returned the *original* text when the safety gate raised, which is
   fail-open: the one input nobody could vet was the one that got stored.
3. The Postgres adapter wrote both columns but did not select them back, so every
   read returned ``None``. Nothing raised, and behaviour cloning simply found no
   eligible examples.

None of the three would show up in a test that only checks a trace was saved,
which is why they survived to review.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory
from amfs.memory import _get_sdk_executor
from amfs_core import capture
from amfs_core.models import OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture(autouse=True)
def _clear_gate_cache():
    """The gate is cached process-wide; a stub must not leak between tests."""
    capture._GATE_CACHE.clear()
    yield
    capture._GATE_CACHE.clear()


class _Decision:
    def __init__(self, allowed: bool, value: str | None = None) -> None:
        self.allowed = allowed
        self.entry = type("E", (), {"value": value})()


def _install_gate(monkeypatch, gate) -> None:
    """Seed the cache so no import of the optional Pro package is attempted."""
    capture._GATE_CACHE["gate"] = gate


# --- The scanner itself -----------------------------------------------------


def test_absent_text_is_left_absent() -> None:
    assert capture.scan_captured_text(None) is None
    assert capture.scan_captured_text("") is None


def test_a_redacted_value_replaces_the_original(monkeypatch) -> None:
    class Gate:
        def check_write(self, entry):
            return _Decision(True, "token=[REDACTED]")

    _install_gate(monkeypatch, Gate())
    assert capture.scan_captured_text("token=sk-live-123") == "token=[REDACTED]"


def test_blocked_text_is_dropped_not_stored(monkeypatch) -> None:
    class Gate:
        def check_write(self, entry):
            return _Decision(False)

    _install_gate(monkeypatch, Gate())
    assert capture.scan_captured_text("AWS_SECRET_ACCESS_KEY=abc") is None


def test_a_gate_that_raises_drops_the_text(monkeypatch) -> None:
    """Fail closed. Returning the text here is what review flagged.

    The gate exists and simply failed, so the text is precisely what could not be
    vetted. Storing it is the one outcome this function is meant to prevent.
    """

    class Gate:
        def check_write(self, entry):
            raise RuntimeError("gate exploded")

    _install_gate(monkeypatch, Gate())
    assert capture.scan_captured_text("password=hunter2") is None


def test_an_oss_install_without_the_safety_package_stores_the_text() -> None:
    """No gate available is a different case from a gate that failed.

    Capture is opt-in and the caller asked for it, so on an OSS install the text
    is stored rather than silently discarded. This is why the ImportError branch
    cannot simply be folded into the general failure branch.
    """
    assert capture._GATE_CACHE == {}
    # SafetyGate is a Pro package; in this environment the import fails and the
    # text passes through unchanged.
    pytest.importorskip
    result = capture.scan_captured_text("plain prompt text")
    try:
        import amfs_safety  # noqa: F401
    except ImportError:
        assert result == "plain prompt text"
    else:
        pytest.skip("amfs_safety installed; the ImportError branch is unreachable")


def test_oversized_text_is_truncated_before_scanning(monkeypatch) -> None:
    seen: list[int] = []

    class Gate:
        def check_write(self, entry):
            seen.append(len(entry.value))
            return _Decision(True, entry.value)

    _install_gate(monkeypatch, Gate())
    out = capture.scan_captured_text("x" * (capture.MAX_CAPTURED_CHARS + 5_000))
    assert seen == [capture.MAX_CAPTURED_CHARS]
    assert out is not None and len(out) == capture.MAX_CAPTURED_CHARS


# --- The SDK applies it -----------------------------------------------------


def _flush_bg() -> None:
    _get_sdk_executor().submit(lambda: None).result(timeout=5)


def test_commit_outcome_scans_captured_text(monkeypatch, tmp_path) -> None:
    """The fix for the MCP path.

    ``amfs_commit_outcome`` calls straight through to this method, so scanning at
    trace construction is what makes the tool's documented promise true for a
    client on a direct adapter.
    """

    class Gate:
        def check_write(self, entry):
            return _Decision(True, entry.value.replace("sk-live-123", "[REDACTED]"))

    _install_gate(monkeypatch, Gate())

    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.commit_outcome(
        "task-1",
        OutcomeType.SUCCESS,
        task_input="deploy with key sk-live-123",
        response_text="used sk-live-123",
    )
    _flush_bg()

    trace = mem._last_trace
    assert trace is not None
    assert "sk-live-123" not in (trace.task_input or "")
    assert "sk-live-123" not in (trace.response_text or "")
    assert "[REDACTED]" in (trace.task_input or "")


def test_commit_outcome_drops_text_the_gate_blocks(monkeypatch, tmp_path) -> None:
    class Gate:
        def check_write(self, entry):
            return _Decision(False)

    _install_gate(monkeypatch, Gate())

    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.commit_outcome(
        "task-2", OutcomeType.SUCCESS, task_input="secret", response_text="secret"
    )
    _flush_bg()

    trace = mem._last_trace
    assert trace is not None
    assert trace.task_input is None
    assert trace.response_text is None


def test_a_trace_without_capture_is_unaffected(tmp_path) -> None:
    """Committing an outcome with no captured text must not start failing."""
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.commit_outcome("task-3", OutcomeType.SUCCESS)
    _flush_bg()

    trace = mem._last_trace
    assert trace is not None
    assert trace.task_input is None
    assert trace.response_text is None


# --- The columns are read back ---------------------------------------------


def test_the_postgres_read_queries_select_the_capture_columns() -> None:
    """A source check, because the bug was an omission from a SELECT list.

    A round trip against a live database is the real proof and lives in
    tests/integration/test_postgres_adapter.py. This runs everywhere and names the
    failure directly: ``_row_to_trace`` reads these with ``row.get``, so a missing
    column yields ``None`` instead of raising, and the loss is silent.
    """
    import inspect

    from amfs_postgres import adapter as pg

    for fn in (pg.PostgresAdapter.get_trace, pg.PostgresAdapter.list_traces):
        source = inspect.getsource(fn)
        assert "task_input" in source, f"{fn.__name__} does not select task_input"
        assert "response_text" in source, f"{fn.__name__} does not select response_text"
