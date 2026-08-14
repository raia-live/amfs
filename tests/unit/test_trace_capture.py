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

import sys
import types

import pytest
from amfs import AgentMemory
from amfs.memory import _get_sdk_executor
from amfs_core import capture
from amfs_core.models import OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture(autouse=True)
def _clear_gate_cache():
    """Gates are cached across calls; a stub must not leak between tests."""
    capture._GATE_CACHE.clear()
    capture._ADAPTERLESS_GATE.clear()
    yield
    capture._GATE_CACHE.clear()
    capture._ADAPTERLESS_GATE.clear()


class _Decision:
    def __init__(self, allowed: bool, value: str | None = None) -> None:
        self.allowed = allowed
        self.entry = type("E", (), {"value": value})()


def _install_gate(monkeypatch, gate) -> None:
    """Stand in for the gate so no import of the optional Pro package is attempted.

    Replaces the lookup rather than seeding a cache entry, because the real cache
    is keyed by adapter and these tests reach it down several paths — some with no
    adapter, some through an ``AgentMemory`` holding a real one.
    """
    monkeypatch.setattr(capture, "_gate_for", lambda adapter: gate)


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
    assert len(capture._GATE_CACHE) == 0
    assert capture._ADAPTERLESS_GATE == []
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


def test_each_adapter_gets_a_gate_bound_to_itself(monkeypatch) -> None:
    """One gate per adapter, not one per process.

    ``SafetyGate`` keeps the adapter it was built with and its validator reads
    through it, so a single shared instance would vet every caller's capture
    through whichever adapter arrived first. In the hosted server that is one
    tenant's connection answering every other tenant's request.
    """
    built = []

    class Gate:
        def __init__(self, adapter=None):
            self.adapter = adapter
            built.append(adapter)

    module = types.ModuleType("amfs_safety")
    module.SafetyGate = Gate
    monkeypatch.setitem(sys.modules, "amfs_safety", module)

    class Adapter:
        pass

    first, second = Adapter(), Adapter()
    assert capture._gate_for(first).adapter is first
    assert capture._gate_for(second).adapter is second
    # Still cached: the same adapter reuses its gate instead of rebuilding the
    # detectors on every captured field.
    assert capture._gate_for(first) is capture._gate_for(first)
    assert built == [first, second]


def test_an_unhashable_adapter_still_gets_a_correctly_bound_gate(monkeypatch) -> None:
    """Losing the cache is acceptable; binding the wrong adapter is not."""

    class Gate:
        def __init__(self, adapter=None):
            self.adapter = adapter

    module = types.ModuleType("amfs_safety")
    module.SafetyGate = Gate
    monkeypatch.setitem(sys.modules, "amfs_safety", module)

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

    adapter = Unhashable()
    assert capture._gate_for(adapter).adapter is adapter


def test_a_non_string_from_the_gate_is_dropped_not_replaced_with_the_original(
    monkeypatch,
) -> None:
    """Allowed-but-not-a-string is still not something we can store.

    Returning the caller's original text here would return the one value known
    *not* to be what the gate approved: if it redacted into a non-string form, the
    unredacted secret would go straight through.
    """

    class Gate:
        def check_write(self, entry):
            return _Decision(True, {"redacted": True})

    _install_gate(monkeypatch, Gate())
    assert capture.scan_captured_text("token=sk-live-123") is None


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


def test_the_capture_travels_with_the_outcome_record(monkeypatch, tmp_path) -> None:
    """The SaaS ordering bug.

    On the HTTP adapter the outcome record reaches the server first, and the server
    seals its immutable trace from that call — the copy Pro export and training
    actually read. A capture that travelled only on the later ``save_trace`` was
    absent from the sealed trace, so the flywheel saw empty prompts.
    """

    class Gate:
        def check_write(self, entry):
            return _Decision(True, entry.value)

    _install_gate(monkeypatch, Gate())

    seen: list[object] = []

    class _RecordingAdapter(FilesystemAdapter):
        def commit_outcome(self, record):
            seen.append(record)
            return super().commit_outcome(record)

    adapter = _RecordingAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.commit_outcome(
        "task-4",
        OutcomeType.SUCCESS,
        task_input="restart the worker",
        response_text="scaled to zero and back",
    )
    _flush_bg()

    assert seen, "the adapter's commit_outcome should have been called"
    assert seen[0].task_input == "restart the worker"
    assert seen[0].response_text == "scaled to zero and back"


def test_the_record_carries_scanned_text_not_raw(monkeypatch, tmp_path) -> None:
    """Nothing raw may leave the process.

    The record goes over the wire before the trace does, so scanning only at trace
    construction would put the unredacted secret on the first request.
    """

    class Gate:
        def check_write(self, entry):
            return _Decision(True, entry.value.replace("sk-live-123", "[REDACTED]"))

    _install_gate(monkeypatch, Gate())

    seen: list[object] = []

    class _RecordingAdapter(FilesystemAdapter):
        def commit_outcome(self, record):
            seen.append(record)
            return super().commit_outcome(record)

    adapter = _RecordingAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.commit_outcome("task-5", OutcomeType.SUCCESS, task_input="key sk-live-123")
    _flush_bg()

    assert seen
    assert "sk-live-123" not in (seen[0].task_input or "")


def test_the_http_adapter_forwards_the_capture_to_the_server() -> None:
    """The adapter builds the /outcomes body by hand, so the fields are droppable.

    This asserts on the request body rather than the record, because the record
    carrying the capture is useless if the transport leaves it out — which is
    exactly what it did.
    """
    from datetime import UTC, datetime

    from amfs_adapter_http.adapter import HttpAdapter
    from amfs_core.models import OutcomeRecord

    sent: dict[str, object] = {}

    adapter = HttpAdapter.__new__(HttpAdapter)

    def _post(path, body):
        sent["path"] = path
        sent["body"] = body
        return {"entries": []}

    adapter._post = _post  # type: ignore[method-assign]

    adapter.commit_outcome(
        OutcomeRecord(
            outcome_ref="task-6",
            outcome_type=OutcomeType.SUCCESS,
            committed_at=datetime.now(UTC),
            agent_id="a",
            task_input="the ask",
            response_text="the answer",
        )
    )

    assert sent["path"] == "/api/v1/outcomes"
    body = sent["body"]
    assert isinstance(body, dict)
    assert body["task_input"] == "the ask"
    assert body["response_text"] == "the answer"


def test_the_posted_traces_capture_is_scanned_as_its_own_agent() -> None:
    """``POST /api/v1/traces`` knows whose text this is; it should say so.

    The scan took its identity from the server's shared memory handle, which on
    this entry point is the process default and not the client that posted the
    trace. No redaction depends on it today — the gate matches on content and on
    a fixed capture namespace — so this is provenance rather than a live bug. It
    is asserted because the failure mode is silent: the first per-agent capture
    rule would apply to the wrong agent, and nothing would say so.

    Driven through the endpoint rather than the helper. Calling the helper with
    the identity spelled out only proves it forwards what it is given, which was
    never in doubt — the endpoint choosing to give it is the whole fix.
    """
    import asyncio
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from amfs_core.models import DecisionTrace
    from amfs_http import server as http_server

    seen: list[dict[str, str]] = []

    def _record(text, *, adapter=None, agent_id="", session_id=""):
        seen.append({"agent_id": agent_id, "session_id": session_id})
        return text

    trace = DecisionTrace(
        session_id="posted-session",
        agent_id="posting-agent",
        started_at=datetime.now(UTC),
        task_input="the ask",
    )
    mem = SimpleNamespace(
        _adapter=SimpleNamespace(save_trace=lambda t: t),
        agent_id="server-default",
        session_id="server-session",
        namespace="ns",
    )

    original_scan = http_server.scan_captured_text
    original_mem = http_server._get_memory
    original_link = http_server._link_agent_owner_once
    http_server.scan_captured_text = _record  # type: ignore[assignment]
    http_server._get_memory = lambda: mem  # type: ignore[assignment]
    http_server._link_agent_owner_once = lambda *a, **k: None  # type: ignore[assignment]
    try:
        request = SimpleNamespace(
            json=lambda: asyncio.sleep(0, result=trace.model_dump(mode="json"))
        )
        asyncio.run(http_server.save_trace(request))
    finally:
        http_server.scan_captured_text = original_scan  # type: ignore[assignment]
        http_server._get_memory = original_mem  # type: ignore[assignment]
        http_server._link_agent_owner_once = original_link  # type: ignore[assignment]

    assert seen, "the endpoint must scan captured text"
    assert all(s["agent_id"] == "posting-agent" for s in seen), seen
    assert all(s["session_id"] == "posted-session" for s in seen), seen


def test_a_posted_actions_result_is_scanned_by_the_endpoint() -> None:
    """The result reaches this endpoint as raw text in an HTTP body.

    Arguments were scanned here from the start and the result was not, so a tool
    that echoed a credential back had it stored verbatim. Driven through the
    endpoint because the omission was in what the endpoint chose to scan.
    """
    import asyncio
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from amfs_core.models import DecisionTrace, ToolCall
    from amfs_http import server as http_server

    trace = DecisionTrace(
        session_id="posted-session",
        agent_id="posting-agent",
        started_at=datetime.now(UTC),
        tool_calls=[
            ToolCall(
                tool_name="create_api_key",
                arguments={"scope": "read"},
                result_summary="issued sk-live-99",
            )
        ],
    )
    mem = SimpleNamespace(
        _adapter=SimpleNamespace(save_trace=lambda t: t),
        agent_id="server-default",
        session_id="server-session",
        namespace="ns",
    )

    def _redact(text, *, adapter=None, agent_id="", session_id=""):
        return None if text is None else text.replace("sk-live-99", "[REDACTED]")

    original_scan = http_server.scan_captured_text
    original_args = http_server.scan_captured_arguments
    original_mem = http_server._get_memory
    original_link = http_server._link_agent_owner_once
    http_server.scan_captured_text = _redact  # type: ignore[assignment]
    # Held aside so a blocked argument cannot be what drops the action; the result
    # is the subject here.
    http_server.scan_captured_arguments = (  # type: ignore[assignment]
        lambda args, **kwargs: args
    )
    http_server._get_memory = lambda: mem  # type: ignore[assignment]
    http_server._link_agent_owner_once = lambda *a, **k: None  # type: ignore[assignment]
    try:
        request = SimpleNamespace(
            json=lambda: asyncio.sleep(0, result=trace.model_dump(mode="json"))
        )
        result = asyncio.run(http_server.save_trace(request))
    finally:
        http_server.scan_captured_text = original_scan  # type: ignore[assignment]
        http_server.scan_captured_arguments = original_args  # type: ignore[assignment]
        http_server._get_memory = original_mem  # type: ignore[assignment]
        http_server._link_agent_owner_once = original_link  # type: ignore[assignment]

    assert result["tool_calls"][0]["result_summary"] == "issued [REDACTED]"
    assert result["tool_calls"][0]["tool_name"] == "create_api_key"


def test_the_scan_identity_still_falls_back_to_the_handle() -> None:
    """Every other call site passes no identity and must keep the handle's."""
    from types import SimpleNamespace

    from amfs_http import server as http_server

    seen: list[dict[str, str]] = []

    def _record(text, *, adapter=None, agent_id="", session_id=""):
        seen.append({"agent_id": agent_id, "session_id": session_id})
        return text

    original = http_server.scan_captured_text
    http_server.scan_captured_text = _record  # type: ignore[assignment]
    try:
        http_server._scan_captured_text(
            SimpleNamespace(_adapter=None, agent_id="handle-agent", session_id="s0"),
            "the ask",
        )
    finally:
        http_server.scan_captured_text = original  # type: ignore[assignment]

    assert seen == [{"agent_id": "handle-agent", "session_id": "s0"}]


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
