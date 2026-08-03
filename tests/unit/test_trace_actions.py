"""Recorded actions: the action half of a supervised training pair.

``task_input`` gave a trace the request an agent was given. On its own that is half
an example — training predicts the *action* from the request, and until now nothing
recorded one. ``ToolCall`` existed on the Pro trace model and a recorder method
existed to append to it, but no caller anywhere invoked it, so the column was
always empty and any dataset built from real traffic came back with zero rows.

The tests here pin the parts of the path that can fail silently:

1. Arguments are caller-supplied structures that routinely carry credentials, and
   a nested one is no less a credential for being a level down.
2. An argument the gate blocks must drop the whole action. Keeping the action
   minus one argument leaves a plausible-looking example that teaches a call with
   a missing parameter, which is worse than having no example.
3. The action has to travel on the outcome record, not only on the trace. The
   record reaches the server first and the server seals from it — the same
   ordering that left captured prompts out of the sealed copy.
4. Actions must not survive into the next trace, or one task's work is attributed
   to the following one.
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
    capture._GATE_CACHE["gate"] = gate


def _flush_bg() -> None:
    _get_sdk_executor().submit(lambda: None).result(timeout=5)


# --- Scanning arguments -----------------------------------------------------


def test_arguments_pass_through_when_there_is_no_gate() -> None:
    """An OSS install has no safety package, and recording is opt-in."""
    args = {"service": "checkout", "to_version": "v41"}
    assert capture.scan_captured_arguments(args) == args


def test_a_non_dict_becomes_an_empty_dict() -> None:
    """Callers reach this through an MCP boundary, so the type is not guaranteed."""
    assert capture.scan_captured_arguments(None) == {}
    assert capture.scan_captured_arguments("not a dict") == {}


def test_a_redacted_argument_keeps_the_action(monkeypatch) -> None:
    class Gate:
        def check_write(self, entry):
            return _Decision(True, entry.value.replace("sk-live-123", "[REDACTED]"))

    _install_gate(monkeypatch, Gate())
    scanned = capture.scan_captured_arguments({"token": "sk-live-123", "n": "keep"})
    assert scanned == {"token": "[REDACTED]", "n": "keep"}


def test_a_blocked_argument_drops_the_whole_action(monkeypatch) -> None:
    """The central judgement of this feature.

    Returning the action with the offending argument removed would leave something
    that still looks like a valid example. A model trained on it learns to call the
    tool without the parameter — a worse outcome than one fewer example.
    """

    class Gate:
        def check_write(self, entry):
            return _Decision("AWS_SECRET" not in entry.value, entry.value)

    _install_gate(monkeypatch, Gate())
    assert capture.scan_captured_arguments(
        {"safe": "fine", "creds": "AWS_SECRET=abc"}
    ) is None


def test_a_credential_nested_in_a_container_is_still_found(monkeypatch) -> None:
    """Arguments are arbitrary caller structures, not flat string maps."""

    class Gate:
        def check_write(self, entry):
            return _Decision("AWS_SECRET" not in entry.value, entry.value)

    _install_gate(monkeypatch, Gate())
    assert capture.scan_captured_arguments(
        {"env": {"vars": [{"value": "AWS_SECRET=abc"}]}}
    ) is None


def test_non_string_scalars_are_left_alone(monkeypatch) -> None:
    """There is nothing in an int or a bool for the gate to redact.

    Passing them through the scanner would be wasted calls, and coercing them to
    strings would change the recorded action.
    """
    calls: list[str] = []

    class Gate:
        def check_write(self, entry):
            calls.append(entry.value)
            return _Decision(True, entry.value)

    _install_gate(monkeypatch, Gate())
    args = {"count": 3, "dry_run": False, "ratio": 1.5, "nothing": None, "s": "x"}
    assert capture.scan_captured_arguments(args) == args
    assert calls == ["x"]


def test_an_empty_string_argument_is_not_mistaken_for_a_block() -> None:
    """The text scanner returns None for empty input, which is not a block.

    Treating it as one would drop every action carrying an empty-string argument.
    """
    assert capture.scan_captured_arguments({"note": ""}) == {"note": ""}


# --- Scanning results -------------------------------------------------------


def test_a_secret_in_an_action_result_is_redacted(tmp_path, monkeypatch) -> None:
    """The result is caller text as much as the arguments are.

    A tool that mints a credential echoes it back in its result, so a result that
    skipped the gate would put the credential into a trace and from there into a
    dataset that leaves our infrastructure for a tuning job.
    """

    class Gate:
        def check_write(self, entry):
            return _Decision(True, entry.value.replace("sk-live-99", "[REDACTED]"))

    _install_gate(monkeypatch, Gate())
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action("create_api_key", {"scope": "read"}, result="key sk-live-99")
    mem.commit_outcome("task-r1", OutcomeType.SUCCESS)
    _flush_bg()

    assert mem._last_trace.tool_calls[0].result_summary == "key [REDACTED]"


def test_a_blocked_result_costs_the_result_not_the_action(tmp_path, monkeypatch) -> None:
    """Deliberately unlike a blocked argument, which drops the whole action.

    The training target is the tool and its arguments, so an action without its
    result is still a usable example — dropping the action would throw away a good
    one to avoid text we are already discarding.
    """

    class Gate:
        def check_write(self, entry):
            return _Decision("AWS_SECRET" not in entry.value, entry.value)

    _install_gate(monkeypatch, Gate())
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action("read_env", {"path": "/etc/app"}, result="AWS_SECRET=abc")
    mem.commit_outcome("task-r2", OutcomeType.SUCCESS)
    _flush_bg()

    action = mem._last_trace.tool_calls[0]
    assert action.tool_name == "read_env"
    assert action.arguments == {"path": "/etc/app"}
    assert action.result_summary == ""


def test_a_relayed_result_is_scanned_too(tmp_path, monkeypatch) -> None:
    """The relay path takes results straight from an HTTP body."""

    class Gate:
        def check_write(self, entry):
            return _Decision(True, entry.value.replace("sk-live-99", "[REDACTED]"))

    _install_gate(monkeypatch, Gate())
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.commit_outcome(
        "task-r3",
        OutcomeType.SUCCESS,
        tool_calls=[{"tool_name": "mint", "result_summary": "sk-live-99"}],
    )
    _flush_bg()

    assert mem._last_trace.tool_calls[0].result_summary == "[REDACTED]"


# --- The SDK path -----------------------------------------------------------


def test_a_recorded_action_reaches_the_trace(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action(
        "deploy_rollback",
        {"service": "checkout", "to_version": "v41"},
        result='{"status": "rolled_back"}',
        duration_ms=1430,
    )
    mem.commit_outcome("task-1", OutcomeType.SUCCESS, task_input="roll back checkout")
    _flush_bg()

    trace = mem._last_trace
    assert trace is not None
    assert len(trace.tool_calls) == 1
    action = trace.tool_calls[0]
    assert action.tool_name == "deploy_rollback"
    assert action.arguments == {"service": "checkout", "to_version": "v41"}
    assert action.duration_ms == 1430
    assert action.result_hash, "the result should be hashed even though it is truncated"


def test_a_trace_carries_both_halves_of_the_training_pair(tmp_path) -> None:
    """What the dataset builder requires, asserted together.

    It filters on a non-empty ``task_input`` *and* a non-empty ``tool_calls``, so a
    trace with one and not the other is skipped. This is the shape that counts.
    """
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action("deploy_rollback", {"service": "checkout"})
    mem.commit_outcome(
        "task-2", OutcomeType.SUCCESS, task_input="checkout is erroring after the deploy"
    )
    _flush_bg()

    trace = mem._last_trace
    assert trace.task_input
    assert trace.tool_calls


def test_the_action_travels_with_the_outcome_record(tmp_path) -> None:
    """The ordering bug that left sealed traces empty, in its action form.

    The record reaches the server before the trace does and the server seals from
    that call, so an action carried only on the later ``save_trace`` is missing
    from the sealed copy that training reads.
    """
    seen: list[object] = []

    class _RecordingAdapter(FilesystemAdapter):
        def commit_outcome(self, record):
            seen.append(record)
            return super().commit_outcome(record)

    adapter = _RecordingAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action("refund_payment", {"charge_id": "ch_123"})
    mem.commit_outcome("task-3", OutcomeType.SUCCESS)
    _flush_bg()

    assert seen, "the adapter's commit_outcome should have been called"
    assert [t["tool_name"] for t in seen[0].tool_calls] == ["refund_payment"]


def test_the_record_carries_scanned_arguments_not_raw(monkeypatch, tmp_path) -> None:
    """Nothing raw may leave the process, arguments included."""

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
    mem.record_action("call_api", {"token": "sk-live-123"})
    mem.commit_outcome("task-4", OutcomeType.SUCCESS)
    _flush_bg()

    assert seen
    assert seen[0].tool_calls[0]["arguments"] == {"token": "[REDACTED]"}


def test_an_action_the_gate_blocks_is_absent_from_the_trace(monkeypatch, tmp_path) -> None:
    class Gate:
        def check_write(self, entry):
            return _Decision("AWS_SECRET" not in entry.value, entry.value)

    _install_gate(monkeypatch, Gate())

    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action("safe_action", {"ok": "yes"})
    mem.record_action("leaky_action", {"creds": "AWS_SECRET=abc"})
    mem.commit_outcome("task-5", OutcomeType.SUCCESS)
    _flush_bg()

    names = [t.tool_name for t in mem._last_trace.tool_calls]
    assert names == ["safe_action"]


def test_actions_do_not_survive_into_the_next_trace(tmp_path) -> None:
    """Committing ends the window the tracker describes.

    Without this, every later trace in the process would re-report the first
    task's actions and the dataset would fill with duplicated examples.
    """
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action("first_action", {})
    mem.commit_outcome("task-6", OutcomeType.SUCCESS)
    _flush_bg()
    assert len(mem._last_trace.tool_calls) == 1

    mem.commit_outcome("task-7", OutcomeType.SUCCESS)
    _flush_bg()
    assert mem._last_trace.tool_calls == []


# --- The relayed (server) path ----------------------------------------------


def test_supplied_actions_are_used_instead_of_the_session_log(tmp_path) -> None:
    """How the HTTP server passes a remote client's actions through."""
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action("local_action", {})
    mem.commit_outcome(
        "task-8",
        OutcomeType.SUCCESS,
        tool_calls=[{"tool_name": "relayed_action", "arguments": {"a": 1}}],
    )
    _flush_bg()

    names = [t.tool_name for t in mem._last_trace.tool_calls]
    assert names == ["relayed_action"]


def test_a_malformed_relayed_action_does_not_cost_the_commit(tmp_path) -> None:
    """Actions over HTTP are free-form JSON, so this arrives from outside.

    The memories being written matter more than the malformed action, so the bad
    entry is dropped and everything else still lands. Constructing the trace
    without this guard raises and the whole commit is lost.
    """
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.write("repo/mod", "k", "a memory worth keeping")
    mem.commit_outcome(
        "task-8b",
        OutcomeType.SUCCESS,
        tool_calls=[
            {"arguments": {"a": 1}},  # no tool name
            {"tool_name": "bad_duration", "duration_ms": "not-a-number"},
            "not even a dict",
            {"tool_name": "null_success", "success": None},  # kept, defaulted
            {"tool_name": "good_action", "arguments": {"a": 1}},
        ],
    )
    _flush_bg()

    kept = mem._last_trace.tool_calls
    assert [t.tool_name for t in kept] == ["null_success", "good_action"]
    assert kept[0].success is True, "an explicit null means 'not told', not 'drop it'"
    assert mem.read("repo/mod", "k").value == "a memory worth keeping"


def test_an_empty_supplied_list_means_no_actions(tmp_path) -> None:
    """Not "fall back to the tracker".

    ``mem`` is shared across requests on the server, so falling back on an empty
    list would attribute whatever happened to be buffered there to this caller.
    """
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.record_action("someone_elses_action", {})
    mem.commit_outcome("task-9", OutcomeType.SUCCESS, tool_calls=[])
    _flush_bg()

    assert mem._last_trace.tool_calls == []


def test_the_http_adapter_forwards_actions_to_the_server() -> None:
    """The /outcomes body is built by hand, so a field is easy to drop.

    Asserted on the request body rather than the record, because the body is what
    the server actually receives.
    """
    from datetime import datetime, timezone

    from amfs_adapter_http.adapter import HttpAdapter
    from amfs_core.models import OutcomeRecord

    sent: dict[str, object] = {}

    adapter = HttpAdapter(base_url="http://localhost:9", api_key="k")
    adapter._post = lambda path, body: (  # type: ignore[method-assign]
        sent.update({"path": path, "body": body}) or {"entries": []}
    )
    adapter.commit_outcome(
        OutcomeRecord(
            outcome_ref="task-10",
            outcome_type=OutcomeType.SUCCESS,
            committed_at=datetime.now(timezone.utc),
            agent_id="a",
            tool_calls=[{"tool_name": "deploy_rollback", "arguments": {}}],
        )
    )

    assert sent["path"] == "/api/v1/outcomes"
    assert sent["body"]["tool_calls"] == [
        {"tool_name": "deploy_rollback", "arguments": {}}
    ]
