"""``amfs replay serve`` and ``amfs replay simulate``, end to end on a port."""

from __future__ import annotations

import json
import sys
import threading
import types
from typing import Any

import pytest
import typer
from amfs.replay import ReplayReceiver, make_server
from amfs_cli.replay import app as replay_app
from amfs_cli.replay import load_runner
from typer.testing import CliRunner

SECRET = "whsec_cli_test"
runner = CliRunner()


@pytest.fixture
def runner_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A module on ``sys.modules`` exposing ``run`` and ``agent.answer``."""
    mod = types.ModuleType("fake_agent_replay")
    calls: list[tuple[str, str]] = []

    def run(task_input: str, memory: Any) -> str:
        calls.append((task_input, memory.branch))
        return f"handled: {task_input}"

    mod.run = run  # type: ignore[attr-defined]
    mod.calls = calls  # type: ignore[attr-defined]
    mod.agent = types.SimpleNamespace(answer=run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_agent_replay", mod)
    return mod


class TestLoadRunner:
    def test_imports_module_colon_callable_including_dotted_attributes(self, runner_module) -> None:
        assert load_runner("fake_agent_replay:run") is runner_module.run
        assert load_runner("fake_agent_replay:agent.answer") is runner_module.run

    @pytest.mark.parametrize(
        "spec",
        [
            "nomodule",
            "fake_agent_replay:",
            "no_such_module_xyz:run",
            "fake_agent_replay:missing",
            "fake_agent_replay:calls",
        ],
    )
    def test_refuses_what_it_cannot_load(self, runner_module, spec: str) -> None:
        with pytest.raises(typer.BadParameter):
            load_runner(spec)


def _serve(receiver: ReplayReceiver):
    server = make_server(receiver, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/amfs/replay"


def test_simulate_drives_a_receiver_end_to_end(runner_module, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AMFS_REPLAY_SECRET", SECRET)
    # The receiver the CLI's `serve` would build, on a filesystem memory in tmp.
    from amfs import AgentMemory
    from amfs_filesystem.adapter import FilesystemAdapter

    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    memories: list[AgentMemory] = []

    def factory(request):
        mem = AgentMemory(agent_id=request.agent_id, adapter=adapter, branch=request.branch)
        memories.append(mem)
        return mem

    receiver = ReplayReceiver(
        secret=SECRET,
        run=load_runner("fake_agent_replay:run"),
        memory_factory=factory,
        background=False,
    )
    server, url = _serve(receiver)
    try:
        result = runner.invoke(replay_app, ["simulate", url, "--ping"])
        assert result.exit_code == 0, result.output
        assert "HTTP 200" in result.output and '"ping"' in result.output

        result = runner.invoke(
            replay_app,
            [
                "simulate",
                url,
                "--task",
                "refund the duplicate charge",
                "--branch",
                "repair/abc",
                "--agent",
                "billing-agent",
                "--expected",
                '{"action": "resolve:refund"}',
            ],
        )
        assert result.exit_code == 0, result.output
        assert "HTTP 200" in result.output
        assert runner_module.calls == [("refund the duplicate charge", "repair/abc")]
        trace = memories[-1]._last_trace
        assert trace.session_metadata.attributes["memory_branch"] == "repair/abc"
        assert trace.session_metadata.attributes["case_id"]
        assert trace.response_text == "handled: refund the duplicate charge"

        # The wrong secret is refused and the command exits non-zero.
        result = runner.invoke(replay_app, ["simulate", url, "--secret", "nope"])
        assert result.exit_code == 1 and "HTTP 401" in result.output

        # A payload file is sent as-is.
        payload = tmp_path / "p.json"
        payload.write_text(json.dumps({"event": "ping", "delivery_id": "d-file", "agent_id": "x"}))
        result = runner.invoke(replay_app, ["simulate", url, "--payload", str(payload)])
        assert result.exit_code == 0 and '"agent_id": "x"' in result.output
    finally:
        server.shutdown()
        server.server_close()


def test_serve_needs_a_secret_unless_insecure(runner_module, monkeypatch) -> None:
    monkeypatch.delenv("AMFS_REPLAY_SECRET", raising=False)
    result = runner.invoke(replay_app, ["serve", "--run", "fake_agent_replay:run", "--port", "0"])
    assert result.exit_code != 0
    assert "no secret" in result.output


def test_serve_binds_and_serves_the_runner(runner_module, tmp_path, monkeypatch) -> None:
    """``serve`` wires the runner, the secret and the inline flag into a
    receiver and hands it to the stdlib server. The server is stubbed so the
    command returns; what it was given is what is asserted."""
    import amfs.replay as replay_module

    given: dict[str, Any] = {}

    def fake_serve(receiver, *, host, port, path, ready):
        given.update(receiver=receiver, host=host, port=port, path=path)
        ready(f"http://{host}:{port}{path}")

    monkeypatch.setattr(replay_module, "serve", fake_serve)
    result = runner.invoke(
        replay_app,
        [
            "serve",
            "--run",
            "fake_agent_replay:run",
            "--secret",
            SECRET,
            "--port",
            "9999",
            "--path",
            "/hooks/replay",
            "--inline",
            "--agent",
            "cli-agent",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "listening at http://0.0.0.0:9999/hooks/replay" in result.output
    receiver = given["receiver"]
    assert isinstance(receiver, ReplayReceiver)
    assert (
        receiver._secret == SECRET
        and receiver._executor is None
        and receiver._agent_id == "cli-agent"
    )
    assert receiver._run is runner_module.run

    monkeypatch.delenv("AMFS_REPLAY_SECRET", raising=False)
    result = runner.invoke(replay_app, ["serve", "--run", "fake_agent_replay:run", "--insecure"])
    assert result.exit_code == 0, result.output
    assert "accepting unsigned" in result.output and given["receiver"]._secret is None
