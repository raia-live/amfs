"""The customer's end of a replay: signature, parse, run on the branch, commit
with the attributes the grader reads — and the branch stamp every committed
outcome carries when the memory is not on ``main``."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from amfs import AgentMemory
from amfs.memory import MEMORY_BRANCH_ATTRIBUTE
from amfs.replay import (
    CASE_ID_ATTRIBUTE,
    DELIVERY_HEADER,
    EVENT_HEADER,
    SIGNATURE_HEADER,
    PayloadError,
    ReplayReceiver,
    ReplayRequest,
    ReplayResult,
    SignatureError,
    make_server,
    parse_replay_request,
    sign_replay_body,
    verify_replay_signature,
)
from amfs_core.models import OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter

SECRET = "whsec_test_0123456789abcdef"


@pytest.fixture(autouse=True)
def _no_branch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AMFS_BRANCH", raising=False)


def _payload(**over: Any) -> dict[str, Any]:
    base = {
        "event": "replay_requested",
        "delivery_id": str(uuid4()),
        "sent_at": datetime.now(UTC).isoformat(),
        "fix_id": "0d1f3a6c-2b4e-4f60-9a1e-7c8d9e0f1a2b",
        "agent_id": "support-agent",
        "branch": "repair/0d1f3a6c",
        "case_id": "9b8a7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d",
        "case_set_id": str(uuid4()),
        "task_input": "customer says the invoice email never arrived",
        "expected": {"action": "resolve:resend_email"},
        "deadline_at": (datetime.now(UTC) + timedelta(hours=6)).isoformat(),
        "instructions": "Run the agent on task_input with its memory on branch.",
    }
    base.update(over)
    return base


def _signed(payload: dict[str, Any], secret: str | None = SECRET) -> tuple[dict[str, str], bytes]:
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        EVENT_HEADER: payload["event"],
        DELIVERY_HEADER: payload.get("delivery_id", ""),
    }
    if secret:
        headers[SIGNATURE_HEADER] = sign_replay_body(secret, body)
    return headers, body


# ---------------------------------------------------------------------------
# Signature and parse
# ---------------------------------------------------------------------------


class TestTheSignature:
    def test_matches_the_senders_scheme_and_is_constant_time_safe_on_junk(self) -> None:
        body = b'{"a": 1}'
        header = sign_replay_body(SECRET, body)
        assert header.startswith("sha256=") and len(header) == 7 + 64
        assert verify_replay_signature(SECRET, body, header)
        assert verify_replay_signature(SECRET, body, header.upper().replace("SHA256", "sha256"))
        assert not verify_replay_signature(SECRET, body + b" ", header)
        assert not verify_replay_signature("other", body, header)
        for junk in (None, "", "sha256=", "md5=abc", "abc", "sha256"):
            assert not verify_replay_signature(SECRET, body, junk)

    def test_a_bad_signature_is_a_401_and_the_sender_stops_retrying(self) -> None:
        headers, body = _signed(_payload(), secret="wrong")
        with pytest.raises(SignatureError) as exc:
            parse_replay_request(headers, body, secret=SECRET)
        assert exc.value.status == 401
        # Header names are matched case-insensitively, as frameworks re-case them.
        headers, body = _signed(_payload())
        lowered = {k.lower(): v for k, v in headers.items()}
        assert parse_replay_request(lowered, body, secret=SECRET).case_id

    def test_no_secret_means_no_verification(self) -> None:
        headers, body = _signed(_payload(), secret=None)
        req = parse_replay_request(headers, body, secret=None)
        assert req.branch == "repair/0d1f3a6c"


class TestTheParse:
    def test_reads_every_field_the_sender_frames(self) -> None:
        payload = _payload()
        headers, body = _signed(payload)
        req = parse_replay_request(headers, body, secret=SECRET)
        assert isinstance(req, ReplayRequest)
        assert (req.fix_id, req.case_id, req.branch) == (
            payload["fix_id"],
            payload["case_id"],
            payload["branch"],
        )
        assert req.task_input == payload["task_input"]
        assert req.expected == {"action": "resolve:resend_email"}
        assert req.deadline_at is not None and req.deadline_at.tzinfo is not None
        assert req.outcome_ref == "replay:0d1f3a6c:9b8a7c6d"
        assert req.attributes[CASE_ID_ATTRIBUTE] == payload["case_id"]
        assert req.attributes[MEMORY_BRANCH_ATTRIBUTE] == payload["branch"]
        assert not req.past_deadline()

    @pytest.mark.parametrize(
        "over, needle",
        [
            ({"case_id": ""}, "case_id"),
            ({"branch": None}, "branch"),
            ({"task_input": ""}, "task_input"),
            ({"branch": "main"}, "main"),
            ({"event": "something_else"}, "unknown event"),
        ],
    )
    def test_refuses_what_it_cannot_act_on(self, over: dict[str, Any], needle: str) -> None:
        headers, body = _signed(_payload(**over))
        with pytest.raises(PayloadError) as exc:
            parse_replay_request(headers, body, secret=SECRET)
        assert exc.value.status == 400 and needle in str(exc.value)

    def test_not_json_is_a_400(self) -> None:
        body = b"<html>"
        headers = {SIGNATURE_HEADER: sign_replay_body(SECRET, body)}
        with pytest.raises(PayloadError):
            parse_replay_request(headers, body, secret=SECRET)

    def test_a_ping_parses_to_a_ping(self) -> None:
        headers, body = _signed(
            {"event": "ping", "delivery_id": "d-1", "agent_id": "support-agent"}
        )
        req = parse_replay_request(headers, body, secret=SECRET)
        assert req.event == "ping" and req.agent_id == "support-agent"


# ---------------------------------------------------------------------------
# The receiver
# ---------------------------------------------------------------------------


class _World:
    """A filesystem-backed memory with the fix on a repair branch and not on
    main, and a runner that answers from whatever it can read."""

    def __init__(self, tmp_path) -> None:
        self.adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
        main = AgentMemory(agent_id="support-agent", adapter=self.adapter)
        main.write(
            "acme/support",
            "fix-resend",
            "resend the invoice email",
            confidence=0.9,
            branch="repair/0d1f3a6c",
        )
        main.close()
        self.memories: list[AgentMemory] = []
        self.runs: list[tuple[str, str]] = []

    def memory_factory(self, request: ReplayRequest) -> AgentMemory:
        mem = AgentMemory(agent_id="support-agent", adapter=self.adapter, branch=request.branch)
        self.memories.append(mem)
        return mem

    def run(self, task_input: str, memory: AgentMemory) -> str:
        self.runs.append((task_input, memory.branch))
        entry = memory.read("acme/support", "fix-resend")
        return f"I will {entry.value}." if entry else "I do not know what to do."

    @property
    def last_trace(self):
        return self.memories[-1]._last_trace


def test_inline_receiver_runs_on_the_branch_and_commits_the_graded_attributes(tmp_path) -> None:
    world = _World(tmp_path)
    receiver = ReplayReceiver(
        secret=SECRET, run=world.run, memory_factory=world.memory_factory, background=False
    )
    payload = _payload()
    status, answer = receiver.handle(*_signed(payload))
    assert status == 200 and answer["ran"] is True
    assert (
        answer["outcome_type"] == "success" and answer["outcome_ref"] == "replay:0d1f3a6c:9b8a7c6d"
    )
    assert world.runs == [(payload["task_input"], "repair/0d1f3a6c")], "ran on the branch"

    trace = world.last_trace
    attrs = trace.session_metadata.attributes
    assert attrs[CASE_ID_ATTRIBUTE] == payload["case_id"]
    assert attrs[MEMORY_BRANCH_ATTRIBUTE] == "repair/0d1f3a6c"
    assert attrs["fix_id"] == payload["fix_id"]
    assert trace.task_input == payload["task_input"]
    assert trace.response_text == "I will resend the invoice email."
    assert trace.outcome_type == OutcomeType.SUCCESS
    assert [e.key for e in trace.causal_entries] == ["fix-resend"], (
        "the read on the branch is the cause"
    )


def test_a_duplicate_delivery_is_acknowledged_without_a_second_run(tmp_path) -> None:
    world = _World(tmp_path)
    receiver = ReplayReceiver(
        secret=SECRET, run=world.run, memory_factory=world.memory_factory, background=False
    )
    payload = _payload()
    first = receiver.handle(*_signed(payload))
    second = receiver.handle(*_signed(payload))
    assert first[0] == 200 and second[0] == 200
    assert second[1]["duplicate"] is True and second[1]["outcome_type"] == "success"
    assert len(world.runs) == 1
    # A different delivery of the same case is a run of its own.
    receiver.handle(*_signed(_payload(delivery_id=str(uuid4()))))
    assert len(world.runs) == 2


def test_a_ping_and_the_refusals_answer_without_running(tmp_path) -> None:
    world = _World(tmp_path)
    receiver = ReplayReceiver(
        secret=SECRET, run=world.run, memory_factory=world.memory_factory, background=False
    )
    status, answer = receiver.handle(
        *_signed({"event": "ping", "delivery_id": "p1", "agent_id": "a"})
    )
    assert (status, answer["ok"], answer["event"]) == (200, True, "ping")

    status, answer = receiver.handle(*_signed(_payload(), secret="wrong"))
    assert status == 401 and answer["code"] == "bad_signature"

    status, answer = receiver.handle(*_signed(_payload(case_id="")))
    assert status == 400 and answer["code"] == "bad_payload"

    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    status, answer = receiver.handle(*_signed(_payload(deadline_at=past)))
    assert status == 410 and answer["code"] == "past_deadline"
    assert world.runs == []


def test_a_runner_that_raises_is_committed_as_a_failure_not_left_missing(tmp_path) -> None:
    world = _World(tmp_path)

    def boom(task_input: str, memory: AgentMemory) -> str:
        memory.read("acme/support", "fix-resend")
        raise RuntimeError("model endpoint down")

    receiver = ReplayReceiver(
        secret=SECRET, run=boom, memory_factory=world.memory_factory, background=False
    )
    status, answer = receiver.handle(*_signed(_payload()))
    assert status == 200 and answer["outcome_type"] == "failure"
    trace = world.last_trace
    assert trace.outcome_type == OutcomeType.FAILURE
    assert "RuntimeError: model endpoint down" in (trace.response_text or "")
    assert trace.session_metadata.attributes[CASE_ID_ATTRIBUTE]
    assert trace.session_metadata.attributes["replay_error"] == "RuntimeError"


def test_the_runner_may_answer_with_a_result_a_pair_or_a_dict(tmp_path) -> None:
    world = _World(tmp_path)
    answers = iter(
        [
            ReplayResult(
                response_text="explicit",
                outcome_type=OutcomeType.MINOR_FAILURE,
                attributes={"model": "gpt-x"},
            ),
            ("as a pair", "failure"),
            {
                "response_text": "as a dict",
                "outcome_type": "success",
                "tool_calls": [{"tool_name": "resolve", "arguments": {"action": "resend_email"}}],
            },
        ]
    )
    receiver = ReplayReceiver(
        secret=SECRET,
        run=lambda t, m: next(answers),
        memory_factory=world.memory_factory,
        background=False,
    )
    got = [receiver.handle(*_signed(_payload()))[1]["outcome_type"] for _ in range(3)]
    assert got == ["minor_failure", "failure", "success"]
    traces = [m._last_trace for m in world.memories]
    assert traces[0].session_metadata.attributes["model"] == "gpt-x"
    assert traces[0].session_metadata.attributes[CASE_ID_ATTRIBUTE], (
        "the grader's keys win over the runner's bag"
    )
    assert traces[1].response_text == "as a pair"
    assert traces[2].tool_calls and traces[2].tool_calls[0].tool_name == "resolve"


def test_the_background_receiver_acknowledges_at_once_and_runs_after(tmp_path) -> None:
    world = _World(tmp_path)
    gate = threading.Event()

    def slow(task_input: str, memory: AgentMemory) -> str:
        assert gate.wait(5), "the test released the runner"
        return world.run(task_input, memory)

    receiver = ReplayReceiver(secret=SECRET, run=slow, memory_factory=world.memory_factory)
    try:
        payload = _payload()
        status, answer = receiver.handle(*_signed(payload))
        assert status == 202 and answer["accepted"] is True
        assert world.runs == [], "the sender got its 202 before the agent ran"
        # A redelivery while the run is in flight is acknowledged, not re-run.
        assert receiver.handle(*_signed(payload))[0] == 200
        gate.set()
        assert receiver.drain(timeout=5)
        assert len(world.runs) == 1
        final = receiver.status(payload["delivery_id"])
        assert final and final["ran"] is True and final["outcome_type"] == "success"
        assert (
            world.last_trace.session_metadata.attributes[MEMORY_BRANCH_ATTRIBUTE]
            == "repair/0d1f3a6c"
        )
    finally:
        receiver.close()


def test_a_runner_that_outlasts_the_timeout_is_committed_as_a_failure(tmp_path) -> None:
    """A hung runner would otherwise get its 202, have its delivery remembered
    so retries are suppressed, and never commit — the case is never graded.
    The bound commits a failure at the deadline; the runner's late answer,
    when it comes, is not committed over it."""
    world = _World(tmp_path)
    gate = threading.Event()

    def hung(task_input: str, memory: AgentMemory) -> str:
        assert gate.wait(5), "the test released the runner"
        return world.run(task_input, memory)

    receiver = ReplayReceiver(
        secret=SECRET, run=hung, memory_factory=world.memory_factory, run_timeout=0.2
    )
    try:
        payload = _payload()
        status, _ = receiver.handle(*_signed(payload))
        assert status == 202
        assert receiver.drain(timeout=5), "the receiver did not wait on the hung runner"
        final = receiver.status(payload["delivery_id"])
        assert final and final["ran"] is True and final["outcome_type"] == "failure"
        trace = world.last_trace
        assert trace.outcome_type == OutcomeType.FAILURE
        assert trace.session_metadata.attributes["replay_error"] == "timeout"
        assert trace.session_metadata.attributes[CASE_ID_ATTRIBUTE] == payload["case_id"]
        assert "did not finish within 0.2s" in trace.response_text
        committed = len(world.memories)
        # The runner finishes late; nothing is committed on top of the failure.
        gate.set()
        deadline = time.monotonic() + 2
        while not world.runs and time.monotonic() < deadline:
            time.sleep(0.01)
        assert world.runs, "the late runner did run to completion"
        time.sleep(0.05)
        assert len(world.memories) == committed
        assert world.last_trace.outcome_type == OutcomeType.FAILURE
    finally:
        receiver.close()


def test_the_timeout_leaves_a_prompt_runner_alone_and_inline_runs_unbounded(tmp_path) -> None:
    world = _World(tmp_path)
    receiver = ReplayReceiver(
        secret=SECRET, run=world.run, memory_factory=world.memory_factory, run_timeout=5
    )
    try:
        payload = _payload()
        receiver.handle(*_signed(payload))
        assert receiver.drain(timeout=5)
        assert receiver.status(payload["delivery_id"])["outcome_type"] == "success"
    finally:
        receiver.close()

    def slow(task_input: str, memory: AgentMemory) -> str:
        time.sleep(0.3)
        return world.run(task_input, memory)

    inline = ReplayReceiver(
        secret=SECRET, run=slow, memory_factory=world.memory_factory,
        background=False, run_timeout=0.05,
    )
    status, answer = inline.handle(*_signed(_payload(delivery_id="d-inline")))
    assert status == 200 and answer["outcome_type"] == "success", "inline runs are not bounded"


def test_the_default_memory_is_an_agent_memory_on_the_branch(tmp_path, monkeypatch) -> None:
    """No factory given: the receiver builds ``AgentMemory(agent_id, branch)``
    from the process configuration and closes what it opened."""
    seen: dict[str, Any] = {}

    class FakeMemory:
        def __init__(self, *, agent_id: str, branch: str) -> None:
            seen["agent_id"], seen["branch"] = agent_id, branch
            self.branch = branch
            self.closed = False
            self._last_trace = None

        def commit_outcome(self, ref, outcome, **kw) -> list:
            seen["commit"] = (ref, outcome, kw["attributes"])
            return []

        def close(self) -> None:
            self.closed = True
            seen["closed"] = True

    import amfs.memory as memory_module

    monkeypatch.setattr(memory_module, "AgentMemory", FakeMemory)
    receiver = ReplayReceiver(secret=None, run=lambda t, m: "done", background=False)
    payload = _payload(agent_id="")
    status, answer = receiver.handle(*_signed(payload, secret=None))
    assert status == 200 and answer["outcome_type"] == "success"
    assert seen["agent_id"] == "replay" and seen["branch"] == "repair/0d1f3a6c"
    assert seen["closed"] is True
    assert seen["commit"][2][CASE_ID_ATTRIBUTE] == payload["case_id"]

    receiver = ReplayReceiver(
        secret=None, run=lambda t, m: "done", agent_id="from-cli", background=False
    )
    receiver.handle(*_signed(_payload(agent_id=""), secret=None))
    assert seen["agent_id"] == "from-cli"


# ---------------------------------------------------------------------------
# The stdlib server
# ---------------------------------------------------------------------------


def test_the_stdlib_server_mounts_the_receiver(tmp_path) -> None:
    world = _World(tmp_path)
    receiver = ReplayReceiver(
        secret=SECRET, run=world.run, memory_factory=world.memory_factory, background=False
    )
    server = make_server(receiver, port=0)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://{host}:{port}"
        with urllib.request.urlopen(f"{base}/amfs/replay") as resp:  # the health check
            assert resp.status == 200 and json.loads(resp.read())["receiver"] == "amfs-replay"

        headers, body = _signed(_payload())
        req = urllib.request.Request(
            f"{base}/amfs/replay", data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            answer = json.loads(resp.read())
        assert answer["outcome_type"] == "success" and len(world.runs) == 1

        headers, body = _signed(_payload(), secret="wrong")
        req = urllib.request.Request(
            f"{base}/amfs/replay", data=body, headers=headers, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 401

        req = urllib.request.Request(f"{base}/elsewhere", data=body, headers=headers, method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# The branch stamp on every commit
# ---------------------------------------------------------------------------


class TestTheBranchStamp:
    def test_a_memory_on_a_branch_stamps_memory_branch_on_its_outcomes(self, tmp_path) -> None:
        adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        mem.write("acme/x", "k", "v", confidence=0.8)
        mem.commit_outcome("o-1", OutcomeType.SUCCESS, task_input="t")
        assert (
            mem._last_trace.session_metadata.attributes[MEMORY_BRANCH_ATTRIBUTE] == "repair/fix-1"
        )

        # checkout moves the stamp with the branch; main carries none.
        mem.checkout("main")
        mem.commit_outcome("o-2", OutcomeType.SUCCESS, task_input="t")
        meta = mem._last_trace.session_metadata
        assert meta is None or MEMORY_BRANCH_ATTRIBUTE not in (meta.attributes or {})

    def test_the_env_var_carries_the_stamp_and_a_caller_may_override_it(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AMFS_BRANCH", "canary/abc")
        adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter)
        assert mem.branch == "canary/abc"
        mem.commit_outcome("o-1", OutcomeType.SUCCESS, task_input="t")
        assert mem._last_trace.session_metadata.attributes[MEMORY_BRANCH_ATTRIBUTE] == "canary/abc"
        mem.commit_outcome(
            "o-2",
            OutcomeType.SUCCESS,
            task_input="t",
            attributes={MEMORY_BRANCH_ATTRIBUTE: "canary/xyz"},
        )
        assert mem._last_trace.session_metadata.attributes[MEMORY_BRANCH_ATTRIBUTE] == "canary/xyz"

    def test_the_stamp_does_not_count_against_the_attribute_cap(self, tmp_path) -> None:
        from amfs.memory import SESSION_ATTRIBUTES_MAX_KEYS

        adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        full = {f"k{i}": i for i in range(SESSION_ATTRIBUTES_MAX_KEYS)}
        mem.commit_outcome("o-1", OutcomeType.SUCCESS, task_input="t", attributes=full)
        attrs = mem._last_trace.session_metadata.attributes
        assert len(attrs) == SESSION_ATTRIBUTES_MAX_KEYS + 1
        assert attrs[MEMORY_BRANCH_ATTRIBUTE] == "repair/fix-1"
        # The server validates the wire bag with the same function, so the
        # 21-key bag the SDK just sent is accepted there too — and a 21st
        # caller key still is not.
        from amfs.memory import validate_session_attributes

        assert len(validate_session_attributes(attrs)) == SESSION_ATTRIBUTES_MAX_KEYS + 1
        with pytest.raises(ValueError, match="at most 20"):
            validate_session_attributes({**full, "k20": 20})
        with pytest.raises(ValueError, match="at most 20"):
            validate_session_attributes({**full, "k20": 20, MEMORY_BRANCH_ATTRIBUTE: "b"})
