"""How many immutable traces one committed outcome is worth.

An SDK client on the HTTP adapter commits an outcome in two requests: the outcome
record goes to ``POST /api/v1/outcomes``, and the decision trace follows on ``POST
/api/v1/traces``. Both endpoints seal an immutable trace, so one outcome produced
two — and the two are not equals.

The trace sealed at ``/outcomes`` is not the caller's. The server has none of the
caller's to seal, so it assembles one on its single shared ``AgentMemory`` handle.
The actions and the attribute bag are passed in the body precisely because that
handle may not be trusted to hold them, but the causal entries, query events, state
diff and session window are still read straight off its tracker — so they belong to
whichever requests last touched it, and the session id is the server process's. Its
duration is measured from that process's clock, which is why no effort or duration
figure taken over sealed traces meant anything.

Two rows per outcome doubled every count and average computed over them, and the
fabricated row was chained by ``get_latest_hash(session_id)`` under the process's
own session — one hash chain shared by every account served by that process.

The fix is a caller declaring that its own trace is on the way, so the server seals
once, from the copy that was right all along: the client's, which carries its real
session, its real window and only its own reads.

The Pro trace package is closed source and absent here, so the seal path is stubbed
at the module boundary, as in ``test_seal_attribution``. That the path cannot run at
all in an OSS install is why both of these reached production untested.
"""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from amfs_core.models import DecisionTrace, OutcomeRecord, OutcomeType
from amfs_http import server as http_server
from amfs_http.models import OutcomeRequest

SERVER_DEFAULT = "amfs-server"
SERVER_SESSION = "server-session"
CALLER = "sre-agent"
CLIENT_SESSION = "client-session"


class _SharedHandle:
    """The server's one memory handle, and the tracker state it carries over.

    ``session_id`` is the process's, not any caller's, and the trace this builds
    reports it — which is the whole reason a trace sealed here is untrustworthy.
    """

    def __init__(self) -> None:
        self._tagger = SimpleNamespace(agent_id=SERVER_DEFAULT)
        self.session_id = SERVER_SESSION
        self.namespace = "test"
        self._last_trace: DecisionTrace | None = None
        self._adapter = SimpleNamespace(
            ensure_agent=lambda *a, **k: None,
            save_trace=lambda t: t,
        )

    @property
    def agent_id(self) -> str:
        return self._tagger.agent_id

    def as_agent(self, agent_id: str) -> "_SharedHandle":
        # What ``AgentMemory.as_agent`` does: a handle onto the same store with
        # its own identity and its own trace slot, sharing nothing mutable.
        clone = copy.copy(self)
        clone._tagger = SimpleNamespace(agent_id=agent_id)
        clone._last_trace = None
        return clone

    def commit_outcome(self, outcome_ref, outcome_type, **kwargs) -> list:
        self._last_trace = DecisionTrace(
            agent_id=self.agent_id,
            session_id=self.session_id,
            outcome_ref=outcome_ref,
            outcome_type=outcome_type.value,
            task_input=kwargs.get("task_input"),
            # The window this handle would report: measured from the server
            # process's tracker, and nothing to do with the caller's session.
            session_started_at=datetime(2026, 1, 1, tzinfo=UTC),
            session_ended_at=datetime(2026, 1, 1, 9, tzinfo=UTC),
        )
        return []


@pytest.fixture
def sealed(monkeypatch) -> list[Any]:
    """Stand the Pro seal path up on stubs and collect everything sealed."""
    records: list[Any] = []

    monkeypatch.setattr(http_server, "_HAS_PRO_TRACES", True, raising=False)

    def _map(oss_trace, **kw):
        fields = {
            k: getattr(oss_trace, k, None)
            for k in (
                "outcome_ref",
                "outcome_type",
                "task_input",
                "agent_id",
                "session_started_at",
                "session_ended_at",
            )
        }
        fields.update(kw)
        return SimpleNamespace(**fields)

    monkeypatch.setattr(
        http_server, "_pro_immutable_from_oss_trace", _map, raising=False
    )
    monkeypatch.setattr(
        http_server, "_pro_finalize_spans", lambda imm: imm, raising=False
    )
    monkeypatch.setattr(http_server, "seal", lambda imm, *a, **kw: imm, raising=False)
    monkeypatch.setattr(http_server, "get_signing_key", lambda: "key", raising=False)
    monkeypatch.setattr(http_server, "get_signing_key_id", lambda: "kid", raising=False)

    class _Store:
        def get_latest_hash(self, session_id):
            return None

        def save(self, trace):
            records.append(trace)
            return SimpleNamespace(
                id="00000000-0000-0000-0000-000000000001", **trace.__dict__
            )

    monkeypatch.setattr(http_server, "_get_immutable_store", lambda: _Store())
    monkeypatch.setattr(http_server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(http_server, "_audit_log", lambda *a, **k: None)
    return records


def _commit(handle: _SharedHandle, monkeypatch, *, trace_follows: bool) -> dict:
    """``POST /api/v1/outcomes``, driven through the endpoint."""
    monkeypatch.setattr(http_server, "_get_memory", lambda: handle)
    request = SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.1"), state=SimpleNamespace()
    )
    return asyncio.run(
        http_server.commit_outcome(
            OutcomeRequest(
                outcome_ref="deploy-142",
                outcome_type="success",
                agent_id=CALLER,
                task_input="roll api back to v41",
                trace_follows=trace_follows,
            ),
            request,
            None,
        )
    )


def _post_trace(handle: _SharedHandle, monkeypatch) -> dict:
    """``POST /api/v1/traces``, the second half of an SDK commit.

    The trace carries the client's own session and the window its tracker
    measured — the things the server cannot know and did not ask for.
    """
    monkeypatch.setattr(http_server, "_get_memory", lambda: handle)
    body = DecisionTrace(
        agent_id=CALLER,
        session_id=CLIENT_SESSION,
        outcome_ref="deploy-142",
        outcome_type="success",
        task_input="roll api back to v41",
        session_started_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        session_ended_at=datetime(2026, 6, 1, 12, 4, tzinfo=UTC),
    ).model_dump(mode="json")

    class _Req:
        client = SimpleNamespace(host="10.0.0.1")
        state = SimpleNamespace()

        async def json(self):
            return body

    return asyncio.run(http_server.save_trace(_Req(), None))


def test_an_sdk_commit_seals_one_trace_not_two(sealed, monkeypatch) -> None:
    handle = _SharedHandle()
    _commit(handle, monkeypatch, trace_follows=True)
    _post_trace(handle, monkeypatch)

    assert sealed, "the seal path did not run; the stubs no longer match the server"
    assert len(sealed) == 1, (
        f"{len(sealed)} immutable traces for one outcome: "
        + ", ".join(f"session={getattr(s, 'session_id', None)}" for s in sealed)
    )


def test_the_one_that_survives_is_the_callers_own(sealed, monkeypatch) -> None:
    """Which of the two is kept is the entire point.

    Keeping the server's copy would leave the count right and every figure taken
    from it still wrong, so this pins the session the surviving row reports.
    """
    handle = _SharedHandle()
    _commit(handle, monkeypatch, trace_follows=True)
    _post_trace(handle, monkeypatch)

    assert sealed
    assert sealed[0].session_id == CLIENT_SESSION, (
        f"sealed under {sealed[0].session_id!r}: the surviving trace is the one the "
        "server assembled on its shared handle, not the caller's"
    )


def test_the_surviving_trace_carries_the_clients_session_window(
    sealed, monkeypatch
) -> None:
    """The window is what every duration and effort figure is computed from.

    The server's copy reports its own process clock — nine hours here — and the
    caller's reports the four minutes it actually took.
    """
    handle = _SharedHandle()
    _commit(handle, monkeypatch, trace_follows=True)
    _post_trace(handle, monkeypatch)

    assert sealed
    window = sealed[0].session_ended_at - sealed[0].session_started_at
    assert window.total_seconds() == 4 * 60, (
        f"window of {window}: this is the server process's clock, not the session's"
    )


def test_a_client_that_posts_no_trace_is_still_sealed_for(
    sealed, monkeypatch
) -> None:
    """A direct REST caller never posts a trace, and the flag defaults off.

    For it the server's assembled copy is not the poorer of two, it is the only one
    there is — so the seal here has to stay reachable.
    """
    handle = _SharedHandle()
    result = _commit(handle, monkeypatch, trace_follows=False)

    assert len(sealed) == 1
    assert sealed[0].session_id == SERVER_SESSION
    assert result["immutable_trace_id"]


def test_the_skipped_seal_is_not_reported_as_one(sealed, monkeypatch) -> None:
    """The response must not name a trace it did not seal.

    A client that read ``immutable_trace_id`` as confirmation would otherwise be
    told an id that belongs to nothing.
    """
    handle = _SharedHandle()
    result = _commit(handle, monkeypatch, trace_follows=True)

    assert not sealed
    assert "immutable_trace_id" not in result
    assert result["outcome_ref"] == "deploy-142"


def test_the_sdk_declares_that_its_trace_is_coming(monkeypatch, tmp_path) -> None:
    """Declared by ``AgentMemory.commit_outcome``, which is what makes it true.

    It builds and posts the trace on every path out of itself. An adapter's own
    ``commit_outcome``, called directly, promises nothing of the sort — so the flag
    is set here and not there.
    """
    from amfs import AgentMemory
    from amfs_filesystem.adapter import FilesystemAdapter

    seen: list[OutcomeRecord] = []

    class _RecordingAdapter(FilesystemAdapter):
        def commit_outcome(self, record):
            seen.append(record)
            return super().commit_outcome(record)

    adapter = _RecordingAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.commit_outcome("deploy-142", OutcomeType.SUCCESS)

    assert seen, "the adapter's commit_outcome should have been called"
    assert seen[0].trace_follows is True


def test_a_record_nobody_promised_for_declares_nothing() -> None:
    """The default is what a direct adapter caller gets, and it has to be off."""
    record = OutcomeRecord(
        outcome_ref="deploy-142",
        outcome_type=OutcomeType.SUCCESS,
        committed_at=datetime.now(UTC),
        agent_id=CALLER,
    )
    assert record.trace_follows is False


def test_the_http_adapter_forwards_the_declaration() -> None:
    """The adapter builds the ``/outcomes`` body by hand, so a field is droppable.

    Asserted on the body rather than the record for the same reason the capture
    fields are: a record that declares it changes nothing if the transport does not
    carry it, and the server would go on sealing its second copy.
    """
    from amfs_adapter_http.adapter import HttpAdapter

    sent: dict[str, Any] = {}

    adapter = HttpAdapter.__new__(HttpAdapter)

    def _post(path, body):
        sent["body"] = body
        return {"entries": []}

    adapter._post = _post  # type: ignore[method-assign]

    def _commit_with(trace_follows: bool) -> dict:
        adapter.commit_outcome(
            OutcomeRecord(
                outcome_ref="deploy-142",
                outcome_type=OutcomeType.SUCCESS,
                committed_at=datetime.now(UTC),
                agent_id=CALLER,
                trace_follows=trace_follows,
            )
        )
        return sent["body"]

    assert _commit_with(True)["trace_follows"] is True
    # Left out rather than sent as false, so an older server that does not know the
    # field is not handed one, and its behaviour is unchanged.
    assert "trace_follows" not in _commit_with(False)
