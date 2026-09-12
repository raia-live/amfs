"""How many rows the ``decision_traces`` table gets for one committed outcome.

The companion to ``test_seal_once_per_outcome``, one layer down. That one stopped
``POST /api/v1/outcomes`` sealing a second *immutable* trace when the caller's own
is on its way. The OSS trace underneath was still written twice.

``AgentMemory.commit_outcome`` persists the trace it builds, and the ``/outcomes``
handler calls it on the server's single shared handle. So the row written there is
assembled from that handle: the causal entries, query events, state diff and
session window come off its tracker and belong to whichever requests last touched
it, and the session is the server process's. The client then posts its own trace to
``/traces``, which is written too.

Two rows per outcome, and not distinguishable by agent — the handler points the
tagger at the caller for exactly the block that builds the first one. So every
count, ratio and average taken over that table was measured against a population
roughly twice its true size, half of it untrue. The read-readiness baseline is one
such figure.

The fix is the declaration the caller already sends, applied one layer deeper.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from amfs import AgentMemory
from amfs_core.models import DecisionTrace, OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter
from amfs_http import server as http_server
from amfs_http.models import OutcomeRequest

CALLER = "sre-agent"
CLIENT_SESSION = "client-session"


class _RecordingAdapter(FilesystemAdapter):
    """A filesystem adapter that keeps every trace it is asked to persist."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.saved: list[DecisionTrace] = []

    def save_trace(self, trace):
        self.saved.append(trace)
        return super().save_trace(trace)


@pytest.fixture
def handle(monkeypatch, tmp_path) -> AgentMemory:
    """The server's one shared handle, wired in as ``_get_memory`` returns it.

    The seal path is off: this is about the OSS row, and an OSS install has no
    Pro trace package anyway.
    """
    adapter = _RecordingAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="http-server", adapter=adapter)

    monkeypatch.setattr(http_server, "_get_memory", lambda: mem)
    monkeypatch.setattr(http_server, "_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(http_server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(http_server, "_HAS_PRO_TRACES", False, raising=False)
    return mem


def _commit(*, trace_follows: bool) -> dict:
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


def _post_trace() -> dict:
    body = DecisionTrace(
        agent_id=CALLER,
        session_id=CLIENT_SESSION,
        outcome_ref="deploy-142",
        outcome_type="success",
        task_input="roll api back to v41",
    ).model_dump(mode="json")

    class _Req:
        client = SimpleNamespace(host="10.0.0.1")
        state = SimpleNamespace()

        async def json(self):
            return body

    return asyncio.run(http_server.save_trace(_Req(), None))


def test_an_sdk_commit_writes_one_trace_row_not_two(handle) -> None:
    _commit(trace_follows=True)
    _post_trace()

    saved = handle._adapter.saved
    assert len(saved) == 1, (
        f"{len(saved)} decision_traces rows for one outcome: "
        + ", ".join(f"session={t.session_id}" for t in saved)
    )


def test_the_row_that_survives_is_the_callers_own(handle) -> None:
    """Which one is kept matters as much as how many.

    Keeping the server's would leave the count right and the row still assembled
    from another request's state.
    """
    _commit(trace_follows=True)
    _post_trace()

    saved = handle._adapter.saved
    assert saved
    assert saved[0].session_id == CLIENT_SESSION, (
        f"kept the row under {saved[0].session_id!r}, which is the shared handle's "
        "session, not the caller's"
    )


def test_the_two_rows_were_not_telling_apart_by_agent(handle) -> None:
    """Why this was invisible, and why no query could have filtered it out.

    The handler points the shared tagger at the caller for exactly the block that
    builds the suppressed row, so both rows named the caller. Nothing about the
    row that was untrue said so.
    """
    _commit(trace_follows=False)
    _post_trace()

    saved = handle._adapter.saved
    assert len(saved) == 2, "this pins the old behaviour; it is the point of the fix"
    assert {t.agent_id for t in saved} == {CALLER}
    assert {t.session_id for t in saved} == {handle.session_id, CLIENT_SESSION}


def test_a_client_that_posts_no_trace_still_gets_its_row(handle) -> None:
    """A direct REST caller writes no trace of its own.

    For it the handle-assembled row is not the worse of two, it is the only record
    of the decision, so the write here has to stay reachable.
    """
    _commit(trace_follows=False)

    saved = handle._adapter.saved
    assert len(saved) == 1
    assert saved[0].session_id == handle.session_id


def test_the_trace_is_still_built_when_it_is_not_written(handle) -> None:
    """Only the write is skipped.

    ``_last_trace`` is what the seal path reaches for, and a caller may read it, so
    suppressing the row must not mean there is no trace to hand on.
    """
    _commit(trace_follows=True)

    assert not handle._adapter.saved
    assert handle._last_trace is not None
    assert handle._last_trace.outcome_ref == "deploy-142"


def test_suppression_is_opt_in_for_everyone_else(tmp_path) -> None:
    """The default has to be to persist.

    Every SDK user calls this method, and for them nothing else writes the trace.
    """
    adapter = _RecordingAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="a", adapter=adapter)
    mem.commit_outcome("deploy-142", OutcomeType.SUCCESS)

    assert len(adapter.saved) == 1

    mem.commit_outcome("deploy-143", OutcomeType.SUCCESS, persist_trace=False)

    assert len(adapter.saved) == 1, "the second commit should have written nothing"
