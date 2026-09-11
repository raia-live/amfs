"""Whose decision the sealed immutable trace says it was.

``POST /api/v1/outcomes`` takes the caller's ``agent_id`` in the body, because one
server process serves every agent on an account. The endpoint points its single
shared ``AgentMemory`` handle at that agent for the duration of the commit and
restores the previous identity in a ``finally``. The immutable trace is sealed
*after* that block, and it read the identity off the handle — so the restore had
already happened and every trace posted by every agent was sealed under the
server's own default identity.

Nothing raises, and the OSS decision trace is written correctly, so the loss is
invisible from the outside. It matters because the sealed traces are the copy a
tuning dataset is built from and they are selected by agent: an agent's whole
history landed under a name that never made a decision, and a model scoped to the
real agent trained on an empty set.

The Pro trace package is closed source and absent here, so the seal path is
stubbed at the module boundary — the names the server imported from it are set on
the module and the store is replaced with a recorder. That the path cannot run at
all in an OSS install is why it reached production with no test.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from amfs_core.models import DecisionTrace
from amfs_http import server as http_server
from amfs_http.models import OutcomeRequest

SERVER_DEFAULT = "amfs-server"
CALLER = "sre-agent"


class _SharedHandle:
    """The server's one memory handle, with the identity swap that bit us.

    Only the parts ``commit_outcome`` touches. ``agent_id`` reads through to the
    tagger exactly as ``AgentMemory`` does, which is what makes the restore
    visible to anything that asks the handle who it is.
    """

    def __init__(self) -> None:
        self._tagger = SimpleNamespace(agent_id=SERVER_DEFAULT)
        self.session_id = "server-session"
        self.namespace = "test"
        self._last_trace: DecisionTrace | None = None
        self._adapter = SimpleNamespace(ensure_agent=lambda *a, **k: None)

    @property
    def agent_id(self) -> str:
        return self._tagger.agent_id

    def commit_outcome(self, outcome_ref, outcome_type, **kwargs) -> list:
        # Built while the tagger still points at the caller, which is why the
        # trace is a trustworthy source for the identity and the handle is not.
        self._last_trace = DecisionTrace(
            agent_id=self.agent_id,
            session_id=self.session_id,
            outcome_ref=outcome_ref,
            outcome_type=outcome_type.value,
            task_input=kwargs.get("task_input"),
        )
        return []


@pytest.fixture
def sealed(monkeypatch) -> list[Any]:
    """Stand the Pro seal path up on stubs and collect what gets sealed."""
    records: list[Any] = []

    monkeypatch.setattr(http_server, "_HAS_PRO_TRACES", True, raising=False)

    # The OSS -> immutable mapping is the Pro package's; the server hands it the
    # trace plus the identity it resolved. The stub keeps both so the assertions
    # below read what the server passed, not what the trace happened to carry.
    def _map(oss_trace, **kw):
        fields = {
            k: getattr(oss_trace, k, None)
            for k in ("outcome_ref", "outcome_type", "task_input", "agent_id")
        }
        fields.update(kw)
        return SimpleNamespace(**fields)

    monkeypatch.setattr(
        http_server, "_pro_immutable_from_oss_trace", _map, raising=False
    )
    monkeypatch.setattr(
        http_server, "_pro_finalize_spans", lambda imm: imm, raising=False
    )
    monkeypatch.setattr(
        http_server, "seal", lambda imm, *a, **kw: imm, raising=False
    )
    monkeypatch.setattr(http_server, "get_signing_key", lambda: "key", raising=False)
    monkeypatch.setattr(http_server, "get_signing_key_id", lambda: "kid", raising=False)

    class _Store:
        def get_latest_hash(self, session_id):
            return None

        def save(self, trace):
            records.append(trace)
            return SimpleNamespace(id="00000000-0000-0000-0000-000000000001",
                                   **trace.__dict__)

    monkeypatch.setattr(http_server, "_get_immutable_store", lambda: _Store())
    monkeypatch.setattr(http_server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(http_server, "_audit_log", lambda *a, **k: None)
    return records


def _post(handle: _SharedHandle, monkeypatch, agent_id: str | None) -> dict:
    """Drive the endpoint, not the helper.

    Calling ``_auto_seal_trace`` with an identity spelled out would only prove it
    seals what it is handed. The bug is in the ordering the endpoint imposes —
    seal after restore — so the restore has to be part of what runs.
    """
    monkeypatch.setattr(http_server, "_get_memory", lambda: handle)
    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.1"), state=SimpleNamespace())
    req = OutcomeRequest(
        outcome_ref="deploy-142",
        outcome_type="success",
        agent_id=agent_id,
        task_input="roll api back to v41",
    )
    return asyncio.run(http_server.commit_outcome(req, request, None))


def test_a_sealed_trace_is_attributed_to_the_agent_that_posted_it(
    sealed, monkeypatch
) -> None:
    handle = _SharedHandle()
    result = _post(handle, monkeypatch, CALLER)

    assert sealed, "the seal path did not run; the stubs no longer match the server"
    assert sealed[0].agent_id == CALLER, (
        f"sealed under {sealed[0].agent_id!r}: the identity came from the shared "
        "handle after the endpoint restored it, not from the caller"
    )
    assert result["outcome_ref"] == "deploy-142"
    assert result["immutable_trace_id"]


def test_the_handles_identity_is_still_restored_after_the_commit(
    sealed, monkeypatch
) -> None:
    """The restore is right; reading the identity from it afterwards was not.

    Sealing before the restore would also have produced the correct name, so this
    pins that the fix did not get it by leaving the process-wide handle pointing
    at the last caller — which every later request would then inherit.
    """
    handle = _SharedHandle()
    _post(handle, monkeypatch, CALLER)

    assert handle.agent_id == SERVER_DEFAULT


def test_a_caller_that_names_no_agent_is_sealed_under_the_handle(
    sealed, monkeypatch
) -> None:
    """``agent_id`` is optional, and older clients omit it.

    With no caller identity the handle's own is the only one there is, so the
    fallback has to stay: a trace sealed under no agent at all would be dropped
    from every dataset instead of merely misfiled.
    """
    handle = _SharedHandle()
    _post(handle, monkeypatch, None)

    assert sealed
    assert sealed[0].agent_id == SERVER_DEFAULT


def test_the_sealed_trace_carries_the_prompt_it_was_committed_with(
    sealed, monkeypatch
) -> None:
    """Attribution is only useful if the training pair travelled with it.

    Asserted alongside the identity because both are read off the same trace: a
    change that fixes the name by reaching for a different object could as easily
    drop the capture that makes the trace worth selecting.
    """
    handle = _SharedHandle()
    _post(handle, monkeypatch, CALLER)

    assert sealed
    assert sealed[0].task_input == "roll api back to v41"
    assert sealed[0].outcome_ref == "deploy-142"
