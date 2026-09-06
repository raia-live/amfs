"""Session attributes and LLM calls: the two things a trace cannot know unless
the SDK is told.

A run's cost and token counts exist only if the agent's LLM calls are recorded;
its dimensions (customer, task type) only if the developer passes them. Both
travel in ``DecisionTrace.session_metadata`` under ``attributes`` / ``llm_calls``,
which the server lifts into the sealed trace. These tests pin that contract for
the Python SDK: what ``set_session_attributes`` / ``record_llm_call`` /
``commit_outcome(attributes=..., llm_calls=...)`` accept, the exact shape they
write, that the shape survives the HTTP adapter's ``model_dump``, and that the
buffers reset once a trace is built.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from amfs import AgentMemory
from amfs.memory import (
    SESSION_ATTRIBUTES_KEY,
    SESSION_LLM_CALLS_KEY,
    validate_session_attributes,
)
from amfs_core.models import DecisionTrace, OutcomeType, SessionMetadata
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture
def mem(tmp_path) -> AgentMemory:
    return AgentMemory(agent_id="deploy-agent", adapter=FilesystemAdapter(tmp_path / "amfs"))


# ---------------------------------------------------------------------------
# SessionMetadata keeps extras
# ---------------------------------------------------------------------------


def test_session_metadata_keeps_and_serialises_extra_keys() -> None:
    meta = SessionMetadata(model="gpt-4o", attributes={"customer": "acme"}, spans=[{"a": 1}])
    dumped = meta.model_dump(mode="json")
    assert dumped["attributes"] == {"customer": "acme"}
    assert dumped["spans"] == [{"a": 1}]

    # Through the trace too — this is the HTTP adapter's wire form.
    trace = DecisionTrace(agent_id="a", session_id="s", session_metadata=meta)
    wire = trace.model_dump(mode="json")["session_metadata"]
    assert wire["attributes"] == {"customer": "acme"}
    assert wire["spans"] == [{"a": 1}]

    # And back from a dict, which is how the server reads a posted body.
    again = DecisionTrace.model_validate(
        {"agent_id": "a", "session_id": "s", "session_metadata": wire}
    )
    assert again.session_metadata is not None
    assert again.session_metadata.model_dump()["attributes"] == {"customer": "acme"}


# ---------------------------------------------------------------------------
# validate_session_attributes
# ---------------------------------------------------------------------------


def test_validate_attributes_accepts_scalars_and_lowercases_keys() -> None:
    out = validate_session_attributes({" Customer ": "acme", "retries": 3, "ok": True, "p": 0.5})
    assert out == {"customer": "acme", "retries": 3, "ok": True, "p": 0.5}


@pytest.mark.parametrize(
    "bad, exc",
    [
        ({"customer": ["acme"]}, TypeError),
        ({"customer": {"id": 1}}, TypeError),
        ({"customer": None}, TypeError),
        ({"": "x"}, ValueError),
        ({"k" * 65: "x"}, ValueError),
        ({"customer": "x" * 257}, ValueError),
        ({f"k{i}": i for i in range(21)}, ValueError),
        ({"nan": float("nan")}, ValueError),
        ("not-a-dict", TypeError),
    ],
)
def test_validate_attributes_rejects_bad_input(bad, exc) -> None:
    with pytest.raises(exc):
        validate_session_attributes(bad)


def test_validate_attributes_none_is_empty() -> None:
    assert validate_session_attributes(None) == {}


# ---------------------------------------------------------------------------
# set_session_attributes / record_llm_call
# ---------------------------------------------------------------------------


def test_set_session_attributes_merges_and_returns_bag(mem) -> None:
    assert mem.set_session_attributes({"customer": "acme"}) == {"customer": "acme"}
    bag = mem.set_session_attributes({"task_type": "deploy", "customer": "globex"})
    assert bag == {"customer": "globex", "task_type": "deploy"}
    assert mem.session_attributes == bag


def test_set_session_attributes_raises_and_leaves_bag_untouched(mem) -> None:
    mem.set_session_attributes({"customer": "acme"})
    with pytest.raises(TypeError):
        mem.set_session_attributes({"bad": object()})
    assert mem.session_attributes == {"customer": "acme"}


def test_record_llm_call_writes_the_llm_call_shape(mem) -> None:
    rec = mem.record_llm_call(
        "gpt-4o", 120, 30, cost_usd=0.0006, latency_ms=412.5, provider="openai", call_id="c1",
    )
    assert rec == {
        "call_id": "c1",
        "model": "gpt-4o",
        "provider": "openai",
        "input_tokens": 120,
        "output_tokens": 30,
        "cost_usd": 0.0006,
        "latency_ms": 412.5,
        "started_at": rec["started_at"],
    }
    assert isinstance(rec["started_at"], str) and rec["started_at"].endswith("+00:00")
    assert mem.session_llm_calls == [rec]


def test_record_llm_call_defaults(mem) -> None:
    rec = mem.record_llm_call("claude-3.5-sonnet", 10, 5)
    assert rec["provider"] == ""
    assert rec["cost_usd"] is None
    assert rec["latency_ms"] is None
    assert rec["call_id"]  # generated


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (dict(model="", input_tokens=1, output_tokens=1), ValueError),
        (dict(model="m", input_tokens="x", output_tokens=1), TypeError),
        (dict(model="m", input_tokens=-1, output_tokens=1), ValueError),
        (dict(model="m", input_tokens=1, output_tokens=1, cost_usd=-0.1), ValueError),
        (dict(model="m", input_tokens=1, output_tokens=1, latency_ms="fast"), ValueError),
    ],
)
def test_record_llm_call_validates(mem, kwargs, exc) -> None:
    with pytest.raises(exc):
        mem.record_llm_call(**kwargs)
    assert mem.session_llm_calls == []


# ---------------------------------------------------------------------------
# commit_outcome merges everything into session_metadata
# ---------------------------------------------------------------------------


def test_commit_outcome_merges_attributes_and_llm_calls_into_session_metadata(mem) -> None:
    mem.session_metadata = SessionMetadata(model="claude-4-opus")
    mem.set_session_attributes({"customer": "acme", "task_type": "deploy"})
    mem.record_llm_call("gpt-4o", 100, 20, cost_usd=0.001, provider="openai")

    mem.commit_outcome(
        "deploy-42",
        OutcomeType.SUCCESS,
        attributes={"task_type": "rollback", "region": "eu"},
        llm_calls=[{"model": "gpt-4o-mini", "input_tokens": 5, "output_tokens": 2}],
    )

    trace = mem._last_trace
    meta = trace.session_metadata
    assert isinstance(meta, SessionMetadata)
    assert meta.model == "claude-4-opus"  # base fields kept
    dumped = meta.model_dump(mode="json")
    # commit_outcome(attributes=) wins over set_session_attributes.
    assert dumped[SESSION_ATTRIBUTES_KEY] == {
        "customer": "acme", "task_type": "rollback", "region": "eu",
    }
    calls = dumped[SESSION_LLM_CALLS_KEY]
    assert [c["model"] for c in calls] == ["gpt-4o", "gpt-4o-mini"]
    # The explicit call was normalised into the same shape as the recorded one.
    assert set(calls[1]) >= {
        "call_id", "model", "provider", "input_tokens", "output_tokens",
        "cost_usd", "latency_ms", "started_at",
    }
    assert calls[1]["provider"] == ""

    # Session timing is on the trace regardless of adapter.
    assert trace.session_started_at is not None
    assert trace.session_ended_at is not None
    assert trace.session_duration_ms is not None and trace.session_duration_ms >= 0

    # Buffers reset for the next window.
    assert mem.session_attributes == {}
    assert mem.session_llm_calls == []
    # ... but the session's own metadata instance was not mutated.
    assert "attributes" not in mem.session_metadata.model_dump()


def test_commit_outcome_with_nothing_extra_leaves_metadata_none(mem) -> None:
    mem.commit_outcome("t-1", OutcomeType.SUCCESS)
    assert mem._last_trace.session_metadata is None


def test_commit_outcome_preserves_extras_already_on_the_metadata(mem) -> None:
    """A Pro recorder puts its span tree on the metadata as an extra key; the
    merge must carry it, and merge into — not replace — an attribute bag it set."""
    meta = SessionMetadata()
    meta.spans = [{"span_id": "s1", "name": "root", "kind": "session"}]
    meta.attributes = {"agent_kind": "deploy"}
    mem.session_metadata = meta
    mem.set_session_attributes({"customer": "acme"})

    mem.commit_outcome("t-2", OutcomeType.SUCCESS)

    dumped = mem._last_trace.session_metadata.model_dump(mode="json")
    assert dumped["spans"] == [{"span_id": "s1", "name": "root", "kind": "session"}]
    assert dumped["attributes"] == {"agent_kind": "deploy", "customer": "acme"}


def test_commit_outcome_rejects_bad_attributes_before_writing(mem) -> None:
    with pytest.raises(TypeError):
        mem.commit_outcome("t-3", OutcomeType.SUCCESS, attributes={"x": object()})
    assert getattr(mem, "_last_trace", None) is None


def test_commit_outcome_skips_unusable_explicit_llm_calls(mem) -> None:
    mem.commit_outcome(
        "t-4", OutcomeType.SUCCESS,
        llm_calls=["nope", {"foo": "bar"}, {"model": "m", "input_tokens": "x"}],
    )
    assert mem._last_trace.session_metadata is None


def test_the_shape_survives_the_http_adapters_wire_form(mem) -> None:
    """``HttpAdapter.save_trace`` posts ``trace.model_dump(mode="json")``; the
    extras must be in it — a subclass instance would have lost them here."""
    mem.set_session_attributes({"customer": "acme"})
    mem.record_llm_call("gpt-4o", 1, 1, provider="openai")
    mem.commit_outcome("t-5", OutcomeType.SUCCESS)

    wire = mem._last_trace.model_dump(mode="json")
    assert wire["session_metadata"]["attributes"] == {"customer": "acme"}
    assert wire["session_metadata"]["llm_calls"][0]["model"] == "gpt-4o"
    assert wire["session_started_at"] and wire["session_ended_at"]
    assert wire["session_duration_ms"] is not None


# ---------------------------------------------------------------------------
# The HTTP server relays a remote client's session_metadata
# ---------------------------------------------------------------------------


def test_outcomes_endpoint_passes_client_session_metadata_to_commit_outcome() -> None:
    """The TypeScript SDK commits through ``POST /api/v1/outcomes``; the trace is
    built server-side from the shared handle, so the client's attributes and
    LLM calls only reach it if the endpoint hands them to ``commit_outcome``."""
    from amfs_http import server as http_server
    from amfs_http.models import OutcomeRequest

    seen: dict[str, object] = {}

    def _commit(outcome_ref, otype, **kwargs):
        seen.update(kwargs, outcome_ref=outcome_ref)
        return []

    mem = SimpleNamespace(
        commit_outcome=_commit,
        _tagger=SimpleNamespace(agent_id="server"),
        _adapter=SimpleNamespace(ensure_agent=lambda *a, **k: None),
        namespace="default",
        _last_trace=None,
    )
    originals = (
        http_server._get_memory, http_server._link_agent_owner_once,
        http_server._audit_log, http_server._auto_seal_trace,
    )
    http_server._get_memory = lambda: mem  # type: ignore[assignment]
    http_server._link_agent_owner_once = lambda *a, **k: None  # type: ignore[assignment]
    http_server._audit_log = lambda *a, **k: None  # type: ignore[assignment]
    http_server._auto_seal_trace = lambda *a, **k: None  # type: ignore[assignment]
    try:
        req = OutcomeRequest(
            outcome_ref="ts-1",
            outcome_type="success",
            agent_id="ts-agent",
            session_metadata={
                "attributes": {"Customer": "acme"},
                "llm_calls": [{"model": "gpt-4o", "input_tokens": 3, "output_tokens": 1}],
            },
        )
        request = SimpleNamespace(client=None)
        asyncio.run(http_server.commit_outcome(req, request, None))
    finally:
        (
            http_server._get_memory, http_server._link_agent_owner_once,
            http_server._audit_log, http_server._auto_seal_trace,
        ) = originals  # type: ignore[assignment]

    assert seen["attributes"] == {"customer": "acme"}
    assert seen["llm_calls"] == [{"model": "gpt-4o", "input_tokens": 3, "output_tokens": 1}]


def test_outcomes_endpoint_rejects_bad_client_attributes_with_422() -> None:
    from amfs_http import server as http_server
    from amfs_http.models import OutcomeRequest
    from fastapi import HTTPException

    mem = SimpleNamespace(
        commit_outcome=lambda *a, **k: pytest.fail("must not commit"),
        _tagger=SimpleNamespace(agent_id="server"),
        _adapter=SimpleNamespace(ensure_agent=lambda *a, **k: None),
        namespace="default",
    )
    original = http_server._get_memory
    http_server._get_memory = lambda: mem  # type: ignore[assignment]
    try:
        req = OutcomeRequest(
            outcome_ref="ts-2", outcome_type="success",
            session_metadata={"attributes": {"customer": ["acme"]}},
        )
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(http_server.commit_outcome(req, SimpleNamespace(client=None), None))
    finally:
        http_server._get_memory = original  # type: ignore[assignment]
    assert exc_info.value.status_code == 422
