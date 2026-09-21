"""The two hooks a layer in front of these routes uses to run a repair canary
on any client.

A hosted server decides, per request, whether the calling session sits in the
canary arm of a live repair canary. It has nowhere to put that decision except
``request.state``: the routes are OSS and the decision is not. So the routes
read two attributes from there — ``memory_branch`` moves reads that named no
branch onto the canary's branch, and ``trace_attributes`` stamps the sealed
trace with which fix and which arm — and behave exactly as before when neither
is set.

Three properties matter and each has a test: a named branch always wins over a
routed one (the tool inspecting the baseline must be able to say ``main``);
writes never follow the route (what the agent learns during a canary is the
user's, not the proposal's); and the server's stamps overwrite the body's
(a client cannot put itself in the canary arm).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

import amfs_http.server as server  # noqa: E402
from amfs_core.models import DecisionTrace, MemoryEntry, Provenance  # noqa: E402
from amfs_http.models import (  # noqa: E402
    AggregateRequest,
    OutcomeRequest,
    RetrieveRequest,
    SearchRequest,
    WriteRequest,
)

CANARY_BRANCH = "repair/fix-7a3c"
STAMPS = {"canary_fix_id": "7a3c", "canary_arm": "canary", "memory_branch": CANARY_BRANCH}


def _request(**state: Any) -> SimpleNamespace:
    """A request as a handler sees it in-process, with what the layer in front set."""
    return SimpleNamespace(
        headers={},
        client=SimpleNamespace(host="10.0.0.1"),
        state=SimpleNamespace(**state),
    )


def _entry(key: str = "k") -> MemoryEntry:
    return MemoryEntry(
        entity_path="acme/billing",
        key=key,
        value="v",
        provenance=Provenance(
            agent_id="writer", session_id="s", written_at=datetime.now(UTC)
        ),
    )


@pytest.fixture
def mem(monkeypatch) -> MagicMock:
    handle = MagicMock()
    handle.namespace = "test"
    handle._tagger = SimpleNamespace(agent_id="srv", session_id="sess")
    handle.read.return_value = _entry()
    handle.list.return_value = []
    handle.search.return_value = []
    monkeypatch.setattr(server, "_memory", handle)
    monkeypatch.setattr(server, "_get_memory", lambda: handle)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda _r: None)
    monkeypatch.setattr(server, "_active_visibility_filter", lambda _r: None)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(server, "_audit_log", lambda *a, **k: None)
    return handle


class TestEffectiveBranch:
    def test_nothing_set_is_main(self) -> None:
        assert server._effective_branch(_request(), None) == "main"
        assert server._effective_branch(None, None) == "main"

    def test_a_routed_session_reads_the_canary_branch(self) -> None:
        assert (
            server._effective_branch(_request(memory_branch=CANARY_BRANCH), None)
            == CANARY_BRANCH
        )

    def test_a_named_branch_wins_over_the_route(self) -> None:
        """``branch=main`` from a canary session still reads main."""
        req = _request(memory_branch=CANARY_BRANCH)
        assert server._effective_branch(req, "main") == "main"
        assert server._effective_branch(req, "other") == "other"

    def test_the_query_sentinel_is_not_a_branch(self) -> None:
        """A handler called in-process gets ``Query(None)`` as its default."""
        from fastapi import Query

        assert server._effective_branch(_request(), Query(None)) == "main"

    def test_blank_and_non_string_state_are_ignored(self) -> None:
        assert server._effective_branch(_request(memory_branch="  "), None) == "main"
        assert server._effective_branch(_request(memory_branch=7), None) == "main"


class TestReadsFollowTheRoute:
    def test_read_entry(self, mem) -> None:
        asyncio.run(
            server._read_entry(_request(memory_branch=CANARY_BRANCH), "acme/billing", "k", None)
        )
        assert mem.read.call_args.kwargs["branch"] == CANARY_BRANCH

    def test_read_entry_unrouted_is_main(self, mem) -> None:
        asyncio.run(server._read_entry(_request(), "acme/billing", "k", None))
        assert mem.read.call_args.kwargs["branch"] == "main"

    def test_read_entry_named_main_stays_main(self, mem) -> None:
        asyncio.run(
            server._read_entry(_request(memory_branch=CANARY_BRANCH), "acme/billing", "k", "main")
        )
        assert mem.read.call_args.kwargs["branch"] == "main"

    def test_list_entries(self, mem) -> None:
        asyncio.run(
            server.list_entries(
                _request(memory_branch=CANARY_BRANCH),
                entity_path="acme/billing",
                branch=None,
                include_superseded=False,
                limit=None,
                offset=0,
                sort=None,
                fields=None,
                _auth=None,
            )
        )
        assert mem.list.call_args.kwargs["branch"] == CANARY_BRANCH

    def test_aggregate(self, mem) -> None:
        asyncio.run(
            server.aggregate_entries_endpoint(
                _request(memory_branch=CANARY_BRANCH),
                AggregateRequest(entity_path="acme/billing"),
                None,
            )
        )
        assert mem.list.call_args.kwargs["branch"] == CANARY_BRANCH

    def test_search(self, mem, monkeypatch) -> None:
        seen: dict[str, Any] = {}

        def _search(sq, **kw):
            seen.update(kw)
            return []

        mem._adapter = SimpleNamespace(search=_search)
        asyncio.run(
            server.search_entries(
                _request(memory_branch=CANARY_BRANCH),
                SearchRequest(entity_path="acme/billing"),
                None,
                None,
            )
        )
        assert seen.get("branch") == CANARY_BRANCH

    def test_request_models_default_to_no_branch(self) -> None:
        """``None`` rather than ``"main"``: the server must be able to tell
        "did not say" from "said main", and the SDK omits the field for main."""
        assert SearchRequest().branch is None
        assert RetrieveRequest(query="q").branch is None
        assert AggregateRequest(entity_path="p").branch is None


class TestWritesStayOnMain:
    def test_a_routed_session_writes_main(self, mem) -> None:
        mem.write.return_value = _entry()
        asyncio.run(
            server.write_entry(
                WriteRequest(entity_path="acme/billing", key="k", value="v"),
                _request(memory_branch=CANARY_BRANCH),
                None,
            )
        )
        assert WriteRequest(entity_path="p", key="k", value="v").branch == "main"
        # Whatever path the write took, the branch it carried was main.
        for call in mem.write.call_args_list:
            assert call.kwargs.get("branch", "main") == "main"


class TestTheSealCarriesTheStamps:
    @pytest.fixture
    def committed(self, mem) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        def _commit(outcome_ref, outcome_type, **kwargs):
            seen.update(kwargs)
            return []

        mem.commit_outcome.side_effect = _commit
        mem.agent_id = "srv"
        mem._adapter = SimpleNamespace(ensure_agent=lambda *a, **k: None, save_trace=lambda t: t)
        return seen

    def _commit(self, request, attributes: dict[str, Any] | None = None) -> None:
        asyncio.run(
            server.commit_outcome(
                OutcomeRequest(
                    outcome_ref="deploy-1",
                    outcome_type="success",
                    session_metadata={"attributes": attributes} if attributes else None,
                ),
                request,
                None,
            )
        )

    def test_stamps_reach_commit_outcome(self, committed) -> None:
        self._commit(_request(trace_attributes=STAMPS), {"customer": "acme"})
        attrs = committed["attributes"]
        assert attrs["customer"] == "acme"
        assert attrs["canary_fix_id"] == "7a3c"
        assert attrs["canary_arm"] == "canary"
        assert attrs["memory_branch"] == CANARY_BRANCH

    def test_the_server_wins_over_the_body(self, committed) -> None:
        """A client cannot put itself in the canary arm."""
        self._commit(
            _request(trace_attributes={"canary_fix_id": "7a3c", "canary_arm": "control"}),
            {"canary_arm": "canary", "canary_fix_id": "forged"},
        )
        assert committed["attributes"]["canary_arm"] == "control"
        assert committed["attributes"]["canary_fix_id"] == "7a3c"

    def test_a_bag_at_the_cap_is_still_accepted(self, committed) -> None:
        """The stamps are the server's, so they do not count against the
        client's twenty keys."""
        full = {f"k{i}": i for i in range(20)}
        self._commit(_request(trace_attributes=STAMPS), full)
        assert len(committed["attributes"]) == 23

    def test_unrouted_is_untouched(self, committed) -> None:
        """No layer spoke for this request: the bag is the client's, as is."""
        self._commit(_request(), {"customer": "acme", "canary_arm": "canary"})
        assert committed["attributes"] == {"customer": "acme", "canary_arm": "canary"}

    def test_once_the_layer_has_spoken_a_clients_canary_claims_are_dropped(
        self, committed
    ) -> None:
        """``trace_attributes={}`` is "decided: not routed". A session nothing
        routed must not be able to vote in a canary it was not in."""
        self._commit(
            _request(trace_attributes={}),
            {"customer": "acme", "canary_fix_id": "7a3c", "canary_arm": "canary"},
        )
        assert committed["attributes"] == {"customer": "acme"}

    def test_no_attributes_and_no_route_stays_none(self, committed) -> None:
        self._commit(_request(), None)
        assert committed["attributes"] is None


class TestTheDeferredTraceCarriesTheStamps:
    """``POST /api/v1/traces`` is the other request a routed session's trace
    arrives on, and it must stamp the same way or SDK sessions are in no arm."""

    @pytest.fixture
    def posted(self, mem, monkeypatch) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        mem._adapter = SimpleNamespace(
            ensure_agent=lambda *a, **k: None,
            save_trace=lambda t: seen.setdefault("saved", t) and t,
        )

        def _seal(_mem, trace=None, *, session_metadata=None):
            seen["sealed_meta"] = session_metadata
            return None

        monkeypatch.setattr(server, "_auto_seal_trace", _seal)
        return seen

    def _post(self, request, body: dict[str, Any]) -> None:
        class _Req:
            client = request.client
            state = request.state
            headers = request.headers

            async def json(self):
                return body

        asyncio.run(server.save_trace(_Req(), None))

    def _body(self, attributes: dict[str, Any] | None) -> dict[str, Any]:
        trace = DecisionTrace(
            agent_id="sre-agent",
            session_id="client-session",
            outcome_ref="deploy-1",
            outcome_type="success",
        ).model_dump(mode="json")
        if attributes is not None:
            trace["session_metadata"] = {"attributes": attributes, "spans": [{"a": 1}]}
        return trace

    def test_stamps_land_on_the_saved_trace_and_the_sealed_metadata(self, posted) -> None:
        self._post(_request(trace_attributes=STAMPS), self._body({"customer": "acme"}))
        saved_attrs = posted["saved"].session_metadata.model_dump()["attributes"]
        assert saved_attrs == {"customer": "acme", **STAMPS}
        # The raw body is what the seal prefers, so it carries them too — and
        # still carries the keys the model does not declare.
        assert posted["sealed_meta"]["attributes"] == {"customer": "acme", **STAMPS}
        assert posted["sealed_meta"]["spans"] == [{"a": 1}]

    def test_a_trace_with_no_metadata_gets_some(self, posted) -> None:
        self._post(_request(trace_attributes=STAMPS), self._body(None))
        saved_attrs = posted["saved"].session_metadata.model_dump()["attributes"]
        assert saved_attrs == STAMPS
        # No raw metadata came in, so the seal falls back to the trace's own.
        assert posted["sealed_meta"] is None

    def test_once_the_layer_has_spoken_a_clients_canary_claims_are_dropped(
        self, posted
    ) -> None:
        """Same rule as /outcomes: ``trace_attributes={}`` is "decided: not
        routed", and this is the path every HttpAdapter commit seals on."""
        self._post(
            _request(trace_attributes={}),
            self._body({"customer": "acme", "canary_fix_id": "7a3c", "canary_arm": "canary"}),
        )
        assert posted["saved"].session_metadata.model_dump()["attributes"] == {"customer": "acme"}
        assert posted["sealed_meta"]["attributes"] == {"customer": "acme"}
        assert posted["sealed_meta"]["spans"] == [{"a": 1}]

    def test_the_server_wins_over_the_body(self, posted) -> None:
        self._post(
            _request(trace_attributes={"canary_fix_id": "7a3c", "canary_arm": "control"}),
            self._body({"canary_fix_id": "forged", "canary_arm": "canary"}),
        )
        attrs = posted["saved"].session_metadata.model_dump()["attributes"]
        assert attrs == {"canary_fix_id": "7a3c", "canary_arm": "control"}
        assert posted["sealed_meta"]["attributes"] == attrs

    def test_unrouted_is_untouched(self, posted) -> None:
        self._post(_request(), self._body({"customer": "acme"}))
        assert posted["saved"].session_metadata.model_dump()["attributes"] == {"customer": "acme"}
        assert posted["sealed_meta"] == {"attributes": {"customer": "acme"}, "spans": [{"a": 1}]}
