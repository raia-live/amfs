"""The SDK's room methods reach the rooms API through the HTTP adapter.

They never had. ``_room_request`` looked for ``_base_url`` and ``_session`` on
the adapter — attributes ``HttpAdapter`` does not have — so every room and
negotiation method raised ``NotImplementedError("... set AMFS_HTTP_URL")``
against the very adapter that message asks for, and nothing here called them.

Each test builds a real ``HttpAdapter`` on a mock transport and asserts what
went over the wire: the method, the path, the body, the query string, and the
headers that make the request the caller's (API key, agent, session).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from amfs import AgentMemory
from amfs_adapter_http import AGENT_ID_HEADER, SESSION_HEADER, HttpAdapter

ROOM = "11111111-2222-3333-4444-555555555555"
NEG = "99999999-8888-7777-6666-555555555555"


def _memory(
    seen: list[httpx.Request],
    responses: dict[tuple[str, str], Any] | None = None,
    *,
    agent_id: str = "agent-a",
) -> AgentMemory:
    """An ``AgentMemory`` on a real ``HttpAdapter`` whose transport records
    every request and answers from *responses* (``(method, path) -> json`` or
    a ready ``httpx.Response``), ``{}`` otherwise."""

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = (responses or {}).get((request.method, request.url.path), {})
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    adapter = HttpAdapter("http://test", "key-a")
    adapter._client = httpx.Client(
        base_url="http://test",
        headers={"X-AMFS-API-Key": "key-a"},
        transport=httpx.MockTransport(_handler),
    )
    return AgentMemory(agent_id, session_id="sess-1", adapter=adapter)


def _body(request: httpx.Request) -> Any:
    return json.loads(request.content) if request.content else None


def _only(seen: list[httpx.Request]) -> httpx.Request:
    assert len(seen) == 1, [f"{r.method} {r.url}" for r in seen]
    return seen[0]


class TestTransport:
    def test_a_room_call_no_longer_raises_not_implemented(self) -> None:
        """The regression itself: with an ``HttpAdapter`` this used to raise."""
        seen: list[httpx.Request] = []
        _memory(seen, {("GET", "/api/v1/rooms"): []}).my_rooms()
        assert _only(seen).url.path == "/api/v1/rooms"

    def test_a_room_call_carries_the_key_and_the_callers_identity(self) -> None:
        """Same connection as every other hosted call, so the same headers: a
        room route authorises on the key and attributes on the agent."""
        seen: list[httpx.Request] = []
        _memory(seen).room_info(ROOM)
        request = _only(seen)
        assert request.headers["X-AMFS-API-Key"] == "key-a"
        assert request.headers[AGENT_ID_HEADER] == "agent-a"
        assert request.headers[SESSION_HEADER] == "sess-1"

    def test_as_agent_sends_that_agents_identity(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).as_agent("agent-b").room_join(ROOM)
        request = _only(seen)
        assert request.headers[AGENT_ID_HEADER] == "agent-b"
        assert _body(request) == {"agent_id": "agent-b"}

    def test_a_refusal_carries_the_servers_detail(self) -> None:
        """A 402 or 409 from a room route explains itself in ``detail``; the
        adapter's error path is what gets that text to the caller."""
        seen: list[httpx.Request] = []
        mem = _memory(seen, {
            ("POST", "/api/v1/rooms"): httpx.Response(
                402, json={"detail": {"error": "room_limit_reached", "limit": 1}},
            ),
        })
        with pytest.raises(httpx.HTTPStatusError) as raised:
            mem.room_create("acme/billing")
        assert raised.value.response.status_code == 402
        assert "room_limit_reached" in str(raised.value)

    def test_a_local_adapter_says_what_to_do(self, tmp_path) -> None:
        """Not the hosted API, so no rooms — and the message names the way
        to get them rather than an environment variable nothing here reads."""
        from amfs_filesystem import FilesystemAdapter

        mem = AgentMemory("agent-a", adapter=FilesystemAdapter(root=tmp_path, namespace="default"))
        with pytest.raises(NotImplementedError) as raised:
            mem.my_rooms()
        message = str(raised.value)
        assert "from_env" in message
        assert "HttpAdapter" in message
        assert "FilesystemAdapter" in message


class TestLifecycle:
    def test_room_create(self) -> None:
        seen: list[httpx.Request] = []
        mem = _memory(seen, {("POST", "/api/v1/rooms"): {"id": ROOM}})
        room = mem.room_create(
            ["acme/billing", "acme/refunds"], name="Billing", description="Shared rules",
        )
        request = _only(seen)
        assert (request.method, request.url.path) == ("POST", "/api/v1/rooms")
        assert _body(request) == {
            "entity_paths": ["acme/billing", "acme/refunds"],
            "display_name": "Billing",
            "description": "Shared rules",
            "discussions_enabled": True,
            "agent_id": "agent-a",
        }
        assert room == {"id": ROOM}

    def test_room_create_takes_one_topic_as_a_string(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_create("acme/billing", discussions=False)
        body = _body(_only(seen))
        assert body["entity_paths"] == ["acme/billing"]
        assert body["discussions_enabled"] is False

    def test_room_create_without_a_topic_is_refused_before_any_request(self) -> None:
        seen: list[httpx.Request] = []
        with pytest.raises(ValueError):
            _memory(seen).room_create([" ", ""])
        assert seen == []

    def test_room_share(self) -> None:
        seen: list[httpx.Request] = []
        mem = _memory(seen, {("POST", f"/api/v1/rooms/{ROOM}/join-links"): {"token": "tok"}})
        link = mem.room_share(ROOM, expires_in_days=1, max_uses=2, role="viewer")
        request = _only(seen)
        assert (request.method, request.url.path) == ("POST", f"/api/v1/rooms/{ROOM}/join-links")
        assert _body(request) == {
            "expires_in_days": 1,
            "max_uses": 2,
            "role": "viewer",
            "clamp_max_uses_to_plan": True,
        }
        assert link == {"token": "tok"}

    def test_room_share_can_ask_for_the_refusal_instead_of_the_clamp(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_share(ROOM, clamp_to_plan=False)
        assert _body(_only(seen)) == {
            "expires_in_days": None,
            "max_uses": None,
            "role": "collaborator",
            "clamp_max_uses_to_plan": False,
        }

    @pytest.mark.parametrize(
        "pasted",
        [
            "tok_abc123",
            "https://amfs.sense-lab.ai/join/tok_abc123",
            "<https://amfs.sense-lab.ai/join/tok_abc123?utm_source=email>.",
            "  https://amfs.sense-lab.ai/j/tok_abc123/  ",
            # Wrapping in any order: the bracket behind the full stop is the
            # ordinary way a link ends a sentence in an email.
            "<https://amfs.sense-lab.ai/join/tok_abc123>.",
            "<https://amfs.sense-lab.ai/join/tok_abc123>,",
            "(https://amfs.sense-lab.ai/join/tok_abc123).",
            "[the room](https://amfs.sense-lab.ai/join/tok_abc123);",
            '"https://amfs.sense-lab.ai/join/tok_abc123".',
            "<tok_abc123>.",
            "https://amfs.sense-lab.ai/join/tok_abc123 .",
        ],
    )
    def test_room_redeem_takes_the_token_or_the_whole_link(self, pasted: str) -> None:
        """The link is what people are sent. The route sits outside
        ``/rooms/{id}`` on purpose: the caller is not a member yet."""
        seen: list[httpx.Request] = []
        mem = _memory(seen, {("POST", "/api/v1/room-join/tok_abc123"): {"room_id": ROOM}})
        assert mem.room_redeem(pasted) == {"room_id": ROOM}
        request = _only(seen)
        assert (request.method, request.url.path) == ("POST", "/api/v1/room-join/tok_abc123")

    @pytest.mark.parametrize("token", ["tok_abc-", "tok_abc_", "-tok_abc", "_tok-abc_"])
    def test_room_redeem_keeps_a_token_that_starts_or_ends_in_a_dash(self, token: str) -> None:
        """Join tokens are URL-safe base64, so ``-`` and ``_`` at either end are
        part of the token and not wrapping to be trimmed."""
        seen: list[httpx.Request] = []
        mem = _memory(seen, {("POST", f"/api/v1/room-join/{token}"): {"room_id": ROOM}})
        mem.room_redeem(f"<https://amfs.sense-lab.ai/join/{token}>.")
        assert _only(seen).url.path == f"/api/v1/room-join/{token}"

    def test_room_redeem_without_a_token_is_refused_before_any_request(self) -> None:
        seen: list[httpx.Request] = []
        with pytest.raises(ValueError):
            _memory(seen).room_redeem("  ")
        assert seen == []

    def test_room_join(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_join(ROOM)
        request = _only(seen)
        assert (request.method, request.url.path) == ("POST", f"/api/v1/rooms/{ROOM}/agents")
        assert _body(request) == {"agent_id": "agent-a"}

    def test_room_leave(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_leave(ROOM)
        request = _only(seen)
        assert (request.method, request.url.path) == (
            "DELETE", f"/api/v1/rooms/{ROOM}/agents/agent-a",
        )
        assert _body(request) is None

    def test_room_leave_escapes_the_agent_id(self) -> None:
        """An agent id is free text; one with a ``?`` in it must stay in the
        path instead of becoming a query string."""
        seen: list[httpx.Request] = []
        _memory(seen, agent_id="agent a?x").room_leave(ROOM)
        request = _only(seen)
        assert request.url.raw_path == f"/api/v1/rooms/{ROOM}/agents/agent%20a%3Fx".encode()
        assert request.url.query == b""

    def test_my_rooms_returns_the_list(self) -> None:
        seen: list[httpx.Request] = []
        rooms = [{"id": ROOM, "role": "owner"}]
        assert _memory(seen, {("GET", "/api/v1/rooms"): rooms}).my_rooms() == rooms
        assert _only(seen).method == "GET"

    def test_my_rooms_unwraps_an_envelope(self) -> None:
        seen: list[httpx.Request] = []
        rooms = [{"id": ROOM}]
        assert _memory(seen, {("GET", "/api/v1/rooms"): {"rooms": rooms}}).my_rooms() == rooms

    def test_room_info(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_info(ROOM)
        request = _only(seen)
        assert (request.method, request.url.path) == ("GET", f"/api/v1/rooms/{ROOM}")

    def test_room_briefing(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_briefing(ROOM)
        request = _only(seen)
        assert (request.method, request.url.path) == ("POST", f"/api/v1/rooms/{ROOM}/briefing")
        assert _body(request) == {"agent_id": "agent-a"}

    def test_room_updates_without_since_sends_no_query(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_updates(ROOM)
        request = _only(seen)
        assert (request.method, request.url.path) == ("GET", f"/api/v1/rooms/{ROOM}/activity")
        assert request.url.query == b""

    def test_room_updates_sends_since_intact(self) -> None:
        """``since`` used to be pasted into the URL unescaped, where the ``+``
        of a UTC offset reads as a space and the server gets a timestamp it
        cannot parse. As a query parameter it arrives as written."""
        seen: list[httpx.Request] = []
        since = "2026-05-01T00:00:00+00:00"
        _memory(seen).room_updates(ROOM, since=since)
        request = _only(seen)
        assert request.url.params["since"] == since
        assert b"+" not in request.url.query

    def test_room_close(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_close(ROOM)
        request = _only(seen)
        assert (request.method, request.url.path) == ("POST", f"/api/v1/rooms/{ROOM}/close")

    def test_room_delete(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_delete(ROOM)
        request = _only(seen)
        assert (request.method, request.url.path) == ("DELETE", f"/api/v1/rooms/{ROOM}")
        assert _body(request) is None


class TestDiscussions:
    def test_room_discuss(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_discuss(
            ROOM, "Is the retry rule still right?", message_type="question",
            addressed_to="agent-b",
        )
        request = _only(seen)
        assert (request.method, request.url.path) == ("POST", f"/api/v1/rooms/{ROOM}/discussions")
        assert _body(request) == {
            "agent_id": "agent-a",
            "content": "Is the retry rule still right?",
            "message_type": "question",
            "addressed_to": "agent-b",
        }

    def test_room_discuss_can_answer_a_message(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_discuss(ROOM, "Yes.", message_type="answer", reply_to=NEG)
        assert _body(_only(seen))["reply_to"] == NEG

    def test_room_discussions_defaults(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).room_discussions(ROOM)
        request = _only(seen)
        assert (request.method, request.url.path) == ("GET", f"/api/v1/rooms/{ROOM}/discussions")
        assert dict(request.url.params) == {"limit": "50"}

    def test_room_discussions_filters(self) -> None:
        seen: list[httpx.Request] = []
        since = "2026-05-01T00:00:00+00:00"
        _memory(seen).room_discussions(ROOM, limit=20, since=since, addressed_to="agent-a")
        assert dict(_only(seen).url.params) == {
            "limit": "20", "since": since, "addressed_to": "agent-a",
        }


class TestNegotiations:
    def test_negotiate_create(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).negotiate_create(ROOM, "Retry budget", description="Pick one", max_rounds=4)
        request = _only(seen)
        assert (request.method, request.url.path) == ("POST", f"/api/v1/rooms/{ROOM}/negotiations")
        assert _body(request) == {
            "title": "Retry budget",
            "description": "Pick one",
            "agent_id": "agent-a",
            "max_rounds": 4,
        }

    def test_negotiate_propose(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).negotiate_propose(ROOM, NEG, proposal={"retries": 3}, rationale="p99")
        request = _only(seen)
        assert (request.method, request.url.path) == (
            "POST", f"/api/v1/rooms/{ROOM}/negotiations/{NEG}/propose",
        )
        assert _body(request) == {
            "agent_id": "agent-a",
            "action": "propose",
            "proposal": {"retries": 3},
            "rationale": "p99",
        }

    def test_negotiate_respond(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).negotiate_respond(ROOM, NEG, action="accept", rationale="agreed")
        request = _only(seen)
        assert request.url.path == f"/api/v1/rooms/{ROOM}/negotiations/{NEG}/propose"
        assert _body(request) == {
            "agent_id": "agent-a",
            "action": "accept",
            "proposal": {},
            "rationale": "agreed",
        }

    def test_negotiate_status(self) -> None:
        seen: list[httpx.Request] = []
        _memory(seen).negotiate_status(ROOM, NEG)
        request = _only(seen)
        assert (request.method, request.url.path) == (
            "GET", f"/api/v1/rooms/{ROOM}/negotiations/{NEG}",
        )
