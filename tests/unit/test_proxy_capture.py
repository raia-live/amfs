"""Byte-faithful capture of proxied exchanges (t2-8e, OSS half).

Pure functions first — SSE reassembly, redaction, the payload's hashes, the
trace's shape — then the mounted route with a fake upstream, streaming and
not, with capture off (nothing held, nothing written) and on (one trace with
no outcome whose capture carries the request as sent and the response as
received).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import httpx
import pytest
from amfs_core.models import DecisionTrace
from amfs_http import capture as cap

# ---------------------------------------------------------------------------
# reassemble_sse
# ---------------------------------------------------------------------------


def _sse(*chunks: dict[str, Any], done: bool = True, extra_lines: tuple[str, ...] = ()) -> bytes:
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.extend(extra_lines)
    if done:
        lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def test_reassemble_sse_joins_content_and_keeps_head_and_usage():
    head = {
        "id": "gen-1",
        "model": "openai/gpt-4o-mini",
        "created": 1,
        "object": "chat.completion.chunk",
    }
    raw = _sse(
        {**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "The "}}]},
        {**head, "choices": [{"index": 0, "delta": {"content": "answer"}}]},
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**head, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    )
    out = cap.reassemble_sse([raw[:17], raw[17:]])  # split mid-line: joined before parsing
    assert out["id"] == "gen-1" and out["model"] == "openai/gpt-4o-mini"
    assert out["object"] == "chat.completion.chunk"
    assert out["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "The answer"},
            "finish_reason": "stop",
        }
    ]
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert out["stream"] is True and out["chunks"] == 4
    assert "unparsed" not in out


def test_reassemble_sse_merges_tool_call_fragments_by_index():
    raw = _sse(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "look", "arguments": ""},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q": '}}]},
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]},
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {"name": "send", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    out = cap.reassemble_sse([raw])
    msg = out["choices"][0]["message"]
    assert msg["content"] is None  # tool-call-only turn, as the non-streaming shape has it
    assert [tc["id"] for tc in msg["tool_calls"]] == ["call_a", "call_b"]
    assert msg["tool_calls"][0]["function"] == {"name": "look", "arguments": '{"q": "x"}'}
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert cap.tool_calls_of(out) == [
        {"tool_name": "look", "arguments": {"q": "x"}, "source": "capture"},
        {"tool_name": "send", "arguments": {}, "source": "capture"},
    ]


def test_reassemble_sse_keeps_non_json_lines_and_stops_at_done():
    raw = _sse(
        {"choices": [{"index": 0, "delta": {"content": "a"}}]},
        extra_lines=(": keep-alive", "data: not json", "event: x"),
    )
    raw += b'data: {"choices": [{"index": 0, "delta": {"content": "AFTER DONE"}}]}\n\n'
    out = cap.reassemble_sse([raw])
    assert out["choices"][0]["message"]["content"] == "a"
    assert out["unparsed"] == ["data: not json", "event: x"]  # comments dropped, the rest kept
    assert out["chunks"] == 1


# ---------------------------------------------------------------------------
# capture_payload / capture_trace
# ---------------------------------------------------------------------------

REQ = {
    "model": "openai/gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "token sk-live-SECRET please"}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"key": "sk-live-SECRET"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok sk-live-SECRET"},
        {"role": "user", "content": "and now?"},
    ],
    "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    "temperature": 0.2,
    "stream": False,
    "amfs_scope": "acme/support",  # not a request key: must not be captured
    "api_key_in_body": "sk-live-SECRET",  # unknown key: must not be captured
}
RESP = {
    "id": "gen-9",
    "object": "chat.completion",
    "model": "openai/gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "use sk-live-SECRET"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
}


def _scanner(text: str) -> str | None:
    if text == "ok sk-live-SECRET":
        return None  # the gate blocks this one outright
    return text.replace("sk-live-SECRET", "***")


def test_capture_payload_redacts_every_text_field_and_hashes_raw_bytes():
    raw_req = json.dumps(REQ).encode()
    raw_resp = json.dumps(RESP).encode()
    payload = cap.capture_payload(
        REQ,
        RESP,
        raw_request=raw_req,
        raw_response=raw_resp,
        upstream_status=200,
        scanner=_scanner,
        scope="acme/support",
        injected_facts=2,
    )
    # only the request keys that shape the exchange; ``stream`` is recorded as
    # ``streamed``, the amfs_* routing keys and unknown keys are not captured
    assert set(payload["request"]) == {"model", "messages", "tools", "temperature"}
    msgs = payload["request"]["messages"]
    assert msgs[1]["content"][0]["text"] == "token *** please"
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"key": "***"}'
    assert msgs[3]["content"] == cap.REDACTED
    assert payload["response"]["choices"][0]["message"]["content"] == "use ***"
    assert payload["response"]["usage"] == RESP["usage"]
    # hashes are of the raw bytes before redaction, so the holder of the
    # original can verify the capture is of that exchange
    assert payload["request_sha256"] == hashlib.sha256(raw_req).hexdigest()
    assert payload["response_sha256"] == hashlib.sha256(raw_resp).hexdigest()
    assert payload["request_bytes"] == len(raw_req) and payload["response_bytes"] == len(raw_resp)
    assert payload["capture_version"] == cap.CAPTURE_VERSION
    assert payload["source"] == cap.CAPTURE_SOURCE_OPENROUTER
    assert payload["scope"] == "acme/support" and payload["injected_facts"] == 2
    assert payload["upstream_status"] == 200 and payload["streamed"] is False
    # the inputs were not mutated
    assert REQ["messages"][1]["content"][0]["text"] == "token sk-live-SECRET please"
    assert RESP["choices"][0]["message"]["content"] == "use sk-live-SECRET"
    # and nothing secret survives anywhere in the payload
    assert "sk-live-SECRET" not in json.dumps(payload)


def test_capture_redacts_reasoning_tool_schemas_and_unknown_fields():
    """Every string leaf is scanned, not a list of known text fields: the
    reasoning a stream reassembles onto the message, the tool schemas the
    request carries, and provider fields this module has never seen."""
    req = {
        **REQ,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "default token sk-live-SECRET for the sandbox",
                    "parameters": {"properties": {"token": {"default": "sk-live-SECRET"}}},
                },
            }
        ],
    }
    resp = {
        **RESP,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "reasoning": "the user's token is sk-live-SECRET so I will",
                    "provider_extension": {"echo": "sk-live-SECRET"},
                },
            }
        ],
    }
    payload = cap.capture_payload(req, resp, scanner=_scanner)
    tool = payload["request"]["tools"][0]["function"]
    assert tool["description"] == "default token *** for the sandbox"
    assert tool["parameters"]["properties"]["token"]["default"] == "***"
    message = payload["response"]["choices"][0]["message"]
    assert message["reasoning"] == "the user's token is *** so I will"
    assert message["provider_extension"]["echo"] == "***"
    # the untouched fields keep their shape and type
    assert payload["request"]["temperature"] == REQ["temperature"]
    assert payload["response"]["usage"] == RESP["usage"]


def test_capture_trace_is_a_decision_trace_with_no_outcome():
    payload = cap.capture_payload(REQ, RESP, scanner=_scanner, scope="acme/support")
    trace = cap.capture_trace(
        payload, agent_id="support-bot", namespace="acct-1", scope="acme/support"
    )
    assert isinstance(trace, DecisionTrace)
    assert trace.outcome_ref is None and trace.outcome_type is None
    assert trace.agent_id == "support-bot" and trace.namespace == "acct-1"
    assert trace.task_input == "and now?"
    assert trace.response_text == "use ***"
    assert trace.tool_calls == []  # the response made no tool calls
    assert trace.session_metadata is not None
    assert trace.session_metadata.attributes == {
        cap.CAPTURE_ATTRIBUTE: cap.CAPTURE_SOURCE_OPENROUTER,
        "source": "proxy",
    }
    dumped = trace.session_metadata.model_dump()
    assert dumped[cap.CAPTURE_KEY]["request"]["messages"] == payload["request"]["messages"]
    # round-trips through the model: the capture is not an undeclared key
    again = DecisionTrace.model_validate(trace.model_dump())
    assert again.session_metadata.model_dump()[cap.CAPTURE_KEY]["request"] == payload["request"]


def test_capture_enabled_is_off_by_default_and_honours_env_and_opt_in(monkeypatch):
    monkeypatch.delenv(cap.CAPTURE_ENV, raising=False)
    cap.set_opt_in(None)
    assert cap.capture_enabled(object()) is False
    monkeypatch.setenv(cap.CAPTURE_ENV, "true")
    assert cap.capture_enabled(None) is True
    monkeypatch.setenv(cap.CAPTURE_ENV, "false")
    cap.set_opt_in(lambda req: getattr(req, "account", None) == "pro")
    assert cap.capture_enabled(type("R", (), {"account": "pro"})()) is True
    assert cap.capture_enabled(type("R", (), {"account": "free"})()) is False
    assert cap.capture_enabled(None) is False  # no request to resolve

    def _boom(req):
        raise RuntimeError("resolver down")

    cap.set_opt_in(_boom)
    assert cap.capture_enabled(object()) is False  # an opt-in that errors is an opt-out
    cap.set_opt_in(None)


# ---------------------------------------------------------------------------
# The mounted route
# ---------------------------------------------------------------------------


class _Adapter:
    def __init__(self):
        self.saved: list[DecisionTrace] = []

    def save_trace(self, trace):
        self.saved.append(trace)
        return trace


class _Memory:
    namespace = "acct-7"

    def __init__(self):
        self._adapter = _Adapter()


def _upstream(seen: list[httpx.Request], *, stream: bool):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        if stream:
            body = _sse(
                {
                    "id": "gen-s",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "str"}}],
                },
                {
                    "id": "gen-s",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "eamed sk-live-SECRET"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
            # a real stream, so the route's ``aiter_raw`` has bytes to iterate
            return httpx.Response(
                200, stream=httpx.ByteStream(body), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=RESP)

    return handler


@pytest.fixture
def mounted(monkeypatch):
    """The route mounted on a bare app, upstream replaced by a mock transport,
    retrieval returning one fact, the secret scanner a deterministic redactor."""
    import importlib

    import amfs_core.capture as core_capture
    from amfs_http import openrouter_proxy as proxy
    from fastapi import FastAPI

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv(cap.CAPTURE_ENV, raising=False)
    cap.set_opt_in(None)
    importlib.reload(proxy)
    monkeypatch.setattr(proxy, "_retrieve_facts", lambda get_memory, scope, query: ["fact one"])
    monkeypatch.setattr(
        core_capture,
        "scan_captured_text",
        lambda text, **kw: text.replace("sk-live-SECRET", "[redacted]"),
    )
    mem = _Memory()
    app = FastAPI()
    proxy.mount_openrouter_proxy(app, get_memory=lambda: mem)
    seen: list[httpx.Request] = []

    def with_upstream(stream: bool):
        monkeypatch.setattr(
            proxy,
            "_upstream_client",
            httpx.AsyncClient(transport=httpx.MockTransport(_upstream(seen, stream=stream))),
        )

    yield app, mem, seen, with_upstream
    cap.set_opt_in(None)


async def _call(app, body: dict[str, Any]):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post(
            "/api/v1/proxy/chat/completions", json=body, headers={"X-AMFS-Agent-Id": "support-bot"}
        )
        content = await resp.aread()
        # the capture is a thread kicked off by create_task: let it finish
        for _ in range(100):
            pending = [
                t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
            ]
            if not pending:
                break
            await asyncio.sleep(0.02)
        return resp.status_code, content


@pytest.mark.parametrize("stream", [False, True])
def test_route_with_capture_off_forwards_and_writes_nothing(mounted, stream):
    app, mem, seen, with_upstream = mounted
    with_upstream(stream)
    status, content = asyncio.run(_call(app, {**REQ, "stream": stream}))
    assert status == 200
    assert mem._adapter.saved == []
    assert len(seen) == 1
    forwarded = json.loads(seen[0].content)
    assert "amfs_scope" not in forwarded and "amfs_agent_id" not in forwarded
    assert (
        forwarded["messages"][0]["role"] == "system"
        and "fact one" in forwarded["messages"][0]["content"]
    )
    assert len(forwarded["messages"]) == len(REQ["messages"]) + 1
    if stream:
        assert b"eamed sk-live-SECRET" in content  # pass-through, untouched
    else:
        assert json.loads(content) == RESP


@pytest.mark.parametrize("stream", [False, True])
def test_route_with_capture_on_saves_one_trace_with_no_outcome(mounted, monkeypatch, stream):
    app, mem, seen, with_upstream = mounted
    with_upstream(stream)
    monkeypatch.setenv(cap.CAPTURE_ENV, "true")
    status, content = asyncio.run(_call(app, {**REQ, "stream": stream}))
    assert status == 200
    # the caller got exactly the upstream bytes: redaction is for the store
    if stream:
        assert b"eamed sk-live-SECRET" in content
    else:
        assert json.loads(content) == RESP
    assert len(mem._adapter.saved) == 1
    trace = mem._adapter.saved[0]
    assert isinstance(trace, DecisionTrace)
    assert trace.outcome_ref is None and trace.outcome_type is None
    assert trace.agent_id == "support-bot" and trace.namespace == "acct-7"
    captured = trace.session_metadata.model_dump()[cap.CAPTURE_KEY]
    # the request as the caller sent it: no injected system turn, no amfs_* keys
    assert captured["request"]["messages"][0] == REQ["messages"][0]
    assert len(captured["request"]["messages"]) == len(REQ["messages"])
    assert "amfs_scope" not in captured["request"]
    assert captured["injected_facts"] == 1 and captured["scope"] == "acme/support"
    assert captured["streamed"] is stream and captured["upstream_status"] == 200
    expected = "streamed [redacted]" if stream else "use [redacted]"
    assert captured["response"]["choices"][0]["message"]["content"] == expected
    assert trace.response_text == expected
    if stream:
        assert captured["response"]["chunks"] == 2 and captured["response"]["id"] == "gen-s"
    assert captured["response_bytes"] > 0 and len(captured["response_sha256"]) == 64
    assert captured["request_bytes"] > 0 and len(captured["request_sha256"]) == 64
    assert trace.task_input == "and now?"
    assert "sk-live-SECRET" not in json.dumps(trace.model_dump(mode="json"))


def test_route_capture_survives_a_failing_store(mounted, monkeypatch):
    app, mem, seen, with_upstream = mounted
    with_upstream(False)
    monkeypatch.setenv(cap.CAPTURE_ENV, "true")

    def _boom(trace):
        raise RuntimeError("store down")

    mem._adapter.save_trace = _boom
    status, content = asyncio.run(_call(app, {**REQ, "stream": False}))
    assert status == 200 and json.loads(content) == RESP  # fail-open: the caller never notices
