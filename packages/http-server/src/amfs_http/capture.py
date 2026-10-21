"""Byte-faithful capture of a proxied chat completion (t2-8e, OSS half).

The memory proxy forwards an OpenAI-shaped request to a frontier model and
hands the answer back. What training needs from that exchange is the whole of
it — the request body as sent (model, every message in the history, the tool
schemas, the sampling options), and the response as received, reassembled
from the SSE deltas when it streamed — kept as it was on the wire rather than
as a paraphrase, and tied to the bytes by hash so a reader can tell a faithful
copy from a reconstruction.

This module is the data shape and the capture surface. It does not select,
render or train — that is the Pro side's — and it is **off by default**:
``capture_enabled(request)`` is true only when ``AMFS_PROXY_CAPTURE=true`` or
when the hosting layer has installed an opt-in resolver (``set_opt_in``) that
says so for this request's account. Captured text goes through the secret
scanner before anything is persisted; a blocked message is replaced by a
marker, never written.

The payload travels as ``session_metadata["capture"]`` on a
:class:`~amfs_core.models.DecisionTrace` with no outcome — a capture is not
evidence and moves no confidence — persisted through the adapter's
``save_trace`` and sealed by whatever seal path the server has. Retention is
the trace store's.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

#: ``session_metadata`` key the payload travels under, and its shape version.
CAPTURE_KEY = "capture"
CAPTURE_VERSION = 1
#: The attribute stamped on a capture trace so a reader can find them.
CAPTURE_ATTRIBUTE = "capture"
CAPTURE_SOURCE_OPENROUTER = "openrouter_proxy"
#: Environment switch. Default off.
CAPTURE_ENV = "AMFS_PROXY_CAPTURE"
#: Request keys kept byte-for-byte. Everything else on the body is dropped so
#: a caller's unrelated extension fields do not ride into the trace store.
REQUEST_KEYS: tuple[str, ...] = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "seed",
    "response_format",
    "parallel_tool_calls",
    "reasoning",
    "n",
)
#: Marker left where the scanner blocked a message outright.
REDACTED = "[redacted by SafetyGate]"

_OptIn = Callable[[Any], bool]
_opt_in: _OptIn | None = None


def set_opt_in(resolver: _OptIn | None) -> None:
    """Install the per-account opt-in. The hosting layer passes a callable
    taking the request and answering whether this account asked for capture;
    ``None`` removes it. The environment switch remains a global override."""
    global _opt_in
    _opt_in = resolver


def capture_enabled(request: Any = None) -> bool:
    """Whether this exchange is captured. Default-off: the env switch, or the
    installed opt-in resolver saying yes for this request."""
    if os.environ.get(CAPTURE_ENV, "false").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if _opt_in is None or request is None:
        return False
    try:
        return bool(_opt_in(request))
    except Exception:  # noqa: BLE001 - an opt-in that errors is an opt-out
        logger.debug("capture opt-in resolver raised; treating as off", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# SSE reassembly
# ---------------------------------------------------------------------------


def reassemble_sse(chunks: Iterable[bytes]) -> dict[str, Any]:
    """Rebuild one chat completion from its streamed deltas.

    Follows the OpenAI streaming shape: each ``data:`` line is a chunk whose
    ``choices[i].delta`` carries a piece of ``content``, ``role``, or a piece
    of a tool call addressed by ``index`` (name and arguments arrive in
    fragments); ``finish_reason`` lands on the last delta of a choice;
    ``usage`` on a trailing chunk when the caller asked for it; ``[DONE]``
    ends the stream. Returns a non-streaming-shaped response — ``choices[]``
    with a full ``message`` — plus ``stream: true`` and the number of chunks
    so a reader knows how it was obtained. Lines that are not JSON are kept
    under ``unparsed`` rather than dropped.
    """
    buffer = b"".join(chunks)
    text = buffer.decode("utf-8", errors="replace")
    choices: dict[int, dict[str, Any]] = {}
    head: dict[str, Any] = {}
    usage: Any = None
    unparsed: list[str] = []
    n_chunks = 0
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            unparsed.append(line)
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            unparsed.append(line)
            continue
        n_chunks += 1
        for key in ("id", "model", "created", "system_fingerprint", "object", "provider"):
            if key in chunk and key not in head:
                head[key] = chunk[key]
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            idx = int(choice.get("index", 0) or 0)
            slot = choices.setdefault(
                idx,
                {
                    "index": idx,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                    "_tool_calls": {},
                },
            )
            delta = choice.get("delta") or {}
            if delta.get("role"):
                slot["message"]["role"] = delta["role"]
            piece = delta.get("content")
            if isinstance(piece, str):
                slot["message"]["content"] += piece
            if delta.get("reasoning") is not None:
                slot["message"]["reasoning"] = (slot["message"].get("reasoning") or "") + str(
                    delta["reasoning"]
                )
            for tc in delta.get("tool_calls") or []:
                t_idx = int(tc.get("index", len(slot["_tool_calls"])) or 0)
                target = slot["_tool_calls"].setdefault(
                    t_idx,
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    target["id"] = tc["id"]
                if tc.get("type"):
                    target["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    target["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    target["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                slot["finish_reason"] = choice["finish_reason"]
    out_choices = []
    for idx in sorted(choices):
        slot = choices[idx]
        tool_calls = [slot["_tool_calls"][k] for k in sorted(slot["_tool_calls"])]
        message = dict(slot["message"])
        if tool_calls:
            message["tool_calls"] = tool_calls
            if message.get("content") == "":
                message["content"] = None
        out_choices.append(
            {"index": idx, "message": message, "finish_reason": slot["finish_reason"]}
        )
    result: dict[str, Any] = {
        **head,
        "object": head.get("object", "chat.completion"),
        "choices": out_choices,
        "stream": True,
        "chunks": n_chunks,
    }
    if usage is not None:
        result["usage"] = usage
    if unparsed:
        result["unparsed"] = unparsed[:50]
    return result


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scan(text: Any, scanner: Callable[[str], str | None] | None) -> Any:
    if scanner is None or not isinstance(text, str) or not text:
        return text
    scanned = scanner(text)
    return REDACTED if scanned is None else scanned


def _redact_tree(value: Any, scanner: Callable[[str], str | None] | None) -> Any:
    """Run *scanner* over **every** string leaf of *value*, whatever the
    shape: message ``content`` as text or parts, ``reasoning``, tool-call
    arguments, tool results, tool schemas, and any field a provider adds
    that this module has never heard of. Structure is preserved; dict keys
    are not scanned (they are field names, not text). Enumerating the known
    text fields was how ``reasoning`` and the ``tools`` schemas were stored
    unscanned."""
    if isinstance(value, str):
        return _scan(value, scanner)
    if isinstance(value, dict):
        return {k: _redact_tree(v, scanner) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_tree(v, scanner) for v in value]
    if isinstance(value, tuple):
        return [_redact_tree(v, scanner) for v in value]
    return value


def _redact_messages(messages: Any, scanner: Callable[[str], str | None] | None) -> Any:
    """Scan every text field a message can carry. A list of messages in, the
    same shape out; kept as the named entry point the tests and the trace
    builder use, implemented as the structural walk."""
    if not isinstance(messages, list):
        return messages
    return _redact_tree(messages, scanner)


def tool_calls_of(response: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool calls the model made, in the :class:`amfs_core.models.ToolCall`
    shape ``DecisionTrace.tool_calls`` carries (``tool_name``, ``arguments``,
    ``source``), parsed from the response's first choice. The result is not
    known at capture time — the caller's agent runs the tool — so each call
    stands as requested. Arguments that are not JSON are kept as text."""
    calls: list[dict[str, Any]] = []
    try:
        message = (response.get("choices") or [{}])[0].get("message") or {}
    except (AttributeError, IndexError, TypeError):
        return calls
    for tc in message.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        raw = fn.get("arguments")
        args: Any = raw
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except ValueError:
                args = {"_raw": raw}
        calls.append(
            {
                "tool_name": str(fn.get("name") or "tool"),
                "arguments": args if isinstance(args, dict) else {"_raw": args},
                "source": "capture",
            }
        )
    return calls


def capture_payload(
    request_body: dict[str, Any],
    response: dict[str, Any],
    *,
    raw_request: bytes | None = None,
    raw_response: bytes | None = None,
    upstream_status: int | None = None,
    scanner: Callable[[str], str | None] | None = None,
    source: str = CAPTURE_SOURCE_OPENROUTER,
    scope: str | None = None,
    injected_facts: int = 0,
) -> dict[str, Any]:
    """The capture, redacted and hashed.

    ``request`` keeps the fields in :data:`REQUEST_KEYS` as the caller sent
    them — the full message history and the tool schemas included —
    ``response`` is the upstream object (reassembled when streamed), both with
    every text field run through *scanner*. The hashes are of the **raw**
    bytes before redaction, so a holder of the original can verify the capture
    is of that exchange; the redacted text itself is what is stored.
    """
    # The whole of both objects, not the fields known to hold text: tool
    # schemas, reasoning, provider extensions and anything else a caller sends
    # or a model returns are stored, so all of it is scanned.
    request = _redact_tree({k: request_body[k] for k in REQUEST_KEYS if k in request_body}, scanner)
    redacted_response = _redact_tree(dict(response or {}), scanner)
    payload: dict[str, Any] = {
        "capture_version": CAPTURE_VERSION,
        "source": source,
        "captured_at": datetime.now(UTC).isoformat(),
        "scope": scope,
        "injected_facts": int(injected_facts),
        "request": request,
        "response": redacted_response,
        "upstream_status": upstream_status,
        "streamed": bool(response.get("stream")) if isinstance(response, dict) else False,
    }
    if raw_request is not None:
        payload["request_sha256"] = _sha256(raw_request)
        payload["request_bytes"] = len(raw_request)
    if raw_response is not None:
        payload["response_sha256"] = _sha256(raw_response)
        payload["response_bytes"] = len(raw_response)
    return payload


def capture_trace(
    payload: dict[str, Any],
    *,
    agent_id: str,
    session_id: str | None = None,
    namespace: str = "default",
    scope: str | None = None,
) -> Any:
    """A :class:`DecisionTrace` carrying *payload*: no outcome, the model's
    tool calls on ``tool_calls``, the last user turn as ``task_input`` and
    the assistant's text as ``response_text`` (what the sealed record and the
    training exporters already read), the whole capture under
    ``session_metadata.capture``."""
    from amfs_core.models import DecisionTrace, SessionMetadata

    messages = (payload.get("request") or {}).get("messages") or []
    task_input = None
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            task_input = (
                c
                if isinstance(c, str)
                else json.dumps(c, ensure_ascii=False)
                if c is not None
                else None
            )
            break
    response = payload.get("response") or {}
    try:
        response_text = (response.get("choices") or [{}])[0].get("message", {}).get("content")
    except (AttributeError, IndexError, TypeError):
        response_text = None
    meta = SessionMetadata(
        attributes={
            CAPTURE_ATTRIBUTE: payload.get("source", CAPTURE_SOURCE_OPENROUTER),
            "source": "proxy",
        },
        **{CAPTURE_KEY: payload},
    )
    return DecisionTrace(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        session_id=session_id or f"capture-{uuid.uuid4().hex[:12]}",
        outcome_ref=None,
        outcome_type=None,
        decision_summary=f"captured {payload.get('source')} exchange"
        + (f" on {scope}" if scope else ""),
        task_input=task_input,
        response_text=response_text if isinstance(response_text, str) else None,
        tool_calls=tool_calls_of(response),
        session_metadata=meta,
        namespace=namespace,
    )


__all__ = [
    "CAPTURE_ATTRIBUTE",
    "CAPTURE_ENV",
    "CAPTURE_KEY",
    "CAPTURE_SOURCE_OPENROUTER",
    "CAPTURE_VERSION",
    "REDACTED",
    "REQUEST_KEYS",
    "capture_enabled",
    "capture_payload",
    "capture_trace",
    "reassemble_sse",
    "set_opt_in",
    "tool_calls_of",
]
