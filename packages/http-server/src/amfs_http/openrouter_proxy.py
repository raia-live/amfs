"""Memory-augmenting OpenRouter proxy (OpenAI-compatible chat completions).

This is the "Option C" proxy: a drop-in replacement for the OpenRouter chat
completions endpoint that (1) retrieves relevant memory for the caller's scope
and injects it as context, (2) forwards the request to OpenRouter unchanged
otherwise, and (3) writes salient facts from the exchange back to memory and
seals a decision trace — all without the client changing anything but the base
URL.

Design principles:
- Fail-open: any memory error (retrieval, injection, write-back) must never
  break the LLM call. On failure we forward the original request untouched.
- Non-blocking: memory retrieval and write-back are synchronous SDK calls, so
  they run in a threadpool (``asyncio.to_thread``) to keep the event loop free.
  Write-back and trace sealing are fire-and-forget background tasks.
- Streaming: ``stream: true`` is passed through as an SSE byte stream; with capture
  on, the bytes are also held and reassembled into one response once it ends.

Configuration (env):
- OPENROUTER_API_KEY         upstream key (required to enable the route)
- OPENROUTER_BASE_URL        default https://openrouter.ai/api/v1
- AMFS_PROXY_INJECT_LIMIT    max memory facts injected (default 5)
- AMFS_PROXY_WRITE_BACK      "true" to enable fact write-back (default false)
- AMFS_PROXY_SAFETY_GATE     "false" to disable the Pro SafetyGate on write-back
                             (default true; no-op when amfs_safety is absent)
- AMFS_PROXY_CAPTURE         "true" to capture each exchange byte-faithfully as a
                             trace with no outcome (default false; a hosting layer
                             can opt accounts in per request via amfs_http.capture.set_opt_in)

TODO(integration-test): tests/integration/test_openrouter_proxy.py exercises the
inject/forward/write-back path against a mocked upstream; it is skipped in unit
CI (requires fastapi TestClient + httpx-mock) and is unvalidated here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from datetime import UTC
from typing import Any

# The route's ``request: Request`` annotation must resolve in *this* module's
# globals: with ``from __future__ import annotations`` FastAPI evaluates the
# string against them, and when ``Request`` was imported inside
# ``mount_openrouter_proxy`` the lookup failed silently — FastAPI then treated
# ``request`` as a required query parameter and every call to the proxy
# returned 422. Module-level import, so the annotation resolves.
try:
    from fastapi import Request
except ImportError:  # pragma: no cover - fastapi is a dependency of this package
    Request = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
INJECT_LIMIT = int(os.environ.get("AMFS_PROXY_INJECT_LIMIT", "5"))
WRITE_BACK = os.environ.get("AMFS_PROXY_WRITE_BACK", "false").lower() == "true"

#: The upstream HTTP client, created on mount. Module-level so a test can put
#: a mock transport behind the mounted route.
_upstream_client: Any = None
# Auto-run the Pro SafetyGate on write-back when available (secret block + PII
# redact). Disabled explicitly with AMFS_PROXY_SAFETY_GATE=false. Soft/optional:
# the gate lives in the Pro ``amfs_safety`` package; OSS-only deployments skip it.
SAFETY_GATE = os.environ.get("AMFS_PROXY_SAFETY_GATE", "true").lower() == "true"

_SAFETY_GATE_CACHE: dict[str, Any] = {}


def _gate_value(mem: Any, scope: str, key: str, value: Any) -> tuple[bool, Any]:
    """Run the optional Pro SafetyGate over a proposed write-back value.

    Returns ``(allowed, value_to_write)`` — the value is redacted when the gate
    masks sensitive content. Fail-open: any import/build/scan error allows the
    original value through (the LLM exchange must never be lost to a gate bug).
    """
    if not SAFETY_GATE:
        return True, value
    try:
        if "gate" not in _SAFETY_GATE_CACHE:
            from amfs_safety import SafetyGate  # Pro package, optional

            adapter = getattr(mem, "_adapter", None)
            _SAFETY_GATE_CACHE["gate"] = SafetyGate(adapter=adapter)
        gate = _SAFETY_GATE_CACHE["gate"]

        from datetime import datetime

        from amfs_core.models import MemoryEntry, Provenance

        entry = MemoryEntry(
            entity_path=scope,
            key=key,
            value=value,
            provenance=Provenance(
                agent_id="openrouter-proxy",
                session_id="proxy",
                written_at=datetime.now(UTC),
            ),
            confidence=0.5,
        )
        decision = gate.check_write(entry)
        if not decision.allowed:
            logger.info("proxy: write-back blocked by SafetyGate (%s)", decision.max_severity)
            return False, value
        return True, decision.entry.value
    except Exception:  # noqa: BLE001 - fail-open
        logger.debug("proxy: SafetyGate unavailable/failed; allowing write", exc_info=True)
        return True, value

_MEMORY_HEADER = (
    "You have access to the following facts from persistent memory. "
    "Treat them as authoritative context; if none are relevant, ignore them.\n"
)


def is_configured() -> bool:
    return bool(API_KEY)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # OpenAI content-parts form
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _retrieve_facts(get_memory: Callable[[], Any], scope: str | None, query: str) -> list[str]:
    """Best-effort memory retrieval. Returns formatted fact strings (fail-open)."""
    if not query:
        return []
    try:
        mem = get_memory()
    except Exception:  # noqa: BLE001
        logger.debug("proxy: get_memory failed", exc_info=True)
        return []
    # Prefer the Pro MultiStrategyRetriever (adaptive RRF + rerank) when the
    # amfs_retrieval package is installed; otherwise SDK full-text/semantic
    # search; otherwise list the scope. All fail-open.
    try:
        from amfs_http.pro_provider import build_pro_retriever, pro_retrieve_values

        adapter = getattr(mem, "_adapter", None)
        retriever = build_pro_retriever(adapter)
        pro_values = pro_retrieve_values(retriever, scope, query, INJECT_LIMIT)
        if pro_values is not None:
            return [
                v if isinstance(v, str) else json.dumps(v, default=str)
                for v in pro_values
            ][:INJECT_LIMIT]
    except Exception:  # noqa: BLE001 - Pro path optional, fall through to OSS
        logger.debug("proxy: pro retrieval skipped", exc_info=True)

    # Prefer full-text/semantic search; fall back to listing the scope.
    try:
        search = getattr(mem, "search", None)
        if callable(search):
            results = search(query=query, entity_path=scope, limit=INJECT_LIMIT)
        else:
            results = mem._adapter.list(scope)[:INJECT_LIMIT]
    except Exception:  # noqa: BLE001
        logger.debug("proxy: memory search failed", exc_info=True)
        return []
    facts: list[str] = []
    for r in results or []:
        value = getattr(r, "value", None)
        if value is None and isinstance(r, dict):
            value = r.get("value")
        if value is not None:
            facts.append(value if isinstance(value, str) else json.dumps(value, default=str))
    return facts[:INJECT_LIMIT]


def _inject(messages: list[dict[str, Any]], facts: list[str]) -> list[dict[str, Any]]:
    if not facts:
        return messages
    block = _MEMORY_HEADER + "\n".join(f"- {f}" for f in facts)
    # Prepend as a system message so it does not overwrite the caller's system prompt.
    return [{"role": "system", "content": block}, *messages]


def _write_back(get_memory: Callable[[], Any], scope: str | None,
                query: str, answer: str) -> None:
    """Persist a salient fact from the exchange + seal a trace (best-effort)."""
    if not WRITE_BACK or not scope or not answer:
        return
    try:
        mem = get_memory()
        key = f"proxy-exchange-{abs(hash(query)) % 10_000_000}"
        value = {"q": query[:500], "a": answer[:1500]}
        allowed, value = _gate_value(mem, scope, key, value)
        if not allowed:
            return
        mem.write(scope, key, value, confidence=0.5)
        commit = getattr(mem, "commit_outcome", None)
        if callable(commit):
            commit("proxy-exchange", "success")
    except Exception:  # noqa: BLE001
        logger.debug("proxy: write-back failed", exc_info=True)


def _capture_exchange(
    get_memory: Callable[[], Any],
    *,
    request_body: dict[str, Any],
    response: dict[str, Any],
    raw_request: bytes,
    raw_response: bytes | None,
    upstream_status: int | None,
    scope: str | None,
    injected_facts: int,
    agent_id: str,
) -> str | None:
    """Byte-faithful capture of one exchange as a trace with no outcome
    (:mod:`amfs_http.capture`, t2-8e). Redacted by the secret scanner before
    anything is written; best-effort, never raises into the request."""
    from amfs_core.capture import scan_captured_text

    from amfs_http import capture as cap

    try:
        mem = get_memory()
        adapter = getattr(mem, "_adapter", None)

        def _scanner(text: str) -> str | None:
            return scan_captured_text(text, adapter=adapter, agent_id=agent_id)

        payload = cap.capture_payload(
            request_body, response, raw_request=raw_request, raw_response=raw_response,
            upstream_status=upstream_status, scanner=_scanner, scope=scope,
            injected_facts=injected_facts,
        )
        trace = cap.capture_trace(
            payload, agent_id=agent_id, namespace=getattr(mem, "namespace", "default"), scope=scope,
        )
        save = getattr(adapter, "save_trace", None)
        if not callable(save):
            return None
        saved = save(trace)
        # Sealed by whatever seal path the server has (Pro traces), so the
        # capture is content-addressed like every other trace. The raw
        # session metadata is passed because the OSS model drops undeclared
        # keys and the payload travels in one.
        try:
            from amfs_http import server as _server

            seal = getattr(_server, "_auto_seal_trace", None)
            if callable(seal):
                meta = trace.session_metadata.model_dump() if trace.session_metadata else None
                seal(mem, saved, session_metadata=meta)
        except Exception:  # noqa: BLE001 - the OSS row stands on its own
            logger.debug("proxy: capture seal skipped", exc_info=True)
        return getattr(saved, "id", None) or trace.id
    except Exception:  # noqa: BLE001
        logger.debug("proxy: capture failed", exc_info=True)
        return None


def mount_openrouter_proxy(app, *, get_memory: Callable[[], Any]) -> None:
    """Mount POST /api/v1/proxy/chat/completions if OpenRouter is configured."""
    if not is_configured():
        logger.info("OpenRouter proxy disabled (OPENROUTER_API_KEY not set)")
        return

    import httpx
    from fastapi.responses import JSONResponse, StreamingResponse

    from amfs_http import capture as cap

    global _upstream_client
    if _upstream_client is None:
        _upstream_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    def _client():
        # Resolved per request so a test (or a reload) can swap the client.
        return _upstream_client

    def _scope_of(request: Request, body: dict[str, Any]) -> str | None:
        return (body.get("amfs_scope")
                or request.headers.get("X-AMFS-Scope")
                or os.environ.get("AMFS_PROXY_DEFAULT_SCOPE"))

    def _upstream_headers(request: Request) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        # Pass through OpenRouter attribution headers when present.
        for h in ("HTTP-Referer", "X-Title"):
            if h in request.headers:
                headers[h] = request.headers[h]
        return headers

    @app.post("/api/v1/proxy/chat/completions")
    async def proxy_chat_completions(request: Request):
        raw = await request.body()
        try:
            body = json.loads(raw)
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        scope = _scope_of(request, body)
        messages = body.get("messages") or []
        query = _last_user_text(messages)
        # Capture is decided per request and off by default (t2-8e): the env
        # switch or the hosting layer's per-account opt-in. The request body
        # is kept as the caller sent it — before injection — because that is
        # the exchange the caller's agent had; the injected facts are counted.
        capturing = cap.capture_enabled(request)
        sent_body = dict(body) if capturing else None
        capture_agent = (request.headers.get("X-AMFS-Agent-Id") or body.get("amfs_agent_id")
                         or "openrouter-proxy")
        injected = 0

        # 1) Retrieve + inject memory (fail-open, off the event loop).
        try:
            facts = await asyncio.to_thread(_retrieve_facts, get_memory, scope, query)
            if facts:
                body = {**body, "messages": _inject(messages, facts)}
                injected = len(facts)
        except Exception:  # noqa: BLE001
            logger.debug("proxy: injection skipped", exc_info=True)

        body.pop("amfs_scope", None)
        body.pop("amfs_agent_id", None)
        if sent_body is not None:
            sent_body.pop("amfs_scope", None)
            sent_body.pop("amfs_agent_id", None)
        url = f"{BASE_URL}/chat/completions"
        headers = _upstream_headers(request)

        def _schedule_capture(
            response: dict[str, Any], raw_response: bytes | None, status: int | None
        ) -> None:
            if sent_body is None:
                return
            asyncio.create_task(asyncio.to_thread(
                _capture_exchange, get_memory,
                request_body=sent_body, response=response, raw_request=raw,
                raw_response=raw_response, upstream_status=status, scope=scope,
                injected_facts=injected, agent_id=capture_agent,
            ))

        # 2a) Streaming: pass the SSE stream straight through. With capture on
        # the bytes are also accumulated and reassembled once the stream ends;
        # with it off nothing is held and the path is as it was.
        if body.get("stream"):
            async def _iter():
                held: list[bytes] = []
                status: int | None = None
                async with _client().stream("POST", url, headers=headers, json=body) as resp:
                    status = resp.status_code
                    async for chunk in resp.aiter_raw():
                        if capturing:
                            held.append(chunk)
                        yield chunk
                if capturing:
                    try:
                        joined = b"".join(held)
                        _schedule_capture(cap.reassemble_sse([joined]), joined, status)
                    except Exception:  # noqa: BLE001
                        logger.debug("proxy: stream capture failed", exc_info=True)
            return StreamingResponse(_iter(), media_type="text/event-stream")

        # 2b) Non-streaming: forward, then schedule background write-back.
        try:
            resp = await _client().post(url, headers=headers, json=body)
        except Exception:  # noqa: BLE001
            logger.warning("proxy: upstream request failed", exc_info=True)
            return JSONResponse({"error": "upstream unavailable"}, status_code=502)

        data = resp.json() if resp.content else {}
        try:
            answer = data["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            answer = ""

        if WRITE_BACK and answer:
            asyncio.create_task(
                asyncio.to_thread(_write_back, get_memory, scope, query, answer))
        if capturing and isinstance(data, dict):
            _schedule_capture(data, resp.content or None, resp.status_code)

        return JSONResponse(data, status_code=resp.status_code)

    logger.info("OpenRouter memory proxy mounted → POST /api/v1/proxy/chat/completions")
