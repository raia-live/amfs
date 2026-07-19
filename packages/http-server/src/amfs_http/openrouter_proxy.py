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
- Streaming: ``stream: true`` is passed through as an SSE byte stream.

Configuration (env):
- OPENROUTER_API_KEY         upstream key (required to enable the route)
- OPENROUTER_BASE_URL        default https://openrouter.ai/api/v1
- AMFS_PROXY_INJECT_LIMIT    max memory facts injected (default 5)
- AMFS_PROXY_WRITE_BACK      "true" to enable fact write-back (default false)
- AMFS_PROXY_SAFETY_GATE     "false" to disable the Pro SafetyGate on write-back
                             (default true; no-op when amfs_safety is absent)

TODO(integration-test): tests/integration/test_openrouter_proxy.py exercises the
inject/forward/write-back path against a mocked upstream; it is skipped in unit
CI (requires fastapi TestClient + httpx-mock) and is unvalidated here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
INJECT_LIMIT = int(os.environ.get("AMFS_PROXY_INJECT_LIMIT", "5"))
WRITE_BACK = os.environ.get("AMFS_PROXY_WRITE_BACK", "false").lower() == "true"
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

        from datetime import datetime, timezone

        from amfs_core.models import MemoryEntry, Provenance

        entry = MemoryEntry(
            entity_path=scope,
            key=key,
            value=value,
            provenance=Provenance(
                agent_id="openrouter-proxy",
                session_id="proxy",
                written_at=datetime.now(timezone.utc),
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


def mount_openrouter_proxy(app, *, get_memory: Callable[[], Any]) -> None:
    """Mount POST /api/v1/proxy/chat/completions if OpenRouter is configured."""
    if not is_configured():
        logger.info("OpenRouter proxy disabled (OPENROUTER_API_KEY not set)")
        return

    import httpx
    from fastapi import Request
    from fastapi.responses import JSONResponse, StreamingResponse

    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

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

        # 1) Retrieve + inject memory (fail-open, off the event loop).
        try:
            facts = await asyncio.to_thread(_retrieve_facts, get_memory, scope, query)
            if facts:
                body = {**body, "messages": _inject(messages, facts)}
        except Exception:  # noqa: BLE001
            logger.debug("proxy: injection skipped", exc_info=True)

        body.pop("amfs_scope", None)
        url = f"{BASE_URL}/chat/completions"
        headers = _upstream_headers(request)

        # 2a) Streaming: pass the SSE stream straight through.
        if body.get("stream"):
            async def _iter():
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk
            # NOTE: write-back on streamed responses requires accumulating the
            # SSE deltas; deferred (TODO) to keep streaming zero-overhead.
            return StreamingResponse(_iter(), media_type="text/event-stream")

        # 2b) Non-streaming: forward, then schedule background write-back.
        try:
            resp = await client.post(url, headers=headers, json=body)
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

        return JSONResponse(data, status_code=resp.status_code)

    logger.info("OpenRouter memory proxy mounted → POST /api/v1/proxy/chat/completions")
