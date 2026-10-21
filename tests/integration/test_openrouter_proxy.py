"""Integration test for the memory-augmenting OpenRouter proxy route.

Exercises the full mount → inject → forward → return path against a mocked
upstream (no real OpenRouter call). Skipped only when fastapi or httpx is not
importable. It used to be gated behind ``AMFS_TEST_PROXY=1`` as well, and so
never ran in CI — which is how the route shipped returning 422 on every call
(``Request`` imported inside the mount function did not resolve the
``request: Request`` annotation under ``from __future__ import annotations``).

Run:
    pytest tests/integration/test_openrouter_proxy.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _build_app(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # Reload module so is_configured() picks up the env var.
    import importlib

    from amfs_http import openrouter_proxy as P
    importlib.reload(P)

    import httpx
    from fastapi import FastAPI

    captured: dict = {}

    class _Resp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {"choices": [{"message": {"role": "assistant",
                                             "content": "PostgreSQL 15."}}]}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    class _Mem:
        def search(self, *, query, entity_path, limit):
            class R:
                value = "The billing service runs on PostgreSQL 15."
            return [R()]

    app = FastAPI()
    P.mount_openrouter_proxy(app, get_memory=lambda: _Mem())
    return app, captured


def test_proxy_injects_memory_and_forwards(monkeypatch):
    from fastapi.testclient import TestClient

    app, captured = _build_app(monkeypatch)
    with TestClient(app) as client:
        resp = client.post("/api/v1/proxy/chat/completions", json={
            "model": "openai/gpt-4o-mini",
            "amfs_scope": "svc/db",
            "messages": [{"role": "user", "content": "which db does billing use?"}],
        })
    assert resp.status_code == 200
    # Upstream received an injected system message with the retrieved fact.
    fwd_messages = captured["json"]["messages"]
    assert fwd_messages[0]["role"] == "system"
    assert "PostgreSQL 15" in fwd_messages[0]["content"]
    # The internal amfs_scope hint was stripped before forwarding upstream.
    assert "amfs_scope" not in captured["json"]
