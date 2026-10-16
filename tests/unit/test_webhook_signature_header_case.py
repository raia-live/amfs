"""A signed webhook is accepted whatever case its signature header arrives in.

HTTP field names are case-insensitive, and ASGI servers deliver them
lowercased, so the ``dict(request.headers)`` the webhook route hands the
ingester holds ``x-webhook-signature``. The ingester looked the header up by
its configured spelling, ``X-Webhook-Signature``, found nothing, and rejected
the request. With ``AMFS_CONNECTOR_{NAME}_SECRET`` set, as the connectors guide
says to, every delivery failed verification, correctly signed or not.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import amfs_http.server as server
import pytest
from amfs import AgentMemory
from amfs_connectors import WebhookConfig, WebhookIngester
from amfs_filesystem.adapter import FilesystemAdapter
from fastapi.testclient import TestClient

SECRET = "whsec_test"
BODY = json.dumps({"id": "evt-0001", "status": "deployed"}).encode()


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _ingester() -> WebhookIngester:
    config = WebhookConfig(
        name="deploys", connector_type="webhook", entity_path="deploys", secret=SECRET
    )
    return WebhookIngester(config)


class TestVerifySignature:
    def test_a_lowercased_header_is_found(self) -> None:
        assert _ingester().verify_signature(BODY, {"x-webhook-signature": _sign(BODY)})

    def test_the_configured_spelling_still_works(self) -> None:
        assert _ingester().verify_signature(BODY, {"X-Webhook-Signature": _sign(BODY)})

    def test_a_wrong_signature_is_still_rejected(self) -> None:
        headers = {"x-webhook-signature": _sign(BODY, secret="not-the-secret")}
        assert not _ingester().verify_signature(BODY, headers)

    def test_a_missing_header_is_still_rejected(self) -> None:
        assert not _ingester().verify_signature(BODY, {"content-type": "application/json"})


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    mem = AgentMemory(agent_id="amfs-server", adapter=FilesystemAdapter(tmp_path))
    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.delenv("AMFS_API_KEYS", raising=False)
    monkeypatch.delenv("AMFS_POSTGRES_DSN", raising=False)
    monkeypatch.setenv("AMFS_CONNECTOR_DEPLOYS_SECRET", SECRET)
    return TestClient(server.app)


class TestTheWebhookRoute:
    def test_a_correctly_signed_delivery_is_persisted(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/webhooks/deploys",
            content=BODY,
            headers={"X-Webhook-Signature": _sign(BODY), "Content-Type": "application/json"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert [r["success"] for r in body["results"]] == [True]
        assert body["persisted"] == 1

    def test_a_badly_signed_delivery_is_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/webhooks/deploys",
            content=BODY,
            headers={
                "X-Webhook-Signature": _sign(BODY, secret="not-the-secret"),
                "Content-Type": "application/json",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["persisted"] == 0
        assert body["results"][0]["error"] == "Invalid webhook signature"
