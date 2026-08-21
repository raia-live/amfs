"""The origin root answers 200 without auth.

Generic API gateways and connector validators (e.g. Fly.io Sprites' custom_api
"test request") probe ``base_url/`` to confirm the service is live before saving
a connector. A 404 there makes them refuse the connector, so the root must be a
reachable, unauthenticated 200 — the same contract as ``/health``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

import amfs_http.server as server  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(server.app)


def test_root_returns_200_without_auth(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # Reports the same payload as /health so validators see a live service.
    assert body == client.get("/health").json()


def test_root_needs_no_api_key(client: TestClient) -> None:
    # No X-AMFS-API-Key / Authorization header at all: still 200. The route is
    # outside the /api/ prefix that carries auth, so it never reaches the gate.
    assert client.get("/").status_code == 200
