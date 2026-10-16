"""The webhook route hands an installed connector the payload it is written for.

``ConnectorABC.transform`` takes the decoded payload dict; that is how
connectors are written and how ``amfs_connectors.testing`` exercises them.
``WebhookIngester`` calls its transforms with the ``WebhookEvent`` it built.
The route registered the connector's bound method as the ingester's catch-all
transform, so PagerDuty, the connector the guide documents, was handed a
``WebhookEvent``, and its first ``.get()`` raised: a 500 on every delivery.
"""

from __future__ import annotations

from pathlib import Path

import amfs_http.server as server
import pytest
from amfs import AgentMemory
from amfs_filesystem.adapter import FilesystemAdapter
from fastapi.testclient import TestClient


def _incident(event_type: str) -> dict:
    """A PagerDuty v3 webhook body, trimmed to what the connector reads."""
    return {
        "event": {
            "id": "01DEN0XW5S0JTJDQ0M5B4XZW6S",
            "event_type": event_type,
            "data": {
                "id": "PGR0VU2",
                "title": "Checkout error rate above 5%",
                "severity": "critical",
                "status": "triggered",
                "service": {"summary": "checkout"},
                "html_url": "https://acme.pagerduty.com/incidents/PGR0VU2",
            },
        }
    }


@pytest.fixture
def mem(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AgentMemory:
    memory = AgentMemory(agent_id="amfs-server", adapter=FilesystemAdapter(tmp_path))
    monkeypatch.setattr(server, "_memory", memory)
    monkeypatch.setattr(server, "_get_memory", lambda: memory)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.delenv("AMFS_API_KEYS", raising=False)
    monkeypatch.delenv("AMFS_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("AMFS_CONNECTOR_PAGERDUTY_SECRET", raising=False)
    return memory


def test_a_triggered_incident_is_written_where_the_connector_says(mem: AgentMemory) -> None:
    resp = TestClient(server.app).post(
        "/api/v1/webhooks/pagerduty", json=_incident("incident.triggered")
    )

    assert resp.status_code == 200
    assert resp.json()["persisted"] == 1

    entry = mem.read("incidents/checkout", "pd-PGR0VU2")
    assert entry is not None
    assert entry.value["title"] == "Checkout error rate above 5%"
    assert entry.value["severity"] == "critical"
    assert entry.provenance.agent_id == "webhook/pagerduty"


def test_an_acknowledgement_comes_back_as_context(mem: AgentMemory) -> None:
    resp = TestClient(server.app).post(
        "/api/v1/webhooks/pagerduty", json=_incident("incident.acknowledged")
    )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [(r["action"], r["key"]) for r in results] == [("context", "pd-PGR0VU2")]
