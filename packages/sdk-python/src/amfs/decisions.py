"""Hosted decision client. Does not execute actions or mutate session outcomes."""
from __future__ import annotations

import os
from uuid import UUID, uuid4

import httpx

from amfs_core.decisions import DecisionRequest, DecisionResponse, OutcomeEvent


class DecisionClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, *,
                 timeout: float = 60, transport: httpx.BaseTransport | None = None):
        base_url = base_url or os.environ.get("AMFS_HTTP_URL")
        api_key = api_key or os.environ.get("AMFS_API_KEY")
        if not base_url or not api_key:
            raise ValueError("base_url/AMFS_HTTP_URL and api_key/AMFS_API_KEY are required")
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                  headers={"X-AMFS-API-Key": api_key}, transport=transport)

    def decide(self, request: DecisionRequest, *, idempotency_key: str | None = None) -> DecisionResponse:
        response = self._http.post("/api/v1/decisions", json=request.model_dump(mode="json"),
                                   headers={"Idempotency-Key": idempotency_key or str(uuid4())})
        response.raise_for_status()
        return DecisionResponse.model_validate(response.json())

    def report_outcome(self, decision_id: UUID | str, event: OutcomeEvent) -> dict:
        response = self._http.post(f"/api/v1/decisions/{UUID(str(decision_id))}/outcomes",
                                   json=event.model_dump(mode="json"))
        response.raise_for_status()
        return response.json()

    def get(self, decision_id: UUID | str) -> dict:
        response = self._http.get(f"/api/v1/decisions/{UUID(str(decision_id))}")
        response.raise_for_status()
        return response.json()

    def create_model(self, name: str, spec: dict) -> dict:
        response = self._http.post("/api/v1/decision-models", json={"name": name, "spec": spec})
        response.raise_for_status()
        return response.json()

    def list_models(self) -> list[dict]:
        response = self._http.get("/api/v1/decision-models")
        response.raise_for_status()
        return response.json()["models"]

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
