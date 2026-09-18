"""Hosted decision client. Does not execute actions or mutate session outcomes."""
from __future__ import annotations

import os
from uuid import UUID, uuid4
from urllib.parse import quote

from .decision_types import (DecisionDataset, DecisionDetail, DecisionHistory, DecisionModel,
    DecisionModelSpec, DecisionTrainingJob, DecisionUsage, DecisionVersion, ServingMode)

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

    def get(self, decision_id: UUID | str) -> DecisionDetail:
        response = self._http.get(f"/api/v1/decisions/{UUID(str(decision_id))}")
        response.raise_for_status()
        return response.json()

    def create_model(self, name: str, spec: DecisionModelSpec) -> DecisionModel:
        response = self._http.post("/api/v1/decision-models", json={"name": name, "spec": spec})
        response.raise_for_status()
        return response.json()

    def list_models(self) -> list[DecisionModel]:
        response = self._http.get("/api/v1/decision-models")
        response.raise_for_status()
        return response.json()["models"]

    def _model_request(self, name: str, suffix: str = "", *, method: str = "GET",
                       body: dict | None = None, params: dict | None = None,
                       idempotency_key: str | None = None):
        response = self._http.request(method, f"/api/v1/decision-models/{quote(name, safe='')}{suffix}",
            json=body, params=params,
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None)
        response.raise_for_status()
        return response.json()

    def get_model(self, name: str) -> DecisionModel:
        return self._model_request(name)

    def set_mode(self, name: str, serving_mode: ServingMode) -> DecisionModel:
        if serving_mode not in {"observe", "route"}:
            raise ValueError("serving_mode must be observe or route")
        return self._model_request(name, method="PATCH", body={"serving_mode": serving_mode})

    def list_versions(self, name: str) -> list[DecisionVersion]:
        return self._model_request(name, "/versions")["versions"]

    def list_datasets(self, name: str) -> list[DecisionDataset]:
        return self._model_request(name, "/datasets")["datasets"]

    def list_jobs(self, name: str) -> list[DecisionTrainingJob]:
        return self._model_request(name, "/jobs")["jobs"]

    def enqueue_training(self, name: str, *, version: str, dataset_id: str,
                         idempotency_key: str) -> DecisionTrainingJob:
        """Reuse the same key and payload when retrying an uncertain response."""
        if not 1 <= len(idempotency_key) <= 128:
            raise ValueError("idempotency_key must contain 1 to 128 characters")
        return self._model_request(name, "/jobs", method="POST",
            body={"version": version, "dataset_id": dataset_id}, idempotency_key=idempotency_key)

    def activate_version(self, name: str, version: str, *,
                         expected_active_version: str | None) -> DecisionModel:
        """Compare-and-set activation; pass None for first activation. Starts in observe mode."""
        return self._model_request(name, f"/versions/{quote(version, safe='')}/activate", method="POST",
            body={"expected_active_version": expected_active_version})

    def history(self, name: str, *, limit: int = 50, cursor: str | None = None) -> DecisionHistory:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        params = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        return self._model_request(name, "/decisions", params=params)

    def usage(self, name: str, *, days: int = 30) -> DecisionUsage:
        if not 1 <= days <= 90:
            raise ValueError("days must be between 1 and 90")
        return self._model_request(name, "/usage", params={"days": days})

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
