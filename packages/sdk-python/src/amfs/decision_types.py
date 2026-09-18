"""Public hosted decision management responses (JSON wire representation)."""
from typing import Any, Literal, NotRequired, TypedDict

ServingMode = Literal["observe", "route"]


class DecisionModelSpec(TypedDict):
    decision: str
    spec_version: str
    questions: dict[str, dict[str, Any]]
    valid_tuples: NotRequired[list[dict[str, str]]]


class DecisionModel(TypedDict):
    id: str
    name: str
    spec: DecisionModelSpec
    spec_hash: str
    serving_mode: ServingMode
    active_version: str | None
    created_at: str


class DecisionVersion(TypedDict):
    version: str
    artifact_sha256: str
    evaluation: dict[str, Any]
    calibration: dict[str, Any]
    created_at: str


class DecisionDataset(TypedDict):
    id: str
    name: str
    manifest_sha256: str


class DecisionTrainingJob(TypedDict):
    id: str
    model_id: str
    version: str
    dataset_id: str
    state: Literal["queued", "submitting", "running", "succeeded", "failed"]
    attempts: int
    error_code: str | None
    created_at: str
    updated_at: str


class DecisionUsage(TypedDict):
    decision_count: int
    answered_units: int
    automated_decisions: int
    automation_rate: float | None
    days: int
    since: str
    by_engine: list[dict[str, Any]]
    daily: list[dict[str, Any]]
    billing_status: Literal["preview_unbilled"]
    cost_usd: None


class DecisionSummary(TypedDict):
    decision_id: str
    created_at: str
    answered_units: int
    automate: bool
    answers: dict[str, dict[str, Any]]


class DecisionHistory(TypedDict):
    decisions: list[DecisionSummary]
    next_cursor: str | None
    usage: DecisionUsage


class DecisionDetail(TypedDict):
    request: dict[str, Any]
    response: dict[str, Any]
    outcomes: list[dict[str, Any]]
    integrity: dict[str, Any]
