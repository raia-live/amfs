"""Versioned decision envelopes, independent of historical ToolCall seals.

Candidate IDs are opaque identities; descriptions carry semantics. A decision
is a recommendation, never evidence that an action was executed successfully.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_serializer, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class VerificationCheck(Contract):
    id: Identifier
    instructions: str = Field(min_length=1, max_length=2048)
    observation_tool: Identifier
    window_seconds: int = Field(ge=1, le=604800)


class Candidate(Contract):
    id: Identifier
    description: str = Field(min_length=1, max_length=4096)
    kind: Literal["action", "observation", "defer"] = "action"
    cost: float = Field(default=0, ge=0, allow_inf_nan=False)
    verification: list[VerificationCheck] = Field(default_factory=list, max_length=16)


class Question(Contract):
    type: Literal["choice", "boolean", "score"] = "choice"
    instructions: str = Field(min_length=1, max_length=4096)
    candidates: list[Candidate] = Field(min_length=2, max_length=256)
    depends_on: list[Identifier] = Field(default_factory=list, max_length=16)
    # Score levels are ordered and explicit, rather than inferred from IDs.
    levels: list[float] | None = None

    @model_validator(mode="after")
    def validate_question(self) -> Question:
        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("dependencies must be unique")
        if self.type == "boolean" and ids != ["false", "true"]:
            raise ValueError("boolean candidates must be ordered false, true")
        if self.type == "score":
            if (self.levels is None or len(self.levels) != len(ids)
                    or any(not math.isfinite(x) for x in self.levels)
                    or any(a >= b for a, b in zip(self.levels, self.levels[1:]))):
                raise ValueError("score requires one strictly increasing finite level per candidate")
        elif self.levels is not None:
            raise ValueError("levels are only valid for score questions")
        return self


class DecisionRequest(Contract):
    schema_version: Literal["decision.v1"] = "decision.v1"
    decision: Identifier
    spec_version: Identifier
    state: JsonValue
    questions: dict[Identifier, Question] = Field(min_length=1, max_length=16)
    model: Identifier = "fallback"
    mode: Literal["observe", "route"] = "observe"
    max_error_rate: float = Field(default=0.02, gt=0, lt=1, allow_inf_nan=False)
    # Permission filtering must be done by the caller's trusted policy layer.
    # This can only narrow the declared candidate set; it grants no tool access.
    allowed_candidates: dict[Identifier, list[Identifier]] = Field(default_factory=dict)
    valid_tuples: list[dict[Identifier, Identifier]] = Field(default_factory=list, max_length=256)
    history: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=64)

    def layers(self) -> list[list[str]]:
        remaining = set(self.questions)
        resolved: set[str] = set()
        layers: list[list[str]] = []
        while remaining:
            layer = sorted(k for k in remaining if set(self.questions[k].depends_on) <= resolved)
            if not layer:
                raise ValueError("question dependencies contain a cycle or an unknown question")
            layers.append(layer)
            resolved.update(layer)
            remaining.difference_update(layer)
        return layers

    @property
    def spec_hash(self) -> str:
        return fingerprint({
            "decision": self.decision, "version": self.spec_version,
            "questions": {k: v.model_dump(mode="json") for k, v in self.questions.items()},
            "valid_tuples": self.valid_tuples,
            "allowed_candidates": self.allowed_candidates,
        })

    @model_validator(mode="after")
    def validate_request(self) -> DecisionRequest:
        self.layers()
        for name, allowed in self.allowed_candidates.items():
            if name not in self.questions:
                raise ValueError("unknown question in allowed_candidates")
            ids = {c.id for c in self.questions[name].candidates}
            if len(allowed) != len(set(allowed)) or not set(allowed) <= ids:
                raise ValueError("allowed_candidates must contain unique declared candidate IDs")
        # Full tuples only: partial constraints require a solver with explicit semantics.
        for row in self.valid_tuples:
            if set(row) != set(self.questions):
                raise ValueError("valid_tuples must assign every question")
            if any(v not in {c.id for c in self.questions[k].candidates} for k, v in row.items()):
                raise ValueError("valid tuple names an unknown candidate")
        if len(canonical_json(self).encode()) > 256_000:
            raise ValueError("decision request exceeds 256KB")
        return self


class RiskEvidence(Contract):
    calibration_id: str
    basis: Literal["adjudicated_decision_error"] = "adjudicated_decision_error"
    sample_size: int = Field(ge=0)
    errors: int = Field(ge=0)
    upper_error_bound: Probability
    confidence_level: Probability
    expires_at: datetime


class BranchEstimates(Contract):
    """Model estimates from experimental branch learning, not calibrated risks."""
    completion: Probability
    harm: Probability


class ExperimentalDecisionDiagnostics(Contract):
    schema_version: Literal["decision-experimental.v1"] = "decision-experimental.v1"
    architecture: Literal["experimental-recovery-vector.v1"]
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    continuation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    branches: dict[Literal["skip", "acquire"], BranchEstimates]
    acquisition_cost: float = Field(ge=0, allow_inf_nan=False)
    remaining_cost: float = Field(ge=0, allow_inf_nan=False)
    policy_reason: str = Field(min_length=1, max_length=128)
    observe_only: Literal[True] = True
    research_gate_status: Literal["failed"] = "failed"

    @model_validator(mode="after")
    def complete_branches(self):
        if set(self.branches) != {"skip", "acquire"}:
            raise ValueError("both experimental branch estimates are required")
        return self


class JointAssignment(Contract):
    check: Literal["skip", "inspect"]
    action: Literal["retry", "wait", "review"]


class GraphDecisionDiagnostics(Contract):
    schema_version: Literal["decision-graph-experimental.v1"] = "decision-graph-experimental.v1"
    architecture: Literal["experimental-relation-graph.v1"]
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    compiler_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    runtime_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assignment: JointAssignment | None
    independent_assignment: JointAssignment | None
    joint_score: float | None = Field(allow_inf_nan=False)
    independent_score: float | None = Field(allow_inf_nan=False)
    feasible_assignments: int = Field(ge=0, le=6)
    solver_status: Literal["feasible", "review"]
    remaining_cost: float = Field(ge=0, allow_inf_nan=False)
    selected_cost: float | None = Field(ge=0, allow_inf_nan=False)
    policy_reason: str = Field(min_length=1, max_length=128)
    evidence_authority: Literal["caller_asserted_diagnostic_only"] = "caller_asserted_diagnostic_only"
    observe_only: Literal[True] = True
    research_gate_status: Literal["failed"] = "failed"

    @model_validator(mode="after")
    def consistent_assignment(self):
        values = (self.assignment, self.independent_assignment, self.joint_score,
                  self.independent_score, self.selected_cost)
        if self.solver_status == "feasible":
            if any(value is None for value in values) or self.feasible_assignments < 1:
                raise ValueError("feasible graph requires complete joint and control diagnostics")
        elif any(value is not None for value in values) or self.feasible_assignments != 0:
            raise ValueError("review graph cannot carry a partial assignment")
        return self


class DecisionAnswer(Contract):
    value: str | bool | float | None = None
    candidate_id: str | None = None
    distribution: dict[str, Probability] = Field(default_factory=dict)
    decision_probability: Probability | None = None
    estimated_success: Probability | None = None
    automate: bool = False
    disposition: Literal["act", "gather", "defer"] = "defer"
    reason: str
    served_by: str
    model_version: str
    risk: RiskEvidence | None = None
    verification: list[VerificationCheck] = Field(default_factory=list)
    candidate_distribution: dict[str, Probability] | None = None
    experimental: ExperimentalDecisionDiagnostics | GraphDecisionDiagnostics | None = None

    @model_serializer(mode="wrap")
    def preserve_baseline_wire(self, handler):
        # Old signed envelopes and idempotent replays must retain their exact
        # shape. An absent experimental payload must not add a null field.
        value = handler(self)
        if self.experimental is None:
            value.pop("experimental", None)
        return value

    @model_validator(mode="after")
    def distribution_is_normalized(self) -> DecisionAnswer:
        if self.experimental is not None and (
            self.automate or self.risk is not None or self.distribution
            or self.decision_probability is not None or self.estimated_success is not None
            or self.candidate_distribution is not None or self.disposition != "defer"
        ):
            raise ValueError("experimental estimates are observe-only and cannot be probability or risk certificates")
        if self.distribution and not math.isclose(sum(self.distribution.values()), 1, abs_tol=1e-6):
            raise ValueError("distribution must sum to one")
        return self


class DecisionResponse(Contract):
    schema_version: Literal["decision.v1"] = "decision.v1"
    decision_id: UUID
    spec_hash: str
    answers: dict[str, DecisionAnswer]
    # Request-level risk must not be inferred by multiplying per-question confidence.
    automate: bool = False
    capture_status: Literal["durable"] = "durable"
    content_hash: str
    answered_units: int = Field(ge=0)
    created_at: datetime


class OutcomeEvent(Contract):
    event_id: UUID
    question: Identifier
    candidate_id: Identifier
    execution: Literal["executed", "overridden", "not_executed"]
    outcome: Literal["success", "failure", "unknown"]
    observed_at: datetime
    source: Identifier
    evidence_ref: str | None = Field(default=None, max_length=2048)
    verification: Literal["self_reported", "external"] = "self_reported"
    supersedes: UUID | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> OutcomeEvent:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must include a timezone")
        if self.execution == "not_executed" and self.outcome != "unknown":
            raise ValueError("an unexecuted action has no observed outcome")
        if self.verification == "external" and not self.evidence_ref:
            raise ValueError("external verification requires evidence_ref")
        return self
