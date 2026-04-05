---
title: API Reference
layout: default
parent: Reference
nav_order: 1
description: "Complete API reference for AgentMemory, MemoryEntry, and related classes."
---

# API Reference
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## AgentMemory

The primary interface for reading, writing, and managing agent memory.

### Constructor

```python
AgentMemory(
    agent_id: str,
    *,
    session_id: str | None = None,
    config_path: Path | None = None,
    adapter: AdapterABC | None = None,
    ttl_sweep_interval: float | None = None,
    decay_half_life_days: float | None = None,
    embedder: EmbedderABC | None = None,
    conflict_policy: ConflictPolicy = ConflictPolicy.LAST_WRITE_WINS,
    on_conflict: Callable | None = None,
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `agent_id` | `str` | required | Unique identifier for this agent |
| `session_id` | `str?` | auto-generated | Session ID for provenance |
| `config_path` | `Path?` | auto-discovered | Path to `amfs.yaml` |
| `adapter` | `AdapterABC?` | from config | Pre-configured adapter |
| `ttl_sweep_interval` | `float?` | `None` | Seconds between TTL sweeps |
| `decay_half_life_days` | `float?` | `None` | Confidence decay half-life |
| `embedder` | `EmbedderABC?` | `None` | Embedder for semantic search |
| `conflict_policy` | `ConflictPolicy` | `LAST_WRITE_WINS` | Concurrent write strategy |
| `on_conflict` | `Callable?` | `None` | Custom conflict resolver |

---

### read

```python
read(
    entity_path: str,
    key: str,
    *,
    min_confidence: float = 0.0,
) -> MemoryEntry | None
```

Returns the current version of the entry, or `None` if not found or below confidence threshold.

---

### write

```python
write(
    entity_path: str,
    key: str,
    value: Any,
    *,
    confidence: float = 1.0,
    ttl_at: datetime | None = None,
    pattern_refs: list[str] | None = None,
    memory_type: MemoryType = MemoryType.FACT,
    artifact_refs: list[ArtifactRef] | None = None,
) -> MemoryEntry
```

Creates a new version of the entry. If the key already exists, the previous version is superseded (CoW). The `memory_type` parameter controls decay behavior — `belief` decays 2× faster, `experience` decays 1.5× slower.

The optional `artifact_refs` parameter links external blobs (S3 objects, files, URLs) to this entry. See [ArtifactRef](#artifactref) below.

---

### list

```python
list(
    entity_path: str | None = None,
    *,
    include_superseded: bool = False,
) -> list[MemoryEntry]
```

Returns all current entries, optionally filtered by entity. Set `include_superseded=True` for full version history.

---

### search

```python
search(
    *,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    agent_id: str | None = None,
    sort_by: str = "confidence",
    limit: int = 20,
) -> list[MemoryEntry]
```

Search across all entries with filters.

---

### semantic_search

```python
semantic_search(
    text: str,
    *,
    top_k: int = 10,
) -> list[tuple[MemoryEntry, float]]
```

Search by meaning using vector similarity. Requires an `embedder` to be configured.

---

### watch

```python
watch(
    entity_path: str,
    callback: Callable[[MemoryEntry], None],
) -> WatchHandle
```

Register a callback for real-time change notifications. Returns a handle with a `cancel()` method.

---

### commit_outcome

```python
commit_outcome(
    outcome_ref: str,
    outcome_type: OutcomeType,
    causal_entry_keys: list[str] | None = None,
    *,
    causal_confidence: float = 1.0,
    decision_summary: str | None = None,
) -> list[MemoryEntry]
```

Record an outcome and update confidence on causal entries. If `causal_entry_keys` is `None`, uses auto-causal linking (all entries read in this session).

The optional `decision_summary` parameter adds a human-readable description of the decision to the persisted trace.

When called, the trace automatically captures:
- **Causal entry snapshots** with full `value`, `memory_type`, `written_by`, and `read_at` timestamps
- **Query events** from all `search()` and `list()` calls during the session, with parameters, result counts, and per-operation latency
- **Error events** from any failed operations
- **Session timing** — `session_started_at`, `session_ended_at`, `session_duration_ms`
- **State diff** — entries created, updated, and confidence changes

---

### history

```python
history(
    entity_path: str,
    key: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[MemoryEntry]
```

Returns all versions of an entry, optionally filtered by time range. Entries are sorted by version ascending.

---

### record_context

```python
record_context(
    label: str,
    summary: str,
    *,
    source: str | None = None,
) -> None
```

Record external context (tool call, API response, database query) in the causal chain without writing to storage. These appear in the `external_contexts` field of `explain()` output, making decision traces complete.

---

### explain

```python
explain(
    outcome_ref: str | None = None,
) -> dict[str, Any]
```

Returns the causal chain for the current session: which AMFS entries were read, which external contexts were recorded, and their details. If `outcome_ref` is provided, labels the explanation with that reference.

Returns:

```python
{
    "outcome_ref": str | None,
    "agent_id": str,
    "session_id": str,
    "causal_chain_length": int,
    "causal_entries": list[dict],       # AMFS entries that were read
    "external_contexts": list[dict],    # tool/API inputs via record_context()
}
```

---

### briefing

```python
briefing(
    entity_path: str | None = None,
    agent_id: str | None = None,
    limit: int = 10,
) -> list[Digest]
```

Get a ranked briefing of compiled knowledge digests from the Memory Cortex. Returns pre-compiled `Digest` objects ranked by relevance to the given entity or agent context. If no Cortex is running, returns an empty list.

---

### stats

```python
stats() -> MemoryStats
```

Returns memory statistics.

---

## MemoryEntry

```python
class MemoryEntry:
    amfs_version: str               # Protocol version ("0.2.0")
    entity_path: str                # Entity scope
    key: str                        # Entry key
    version: int                    # Version number
    value: Any                      # Stored data
    provenance: Provenance          # Authorship metadata
    confidence: float               # Trust score
    outcome_count: int              # Outcomes applied
    memory_type: MemoryType         # fact, belief, or experience
    ttl_at: datetime | None         # Expiration timestamp
    embedding: list[float] | None   # Vector embedding
    artifact_refs: list[ArtifactRef]  # Linked external blobs

    @property
    def entry_key(self) -> str:
        """Canonical reference: 'entity_path/key'"""

    @property
    def provenance_tier(self) -> ProvenanceTier:
        """Computed quality tier based on agent ID and outcome history."""
```

---

## ArtifactRef

Link memory entries to external blobs — model weights, datasets, logs, screenshots, or any binary artifact stored outside AMFS.

```python
class ArtifactRef:
    uri: str                    # S3 URI, file path, or URL
    media_type: str | None      # MIME type (e.g. "application/json")
    label: str | None           # Human-readable label
    size_bytes: int | None      # File size in bytes
```

Example:

```python
from amfs_core.models import ArtifactRef

mem.write(
    "training-pipeline",
    "model-v3-checkpoint",
    {"epoch": 42, "loss": 0.023},
    confidence=0.95,
    artifact_refs=[
        ArtifactRef(
            uri="s3://my-bucket/models/v3/checkpoint.pt",
            media_type="application/octet-stream",
            label="Model checkpoint",
            size_bytes=1_500_000_000,
        ),
    ],
)
```

---

## Provenance

```python
class Provenance:
    agent_id: str           # Who wrote it
    session_id: str         # Which session
    written_at: datetime    # When
    pattern_refs: list[str] # Cross-references
```

---

## OutcomeType

```python
class OutcomeType(str, Enum):
    CRITICAL_FAILURE = "critical_failure"  # × 1.15
    FAILURE = "failure"                    # × 1.10
    MINOR_FAILURE = "minor_failure"        # × 1.08
    SUCCESS = "success"                    # × 0.97

    # Legacy aliases (deprecated — will be removed in a future version)
    P1_INCIDENT = "critical_failure"
    P2_INCIDENT = "failure"
    REGRESSION = "minor_failure"
    CLEAN_DEPLOY = "success"
```

---

## MemoryType

```python
class MemoryType(str, Enum):
    FACT = "fact"              # Objective knowledge, standard decay
    BELIEF = "belief"          # Subjective inference, 2× faster decay
    EXPERIENCE = "experience"  # Action log, 1.5× slower decay
```

---

## ProvenanceTier

```python
class ProvenanceTier(int, Enum):
    PRODUCTION_VALIDATED = 1   # Production agent + outcomes applied
    PRODUCTION_OBSERVED = 2    # Production agent, no outcomes yet
    DEVELOPMENT = 3            # Dev/test environment
    MANUAL = 4                 # Manually seeded
```

---

## ConflictPolicy

```python
class ConflictPolicy(str, Enum):
    LAST_WRITE_WINS = "last_write_wins"
    RAISE = "raise"
```

---

## DigestType

```python
class DigestType(str, Enum):
    ENTITY = "entity"              # Summary of all knowledge about an entity
    AGENT_BRIEF = "agent_brief"    # Summary of an agent's knowledge and activity
    SOURCE = "source"              # Summary of external data from a connector
    CONNECTION_MAP = "connection_map"  # Cross-entity relationships (Pro)
```

---

## Digest

A compiled knowledge digest produced by the Memory Cortex.

```python
class Digest:
    digest_type: DigestType
    scope: str                          # Entity path, agent ID, or source ID
    summary: dict[str, Any]             # Structured summary (varies by type)
    entry_count: int                    # Number of source entries
    source_agents: list[str]            # Agents that contributed
    compiled_at: datetime               # When this digest was last compiled
    staleness_ms: int                   # Age since compilation (set at query time)
    anticipation_score: float           # Outcome-calibrated relevance (0.0–1.0)
    namespace: str                      # Memory namespace
```

---

## MemoryStats

```python
class MemoryStats:
    total_entries: int
    total_outcomes: int
    entities: list[str]
```

---

## MCP Tools

When used via MCP, the following tool signatures are exposed:

### amfs_read

```
amfs_read(entity_path: str, key: str) -> str (JSON)
```

### amfs_write

```
amfs_write(
    entity_path: str,
    key: str,
    value: str,
    confidence: float = 1.0,
    pattern_refs: list[str] | None = None,
    memory_type: str = "fact",  # "fact" | "belief" | "experience"
    artifact_refs: list[dict] | None = None,
) -> str (JSON)
```

{: .note }
`value` is passed as a string. If it's valid JSON, it's parsed automatically; otherwise stored as a plain string.

Each item in `artifact_refs` should be a dict with `uri` (required), and optionally `media_type`, `label`, and `size_bytes`.

### amfs_search

```
amfs_search(
    query: str | None = None,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    agent_id: str | None = None,
    sort_by: str = "confidence",
    limit: int = 20,
) -> str (JSON)
```

### amfs_list

```
amfs_list(entity_path: str | None = None) -> str (JSON)
```

### amfs_stats

```
amfs_stats() -> str (JSON)
```

### amfs_commit_outcome

```
amfs_commit_outcome(
    outcome_ref: str,
    outcome_type: str,  # "critical_failure" | "failure" | "minor_failure" | "success" (legacy: "p1_incident" | "p2_incident" | "regression" | "clean_deploy")
) -> str (JSON)
```

### amfs_record_context

```
amfs_record_context(
    label: str,
    summary: str,
    source: str = "",
) -> str (JSON)
```

Record external context (tool call, API response) in the causal chain. Appears in `amfs_explain()` output.

### amfs_history

```
amfs_history(
    entity_path: str,
    key: str,
    since: str | None = None,  # ISO 8601 datetime
    until: str | None = None,  # ISO 8601 datetime
) -> str (JSON)
```

Returns all versions of an entry, optionally bounded by a time range. Dates are ISO 8601 strings.

### amfs_explain

```
amfs_explain(
    outcome_ref: str | None = None,
) -> str (JSON)
```

Returns the causal read chain for the current session: which entries were read and their details.

---

## HTTP REST API

When using the [HTTP API server](/amfs/guides/http-server/), the following REST endpoints are available:

### Entries

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/entries/{entity_path}/{key}` | Read current version |
| `POST` | `/api/v1/entries` | Write new entry (CoW) |
| `GET` | `/api/v1/entries` | List entries |
| `GET` | `/api/v1/entries/{entity_path}/{key}/history` | Version history |
| `GET` | `/api/v1/search` | Search with filters |

### Outcomes

| Method | Path | Description |
|:-------|:-----|:------------|
| `POST` | `/api/v1/outcomes` | Commit outcome |
| `GET` | `/api/v1/outcomes` | List outcomes |

### Decision Traces

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/traces` | List decision traces (supports `?outcome_type=`, `?agent_id=`, `?limit=`) |
| `GET` | `/api/v1/traces/{trace_id}` | Get full trace detail with causal entries, external contexts, query/error events, state diff |

### Agents

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/agents` | List agents with entry counts, entities touched, and last active time |
| `GET` | `/api/v1/agents/{agent_id}/memory-graph` | Get agent's memory graph (entities and entries touched) |
| `GET` | `/api/v1/agents/{agent_id}/activity` | Get agent's activity timeline (writes, outcomes, traces) |

### Observability

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/stats` | Memory statistics |
| `POST` | `/api/v1/context` | Record external context |
| `GET` | `/api/v1/explain` | Causal trace |
| `GET` | `/api/v1/stream` | SSE event stream |
| `GET` | `/api/v1/admin/usage` | Usage statistics and metrics |
| `GET` | `/health` | Health check |

### Admin — API Keys

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/admin/api-keys` | List API keys |
| `POST` | `/api/v1/admin/api-keys` | Create a new API key |
| `DELETE` | `/api/v1/admin/api-keys/{key_id}` | Revoke an API key |

### Admin — Audit Log

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/admin/audit` | List audit log entries |

Authentication is via the `X-AMFS-API-Key` header. Set `AMFS_API_KEYS` to enable. Interactive API docs are available at `/docs` (Swagger UI).

---

## Pro MCP Tools

The following tools are available only with the AMFS Pro MCP server.

### amfs_critique

```
amfs_critique() -> str (JSON)
```

Run the Memory Critic to detect toxic, stale, contradictory, uncalibrated, and orphaned entries.

### amfs_briefing

```
amfs_briefing(
    entity_path: str | None = None,
    agent_id: str | None = None,
    limit: int = 10,
) -> str (JSON list of Digest objects)
```

Get a compiled knowledge briefing — what you should know right now. Returns pre-compiled digests from the Memory Cortex ranked by relevance. Includes entity summaries, agent brain briefs, and external source summaries.

### amfs_distill

```
amfs_distill(
    min_confidence: float = 0.3,
    max_entries: int = 500,
) -> str (JSON)
```

Generate a distilled memory set for bootstrapping new agents.

### amfs_validate

```
amfs_validate(
    entity_path: str,
    key: str,
    value: str,
    confidence: float = 1.0,
    memory_type: str = "fact",
) -> str (JSON)
```

Validate a proposed memory write against safety checks (contradiction detection, temporal consistency, confidence thresholds).

### amfs_retrieve

```
amfs_retrieve(
    query: str,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 10,
) -> str (JSON)
```

Multi-strategy retrieval combining semantic, keyword, temporal, confidence, and learned ranking signals via Reciprocal Rank Fusion. When a learned model is trained (via `amfs_retrain`), it automatically contributes to ranking.

### amfs_retrain

```
amfs_retrain(
    entity_path: str | None = None,
) -> str (JSON)
```

Train (or retrain) the learned ranking model from outcome data. Requires at least 20 outcome-linked entries. Returns training metrics including accuracy, sample counts, and feature importances. Once trained, the model enhances `amfs_retrieve` results automatically.

### amfs_calibrate

```
amfs_calibrate(
    entity_path: str | None = None,
    per_entity: bool = false,
) -> str (JSON)
```

Learn optimal confidence multipliers from historical outcome data. Returns calibrated multipliers and estimated decay half-life. Set `per_entity=true` to also produce entity-specific overrides.

### amfs_export_training_data

```
amfs_export_training_data(
    format: str = "sft",
    entity_path: str | None = None,
    limit: int = 10000,
) -> str (JSON)
```

Export decision traces as fine-tuning datasets. Format options: `"sft"` (supervised fine-tuning), `"dpo"` (direct preference optimization), `"reward_model"` (reward model training). See the [ML Layer guide](/amfs/guides/ml-layer/) for format details.

### amfs_record_llm_call

```
amfs_record_llm_call(
    model: str,
    provider: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: float = 0,
    cost_usd: float = 0,
    temperature: float | None = None,
    max_tokens: int | None = None,
    finish_reason: str | None = None,
    error: str | None = None,
) -> str (JSON)
```

Record an LLM call in the current decision trace. Captures model, provider, token counts, cost, latency, and sampling parameters. Aggregated as `total_llm_calls`, `total_tokens`, and `total_cost_usd` in the trace.
