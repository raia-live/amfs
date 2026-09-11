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
    importance_evaluator: ImportanceEvaluator | None = None,
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
| `importance_evaluator` | `ImportanceEvaluator?` | `None` | Scores entries on write (sets `importance_score` and `importance_dimensions`) |
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
    branch: str = "main",
) -> MemoryEntry | None
```

Returns the current version of the entry, or `None` if not found or below confidence threshold. The `branch` parameter selects which branch to read from (defaults to `main`).

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
    shared: bool = True,
    branch: str = "main",
) -> MemoryEntry
```

Creates a new version of the entry. If the key already exists, the previous version is superseded (CoW). The `memory_type` parameter controls decay behavior — `belief` decays 2× faster, `experience` decays 1.5× slower.

The `branch` parameter determines which branch the entry is written to (defaults to `main`). Every write is also logged as an event on the agent's [git-like timeline](/amfs/concepts/git-timeline/).

The optional `artifact_refs` parameter links external blobs (S3 objects, files, URLs) to this entry. See [ArtifactRef](#artifactref) below.

---

### list

```python
list(
    entity_path: str | None = None,
    *,
    include_superseded: bool = False,
    branch: str = "main",
) -> list[MemoryEntry]
```

Returns all current entries, optionally filtered by entity. Set `include_superseded=True` for full version history. The `branch` parameter selects which branch to list from.

---

### search

```python
search(
    *,
    query: str | None = None,
    entity_path: str | None = None,
    entity_paths: list[str] | None = None,
    min_confidence: float = 0.0,
    max_confidence: float | None = None,
    agent_id: str | None = None,
    since: datetime | None = None,
    pattern_ref: str | None = None,
    sort_by: str = "confidence",
    limit: int = 100,
    depth: int = 3,
    recall_config: RecallConfig | None = None,
) -> list[MemoryEntry] | list[ScoredEntry]
```

Search across all entries with rich filters. `depth` controls progressive retrieval: `1` = Hot tier only, `2` = Hot + Warm, `3` = all tiers (default).

When `query` is provided, the text is forwarded to the adapter for full-text search (Postgres tsvector). When `recall_config` is also set, returns `ScoredEntry` objects sorted by composite recall score with a `breakdown` dict.

| Behavior | `query` set | `query` not set |
|:---------|:-----------|:----------------|
| Postgres adapter | tsvector `@@` filter + `ts_rank` ordering | standard SQL filter |
| Filesystem/S3 adapter | Python substring fallback | standard filter |
| With `recall_config` | Real cosine similarity in semantic component | Semantic component is 0.0 |

---

### graph_neighbors

```python
graph_neighbors(
    entity: str,
    *,
    relation: str | None = None,
    direction: str = "both",
    min_confidence: float = 0.0,
    depth: int = 1,
    limit: int = 50,
) -> list[GraphEdge]
```

Traverse the knowledge graph from an entity. Returns edges connecting the entity to other entities, agents, and outcomes. Multi-hop traversal is supported via `depth > 1` (Postgres adapter uses recursive CTE). The Filesystem and S3 adapters return an empty list.

| Parameter | Description |
|:----------|:------------|
| `entity` | Entity to explore (e.g. `"checkout-service/retry-pattern"`) |
| `relation` | Filter by relation type (e.g. `"references"`, `"informed"`, `"learned_from"`) |
| `direction` | `"outgoing"`, `"incoming"`, or `"both"` |
| `depth` | Traversal depth (1 = direct neighbors) |

---

### timeline

```python
timeline(
    *,
    branch: str | None = None,
    event_type: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[Event]
```

Returns the git-like event log for this agent. Every write, outcome, cross-agent read, and brain brief is recorded as an event. See [Git-like Timeline](/amfs/concepts/git-timeline/) for details.

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
    recall_count: int               # Times read (in-place, no new version)
    memory_type: MemoryType         # fact, belief, or experience
    tier: int                       # 1=Hot, 2=Warm, 3=Archive
    priority_score: float | None    # Composite score for tier assignment
    importance_score: float | None  # Multi-dimensional importance (0.0–1.0)
    importance_dimensions: dict[str, float] | None  # Per-dimension breakdown
    branch: str                     # Branch name ("main" by default)
    shared: bool                    # Visible to other agents
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

## RecallConfig

```python
class RecallConfig:
    semantic_weight: float = 0.5   # Cosine similarity (requires embedder)
    recency_weight: float = 0.3    # Exponential decay by age
    confidence_weight: float = 0.2 # Entry confidence score
    recency_half_life_days: float = 30.0
```

When no embedder is configured or an entry lacks an embedding, the semantic component scores 0.0 and the remaining weights dominate.

---

## ScoredEntry

```python
class ScoredEntry:
    entry: MemoryEntry
    score: float                # Composite recall score
    breakdown: dict[str, float] # Per-signal contributions
```

---

## MemoryTier

```python
class MemoryTier(IntEnum):
    HOT = 1
    WARM = 2
    ARCHIVE = 3
```

---

## TierConfig

```python
class TierConfig:
    hot_capacity: int = 50        # Max entries in Hot tier
    warm_capacity: int = 200      # Max entries in Warm tier
    recency_weight: float = 0.4   # Weight for recency in priority score
    confidence_weight: float = 0.3
    importance_weight: float = 0.3
```

---

## PriorityScorer

```python
from amfs_core.tiering import PriorityScorer

scorer = PriorityScorer(config=TierConfig())
score = scorer.score(entry)  # Returns float
```

Computes `S = (alpha * importance + beta * recency) * freq_boost * time_decay`.

---

## TierAssigner

```python
from amfs_core.tiering import TierAssigner

assigner = TierAssigner(config=TierConfig())
assignments = assigner.assign(entries)  # Returns list[(entry_key, tier, score)]
```

Sorts entries by priority score and assigns them to Hot, Warm, or Archive based on configured capacities.

---

## ImportanceEvaluator

```python
from amfs_core.importance import ImportanceEvaluator, NoOpEvaluator

class ImportanceEvaluator(ABC):
    def evaluate(self, entity_path: str, key: str, value: Any) -> tuple[float | None, dict[str, float]]:
        """Returns (overall_score, dimension_breakdown)."""

class NoOpEvaluator(ImportanceEvaluator):
    """Returns (None, {}) — zero overhead. Used by default."""
```

Pass a custom evaluator to `AgentMemory(importance_evaluator=...)` to score entries on write. The Pro edition provides `LLMImportanceEvaluator` with 3-dimension scoring.

---

## AdapterABC (new methods)

```python
def increment_recall_count(self, entity_path: str, key: str, *, branch: str = "main") -> None:
    """In-place update — does NOT create a new CoW version."""

def update_tiers(self, updates: list[tuple[str, str, int, float]], *, branch: str = "main") -> None:
    """Batch update (entity_path, key, tier, priority_score) tuples."""
```

---

## GraphEdge

```python
class GraphEdge:
    source_entity: str
    source_type: str       # "entry", "agent", "outcome"
    relation: str          # "references", "informed", "learned_from", "co_occurs_with", "read", "wrote"
    target_entity: str
    target_type: str
    confidence: float      # Edge confidence (0.0–1.0)
    evidence_count: int    # Times this edge has been reinforced
    first_seen: datetime
    last_seen: datetime
    provenance: dict | None
```

Graph edges are materialized automatically:

| Trigger | Edge created |
|:--------|:-------------|
| `write(pattern_refs=["x"])` | `entry → references → x` |
| `commit_outcome()` | `entry → informed → outcome`, `agent → read → entry`, co-occurrence edges |
| `read_from(agent_id)` | `this_agent → learned_from → other_agent` |

---

## GraphNeighborQuery

```python
class GraphNeighborQuery:
    entity: str
    relation: str | None = None
    direction: str = "both"   # "outgoing", "incoming", "both"
    min_confidence: float = 0.0
    depth: int = 1
    limit: int = 50
```

---

## DigestType

```python
class DigestType(str, Enum):
    ENTITY = "entity"              # Summary of all knowledge about an entity
    AGENT_BRIEF = "agent_brief"    # Summary of an agent's knowledge and activity
    SOURCE = "source"              # Summary of external data from a connector
    CONNECTION_MAP = "connection_map"  # Cross-entity relationships from the knowledge graph
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

**Response format:**

The response wraps the stored entry with a quality assessment:

```json
{
  "entry": { /* full MemoryEntry fields */ },
  "quality": {
    "score": 0.85,
    "action": "stored_ok",
    "issues": []
  }
}
```

When quality issues are detected, `action` is `"stored_with_suggestions"` and `issues` contains actionable feedback:

```json
{
  "entry": { /* ... */ },
  "quality": {
    "score": 0.55,
    "action": "stored_with_suggestions",
    "issues": [
      {
        "type": "too_short",
        "message": "Value is only 12 characters...",
        "suggestion": "Rewrite with specific details..."
      }
    ]
  }
}
```

Issue types: `too_short`, `unstructured`, `missing_pattern_refs`, `belief_no_rationale`, `overconfident_belief`.

Quality feedback can be disabled by setting `AMFS_QUALITY_FEEDBACK=0`.

### amfs_search

```
amfs_search(
    query: str | None = None,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    agent_id: str | None = None,
    sort_by: str = "confidence",
    limit: int = 20,
    depth: int = 3,
) -> str (JSON)
```

`depth` controls progressive retrieval: `1` = Hot only, `2` = Hot + Warm, `3` = all (default).

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

### amfs_retrieve

```
amfs_retrieve(
    query: str,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    semantic_weight: float = 0.5,
    recency_weight: float = 0.3,
    confidence_weight: float = 0.2,
) -> str (JSON)
```

Find the most relevant memories for a natural language query. Blends semantic similarity, recency, and confidence into a single ranked list. Returns `ScoredEntry`-shaped results with score breakdowns. Requires an embedder for the semantic signal; without one, ranking uses recency and confidence only.

### amfs_graph_neighbors

```
amfs_graph_neighbors(
    entity: str,
    relation: str | None = None,
    direction: str = "both",
    min_confidence: float = 0.0,
    depth: int = 1,
    limit: int = 50,
) -> str (JSON)
```

Explore the knowledge graph around an entity. Returns edges with relation types, confidence, and evidence counts. Use `depth > 1` for multi-hop traversal (Postgres adapter only).

### amfs_timeline

```
amfs_timeline(
    agent_id: str | None = None,
    branch: str | None = None,
    event_type: str | None = None,
    since: str | None = None,
    limit: int = 100,
) -> str (JSON)
```

Returns the git-like event log for an agent. Every write, outcome, cross-agent read, and brain brief is recorded as an event. See [Git-like Timeline](/amfs/concepts/git-timeline/).

---

## HTTP REST API

When using the [HTTP API server](/amfs/guides/http-server/), the following REST endpoints are available:

### Entries

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/entries/{entity_path}/{key}?branch=main` | Read current version (branch-aware) |
| `POST` | `/api/v1/entries` | Write new entry (CoW, supports `branch` in body) |
| `GET` | `/api/v1/entries?branch=main` | List entries (branch-aware) |
| `GET` | `/api/v1/entries/{entity_path}/{key}/history` | Version history |
| `POST` | `/api/v1/search` | Search with filters (supports `branch` in body) |

All entry endpoints accept a `branch` parameter (query param for GET, body field for POST). Defaults to `main`. When targeting a non-main branch with the Pro branching module installed, the caller's API key is checked against the branch access grants.

### Agents & Timeline

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/agents` | List agents with entry counts, entities touched, and last active time |
| `GET` | `/api/v1/agents/{agent_id}/memory-graph` | Get agent's memory graph (entities and entries touched) |
| `GET` | `/api/v1/agents/{agent_id}/activity` | Get agent's activity timeline (writes, outcomes, traces). Keyset-paginated: `?limit=`, `?cursor=`, `?since=`, `?until=` |
| `GET` | `/api/v1/agents/{agent_id}/timeline` | Git-like event log (every write, outcome, read, brief). Keyset-paginated: `?limit=`, `?cursor=`, `?since=`, `?until=`, `?event_type=` |

Paginated list endpoints return the existing list key plus `next_cursor` (opaque string, `null` on the last page) and `has_more`. Pass `next_cursor` back as `?cursor=` to fetch the next page; `offset` is still honoured when no cursor is given. Page size is capped at 1000.

### Outcomes

| Method | Path | Description |
|:-------|:-----|:------------|
| `POST` | `/api/v1/outcomes` | Commit outcome |
| `GET` | `/api/v1/outcomes` | List outcomes |

### Decision Traces

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/traces` | List decision traces (supports `?outcome_type=`, `?agent_id=`, `?entity_path=`, `?limit=`, `?offset=`, `?cursor=`, `?since=` (inclusive), `?until=` (exclusive), ISO-8601). Returns `traces`, `next_cursor`, `has_more` |
| `GET` | `/api/v1/traces/{trace_id}` | Get full trace detail with causal entries, external contexts, query/error events, state diff |

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

### Branching (Pro)

These endpoints are available when the `amfs-branching` module is installed:

| Method | Path | Description |
|:-------|:-----|:------------|
| `POST` | `/api/v1/branches` | Create a branch |
| `GET` | `/api/v1/branches` | List branches |
| `GET` | `/api/v1/branches/{name}` | Get branch details |
| `DELETE` | `/api/v1/branches/{name}` | Close a branch |
| `GET` | `/api/v1/branches/{name}/diff` | Diff branch vs. main |
| `POST` | `/api/v1/branches/{name}/merge` | Merge branch into main |
| `POST` | `/api/v1/branches/{name}/access` | Grant access to a branch |
| `GET` | `/api/v1/branches/{name}/access` | List access grants |
| `DELETE` | `/api/v1/branches/{name}/access/{type}/{id}` | Revoke access |
| `POST` | `/api/v1/branches/{name}/cherry-pick` | Cherry-pick entries |
| `POST` | `/api/v1/pull-requests` | Create a pull request |
| `GET` | `/api/v1/pull-requests` | List pull requests |
| `GET` | `/api/v1/pull-requests/{id}` | Get PR with reviews |
| `POST` | `/api/v1/pull-requests/{id}/reviews` | Add a review |
| `POST` | `/api/v1/pull-requests/{id}/merge` | Merge a PR |
| `POST` | `/api/v1/pull-requests/{id}/close` | Close a PR |
| `POST` | `/api/v1/tags` | Create a snapshot tag |
| `GET` | `/api/v1/tags` | List tags |
| `DELETE` | `/api/v1/tags/{name}` | Delete a tag |
| `POST` | `/api/v1/rollback` | Rollback to timestamp or event |
| `POST` | `/api/v1/rollback/tag/{name}` | Rollback to a tag |
| `POST` | `/api/v1/fork` | Fork agent memory to a new agent |

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
    depth: int = 3,
) -> str (JSON)
```

Multi-strategy retrieval combining semantic, keyword, temporal, confidence, and learned ranking signals via Reciprocal Rank Fusion. `depth` controls tier scope (same as `amfs_search`). When a learned model is trained (via `amfs_retrain`), it automatically contributes to ranking.

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

> **Pro only.** This tool is implemented only in the private AMFS Pro MCP server. The OSS `amfs-mcp-server` package does not register it; calling it there returns an unknown-tool error.

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
