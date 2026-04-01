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
) -> MemoryEntry
```

Creates a new version of the entry. If the key already exists, the previous version is superseded (CoW).

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
) -> list[MemoryEntry]
```

Record an outcome and update confidence on causal entries. If `causal_entry_keys` is `None`, uses auto-causal linking (all entries read in this session).

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
    amfs_version: str       # Protocol version ("0.1.0")
    entity_path: str        # Entity scope
    key: str                # Entry key
    version: int            # Version number
    value: Any              # Stored data
    provenance: Provenance  # Authorship metadata
    confidence: float       # Trust score
    outcome_count: int      # Outcomes applied
    ttl_at: datetime | None # Expiration timestamp
    embedding: list[float] | None  # Vector embedding

    @property
    def entry_key(self) -> str:
        """Canonical reference: 'entity_path/key'"""
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
    P1_INCIDENT = "p1_incident"      # × 1.15
    P2_INCIDENT = "p2_incident"      # × 1.10
    REGRESSION = "regression"        # × 1.08
    CLEAN_DEPLOY = "clean_deploy"    # × 0.97
```

---

## ConflictPolicy

```python
class ConflictPolicy(str, Enum):
    LAST_WRITE_WINS = "last_write_wins"
    RAISE = "raise"
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
) -> str (JSON)
```

{: .note }
`value` is passed as a string. If it's valid JSON, it's parsed automatically; otherwise stored as a plain string.

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
    outcome_type: str,  # "p1_incident" | "p2_incident" | "regression" | "clean_deploy"
) -> str (JSON)
```
