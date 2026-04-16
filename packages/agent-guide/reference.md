# AMFS Tool Reference

Detailed reference for all AMFS MCP tools. For behavioral guidance on when and how to use these, see [AGENT_MEMORY_GUIDE.md](AGENT_MEMORY_GUIDE.md).

---

## Identity & Context

### amfs_set_identity

Set agent identity for this conversation. Call before any other AMFS operation.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `name` | `str` | Yes | Kebab-case role name (e.g. `"api-agent"`) |
| `description` | `str` | No | What you're working on right now |

### amfs_briefing

Get compiled knowledge digests from the Memory Cortex. Returns ranked, pre-compiled summaries covering what other agents know, recent risks, and confidence-ranked facts.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity_path` | `str` | No | Scope to a specific entity (e.g. `"myapp/auth"`) |
| `agent_id` | `str` | No | Filter digests by agent |
| `limit` | `int` | No | Max digests to return (default: 10) |

---

## Read & Write

### amfs_read

Read a memory entry by entity path and key.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity_path` | `str` | Yes | Entity scope |
| `key` | `str` | Yes | Entry key |

### amfs_write

Write knowledge with automatic provenance tracking.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity_path` | `str` | Yes | Entity scope |
| `key` | `str` | Yes | Entry key |
| `value` | `any` | Yes | The knowledge (any JSON-serializable data) |
| `confidence` | `float` | No | Trust score 0.0–1.0 (default: 1.0) |
| `pattern_refs` | `list[str]` | No | Keys of related entries for cross-referencing |
| `memory_type` | `str` | No | `"fact"` (default), `"belief"`, or `"experience"` |
| `artifact_refs` | `list` | No | References to external artifacts |
| `shared` | `bool` | No | Whether other agents can see this entry (default: true) |

### amfs_search

Search entries with filters and progressive retrieval.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `query` | `str` | No | Text search query |
| `entity_path` | `str` | No | Scope to entity |
| `min_confidence` | `float` | No | Minimum confidence filter |
| `max_confidence` | `float` | No | Maximum confidence filter |
| `agent_id` | `str` | No | Filter by writing agent |
| `since` | `str` | No | ISO datetime — entries written after this |
| `pattern_ref` | `str` | No | Filter by pattern reference |
| `sort_by` | `str` | No | Sort field |
| `limit` | `int` | No | Max results |
| `depth` | `int` | No | Tier depth: 1=Hot only, 2=Hot+Warm, 3=All (default) |

### amfs_retrieve

Natural language retrieval with semantic + recency + confidence scoring.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `query` | `str` | Yes | Natural language query |
| `entity_path` | `str` | No | Scope to entity |
| `min_confidence` | `float` | No | Minimum confidence filter |
| `limit` | `int` | No | Max results |

Also accepts weight parameters for tuning semantic, recency, and confidence scoring.

### amfs_list

List entries for an entity.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity_path` | `str` | No | Scope to entity (omit for all) |

### amfs_history

Retrieve version history of an entry.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity_path` | `str` | Yes | Entity scope |
| `key` | `str` | Yes | Entry key |
| `since` | `str` | No | ISO datetime — versions after this |
| `until` | `str` | No | ISO datetime — versions before this |

### amfs_graph_neighbors

Explore the knowledge graph around an entity.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity` | `str` | Yes | Entity path to explore from |
| `relation` | `str` | No | Filter by relation type |
| `direction` | `str` | No | `"out"`, `"in"`, or `"both"` |
| `min_confidence` | `float` | No | Minimum confidence |
| `depth` | `int` | No | Traversal depth |
| `limit` | `int` | No | Max results |

---

## Agent Brain (scoped to you)

### amfs_recall

Recall YOUR OWN memory for a specific key. Only returns entries you wrote.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity_path` | `str` | Yes | Entity scope |
| `key` | `str` | Yes | Entry key |

### amfs_my_entries

List everything YOU have written.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity_path` | `str` | No | Scope to entity (omit for all) |

### amfs_read_from

Read from ANOTHER agent's memory. Creates a tracked knowledge transfer.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `agent_id` | `str` | Yes | The agent to read from |
| `entity_path` | `str` | Yes | Entity scope |
| `key` | `str` | Yes | Entry key |

### amfs_cross_agent_reads

See which other agents' memory you've read in this session.

No parameters.

---

## Decision Traces

### amfs_record_context

Capture decisions, external tool results, or user choices in the causal chain. Call as decisions happen, not at the end.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `label` | `str` | Yes | Category label (e.g. `"user-decision"`, `"datadog-metrics"`) |
| `summary` | `str` | Yes | What was decided or found |
| `source` | `str` | No | Where this came from (e.g. `"chat"`, `"Datadog APM"`) |

### amfs_commit_outcome

Record outcome — snapshots the full decision trace. **FREE (0 ops).**

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `outcome_ref` | `str` | Yes | Reference label (e.g. ticket ID, task name) |
| `outcome_type` | `str` | Yes | `"success"`, `"minor_failure"`, `"failure"`, `"critical_failure"` |

### amfs_explain

Inspect the current session's decision trace (reads, writes, contexts so far).

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `outcome_ref` | `str` | No | Filter by outcome reference |

### amfs_list_traces

Browse persisted decision traces from past sessions.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `entity_path` | `str` | No | Filter by entity |
| `agent_id` | `str` | No | Filter by agent |
| `outcome_type` | `str` | No | Filter by outcome type |
| `limit` | `int` | No | Max traces |

### amfs_get_trace

Retrieve a full decision trace by ID.

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `trace_id` | `str` | Yes | The trace ID to retrieve |

---

## Memory Type Decay Rates

| Type | Base decay | Effect |
|:-----|:-----------|:-------|
| `fact` | Normal | Standard confidence decay over time |
| `belief` | 2x faster | Hypotheses lose relevance quickly unless validated |
| `experience` | 1.5x slower | Action logs retain relevance longer for retracing |

Entries validated by successful outcomes (`amfs_commit_outcome` with `"success"`) decay slower regardless of type.

---

## Conflict Resolution

When multiple agents write to the same key:

- **Default (`last_write_wins`)** — the most recent write wins. This is fine for most cases.
- **`raise` policy** — raises `StaleWriteError` if another agent advanced the version after your read. Use for critical shared state.

If you read an entry and plan to update it, write promptly. The longer you wait, the higher the chance of a conflict.

---

## Advanced Patterns

### Progressive retrieval

Control search scope with `depth` to trade speed for completeness:

```
amfs_search(query="retry", depth=1)  # Hot tier only — fast, high-signal
amfs_search(query="retry", depth=2)  # Hot + Warm
amfs_search(query="retry", depth=3)  # All tiers (default)
```

### Semantic retrieval

Use `amfs_retrieve` for natural language queries when you're not sure of exact keys:

```
amfs_retrieve(query="how do we handle payment failures?", entity_path="myapp/checkout")
```

### Knowledge graph exploration

Traverse the knowledge graph to discover related entities:

```
amfs_graph_neighbors(entity="myapp/checkout", depth=2)
```
