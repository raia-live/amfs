---
title: Memory Entries
layout: default
parent: Core Concepts
nav_order: 1
description: "The fundamental unit of knowledge in AMFS."
---

# Memory Entries

A `MemoryEntry` is the fundamental unit of knowledge in AMFS. It represents a single piece of information that an agent has recorded.

---

## Structure

Every memory entry has these fields:

| Field | Type | Description |
|:------|:-----|:------------|
| `entity_path` | `str` | Hierarchical scope (e.g., `"checkout-service"` or `"myapp/auth"`) |
| `key` | `str` | Unique identifier within the entity (e.g., `"retry-pattern"`) |
| `value` | `Any` | The knowledge itself — any JSON-serializable data |
| `version` | `int` | Version number, incremented on each write |
| `confidence` | `float` | Trust score, modified by outcomes (default: `1.0`) |
| `provenance` | `Provenance` | Who wrote it, when, and from which session |
| `outcome_count` | `int` | Number of outcomes that have affected this entry |
| `ttl_at` | `datetime?` | Optional expiration timestamp |
| `embedding` | `list[float]?` | Optional vector embedding for semantic search |
| `amfs_version` | `str` | Protocol version (currently `"0.1.0"`) |

---

## Entity Paths and Keys

Memory is organized in a two-level hierarchy:

```
entity_path / key
```

- **Entity path** — Groups related entries. Think of it as the "subject" or "scope." Can be hierarchical with `/` separators (e.g., `myapp/checkout-service`).
- **Key** — Identifies a specific piece of knowledge within that entity. Typically descriptive (e.g., `retry-pattern`, `risk-race-condition`, `decision-auth-strategy`).

### Naming Conventions

| Prefix | Use Case | Example |
|:-------|:---------|:--------|
| `pattern-` | Reusable patterns | `pattern-exponential-backoff` |
| `risk-` | Known risks or bugs | `risk-race-condition` |
| `decision-` | Architectural decisions | `decision-auth-jwt` |
| `task-summary-` | What was done and why | `task-summary-refactor-checkout` |

---

## Entry Key (Canonical Reference)

The combination of `entity_path/key` forms the **entry key** — the canonical way to reference an entry across the system:

```
checkout-service/retry-pattern
myapp/auth/decision-jwt-strategy
```

Entry keys are used in `causal_entry_keys` when recording outcomes, and in `pattern_refs` for cross-referencing.

---

## Values

The `value` field accepts any JSON-serializable data:

```python
# String
mem.write("svc", "note", "Use connection pooling")

# Dict
mem.write("svc", "config", {"pool_size": 10, "timeout": 30})

# List
mem.write("svc", "endpoints", ["/api/v1/orders", "/api/v1/users"])

# Nested structures
mem.write("svc", "analysis", {
    "finding": "N+1 query in order listing",
    "impact": "high",
    "suggested_fix": "Add prefetch_related('items')",
})
```

---

## Lifecycle

A memory entry goes through these states:

```
Created (v1, current)
    │
    ▼ write same key
Superseded (v1, superseded)  ←  preserved in history
    │
New version created (v2, current)
    │
    ▼ outcome committed
Confidence updated (v2 superseded → v3 current, new confidence)
    │
    ▼ TTL expires
Archived (confidence → 0.0)
```

Every state transition creates a new version. No data is ever deleted — only superseded.
