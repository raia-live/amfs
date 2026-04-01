---
title: Quick Start
layout: default
parent: Getting Started
nav_order: 2
description: "Write, read, search, and observe memory entries in 5 minutes."
---

# Quick Start

This guide walks you through the core AMFS operations: writing, reading, listing, searching, and observing memory.

---

## Create a Memory Instance

Every agent gets an `AgentMemory` instance identified by an `agent_id`:

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="review-agent")
```

By default, this stores data in `.amfs/` in the current directory using the filesystem adapter.

---

## Write a Memory Entry

Write a key-value pair scoped to an entity (like a service, module, or project):

```python
entry = mem.write(
    "checkout-service",       # entity_path — scope for grouping related entries
    "retry-pattern",          # key — identifier for this piece of knowledge
    {                         # value — any JSON-serializable data
        "pattern": "exponential-backoff",
        "max_retries": 3,
        "base_delay": "200ms",
    },
    confidence=0.85,          # how confident is this observation (0.0–1.0+)
    pattern_refs=["retry-logic"],  # cross-reference tags
)

print(entry.version)      # 1
print(entry.confidence)   # 0.85
print(entry.provenance.agent_id)  # "review-agent"
```

---

## Read It Back

```python
entry = mem.read("checkout-service", "retry-pattern")

print(entry.value)        # {"pattern": "exponential-backoff", ...}
print(entry.version)      # 1
print(entry.confidence)   # 0.85
```

`read()` returns `None` if the key doesn't exist. You can filter by minimum confidence:

```python
entry = mem.read("checkout-service", "retry-pattern", min_confidence=0.5)
```

---

## Update an Entry (Copy-on-Write)

Writing to the same key creates a new version. The old version is preserved as superseded:

```python
mem.write(
    "checkout-service",
    "retry-pattern",
    {"pattern": "exponential-backoff", "max_retries": 5},
    confidence=0.9,
)

entry = mem.read("checkout-service", "retry-pattern")
print(entry.version)  # 2 — new version
```

---

## List Entries

List all current entries, optionally filtered by entity:

```python
# All entries
entries = mem.list()

# Entries for a specific entity
entries = mem.list("checkout-service")
for e in entries:
    print(f"{e.key} v{e.version} confidence={e.confidence}")

# Include version history
all_versions = mem.list("checkout-service", include_superseded=True)
```

---

## Search

Search across all entries with filters:

```python
results = mem.search(
    entity_path="checkout-service",
    min_confidence=0.5,
)
for entry in results:
    print(f"{entry.entity_path}/{entry.key}: {entry.value}")
```

---

## Record Outcomes

When something significant happens — an incident, a regression, a clean deploy — record it. AMFS automatically adjusts confidence scores on related entries:

```python
from amfs import OutcomeType

# A P1 incident related to the retry pattern — confidence increases
updated = mem.commit_outcome(
    outcome_ref="INC-1042",
    outcome_type=OutcomeType.P1_INCIDENT,
    causal_entry_keys=["checkout-service/retry-pattern"],
)
print(updated[0].confidence)  # 0.9 × 1.15 = 1.035

# A clean deploy — confidence decays
updated = mem.commit_outcome(
    outcome_ref="DEP-287",
    outcome_type=OutcomeType.CLEAN_DEPLOY,
    causal_entry_keys=["checkout-service/retry-pattern"],
)
print(updated[0].confidence)  # 1.035 × 0.97 ≈ 1.004
```

{: .tip }
If you don't pass `causal_entry_keys`, AMFS uses **auto-causal linking** — it applies the outcome to every entry the agent read during the current session.

---

## Watch for Changes

Get notified in real-time when entries change:

```python
def on_change(entry):
    print(f"Updated: {entry.entity_path}/{entry.key} v{entry.version}")

handle = mem.watch("checkout-service", on_change)

# ... later, stop watching
handle.cancel()
```

---

## Context Manager

Use `AgentMemory` as a context manager to ensure clean shutdown:

```python
with AgentMemory(agent_id="review-agent") as mem:
    mem.write("svc", "key", "value")
# Watchers and background threads are cleaned up automatically
```

---

## Next Steps

- [Configuration](/amfs/getting-started/configuration/) — YAML config, adapters, and environment variables
- [Core Concepts](/amfs/concepts/) — understand CoW, confidence, and outcome propagation
- [Python SDK Guide](/amfs/guides/python/) — full SDK reference with advanced features
