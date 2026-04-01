---
title: Python SDK
layout: default
parent: Guides
nav_order: 1
description: "Complete guide to the AMFS Python SDK."
---

# Python SDK
{: .no_toc }

The Python SDK provides the `AgentMemory` class — the primary interface for reading, writing, and managing agent memory.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

```bash
pip install amfs
```

---

## Creating an Instance

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="my-agent")
```

### With Custom Configuration

```python
from pathlib import Path

mem = AgentMemory(
    agent_id="my-agent",
    config_path=Path("./custom-amfs.yaml"),
    ttl_sweep_interval=60.0,
    decay_half_life_days=30.0,
)
```

### With a Pre-configured Adapter

```python
from amfs_filesystem import FilesystemAdapter

adapter = FilesystemAdapter(root=Path(".amfs"), namespace="staging")
mem = AgentMemory(agent_id="my-agent", adapter=adapter)
```

---

## Core Operations

### Write

```python
entry = mem.write(
    "checkout-service",          # entity_path
    "retry-pattern",             # key
    {"max_retries": 3},          # value (any JSON-serializable data)
    confidence=0.85,             # optional, default 1.0
    pattern_refs=["retry"],      # optional cross-references
)
```

Write with a TTL (time-to-live):

```python
from datetime import datetime, timedelta, timezone

mem.write(
    "svc", "temp-flag", {"active": True},
    ttl_at=datetime.now(timezone.utc) + timedelta(hours=24),
)
```

### Read

```python
entry = mem.read("checkout-service", "retry-pattern")

if entry:
    print(entry.value)
    print(entry.version)
    print(entry.confidence)
```

With minimum confidence filter:

```python
entry = mem.read("svc", "pattern", min_confidence=0.5)
```

### List

```python
# All entries
entries = mem.list()

# Entries for a specific entity
entries = mem.list("checkout-service")

# Include superseded versions
entries = mem.list("checkout-service", include_superseded=True)
```

### Search

```python
results = mem.search(
    entity_path="checkout-service",   # optional filter
    min_confidence=0.5,               # optional filter
)
```

### Stats

```python
stats = mem.stats()
print(f"Total entries: {stats.total_entries}")
print(f"Total outcomes: {stats.total_outcomes}")
```

---

## Outcomes

### Recording Outcomes

```python
from amfs import OutcomeType

# With explicit causal keys
updated = mem.commit_outcome(
    outcome_ref="INC-1042",
    outcome_type=OutcomeType.P1_INCIDENT,
    causal_entry_keys=["checkout-service/retry-pattern"],
)

# With auto-causal linking (uses everything read in this session)
updated = mem.commit_outcome(
    outcome_ref="DEP-300",
    outcome_type=OutcomeType.CLEAN_DEPLOY,
)
```

### Outcome Types

```python
OutcomeType.P1_INCIDENT    # × 1.15
OutcomeType.P2_INCIDENT    # × 1.10
OutcomeType.REGRESSION     # × 1.08
OutcomeType.CLEAN_DEPLOY   # × 0.97
```

---

## Watch

Get real-time notifications when entries change:

```python
def on_change(entry):
    print(f"{entry.key} updated to v{entry.version}")

handle = mem.watch("checkout-service", on_change)

# Stop watching
handle.cancel()
```

---

## Snapshots

Export and import the full state of your memory:

```python
from amfs_core.snapshot import SnapshotExporter, SnapshotImporter

# Export
exporter = SnapshotExporter(mem.adapter)
exporter.export("backup.json")

# Import into a different adapter
from amfs_filesystem import FilesystemAdapter
target = FilesystemAdapter(root=Path("/new/.amfs"), namespace="restored")
importer = SnapshotImporter(target)
importer.restore("backup.json")
```

---

## Semantic Search

If you configure an embedder, you can search by meaning:

```python
results = mem.semantic_search("how do we handle retries?", top_k=5)
for entry, score in results:
    print(f"{entry.key} (similarity: {score:.3f})")
```

---

## Context Manager

Use `AgentMemory` as a context manager for automatic cleanup:

```python
with AgentMemory(agent_id="my-agent") as mem:
    mem.write("svc", "key", "value")
    entry = mem.read("svc", "key")
# Watchers, TTL sweepers, and background threads are cleaned up
```

---

## Conflict Handling

Handle concurrent writes to the same key:

```python
from amfs_core.models import ConflictPolicy

# Raise an error on conflict
mem = AgentMemory(
    agent_id="my-agent",
    conflict_policy=ConflictPolicy.RAISE,
)

# Custom conflict resolution
def merge(existing, incoming, value):
    return {**existing.value, **value}

mem = AgentMemory(
    agent_id="my-agent",
    on_conflict=merge,
)
```
