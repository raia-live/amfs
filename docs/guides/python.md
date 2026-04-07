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

### With an Importance Evaluator

```python
from amfs_core.importance import ImportanceEvaluator

class MyEvaluator(ImportanceEvaluator):
    def evaluate(self, entity_path, key, value):
        score = 0.8 if "critical" in str(value).lower() else 0.3
        return score, {"criticality": score}

mem = AgentMemory(agent_id="my-agent", importance_evaluator=MyEvaluator())
```

Every `write()` call will automatically score the entry and set `importance_score` and `importance_dimensions`. If the evaluator raises, the write proceeds without scoring.

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
    memory_type=MemoryType.FACT, # optional: fact (default), belief, or experience
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
results = mem.search(entity_path="checkout-service", min_confidence=0.5)
```

Progressive retrieval with `depth` — search only high-priority tiers for fast, high-signal results:

```python
hot_only = mem.search(query="retry strategy", depth=1)   # Hot tier
hot_warm = mem.search(query="retry strategy", depth=2)   # Hot + Warm
all_tiers = mem.search(query="retry strategy")            # All (default)
```

Composite recall scoring (blends semantic similarity, recency, and confidence):

```python
from amfs_core.models import RecallConfig

scored = mem.search(
    query="how do we handle retries?",
    recall_config=RecallConfig(semantic_weight=0.5, recency_weight=0.3, confidence_weight=0.2),
)
for item in scored:
    print(f"{item.entry.key} — score={item.score:.3f}")
```

{: .note }
Semantic scoring requires an `embedder`. Without one, the semantic component is 0.0.

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
    outcome_type=OutcomeType.CRITICAL_FAILURE,
    causal_entry_keys=["checkout-service/retry-pattern"],
)

# With auto-causal linking (uses everything read in this session)
updated = mem.commit_outcome(
    outcome_ref="DEP-300",
    outcome_type=OutcomeType.SUCCESS,
)
```

### Outcome Types

```python
OutcomeType.CRITICAL_FAILURE  # × 1.15
OutcomeType.FAILURE           # × 1.10
OutcomeType.MINOR_FAILURE     # × 1.08
OutcomeType.SUCCESS           # × 0.97
```

---

## Memory Types

Classify entries to control decay behavior:

```python
from amfs import MemoryType

# Facts (default) — objective knowledge, standard decay
mem.write("svc", "config", {"pool_size": 10}, memory_type=MemoryType.FACT)

# Beliefs — subjective inferences, decay 2× faster
mem.write("svc", "hypothesis", "Likely an N+1 query issue", memory_type=MemoryType.BELIEF)

# Experiences — action logs, decay 1.5× slower
mem.write("svc", "action-log", "Added index on user_id", memory_type=MemoryType.EXPERIENCE)
```

---

## History (Temporal Queries)

Retrieve the full version history of an entry with optional time filtering:

```python
from datetime import datetime, timedelta, timezone

# All versions
versions = mem.history("checkout-service", "retry-pattern")
for v in versions:
    print(f"v{v.version} — confidence: {v.confidence} — {v.provenance.written_at}")

# Versions from the last 7 days
since = datetime.now(timezone.utc) - timedelta(days=7)
recent = mem.history("checkout-service", "retry-pattern", since=since)
```

---

## Explainability

Inspect the causal chain — which entries were read during the current session and how they connect to outcomes:

```python
chain = mem.explain()
print(chain["session_id"])
print(chain["causal_keys"])   # list of entity_path/key pairs that were read
print(chain["entries"])       # full entry details for each causal key
```

Filter by outcome reference:

```python
chain = mem.explain(outcome_ref="INC-1042")
```

---

## Tool Context

When agents call external tools or APIs, there are two ways to capture that context in AMFS depending on your needs.

### Record in the causal chain (lightweight)

Use `record_context()` to add external inputs to the causal chain without writing to storage. This makes `explain()` return a complete decision trace:

```python
entry = mem.read("checkout-service", "retry-pattern")

mem.record_context(
    "pagerduty-incidents",
    "3 SEV-1 incidents in the last 24h for checkout-service",
    source="PagerDuty API",
)
mem.record_context(
    "git-log",
    "15 commits since last deploy, 3 touching retry logic",
    source="git",
)

mem.commit_outcome("DEP-500", OutcomeType.SUCCESS)

chain = mem.explain()
print(chain["causal_entries"])     # AMFS entries that were read
print(chain["external_contexts"])  # tool/API inputs that informed the decision
```

### Persist for other agents (durable)

Use `MemoryType.EXPERIENCE` with a TTL to store tool results so downstream agents can retrieve them:

```python
from datetime import datetime, timedelta, timezone

mem.write(
    "checkout-service",
    "tool-result-pagerduty",
    {"incidents": 3, "sev1": True, "last_24h": True},
    memory_type=MemoryType.EXPERIENCE,
    ttl_at=datetime.now(timezone.utc) + timedelta(hours=1),
)
```

The next agent reads it with `mem.read("checkout-service", "tool-result-pagerduty")` instead of re-calling the API.

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

## Briefing (Memory Cortex)

When the Memory Cortex is running, agents can retrieve pre-compiled knowledge digests ranked by relevance. This is how agents consume the "brain brief" — compiled summaries of entities, other agents, and external sources — without having to search through raw memory entries.

### Basic Usage

```python
mem = AgentMemory(agent_id="deploy-agent", adapter=adapter)

# Get a ranked briefing of compiled knowledge
briefs = mem.briefing(entity_path="myapp/checkout-service", limit=5)

for digest in briefs:
    print(digest.digest_type)   # "entity", "agent_brief", "source", or "connection_map"
    print(digest.scope)         # the entity path, agent ID, or source ID
    print(digest.summary)       # structured summary (varies by digest type)
    print(digest.entry_count)   # number of source entries compiled
    print(digest.compiled_at)   # when the digest was last compiled
```

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `entity_path` | `str \| None` | Focus on digests relevant to this entity |
| `agent_id` | `str \| None` | Focus on digests relevant to this agent |
| `limit` | `int` | Maximum number of digests to return (default: 10) |

### Digest Types

| Type | Scope | What It Contains |
|:-----|:------|:-----------------|
| `entity` | Entity path (e.g. `myapp/checkout-service`) | Summary of all knowledge about an entity — key count, average confidence, top keys, narrative |
| `agent_brief` | Agent ID (e.g. `deploy-agent`) | Summary of an agent's knowledge and activity — entries written, entities touched, outcomes |
| `source` | Source ID (e.g. `github`) | Summary of external data from a connector — events ingested, entities touched |
| `connection_map` | Cross-entity scope | Cross-entity relationships (Pro) |

### Workflow Integration

Call `briefing()` at the start of a task to get context before making decisions:

```python
with AgentMemory(agent_id="deploy-agent", adapter=adapter) as mem:
    # Step 1: Get a briefing on what you need to know
    briefs = mem.briefing(entity_path="myapp/checkout-service", limit=5)
    for digest in briefs:
        print(f"[{digest.digest_type}] {digest.scope}: {digest.summary.get('narrative', '')}")

    # Step 2: Read specific entries based on the briefing
    entry = mem.read("myapp/checkout-service", "retry-pattern")

    # Step 3: Do your work, record context
    mem.record_context("ci-pipeline", "All tests passing, deploy ready", source="GitHub Actions")

    # Step 4: Commit the outcome
    mem.commit_outcome("DEP-500", OutcomeType.SUCCESS)
```

If the Cortex is not running, `briefing()` returns an empty list — your agent code can safely call it without checking.

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

## Knowledge Graph

The knowledge graph builds automatically as agents write, commit outcomes, and learn from each other. You can also traverse it directly:

```python
edges = mem.graph_neighbors(
    "checkout-service/retry-pattern",
    direction="both",
    depth=2,
    min_confidence=0.5,
)
for edge in edges:
    print(f"{edge.source_entity} --{edge.relation}--> {edge.target_entity}")
```

| Parameter | Description |
|:----------|:------------|
| `entity` | Starting entity to explore |
| `relation` | Filter by relation type (e.g. `"references"`, `"informed"`) |
| `direction` | `"outgoing"`, `"incoming"`, or `"both"` |
| `depth` | Traversal depth (1 = direct neighbors, >1 for multi-hop) |

{: .note }
Multi-hop traversal (`depth > 1`) requires the Postgres adapter. The Filesystem and S3 adapters return an empty list for graph methods.

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
