---
title: Postgres
layout: default
parent: Adapters
nav_order: 2
description: "The Postgres adapter — shared memory across machines with database-level triggers, full-text search, and vector similarity."
---

# Postgres Adapter
{: .no_toc }

For team sharing and production deployments. Uses PostgreSQL with database-level triggers for outcome propagation, `LISTEN/NOTIFY` for real-time watch, native full-text search via tsvector/GIN, and vector similarity search via pgvector.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

```bash
pip install amfs-adapter-postgres
```

Requires PostgreSQL 14+ with `psycopg3`. For vector search, install the [pgvector](https://github.com/pgvector/pgvector) extension (included in the `pgvector/pgvector` Docker image).

---

## Configuration

### YAML

```yaml
namespace: production
layers:
  primary:
    adapter: postgres
    options:
      dsn: postgresql://user:pass@localhost:5432/amfs_db
```

### Environment Variable

```bash
export AMFS_POSTGRES_DSN="postgresql://user:pass@localhost:5432/amfs_db"
```

### Programmatic

```python
from amfs_postgres import PostgresAdapter

adapter = PostgresAdapter(
    dsn="postgresql://user:pass@localhost:5432/amfs_db",
    namespace="production",
)
```

---

## Schema

The adapter auto-creates three tables and associated triggers:

### `amfs_memory_entries`

| Column | Type | Description |
|:-------|:-----|:------------|
| `id` | `SERIAL` | Primary key |
| `namespace` | `TEXT` | Namespace isolation |
| `entity_path` | `TEXT` | Entity scope |
| `key` | `TEXT` | Entry key |
| `version` | `INT` | Version number |
| `value` | `JSONB` | The stored data |
| `provenance` | `JSONB` | Agent, session, timestamp, pattern_refs |
| `confidence` | `FLOAT` | Trust score |
| `outcome_count` | `INT` | Outcomes applied |
| `recall_count` | `INT` | Times read (in-place update, default: `0`) |
| `priority_score` | `NUMERIC(10,6)` | Composite priority for tier assignment |
| `tier` | `SMALLINT` | Memory tier: `1`=Hot, `2`=Warm, `3`=Archive (default: `3`) |
| `importance_score` | `NUMERIC(6,4)` | Multi-dimensional importance (0.0–1.0) |
| `importance_dimensions` | `JSONB` | Per-dimension breakdown |
| `memory_type` | `TEXT` | Memory type: `fact`, `belief`, or `experience` (default: `fact`) |
| `artifact_refs` | `JSONB` | Linked external blobs (default: `[]`) |
| `search_tsv` | `TSVECTOR` | Auto-generated full-text search vector (GIN-indexed) |
| `embedding` | `VECTOR(384)` | Vector embedding for semantic search (HNSW-indexed, requires pgvector) |
| `superseded_at` | `TIMESTAMP` | When this version was superseded (NULL = current) |

### `amfs_outcomes`

| Column | Type | Description |
|:-------|:-----|:------------|
| `id` | `SERIAL` | Primary key |
| `outcome_ref` | `TEXT` | External reference (ticket ID, deploy ID) |
| `outcome_type` | `TEXT` | One of: `critical_failure`, `failure`, `minor_failure`, `success` |
| `causal_entry_keys` | `TEXT[]` | Array of `entity_path/key` strings |

### `amfs_knowledge_graph`

| Column | Type | Description |
|:-------|:-----|:------------|
| `id` | `UUID` | Primary key |
| `namespace` | `TEXT` | Namespace isolation |
| `source_entity` | `TEXT` | Source node (entity path, agent ID, or outcome ref) |
| `source_type` | `TEXT` | `"entry"`, `"agent"`, or `"outcome"` |
| `relation` | `TEXT` | Edge label (`"references"`, `"informed"`, `"learned_from"`, `"co_occurs_with"`, etc.) |
| `target_entity` | `TEXT` | Target node |
| `target_type` | `TEXT` | Target node type |
| `confidence` | `FLOAT` | Edge confidence (0.0–1.0) |
| `evidence_count` | `INT` | Times this edge has been reinforced |
| `first_seen` | `TIMESTAMPTZ` | When this edge was first created |
| `last_seen` | `TIMESTAMPTZ` | Most recent reinforcement |
| `provenance` | `JSONB` | Contextual metadata (session, trigger) |
| `branch` | `TEXT` | Branch scope (default: `"main"`) |

A unique constraint on `(namespace, branch, source_entity, relation, target_entity)` ensures idempotent upserts — repeated writes increment `evidence_count` and update `last_seen` rather than creating duplicates.

The knowledge graph is populated automatically by the SDK's materializers: `write()` with `pattern_refs`, `commit_outcome()`, and `read_from()` all create edges without extra code. See [API Reference](/amfs/reference/api/#graphedge) for the full model.

#### Graph Methods

| Method | Description |
|:-------|:------------|
| `upsert_graph_edge(edge)` | Insert or merge a graph edge (ON CONFLICT increments evidence) |
| `graph_neighbors(query)` | Recursive CTE traversal — returns edges within `depth` hops |
| `list_graph_edges(namespace, branch)` | List all edges in a namespace/branch |
| `graph_stats(namespace, branch)` | Edge count, unique entities, top relations |

---

## How It Works

### Write

1. `SELECT ... FOR UPDATE` locks the current version row
2. `UPDATE` sets `superseded_at` on the old row
3. `INSERT` creates the new version

### Outcome Propagation

An `AFTER INSERT` trigger on `amfs_outcomes` automatically:

1. Reads the causal entry keys
2. Supersedes the current version of each
3. Inserts a new version with updated confidence (`old × multiplier`)

### Watch (LISTEN/NOTIFY)

An `AFTER INSERT` trigger on `amfs_memory_entries` calls `pg_notify('amfs_write', ...)`. The adapter listens on this channel and dispatches to your callbacks.

---

## Docker Quick Start

```bash
docker run -d \
  --name amfs-pg \
  -e POSTGRES_DB=amfs \
  -e POSTGRES_PASSWORD=amfs \
  -p 5432:5432 \
  postgres:16

export AMFS_POSTGRES_DSN="postgresql://postgres:amfs@localhost:5432/amfs"
```

---

## Search

### Full-Text Search

The adapter automatically maintains a `search_tsv` column (GIN-indexed) that combines the `key`, `entity_path`, and `value` fields. When `SearchQuery.query` is set, the adapter uses `plainto_tsquery` with the `@@` operator for efficient in-database filtering. When `sort_by` is `"confidence"` and a `query` is present, results are ordered by `ts_rank(search_tsv, ...)` first, then by confidence — so textually relevant results float to the top.

### Vector Similarity Search (pgvector)

When an embedder is configured, the adapter stores vector embeddings in a `VECTOR(384)` column with an HNSW index. The `semantic_search()` method uses cosine similarity directly in SQL via pgvector's `<=>` operator.

To use pgvector, install the extension in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

{: .tip }
The `pgvector/pgvector:pg16` Docker image ships with pgvector pre-installed. The `docker-compose.yml` in the repo uses this image.

### Tier Indexes

Two partial indexes accelerate progressive retrieval queries:

- `idx_entries_hot` — `WHERE tier = 1 AND superseded_at IS NULL`
- `idx_entries_warm` — `WHERE tier <= 2 AND superseded_at IS NULL`

When `depth=1`, the query uses the hot index; `depth=2` uses the warm index.

### Connection Pooling

The adapter uses `psycopg_pool.ConnectionPool` for efficient connection management. The pool size is configurable and defaults to sensible limits for most deployments.

---

## When to Use

- Team environments (multiple developers/agents sharing memory)
- Production deployments with full-text and vector search
- When you need database-level consistency guarantees
- When you want memory to survive machine restarts
- When you need efficient search across large memory stores

---

## Next Steps

- [S3 Adapter](/amfs/adapters/s3/) — cloud-native storage for distributed teams
- [HTTP API Server](/amfs/guides/http-server/) — expose AMFS over REST
- [Docker & Kubernetes](/amfs/guides/docker/) — deploy AMFS + Postgres in containers
