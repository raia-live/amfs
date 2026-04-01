---
title: Postgres
layout: default
parent: Adapters
nav_order: 2
description: "The Postgres adapter — shared memory across machines with database-level triggers."
---

# Postgres Adapter
{: .no_toc }

For team sharing and production deployments. Uses PostgreSQL with database-level triggers for outcome propagation and `LISTEN/NOTIFY` for real-time watch.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

```bash
pip install amfs-adapter-postgres
```

Requires PostgreSQL 14+ and `psycopg3`.

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

The adapter auto-creates two tables and associated triggers:

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
| `superseded_at` | `TIMESTAMP` | When this version was superseded (NULL = current) |

### `amfs_outcomes`

| Column | Type | Description |
|:-------|:-----|:------------|
| `id` | `SERIAL` | Primary key |
| `outcome_ref` | `TEXT` | External reference (ticket ID, deploy ID) |
| `outcome_type` | `TEXT` | One of: `p1_incident`, `p2_incident`, `regression`, `clean_deploy` |
| `causal_entry_keys` | `TEXT[]` | Array of `entity_path/key` strings |

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

## When to Use

- Team environments (multiple developers/agents sharing memory)
- Production deployments
- When you need database-level consistency guarantees
- When you want memory to survive machine restarts
