---
title: Configuration
layout: default
parent: Getting Started
nav_order: 3
description: "Configure AMFS with YAML files, environment variables, and programmatic options."
---

# Configuration

AMFS can be configured through YAML files, environment variables, or programmatically.

---

## YAML Configuration

Create an `amfs.yaml` file in your project root:

```yaml
namespace: production
layers:
  primary:
    adapter: filesystem
    options:
      root: .amfs
```

### Config File Discovery

The SDK searches for configuration files by walking up from the current directory. It looks for these filenames in order:

1. `amfs.yaml`
2. `amfs.yml`
3. `.amfs.yaml`
4. `.amfs.yml`

If no config file is found, the SDK uses sensible defaults (filesystem adapter at `.amfs/`, namespace `default`).

### Filesystem Adapter Config

```yaml
namespace: default
layers:
  primary:
    adapter: filesystem
    options:
      root: .amfs           # path to data directory (relative to config file)
```

### Postgres Adapter Config

```yaml
namespace: production
layers:
  primary:
    adapter: postgres
    options:
      dsn: postgresql://user:pass@localhost:5432/amfs_db
```

---

## Environment Variables

Environment variables override config file values:

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_POSTGRES_DSN` | Postgres connection string; overrides adapter to Postgres | — |
| `AMFS_DATA_DIR` | Custom filesystem data directory | `.amfs` |
| `AMFS_AGENT_ID` | Override auto-detected agent identity | Auto-detected |
| `AMFS_TRANSPORT` | MCP transport: `stdio` or `http` | `stdio` |
| `AMFS_HOST` | HTTP server bind host | `0.0.0.0` |
| `AMFS_PORT` | HTTP server bind port | `8000` |
| `AMFS_PATH` | HTTP server URL path | `/mcp` |

---

## Programmatic Configuration

You can bypass config files entirely and configure `AgentMemory` in code:

```python
from amfs import AgentMemory
from amfs_filesystem import FilesystemAdapter
from pathlib import Path

adapter = FilesystemAdapter(
    root=Path("/data/shared-memory"),
    namespace="production",
)

mem = AgentMemory(
    agent_id="my-agent",
    adapter=adapter,
    ttl_sweep_interval=60.0,    # sweep expired entries every 60s
    decay_half_life_days=30.0,  # confidence half-life for decay
)
```

### Constructor Options

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `agent_id` | `str` | Unique identifier for this agent (required) |
| `session_id` | `str` | Session identifier; auto-generated if omitted |
| `config_path` | `Path` | Explicit path to config file |
| `adapter` | `AdapterABC` | Pre-configured adapter instance |
| `ttl_sweep_interval` | `float` | Seconds between TTL sweep runs |
| `decay_half_life_days` | `float` | Confidence decay half-life in days |
| `embedder` | `EmbedderABC` | Embedder for semantic search |
| `conflict_policy` | `ConflictPolicy` | How to handle concurrent write conflicts |
| `on_conflict` | `Callable` | Custom conflict resolution function |

---

## Namespaces

Namespaces isolate memory between environments. Entries in different namespaces don't see each other:

```yaml
# Development
namespace: dev

# Production
namespace: production
```

The default namespace is `default`.

---

## Next Steps

- [Core Concepts](/amfs/concepts/) — understand how AMFS works
- [Adapters](/amfs/adapters/) — filesystem, Postgres, and custom adapters
- [MCP Setup](/amfs/guides/mcp/) — connect AMFS to your AI coding agents
