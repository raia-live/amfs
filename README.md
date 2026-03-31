# AMFS — Agent Memory File System

A filesystem-modeled protocol and SDK that gives multi-agent AI systems a standard way to share, persist, and version memory. AMFS provides a shared, causally-linked memory layer so agents can read each other's findings, track confidence over time, and learn from outcomes.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Review Agent│     │Release Agent│     │  Other Agent │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   │
            ┌──────▼──────┐
            │ AgentMemory │  ← Python/TypeScript SDK
            │   (CoW)     │
            └──────┬──────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    Filesystem  Postgres    Redis
     Adapter    Adapter    Adapter
```

**Key concepts:**

- **MemoryEntry** — A versioned key-value pair with provenance (who wrote it, when, why) and a confidence score
- **Copy-on-Write (CoW)** — Every write creates a new version; old versions are preserved as superseded
- **Outcome back-propagation** — When an incident or clean deploy happens, confidence scores on causal entries are automatically adjusted
- **Adapters** — Pluggable storage backends (filesystem, Postgres, Redis)

## Installation

### Python SDK

```bash
pip install amfs
```

For Postgres support:

```bash
pip install amfs-adapter-postgres
```

### TypeScript SDK

```bash
npm install @amfs/sdk
```

### CLI

```bash
pip install amfs-cli
```

## Quick Start

### Python — Write and read memory

```python
from amfs import AgentMemory, OutcomeType

# Create a memory instance (defaults to filesystem adapter at .amfs/)
mem = AgentMemory(agent_id="review-agent")

# Write a finding
mem.write(
    "checkout-service",           # entity path
    "retry-pattern",              # key
    {                             # value (any JSON-serializable data)
        "pattern": "exponential-backoff",
        "max_retries": 3,
        "base_delay": "200ms",
    },
    confidence=0.85,
    pattern_refs=["retry-logic"],
)

# Read it back
entry = mem.read("checkout-service", "retry-pattern")
print(entry.value)        # {"pattern": "exponential-backoff", ...}
print(entry.version)      # 1
print(entry.confidence)   # 0.85

# Update it (CoW — creates version 2, supersedes version 1)
mem.write(
    "checkout-service",
    "retry-pattern",
    {"pattern": "exponential-backoff", "max_retries": 5},
    confidence=0.9,
)

entry = mem.read("checkout-service", "retry-pattern")
print(entry.version)  # 2

# List all entries for a service
entries = mem.list("checkout-service")
for e in entries:
    print(f"{e.key} v{e.version} confidence={e.confidence}")

# Include version history
all_versions = mem.list("checkout-service", include_superseded=True)
```

### Python — Outcome back-propagation

When an incident happens, confidence on the causal entries increases (meaning higher risk). When a clean deploy happens, confidence decays (meaning the pattern is proving safe).

```python
# A P1 incident caused by the retry pattern — confidence *= 1.15
updated = mem.commit_outcome(
    outcome_ref="INC-1042",
    outcome_type=OutcomeType.P1_INCIDENT,
    causal_entry_keys=["checkout-service/retry-pattern"],
)
print(updated[0].confidence)  # 0.9 * 1.15 = 1.035

# A clean deploy — confidence *= 0.97
updated = mem.commit_outcome(
    outcome_ref="DEP-287",
    outcome_type=OutcomeType.CLEAN_DEPLOY,
    causal_entry_keys=["checkout-service/retry-pattern"],
)
print(updated[0].confidence)  # 1.035 * 0.97 ≈ 1.004
```

Outcome multipliers:

| Outcome | Multiplier | Effect |
|---------|-----------|--------|
| `P1_INCIDENT` | ×1.15 | Increases confidence (higher risk signal) |
| `P2_INCIDENT` | ×1.10 | Increases confidence |
| `REGRESSION` | ×1.08 | Increases confidence |
| `CLEAN_DEPLOY` | ×0.97 | Decays confidence (pattern proving safe) |

### Python — Watch for real-time changes

```python
def on_change(entry):
    print(f"New write: {entry.entity_path}/{entry.key} v{entry.version}")

handle = mem.watch("checkout-service", on_change)

# ... later
handle.cancel()
```

### Python — TTL and lifecycle

```python
from datetime import datetime, timedelta, timezone

# Write an entry that expires in 24 hours
mem = AgentMemory(
    agent_id="review-agent",
    ttl_sweep_interval=60.0,  # sweep every 60 seconds
)

mem.write(
    "checkout-service",
    "temp-flag",
    {"active": True},
    ttl_at=datetime.now(timezone.utc) + timedelta(hours=24),
)
# After 24h, the lifecycle manager archives it (confidence → 0.0)
```

### Python — Context manager

```python
with AgentMemory(agent_id="review-agent") as mem:
    mem.write("svc", "key", "value")
# Automatically cleans up watchers and background threads
```

### Python — YAML configuration

Create `amfs.yaml` in your project root:

```yaml
namespace: production
layers:
  primary:
    adapter: filesystem
    options:
      root: /data/.amfs
```

Or for Postgres:

```yaml
namespace: production
layers:
  primary:
    adapter: postgres
    options:
      dsn: postgresql://user:pass@localhost/amfs_db
```

The SDK auto-discovers the config file:

```python
mem = AgentMemory(agent_id="my-agent")  # finds amfs.yaml automatically
```

### Python — Snapshots

```python
from amfs_core.snapshot import SnapshotExporter, SnapshotImporter

# Export current state
exporter = SnapshotExporter(mem.adapter)
exporter.export("backup.json")

# Restore into a different adapter
from amfs_filesystem import FilesystemAdapter
target = FilesystemAdapter(root=Path("/new-location/.amfs"), namespace="restored")
importer = SnapshotImporter(target)
importer.restore("backup.json")
```

### TypeScript — Write and read memory

```typescript
import { AgentMemory, OutcomeType } from "@amfs/sdk";

const mem = new AgentMemory("review-agent");

// Write
mem.write("checkout-service", "retry-pattern", {
  pattern: "exponential-backoff",
  maxRetries: 3,
});

// Read
const entry = mem.read("checkout-service", "retry-pattern");
console.log(entry?.value);    // { pattern: "exponential-backoff", ... }
console.log(entry?.version);  // 1

// List
const entries = mem.list("checkout-service");

// Outcome
const updated = mem.commitOutcome(
  "INC-1042",
  OutcomeType.P1_INCIDENT,
  ["checkout-service/retry-pattern"],
);
```

### CLI — Inspect and manage memory

```bash
# List all entries
amfs inspect list

# List entries for a specific entity
amfs inspect list checkout-service

# Read a specific key
amfs inspect read checkout-service retry-pattern

# Show version history diff
amfs inspect diff checkout-service retry-pattern

# Export snapshot
amfs snapshot export backup.json

# Export filtered to one entity
amfs snapshot export backup.json --entity checkout-service

# Restore from snapshot
amfs snapshot restore backup.json
```

## Framework Integrations

### CrewAI

```python
from amfs import AgentMemory
from amfs_crewai import AMFSTool

mem = AgentMemory(agent_id="crewai-agent")
tools = AMFSTool(mem).tools()
# Returns [AMFSReadTool, AMFSWriteTool, AMFSListTool]
# Pass these to your CrewAI agent
```

### LangGraph

```python
from amfs import AgentMemory
from amfs_langgraph import AMFSCheckpointer

mem = AgentMemory(agent_id="graph-agent")
checkpointer = AMFSCheckpointer(mem)
# Pass to your LangGraph graph builder
```

### AutoGen

```python
from amfs import AgentMemory
from amfs_autogen import AMFSMemoryStore

mem = AgentMemory(agent_id="autogen-agent")
store = AMFSMemoryStore(mem)
store.add("user_prefs", {"theme": "dark"})
prefs = store.get("user_prefs")
```

### LangChain

```python
from amfs import AgentMemory
from amfs_langchain import AMFSChatMemory

mem = AgentMemory(agent_id="chat-agent")
chat_memory = AMFSChatMemory(mem, session_key="conv-123")
chat_memory.save_context({"input": "hello"}, {"output": "hi there"})
history = chat_memory.load_memory_variables({})
```

## Adapter Contract

All adapters implement 5 operations:

| Operation | Description |
|-----------|-------------|
| `read(entity_path, key)` | Get current version of a key |
| `write(entry)` | Persist a new version (CoW) |
| `list(entity_path?)` | Enumerate entries |
| `watch(entity_path, callback)` | Real-time notifications |
| `commit_outcome(record)` | Back-propagate confidence |

Every adapter must pass the same contract test suite, ensuring identical behavior regardless of storage backend.

## Project Structure

```
amfs/
├── packages/
│   ├── core/                    # Models, ABC, engine, lifecycle, outcome
│   ├── adapters/
│   │   ├── filesystem/          # CoW via atomic rename
│   │   └── postgres/            # psycopg3 + triggers
│   ├── sdk-python/              # AgentMemory, config, factory
│   ├── sdk-typescript/          # @amfs/sdk (TypeScript port)
│   ├── integrations/
│   │   ├── crewai/
│   │   ├── langgraph/
│   │   ├── autogen/
│   │   └── langchain/
│   └── cli/                     # Typer CLI
└── tests/
    ├── unit/
    └── integration/
```

## Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Setup

```bash
git clone https://github.com/raia-live/amfs.git
cd amfs

# Install Python packages in editable mode
uv pip install -e packages/core -e packages/adapters/filesystem -e packages/sdk-python -e packages/cli

# Run Python tests
uv run pytest tests/ -v

# Install and test TypeScript SDK
cd packages/sdk-typescript
npm install
npm test
```

### Running Postgres adapter tests

```bash
# Start Postgres (e.g., via Docker)
docker run -d --name amfs-pg -e POSTGRES_DB=amfs_test -e POSTGRES_PASSWORD=test -p 5432:5432 postgres:16

# Run with DSN
AMFS_TEST_PG_DSN=postgresql://postgres:test@localhost/amfs_test uv run pytest tests/integration/test_postgres_adapter.py -v
```

## How It Works

### Filesystem Adapter

Entries are stored as versioned JSON files with atomic rename for crash safety:

```
.amfs/default/checkout-service/retry-pattern/
├── v001_superseded.json
├── v002_superseded.json
└── v003_current.json
```

Each write: write to `.tmp` file, then `os.rename()` (atomic on POSIX). The old `_current.json` is renamed to `_superseded.json`. Advisory `flock()` prevents concurrent-write conflicts.

### Postgres Adapter

Uses `psycopg3` with database-level triggers for back-propagation:

- **Write**: `SELECT ... FOR UPDATE` to lock the current version, then `INSERT` new version + `UPDATE` old as superseded
- **Outcome**: `INSERT` into `amfs_outcomes` fires a `AFTER INSERT` trigger that automatically supersedes and re-inserts affected entries with updated confidence
- **Watch**: `LISTEN/NOTIFY` on the `amfs_write` channel, fired by an `AFTER INSERT` trigger on `amfs_memory_entries`

### Confidence Model

Confidence starts at 1.0 and is modified by outcomes:

- Incidents **increase** confidence (the pattern is a proven risk factor)
- Clean deploys **decay** confidence (the pattern is proving safe over time)

Formula: `new_confidence = old_confidence * outcome_multiplier * causal_confidence`

Over many clean deploys, confidence trends toward 0 — meaning the risk signal fades. A single P1 incident can spike it back up.
