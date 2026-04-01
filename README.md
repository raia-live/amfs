# AMFS — Agent Memory File System

A filesystem-modeled protocol and SDK that gives multi-agent AI systems a standard way to share, persist, and version memory. AMFS provides a shared, causally-linked memory layer so agents can read each other's findings, track confidence over time, and learn from outcomes.

**[Read the documentation →](https://raia-live.github.io/amfs/)**

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

```bash
pip install amfs                    # Python SDK (includes filesystem adapter)
pip install amfs-adapter-postgres   # Postgres adapter
pip install amfs-cli                # CLI tools
npm install @amfs/sdk               # TypeScript SDK
```

## Quick Start

```python
from amfs import AgentMemory, OutcomeType

mem = AgentMemory(agent_id="review-agent")

# Write a finding
mem.write(
    "checkout-service",
    "retry-pattern",
    {"pattern": "exponential-backoff", "max_retries": 3},
    confidence=0.85,
)

# Read it back
entry = mem.read("checkout-service", "retry-pattern")
print(entry.value)       # {"pattern": "exponential-backoff", ...}
print(entry.confidence)  # 0.85

# Update it — CoW creates version 2, supersedes version 1
mem.write(
    "checkout-service",
    "retry-pattern",
    {"pattern": "exponential-backoff", "max_retries": 5},
    confidence=0.9,
)

# Record an outcome — confidence adjusts automatically
mem.commit_outcome(
    outcome_ref="INC-1042",
    outcome_type=OutcomeType.P1_INCIDENT,
    causal_entry_keys=["checkout-service/retry-pattern"],
)
```

**[See the full Quick Start guide →](https://raia-live.github.io/amfs/getting-started/quickstart/)**

## Features

| Feature | Description |
|:--------|:------------|
| Copy-on-Write versioning | Every write creates a new version. Full history is preserved. |
| Confidence & outcomes | Entries carry confidence scores that evolve based on real-world outcomes. |
| Provenance tracking | Every entry records which agent wrote it, when, and from which session. |
| Multiple adapters | Filesystem (default), Postgres, or build your own. |
| MCP integration | First-class MCP server for Cursor, Claude Code, and any MCP client. |
| Framework integrations | CrewAI, LangGraph, LangChain, AutoGen. |
| CLI tools | Inspect, diff, snapshot, and restore memory from the terminal. |
| Python & TypeScript | SDKs for both languages with the same conceptual API. |

## MCP Setup (Cursor / Claude Code)

Give your AI coding agents persistent, shared memory via MCP:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/amfs", "amfs-mcp-server"],
      "env": {}
    }
  }
}
```

**[Full MCP setup guide →](https://raia-live.github.io/amfs/guides/mcp/)**

## Documentation

Visit **[raia-live.github.io/amfs](https://raia-live.github.io/amfs/)** for the full documentation:

- [Getting Started](https://raia-live.github.io/amfs/getting-started/) — installation, quick start, configuration
- [Core Concepts](https://raia-live.github.io/amfs/concepts/) — memory entries, CoW, confidence, provenance
- [Guides](https://raia-live.github.io/amfs/guides/) — Python SDK, TypeScript SDK, CLI, MCP setup
- [Adapters](https://raia-live.github.io/amfs/adapters/) — filesystem, Postgres, custom adapters
- [Integrations](https://raia-live.github.io/amfs/integrations/) — CrewAI, LangGraph, LangChain, AutoGen
- [API Reference](https://raia-live.github.io/amfs/reference/) — complete API and configuration reference
- [Contributing](https://raia-live.github.io/amfs/contributing/) — development setup, testing, code quality

## Development

```bash
git clone https://github.com/raia-live/amfs.git
cd amfs
uv pip install -e packages/core -e packages/adapters/filesystem -e packages/sdk-python -e packages/cli
uv run pytest tests/ -v
```

**[Contributing guide →](https://raia-live.github.io/amfs/contributing/)**

## License

[Apache License 2.0](LICENSE)
