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
       ┌───────────┴───────────┐
       │ AgentMemory (CoW)     │  ← Python/TypeScript SDK
       └───────────┬───────────┘
                   │
       ┌───────────┴───────────┐
       │ HTTP API / MCP Server │  ← REST + SSE / stdio + HTTP
       └───────────┬───────────┘
                   │
     ┌─────────┬───┴────┬─────────┐
     ▼         ▼        ▼         ▼
 Filesystem  Postgres   S3      Custom
  Adapter    Adapter  Adapter   Adapter
                     (ACS, R2,
                      MinIO)
```

**Key concepts:**

- **MemoryEntry** — A versioned key-value pair with provenance (who wrote it, when, why), a confidence score, and optional artifact references
- **Memory Types** — Classify entries as facts, beliefs, or experiences with type-specific decay rates
- **Copy-on-Write (CoW)** — Every write creates a new version; old versions are preserved as superseded
- **Outcome back-propagation** — When an incident or clean deploy happens, confidence scores on causal entries are automatically adjusted
- **Provenance Tiers** — Entries are automatically tiered by quality: production-validated, observed, dev, or manual
- **Adapters** — Pluggable storage backends (filesystem, Postgres, S3-compatible, or custom)
- **HTTP API** — REST + SSE server for universal access from any language or service
- **Artifact References** — Link memory entries to external blobs in S3, local files, or URLs

## Run with Docker

The fastest way to get AMFS running — no Python install required:

```bash
# HTTP API server with filesystem storage
docker run -p 8080:8080 -v amfs-data:/data ghcr.io/raia-live/amfs

# With Postgres backend
docker run -p 8080:8080 -e AMFS_POSTGRES_DSN=postgresql://user:pass@host:5432/amfs ghcr.io/raia-live/amfs

# Full stack (HTTP server + Postgres) with docker compose
docker compose up
```

**[Docker & Kubernetes guide →](https://raia-live.github.io/amfs/guides/docker/)**

## Installation

```bash
pip install amfs                    # Python SDK (includes filesystem adapter)
pip install amfs-adapter-postgres   # Postgres adapter
pip install amfs-adapter-s3         # S3-compatible adapter (AWS, ACS, MinIO, R2)
pip install amfs-http-server        # HTTP/REST API server
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
| Memory types | Classify entries as `fact`, `belief`, or `experience` with type-specific decay. |
| Provenance tiers | Entries auto-tier by quality: production-validated > observed > dev > manual. |
| Temporal queries | Retrieve the full version history of any entry, filtered by time range. |
| Causal explainability | Inspect which entries were read and how they connect to outcomes. |
| Provenance tracking | Every entry records which agent wrote it, when, and from which session. |
| Artifact references | Link memory entries to external blobs in S3, local files, or URLs. |
| HTTP/REST API | FastAPI server with 12 endpoints, SSE streaming, and API key auth. |
| Multiple adapters | Filesystem (default), Postgres (with full-text + vector search), S3-compatible, or custom. |
| MCP integration | First-class MCP server for Cursor, Claude Code, and any MCP client. |
| Connector ecosystem | Ingest events from PagerDuty, GitHub, Slack, Jira, or build your own. |
| Composite recall scoring | Rank results by relevance, recency, confidence, and outcome history. |
| Framework integrations | CrewAI, LangGraph, LangChain, AutoGen. |
| CLI tools | Inspect, diff, snapshot, and restore memory from the terminal. |
| Docker & Kubernetes | Single-command deployment with Docker or Helm chart. |
| Python & TypeScript | SDKs for both languages with the same conceptual API. |

## HTTP API

Access AMFS from any language or service over HTTP:

```bash
amfs-http --port 8080
```

```bash
# Write
curl -X POST http://localhost:8080/api/v1/entries \
  -H "Content-Type: application/json" \
  -d '{"entity_path": "checkout-service", "key": "retry-pattern", "value": {"max_retries": 3}}'

# Read
curl http://localhost:8080/api/v1/entries/checkout-service/retry-pattern

# Stream real-time events
curl http://localhost:8080/api/v1/stream
```

**[HTTP API guide →](https://raia-live.github.io/amfs/guides/http-server/)**

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

## Connectors

AMFS connectors ingest events from external systems — PagerDuty, GitHub, Slack, Jira — and turn them into structured memory entries. Install built-in connectors or build your own:

```bash
amfs connector install pagerduty   # Install a connector
amfs connector list                # See installed connectors
```

External systems send webhooks to `/api/v1/webhooks/{connector}`. AMFS handles HMAC verification, deduplication, and transformation into memory operations.

**Built-in connectors:** PagerDuty (incident lifecycle), GitHub (PRs, deploys, issues), Slack (messages, threads), Jira (issue transitions, sprints).

**Build your own:** Subclass `ConnectorABC`, define a `connector.yaml` manifest, and publish to PyPI as `amfs-connector-{name}`.

**[Connector guide →](https://raia-live.github.io/amfs/guides/connectors/)**

## OSS vs Pro

AMFS is available in two editions:

**OSS (Apache 2.0)** — The full memory primitive: CoW versioning, confidence scoring, outcome back-propagation, session-level explainability, connector ecosystem, composite recall scoring, filesystem/Postgres/S3 adapters, HTTP API, MCP server, Python & TypeScript SDKs.

**Pro (Proprietary)** — The compounding intelligence layer:
- **Multi-Tenant SaaS** — Account isolation (Postgres RLS), RBAC, scoped API keys, OAuth/OIDC, audit logging, rate limiting, usage quotas
- **Persistent Decision Traces** — Durable causal chains, historical `explain()`, precedent search across all past decisions
- **Cross-System Ingestion** — Webhook endpoint with HMAC verification, deduplication, PagerDuty/Slack/GitHub/Jira connectors
- **Automated Pattern Detection** — Recurring failures, hot entities, stale clusters, confidence drift, configurable alert rules
- **Intelligence Layer** — LLM extraction, memory critic, distiller, safety validator, multi-strategy retrieval, learned ranking
- **ML Layer** — Adaptive confidence calibration, training data export (SFT, DPO, reward model)
- **Dashboard** — Memory explorer, context graph, decision trace explorer, API key console, audit viewer, usage analytics

[See the full comparison →](https://raia-live.github.io/amfs/editions/)

## Documentation

Visit **[raia-live.github.io/amfs](https://raia-live.github.io/amfs/)** for the full documentation:

- [Getting Started](https://raia-live.github.io/amfs/getting-started/) — installation, quick start, configuration, Docker
- [Core Concepts](https://raia-live.github.io/amfs/concepts/) — memory entries, CoW, confidence, provenance
- [OSS vs Pro](https://raia-live.github.io/amfs/editions/) — feature comparison and when to use which
- [AMFS vs Vector Databases](https://raia-live.github.io/amfs/vs-vector-databases/) — when to use which, and how they complement each other
- [AMFS vs Competitors](https://raia-live.github.io/amfs/vs-competitors/) — comparison with Mem0, Cognee, Zep/Graphiti, CrewAI Memory, LangMem
- [Guides](https://raia-live.github.io/amfs/guides/) — Python SDK, TypeScript SDK, CLI, MCP, HTTP API, Docker & Kubernetes
- [Adapters](https://raia-live.github.io/amfs/adapters/) — filesystem, Postgres, S3-compatible, custom adapters
- [Integrations](https://raia-live.github.io/amfs/integrations/) — CrewAI, LangGraph, LangChain, AutoGen
- [API Reference](https://raia-live.github.io/amfs/reference/) — complete API and configuration reference
- [Contributing](https://raia-live.github.io/amfs/contributing/) — development setup, testing, code quality

## Development

```bash
git clone https://github.com/raia-live/amfs.git
cd amfs
uv pip install -e packages/core -e packages/adapters/filesystem -e packages/sdk-python -e packages/cli -e packages/http-server
uv run pytest tests/ -v
```

**[Contributing guide →](https://raia-live.github.io/amfs/contributing/)**

## License

[Apache License 2.0](LICENSE)
