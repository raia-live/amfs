---
title: Home
layout: home
nav_order: 1
description: "AMFS — Agent Memory File System. Shared, versioned memory for multi-agent AI systems."
permalink: /
---

# Agent Memory File System
{: .fs-9 }

A shared, causally-linked memory layer so AI agents can read each other's findings, track confidence over time, and learn from outcomes.
{: .fs-6 .fw-300 }

[Get Started](/amfs/getting-started/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/raia-live/amfs){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What is AMFS?

AMFS is a filesystem-modeled protocol and SDK that gives multi-agent AI systems a standard way to **share**, **persist**, and **version** memory. It works like a version-controlled knowledge base that agents can read from and write to — across sessions, machines, and frameworks.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Review Agent│     │Release Agent│     │  Other Agent │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   │
       ┌───────────┴───────────┐
       │ AgentMemory (CoW)     │  ← Python / TypeScript SDK
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
```

### Why AMFS?

AI agents today are stateless. Each session starts from scratch — no memory of past decisions, patterns, or mistakes. When multiple agents work on the same codebase, they duplicate work and repeat errors.

AMFS solves this by giving agents a **shared memory layer** with:

- **Versioned knowledge** — Every write creates a new version. Nothing is lost.
- **Provenance tracking** — Know which agent wrote what, and when.
- **Confidence scoring** — Entries carry a confidence score that evolves over time based on real-world outcomes.
- **Outcome back-propagation** — Incidents increase confidence on risky patterns. Clean deploys decay it.
- **Pluggable storage** — Filesystem, Postgres (with full-text + vector search), S3-compatible, or custom.
- **HTTP API** — Access AMFS from any language or service over REST. Real-time streaming via SSE.
- **Artifact references** — Link memory entries to external blobs (model weights, datasets, logs) in S3 or elsewhere.

### Quick Example

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="review-agent")

# Agent discovers a risk and records it
mem.write(
    "checkout-service",
    "risk-race-condition",
    "Race condition in order processing under concurrent load",
    confidence=0.85,
)

# Later, another agent reads it before working on the same code
entry = mem.read("checkout-service", "risk-race-condition")
# → "Race condition in order processing under concurrent load"
# → confidence: 0.85
```

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Copy-on-Write versioning** | Every write creates a new version. Full history is preserved. |
| **Confidence scores** | Each entry carries a confidence score that evolves based on outcomes. |
| **Outcome back-propagation** | Link deploys and incidents to memory entries. Confidence adjusts automatically. |
| **Memory types** | Classify entries as `fact`, `belief`, or `experience` with type-specific decay rates. |
| **Provenance tiers** | Automatically tier entries by quality: production-validated, observed, dev, or manual. |
| **Temporal queries** | Retrieve the full version history of any entry, filtered by time range. |
| **Causal explainability** | Inspect which entries were read and how they connect to outcomes. |
| **Enriched decision traces** | Traces capture full entry snapshots, query/error events, session timing, and state diffs. |
| **Per-agent memory graph** | View any agent's complete knowledge footprint across entities. |
| **Provenance tracking** | Every entry records which agent wrote it, when, and from which session. |
| **Artifact references** | Link entries to external blobs in S3, local files, or URLs. |
| **HTTP/REST API** | FastAPI server with 12 endpoints, SSE streaming, and API key auth. |
| **Multiple adapters** | Filesystem (default), Postgres (with full-text + vector search), S3-compatible, or custom. |
| **MCP integration** | First-class MCP server for Cursor, Claude Code, and any MCP-compatible client. |
| **Framework integrations** | Works with CrewAI, LangGraph, LangChain, and AutoGen. |
| **CLI tools** | Inspect, diff, snapshot, and restore memory from the command line. |
| **Docker & Kubernetes** | One-command deployment with Docker or Helm chart. |
| **Python & TypeScript** | SDKs for both languages with the same conceptual API. |

---

## Quick Start with Docker

The fastest way to get AMFS running — no Python install required:

```bash
docker run -p 8080:8080 -v amfs-data:/data ghcr.io/raia-live/amfs
```

Or with Postgres for full-text + vector search:

```bash
docker compose up
```

Then interact via HTTP:

```bash
# Write
curl -X POST http://localhost:8080/api/v1/entries \
  -H "Content-Type: application/json" \
  -d '{"entity_path": "checkout-service", "key": "retry-pattern", "value": {"max_retries": 3}}'

# Read
curl http://localhost:8080/api/v1/entries/checkout-service/retry-pattern
```

[Docker & Kubernetes guide](/amfs/guides/docker/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[HTTP API reference](/amfs/guides/http-server/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

---

## Packages

| Package | Language | Install |
|:--------|:---------|:--------|
| `amfs` | Python | `pip install amfs` |
| `amfs-adapter-postgres` | Python | `pip install amfs-adapter-postgres` |
| `amfs-adapter-s3` | Python | `pip install amfs-adapter-s3` |
| `amfs-http-server` | Python | `pip install amfs-http-server` |
| `amfs-cli` | Python | `pip install amfs-cli` |
| `amfs-mcp-server` | Python | `pip install amfs-mcp-server` |
| `@amfs/sdk` | TypeScript | `npm install @amfs/sdk` |

---

## OSS vs Pro

AMFS is available in two editions. The open-source layer provides the full memory primitive — versioning, confidence, outcomes, enriched decision traces, per-agent memory graphs, adapters, and MCP. The Pro layer adds a multi-tenant SaaS foundation (RBAC, scoped API keys, Postgres RLS), immutable decision trace store with cryptographic integrity and LLM call tracking, OpenTelemetry export, auto entity extraction, cross-system ingestion (PagerDuty, Slack, GitHub webhooks), automated pattern detection and alerting, LLM-powered intelligence, and an interactive dashboard with causal graph visualization.

[Compare editions](/amfs/editions/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[AMFS vs Vector DBs](/amfs/vs-vector-databases/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[AMFS vs Competitors](/amfs/vs-competitors/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

---

## How Agents Use AMFS

```
1. Get a briefing           →  amfs_briefing(entity_path="checkout-service")
2. Search for specifics     →  amfs_search("checkout-service")
3. Read relevant entries    →  amfs_read("checkout-service", "retry-pattern")
4. Do the work              →  (agent performs its task)
5. Write findings           →  amfs_write("checkout-service", "new-pattern", ...)
6. Record outcomes          →  amfs_commit_outcome("DEP-287", "success")
```

Knowledge compounds over time. The Memory Cortex compiles raw entries into ranked digests, so the next agent — on any machine — starts with a pre-compiled briefing of what matters, instead of searching from scratch.

---

## License

AMFS is distributed under the [Apache License 2.0](https://github.com/raia-live/amfs/blob/main/LICENSE).
