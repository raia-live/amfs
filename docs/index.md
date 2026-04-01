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
            ┌──────▼──────┐
            │ AgentMemory │  ← Python / TypeScript SDK
            │   (CoW)     │
            └──────┬──────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    Filesystem  Postgres    Redis
     Adapter    Adapter    Adapter
```

### Why AMFS?

AI agents today are stateless. Each session starts from scratch — no memory of past decisions, patterns, or mistakes. When multiple agents work on the same codebase, they duplicate work and repeat errors.

AMFS solves this by giving agents a **shared memory layer** with:

- **Versioned knowledge** — Every write creates a new version. Nothing is lost.
- **Provenance tracking** — Know which agent wrote what, and when.
- **Confidence scoring** — Entries carry a confidence score that evolves over time based on real-world outcomes.
- **Outcome back-propagation** — Incidents increase confidence on risky patterns. Clean deploys decay it.
- **Pluggable storage** — Start with the local filesystem, scale to Postgres for team sharing.

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
| **Provenance tracking** | Every entry records which agent wrote it, when, and from which session. |
| **Multiple adapters** | Filesystem (default), Postgres, or build your own. |
| **MCP integration** | First-class MCP server for Cursor, Claude Code, and any MCP-compatible client. |
| **Framework integrations** | Works with CrewAI, LangGraph, LangChain, and AutoGen. |
| **CLI tools** | Inspect, diff, snapshot, and restore memory from the command line. |
| **Python & TypeScript** | SDKs for both languages with the same conceptual API. |

---

## Packages

| Package | Language | Install |
|:--------|:---------|:--------|
| `amfs` | Python | `pip install amfs` |
| `amfs-adapter-postgres` | Python | `pip install amfs-adapter-postgres` |
| `amfs-cli` | Python | `pip install amfs-cli` |
| `amfs-mcp-server` | Python | `pip install amfs-mcp-server` |
| `@amfs/sdk` | TypeScript | `npm install @amfs/sdk` |

---

## OSS vs Pro

AMFS is available in two editions. The open-source layer provides the full memory primitive — versioning, confidence, outcomes, adapters, and MCP. The Pro layer adds an intelligence layer with LLM-driven extraction, automated quality auditing, memory distillation, safety validation, and multi-strategy retrieval.

[Compare editions](/amfs/editions/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[AMFS vs Vector DBs](/amfs/vs-vector-databases/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[AMFS vs Competitors](/amfs/vs-competitors/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

---

## How Agents Use AMFS

```
1. Search before working    →  amfs_search("checkout-service")
2. Read relevant entries    →  amfs_read("checkout-service", "retry-pattern")
3. Do the work              →  (agent performs its task)
4. Write findings           →  amfs_write("checkout-service", "new-pattern", ...)
5. Record outcomes          →  amfs_commit_outcome("DEP-287", "clean_deploy")
```

Knowledge compounds over time. The next agent — on any machine — starts with the context that previous agents built, instead of starting from scratch.

---

## License

AMFS is distributed under the [Apache License 2.0](https://github.com/raia-live/amfs/blob/main/LICENSE).
