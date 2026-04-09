# AMFS — Agent Memory File System

[![Python Tests](https://github.com/raia-live/amfs/actions/workflows/test-python.yml/badge.svg)](https://github.com/raia-live/amfs/actions/workflows/test-python.yml)
[![TypeScript Tests](https://github.com/raia-live/amfs/actions/workflows/test-typescript.yml/badge.svg)](https://github.com/raia-live/amfs/actions/workflows/test-typescript.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A shared, causally-linked memory layer for multi-agent AI systems. Agents write versioned findings with confidence scores, read each other's knowledge, and learn from real-world outcomes — turning isolated LLM calls into compounding institutional intelligence.

**[Documentation](https://raia-live.github.io/amfs/)** · **[Roadmap](https://github.com/orgs/raia-live/projects/2)** · **[Contributing](https://raia-live.github.io/amfs/contributing/)**

## Why AMFS

Most agent memory is a vector store with a `save()` and `query()`. AMFS is different:

- **Versioned** — Every write creates an immutable CoW snapshot. See how knowledge evolved.
- **Outcome-aware** — Confidence scores adjust automatically when deploys succeed or incidents happen.
- **Causal** — Every read is tracked. `explain()` shows exactly which memories drove a decision.
- **Multi-agent** — Agents share a single memory layer. One agent's finding is another's context.
- **Pluggable** — Filesystem for dev, Postgres for production, S3 for cloud. Swap without code changes.

## Quick Start

```bash
pip install amfs
```

```python
from amfs import AgentMemory, OutcomeType

mem = AgentMemory(agent_id="review-agent")

mem.write("checkout-service", "retry-pattern",
          {"max_retries": 3, "strategy": "exponential-backoff"},
          confidence=0.85)

entry = mem.read("checkout-service", "retry-pattern")

mem.commit_outcome("INC-1042", OutcomeType.CRITICAL_FAILURE)
```

**[Full quick start guide →](https://raia-live.github.io/amfs/getting-started/quickstart/)**

## Install

```bash
pip install amfs                    # Python SDK
npm install @amfs/sdk               # TypeScript SDK
pip install amfs-http-server        # REST API server
pip install amfs-adapter-postgres   # Postgres backend
pip install amfs-adapter-s3         # S3 backend
pip install amfs-cli                # CLI tools
```

Or run with Docker:

```bash
docker run -p 8080:8080 -v amfs-data:/data ghcr.io/raia-live/amfs
```

## Architecture

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
       │ HTTP API / MCP Server │  ← REST + SSE / MCP stdio
       └───────────┬───────────┘
                   │
     ┌─────────┬───┴────┬─────────┐
     ▼         ▼        ▼         ▼
 Filesystem  Postgres   S3      Custom
  Adapter    Adapter  Adapter   Adapter
```

## Highlights

| | |
|:--|:--|
| **CoW Versioning** | Full version history for every entry. `history()` to replay changes. |
| **Confidence & Outcomes** | Scores adjust when real-world outcomes (deploys, incidents) are recorded. |
| **Causal Explainability** | `explain()` returns the full decision trace: what was read, by whom, and when. |
| **Connectors** | Ingest events from PagerDuty, GitHub, Slack, Jira — or [build your own](https://raia-live.github.io/amfs/guides/connectors/). |
| **Composite Scoring** | Rank results by weighted blend of relevance, recency, and confidence. |
| **Multi-Scope** | Query across scopes with `search(entity_paths=[...])` and visualize with `tree()`. |
| **MCP Server** | First-class support for Cursor, Claude Code, and any MCP client. [Setup →](https://raia-live.github.io/amfs/guides/mcp/) · **[Cursor plugin repo →](https://github.com/raia-live/cursor-plugin)** |
| **Integrations** | [CrewAI](https://raia-live.github.io/amfs/guides/crewai/), LangGraph, LangChain, AutoGen. |
| **Python & TypeScript** | Same API in both languages. |

## OSS vs Pro

AMFS is open source under [Apache 2.0](LICENSE). The OSS edition includes the full memory engine, SDKs, adapters, connectors, HTTP API, MCP server, and CLI.

**[AMFS Pro](https://raia-live.github.io/amfs/editions/)** adds enterprise capabilities: multi-tenant isolation (Postgres RLS + RBAC), persistent decision traces, automated pattern detection, an intelligence layer (LLM extraction, memory critic), and a web dashboard.

**[Full comparison →](https://raia-live.github.io/amfs/editions/)**

## Documentation

**[raia-live.github.io/amfs](https://raia-live.github.io/amfs/)** — Getting started, core concepts, SDK guides, adapter docs, integrations, API reference, and more.

## Roadmap

Track what's been delivered and what's coming next on the **[AMFS Roadmap board](https://github.com/orgs/raia-live/projects/2)**.

## Development

```bash
git clone https://github.com/raia-live/amfs.git && cd amfs
uv pip install -e packages/core -e packages/adapters/filesystem -e packages/sdk-python -e packages/cli -e packages/http-server
uv run pytest tests/ -v
```

**[Contributing guide →](https://raia-live.github.io/amfs/contributing/)**

## License

[Apache License 2.0](LICENSE)
