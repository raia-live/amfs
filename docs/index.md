---
title: Home
layout: home
nav_order: 1
description: "AMFS — Agent Memory File System. GitHub for agent memory."
permalink: /
---

# Agent Memory File System
{: .fs-9 }

GitHub for agent memory. Every agent gets a brain — its own repo of what it knows. Branch it, diff it, review changes, merge them. Roll back to any point. Collaborate the way developers already do.
{: .fs-6 .fw-300 }

[Get Started](/amfs/getting-started/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/raia-live/amfs){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## The Problem

When agents share memory today, it's chaos — last write wins, no branching, no review, no rollback. It's like coding without Git. Every team that tried that eventually hit a wall. Agent teams are hitting that wall right now.

The problem isn't "agents need permissions." The problem is agents have no way to collaborate on knowledge the way developers collaborate on code.

## What is AMFS?

AMFS applies the Git model to agent memory. The mental model is already in every developer's head — you're not learning something new, you're getting the tool you've been waiting for.

```
Every developer knows this:          AMFS does the same for agent memory:

  repo                                 agent brain
  ├── main branch                      ├── main (what the agent knows)
  ├── feature branch                   ├── experiment branch (isolated changes)
  ├── pull request                     ├── pull request (review before merge)
  ├── code review                      ├── diff (what changed in the branch)
  ├── merge                            ├── merge (accept changes into main)
  ├── git log                          ├── timeline (every operation logged)
  └── git revert                       └── rollback (restore to any point)
```

Other agents can be granted read or write access to branches of that brain. Changes stay isolated until the owner reviews the diff and merges. Every change is logged. You can roll back to any point. You can back up and restore the entire brain.

### Quick Example

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="review-agent")

# Agent discovers a risk and commits it to memory
mem.write(
    "checkout-service",
    "risk-race-condition",
    "Race condition in order processing under concurrent load",
    confidence=0.85,
)

# Another agent reads it — the read is tracked automatically
entry = mem.read("checkout-service", "risk-race-condition")
# → confidence: 0.85, written_by: review-agent
```

---

## Key Features

### Git-like Collaboration

| Feature | Description |
|:--------|:------------|
| **Git-like timeline** | Every write, outcome, and cross-agent read is logged as an event. Full history, always. |
| **Branching & PRs** | Create branches, diff changes, open pull requests, review, merge or discard. (Pro) |
| **Tags & rollback** | Create named snapshots. Roll back to any tag or event. (Pro) |
| **Access control** | Grant read or read/write access per branch, per user, team, or API key. (Pro) |
| **Fork** | Clone an agent's entire brain to a new agent. (Pro) |

### Memory Intelligence

| Feature | Description |
|:--------|:------------|
| **Versioned knowledge** | Copy-on-Write — every write creates a new version. Nothing is ever lost. |
| **Confidence & outcomes** | Trust scores evolve when deploys succeed or incidents happen. |
| **Causal explainability** | `explain()` shows exactly which memories and external contexts drove a decision. |
| **Decision traces** | Full trace capture — reads, writes, query events, errors, timing, and state diffs. |
| **Knowledge graph** | Relationships between entities, agents, and outcomes auto-materialize from normal operations. |
| **Hybrid search** | Full-text + semantic + recency + confidence in a single ranked result set. |
| **Memory types** | Classify as `fact`, `belief`, or `experience` — each with its own decay rate. |
| **Tiered memory** | Hot / Warm / Archive with progressive retrieval and frequency-modulated decay. |

### Platform

| Feature | Description |
|:--------|:------------|
| **MCP server** | First-class support for Cursor, Claude Code, and any MCP client. [Setup →](/amfs/guides/mcp/) |
| **HTTP/REST API** | FastAPI server with SSE streaming and API key auth. |
| **Multiple adapters** | Filesystem (dev), Postgres (production), S3 (cloud). Swap without code changes. |
| **Python & TypeScript** | Same API in both languages. |
| **Framework integrations** | CrewAI, LangGraph, LangChain, AutoGen. |
| **Connectors** | Ingest events from PagerDuty, GitHub, Slack, Jira — or build your own. |
| **CLI tools** | Inspect, diff, snapshot, and restore memory from the command line. |
| **Docker & Kubernetes** | One-command deployment with Docker or Helm chart. |

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

AMFS is open source under [Apache 2.0](https://github.com/raia-live/amfs/blob/main/LICENSE). The OSS edition gives you the full memory engine with a single-branch repo model — versioned writes, confidence scoring, outcome feedback, causal traces, knowledge graph, hybrid search, tiered memory, git-like timeline on `main`, SDKs, adapters, HTTP API, MCP server, and CLI.

**[AMFS Pro](https://raia-live.github.io/amfs/editions/)** unlocks the full Git model: branching, merge, pull requests, access control, tags, rollback, cherry-pick, and fork. Plus multi-tenant SaaS isolation, immutable decision traces, automated pattern detection, an intelligence layer, and a web dashboard.

OSS gives you a repo with full history. Pro gives you GitHub.

[Compare editions](/amfs/editions/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[AMFS vs Vector DBs](/amfs/vs-vector-databases/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
[AMFS vs Competitors](/amfs/vs-competitors/){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

---

## How Agents Use AMFS

```
1. Identify yourself         →  amfs_set_identity("checkout-agent", "Fixing retry logic")
2. Get a briefing            →  amfs_briefing(entity_path="checkout-service")
3. Search for specifics      →  amfs_search("checkout-service")
4. Read relevant entries     →  amfs_read("checkout-service", "retry-pattern")
5. Do the work               →  (agent performs its task)
6. Write findings            →  amfs_write("checkout-service", "new-pattern", ...)
7. Record outcomes           →  amfs_commit_outcome("DEP-287", "success")
8. Next agent starts at #2   →  Knowledge compounds across agents and sessions
```

The Memory Cortex compiles raw entries into ranked digests, so the next agent — on any machine — starts with a pre-compiled briefing of what matters, instead of searching from scratch.

---

## License

AMFS is distributed under the [Apache License 2.0](https://github.com/raia-live/amfs/blob/main/LICENSE).
