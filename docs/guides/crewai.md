---
title: CrewAI Integration
layout: default
parent: Guides
nav_order: 8
description: "Use AMFS as the storage backend for CrewAI memory — get versioning, provenance, and outcome tracking on top of CrewAI's agent orchestration."
---

# CrewAI Integration
{: .no_toc }

AMFS plugs into CrewAI as a storage backend, giving your crews versioned, outcome-aware memory that persists across runs and survives restarts.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

[CrewAI](https://crewai.com) provides a powerful framework for orchestrating multi-agent workflows with built-in memory support. By default, CrewAI stores memories in local files or vector databases. AMFS replaces that storage layer while preserving CrewAI's full API — you get CrewAI's agent orchestration **plus** AMFS's versioning, confidence scoring, and outcome back-propagation.

```
┌────────────────────────────────────────┐
│              CrewAI Crew               │
│                                        │
│  ┌──────────┐  ┌──────────┐            │
│  │ Agent A  │  │ Agent B  │  ...       │
│  └────┬─────┘  └────┬─────┘            │
│       │              │                 │
│       └──────┬───────┘                 │
│              ▼                         │
│     ┌──────────────┐                   │
│     │ CrewAI Memory│                   │
│     │   (API)      │                   │
│     └──────┬───────┘                   │
│            │                           │
│            ▼                           │
│   ┌────────────────┐                   │
│   │ AMFSStorage    │  ← storage=...    │
│   │ Backend        │                   │
│   └────────┬───────┘                   │
└────────────┼───────────────────────────┘
             │
             ▼
    ┌────────────────┐
    │  AgentMemory   │  CoW · Provenance · Outcomes
    │  (AMFS)        │
    └────────┬───────┘
             │
       Filesystem / Postgres / S3
```

---

## Installation

```bash
pip install amfs crewai
```

---

## Basic Setup

Use `AMFSStorageBackend` as the storage layer for CrewAI's `Memory`:

```python
from crewai import Agent, Crew, Task, Process
from crewai.memory import Memory
from amfs.integrations.crewai import AMFSStorageBackend

backend = AMFSStorageBackend(
    agent_id="my-crew",
    entity_path="my-project",
)

crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential,
    memory=Memory(storage=backend),
)

result = crew.kickoff()
```

Every memory operation CrewAI performs — short-term, long-term, and entity memory — flows through AMFS and benefits from CoW versioning, provenance tracking, and confidence scoring.

---

## What You Get

### Versioning

CrewAI overwrites memories in place. With AMFS as the backend, every update creates a new CoW version. You can inspect how agent knowledge evolved across crew runs:

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="my-crew")
versions = mem.history("my-project", "agent-a-learnings")
for v in versions:
    print(f"v{v.version}  confidence={v.confidence}  at={v.updated_at}")
```

### Provenance

AMFS tracks which agent wrote each memory entry, in which session, and from which crew run. This is critical when multiple crews or agents share the same memory store:

```python
entry = mem.read("my-project", "research-findings")
print(entry.agent_id)       # "researcher-agent"
print(entry.session_id)     # "crew-run-2024-03-15-001"
print(entry.provenance_tier) # ProvenanceTier.OBSERVED
```

### Outcome Back-Propagation

After a crew run, record whether the result was successful. AMFS adjusts confidence on all entries that were read during the decision process:

```python
from amfs import OutcomeType

mem.commit_outcome(
    outcome_ref="crew-run-042",
    outcome_type=OutcomeType.SUCCESS,
)
```

Entries that contributed to successful outcomes gain confidence. Entries involved in failures lose confidence. Over time, your crew's memory becomes self-curating.

### Decision Traces

Use `explain()` to see the full causal chain — which memories were read, which external contexts were gathered, and what outcome occurred:

```python
chain = mem.explain()
print(chain.reads)     # entries read during this session
print(chain.contexts)  # external context recorded
```

---

## Multi-Crew Setup

When running multiple crews that share knowledge, use separate `agent_id` values but the same storage backend configuration:

```python
research_backend = AMFSStorageBackend(
    agent_id="research-crew",
    entity_path="shared-project",
)

execution_backend = AMFSStorageBackend(
    agent_id="execution-crew",
    entity_path="shared-project",
)
```

Both crews read from and write to the same entity path. AMFS handles provenance automatically — you always know which crew produced which insight.

---

## Using with Postgres

For production deployments, point AMFS at Postgres for durable storage with full-text and vector search:

```python
from amfs import AgentMemory
from amfs_adapter_postgres import PostgresAdapter

adapter = PostgresAdapter(dsn="postgresql://user:pass@host:5432/amfs")

backend = AMFSStorageBackend(
    agent_id="my-crew",
    entity_path="my-project",
    adapter=adapter,
)
```

---

## Connecting to AMFS SaaS

When using AMFS as a hosted service (SaaS), connect through the HTTP API with your API key instead of a direct database connection.

### Environment Variables

Set these before running your crew:

```bash
export AMFS_HTTP_URL="https://amfs-login.sense-lab.ai"
export AMFS_API_KEY="amfs_sk_your_key_here"
```

The `AMFSStorageBackend` will auto-detect the HTTP adapter when `AMFS_HTTP_URL` is set.

### Explicit HttpAdapter

You can also pass the adapter directly:

```python
from amfs import AgentMemory
from amfs_adapter_http import HttpAdapter
from amfs.integrations.crewai import AMFSStorageBackend

adapter = HttpAdapter(
    base_url="https://amfs-login.sense-lab.ai",
    api_key="amfs_sk_your_key_here",
)

backend = AMFSStorageBackend(
    agent_id="my-crew",
    entity_path="my-project",
    adapter=adapter,
)
```

{: .warning }
Never use `AMFS_POSTGRES_DSN` for external agents in multi-tenant mode. Always use `AMFS_HTTP_URL` + `AMFS_API_KEY`.

See the [SaaS Connection Guide](/amfs/guides/saas/) and [Environment Variables](/amfs/reference/environment-variables/) for details.

---

## Example: Research Crew with Persistent Memory

```python
from crewai import Agent, Crew, Task, Process
from crewai.memory import Memory
from amfs.integrations.crewai import AMFSStorageBackend
from amfs import OutcomeType

backend = AMFSStorageBackend(
    agent_id="research-crew",
    entity_path="market-analysis",
)

researcher = Agent(
    role="Market Researcher",
    goal="Find emerging trends in AI infrastructure",
    backstory="Senior analyst with deep knowledge of the AI market.",
)

writer = Agent(
    role="Report Writer",
    goal="Synthesize research into actionable reports",
    backstory="Technical writer specializing in market analysis.",
)

research_task = Task(
    description="Research the latest trends in AI agent memory systems",
    agent=researcher,
)

write_task = Task(
    description="Write a summary report based on the research findings",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=Memory(storage=backend),
)

result = crew.kickoff()

# Record success so future runs benefit from higher-confidence memories
backend.memory.commit_outcome("research-run-001", OutcomeType.SUCCESS)
```

On subsequent runs, the crew starts with all prior research findings — versioned, confidence-scored, and ranked by outcome history.
