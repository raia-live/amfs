---
title: Strands Agents Integration
layout: default
parent: Guides
nav_order: 9
description: "Use AMFS as a persistent memory plugin for Strands Agents — get versioning, provenance, causal tracing, and outcome tracking on top of Strands' agent orchestration."
---

# Strands Agents Integration
{: .no_toc }

AMFS integrates with Strands Agents as a **Plugin**, giving your agents persistent, versioned memory with automatic causal tracing — deeper than tool-only integrations.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

[Strands Agents](https://strandsagents.com) is an open-source SDK for building AI agents with any model. AMFS integrates as a Strands **Plugin** — not just tools — which means your agents get persistent memory **and** automatic tracking of every tool call in the decision trace.

```
┌────────────────────────────────────────┐
│           Strands Agent                │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │        AMFSPlugin                │  │
│  │                                  │  │
│  │  Tools:     @tool methods        │  │
│  │  amfs_read, amfs_write,          │  │
│  │  amfs_search, amfs_recall,       │  │
│  │  amfs_list, amfs_record_context  │  │
│  │                                  │  │
│  │  Hooks:     @hook methods        │  │
│  │  AfterToolCallEvent → auto-trace │  │
│  └──────────┬───────────────────────┘  │
└─────────────┼──────────────────────────┘
              │
              ▼
     ┌────────────────┐
     │  AgentMemory   │  CoW · Provenance · Outcomes
     │  (AMFS)        │
     └────────┬───────┘
              │
        Filesystem / Postgres / S3 / HTTP (SaaS)
```

---

## Installation

```bash
pip install amfs-strands strands-agents
```

---

## Plugin Setup (Recommended)

The `AMFSPlugin` is the recommended way to use AMFS with Strands. It registers memory tools **and** hooks that automatically track every tool call in the AMFS causal chain.

```python
from strands import Agent
from amfs import AgentMemory
from amfs_strands import AMFSPlugin

mem = AgentMemory(agent_id="research-agent")
agent = Agent(
    system_prompt="You are a research assistant with persistent memory.",
    plugins=[AMFSPlugin(mem)],
)

agent("Research the latest AI papers and remember key findings")
```

The plugin registers six tools for your agent:

| Tool | Description |
|------|-------------|
| `amfs_read` | Read a memory entry by entity path and key |
| `amfs_write` | Write a value with confidence and memory type |
| `amfs_search` | Search across entries with text query and filters |
| `amfs_recall` | Agent-scoped read — only this agent's own entries |
| `amfs_list` | List all entries under an entity path |
| `amfs_record_context` | Record external context in the causal chain |

---

## Tools-Only Setup

If you prefer a single tool (like mem0's `mem0_memory`), use the standalone `amfs_memory` tool:

```python
from strands import Agent
from amfs_strands import amfs_memory

agent = Agent(tools=[amfs_memory])
agent("Remember that I prefer Python over TypeScript")
```

The tool supports four actions: `store`, `retrieve`, `search`, and `list`.

Set `AMFS_AGENT_ID` to control the agent identity (defaults to `strands-agent`).

---

## What You Get

### Versioning

Every write creates a new CoW version. You can inspect how agent knowledge evolved:

```python
versions = mem.history("myapp", "retry-pattern")
for v in versions:
    print(f"v{v.version}  confidence={v.confidence}  at={v.provenance.written_at}")
```

### Provenance

AMFS tracks which agent wrote each entry, in which session, and when:

```python
entry = mem.read("myapp", "research-findings")
print(entry.provenance.agent_id)   # "research-agent"
print(entry.provenance.session_id) # auto-generated
```

### Automatic Causal Tracing

The plugin's `AfterToolCallEvent` hook records every non-AMFS tool call into the causal chain. When you call `explain()`, you see the complete picture — not just which memories were read, but which tools the agent used.

### Outcome Back-Propagation

Record whether outcomes were successful. AMFS adjusts confidence on all entries that were read during the decision process:

```python
from amfs import OutcomeType

mem.commit_outcome("task-001", OutcomeType.SUCCESS)
```

### Decision Traces

Use `explain()` to see the full causal chain:

```python
chain = mem.explain()
# Shows: memories read, external contexts, tool calls traced via hooks
```

---

## Connecting to AMFS SaaS

When using AMFS as a hosted service, connect through the HTTP API:

### Environment Variables

```bash
export AMFS_HTTP_URL="https://amfs-login.sense-lab.ai"
export AMFS_API_KEY="amfs_sk_your_key_here"
```

`AgentMemory` auto-detects the HTTP adapter when `AMFS_HTTP_URL` is set.

### Explicit HttpAdapter

```python
from amfs import AgentMemory
from amfs_adapter_http import HttpAdapter
from amfs_strands import AMFSPlugin

adapter = HttpAdapter(
    base_url="https://amfs-login.sense-lab.ai",
    api_key="amfs_sk_your_key_here",
)

mem = AgentMemory(agent_id="my-agent", adapter=adapter)
agent = Agent(plugins=[AMFSPlugin(mem)])
```

{: .warning }
Never use `AMFS_POSTGRES_DSN` for external agents in multi-tenant mode. Always use `AMFS_HTTP_URL` + `AMFS_API_KEY`.

---

## Example: Memory Agent (Replacing mem0)

This example mirrors the [Strands mem0 memory agent](https://strandsagents.com/docs/examples/python/memory_agent/) but uses AMFS:

```python
from strands import Agent
from amfs import AgentMemory
from amfs_strands import AMFSPlugin

mem = AgentMemory(agent_id="memory-agent")
plugin = AMFSPlugin(mem)

agent = Agent(
    system_prompt="""You are a personal assistant with persistent memory.

    Use amfs_write to store information the user shares.
    Use amfs_search to find relevant memories when answering questions.
    Use amfs_recall to check what you already know.
    Use amfs_list to show all stored memories.""",
    plugins=[plugin],
)

# Interactive loop
while True:
    user_input = input("\n> ")
    if user_input.lower() == "exit":
        break
    agent(user_input)

mem.close()
```

---

## Multi-Agent Setup

When running multiple Strands agents that share knowledge, use separate `agent_id` values:

```python
researcher_mem = AgentMemory(agent_id="researcher")
writer_mem = AgentMemory(agent_id="writer")

researcher = Agent(plugins=[AMFSPlugin(researcher_mem)])
writer = Agent(plugins=[AMFSPlugin(writer_mem)])

# Researcher writes findings
researcher("Research AI agent memory systems and save key findings")

# Writer reads researcher's entries via shared AMFS storage
writer("Read the research findings and write a summary report")
```

Both agents read from and write to the same storage. AMFS handles provenance automatically — you always know which agent produced which insight. Use `read_from()` for explicit cross-agent knowledge transfer.

---

## Disabling Auto-Tracing

If you don't want the plugin to automatically record tool calls:

```python
plugin = AMFSPlugin(mem, auto_trace=False)
```

Tools still work; only the `AfterToolCallEvent` hook is disabled.
