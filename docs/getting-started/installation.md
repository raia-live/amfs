---
title: Installation
layout: default
parent: Getting Started
nav_order: 1
description: "Install the AMFS SDK, CLI, and adapters."
---

# Installation

AMFS is distributed as several packages so you can install only what you need.

---

## Python SDK

The core SDK includes the `AgentMemory` class and the filesystem adapter:

```bash
pip install amfs
```

{: .note }
Requires Python 3.11 or later.

### Optional: Postgres adapter

For shared memory across machines and team members:

```bash
pip install amfs-adapter-postgres
```

### Optional: CLI

Command-line tools for inspecting, diffing, and snapshotting memory:

```bash
pip install amfs-cli
```

### Optional: MCP Server

Model Context Protocol server for AI coding agents:

```bash
pip install amfs-mcp-server
```

---

## TypeScript SDK

```bash
npm install @amfs/sdk
```

---

## Framework Integrations

Install the integration package for your framework:

```bash
pip install amfs-crewai       # CrewAI
pip install amfs-langgraph    # LangGraph
pip install amfs-langchain    # LangChain
pip install amfs-autogen      # AutoGen
```

---

## Initialize a Project

After installing, initialize AMFS in your project directory:

```bash
amfs init
```

This creates:

| Path | Purpose |
|:-----|:--------|
| `amfs.yaml` | Configuration file |
| `.amfs/` | Local data directory (filesystem adapter storage) |
| `.gitignore` | Updated to exclude `.amfs/` |

{: .tip }
You can skip `amfs init` if you're using the Postgres adapter or passing configuration programmatically — the SDK works without a config file using sensible defaults.

---

## Verify Installation

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="test")
entry = mem.write("test", "hello", "world")
print(entry.value)  # "world"
print("AMFS is working!")
```

---

## Next Steps

- [Quick Start](/amfs/getting-started/quickstart/) — write, read, and search memory
- [Configuration](/amfs/getting-started/configuration/) — YAML config, adapters, and options
- [Core Concepts](/amfs/concepts/) — understand how AMFS works under the hood
