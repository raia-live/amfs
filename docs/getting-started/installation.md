---
title: Installation
layout: default
parent: Getting Started
nav_order: 1
description: "Install the AMFS SDK, CLI, and adapters."
---

# Installation

AMFS is distributed as several packages so you can install only what you need. The fastest way to get started is with Docker — no Python required.

---

## Docker (Recommended)

Get a running AMFS server in one command:

```bash
docker run -p 8080:8080 -v amfs-data:/data ghcr.io/raia-live/amfs
```

Or with Postgres (full-text + vector search):

```bash
docker compose up
```

See the [Docker & Kubernetes guide](/amfs/guides/docker/) for the full setup.

---

## Python SDK

The core SDK includes the `AgentMemory` class and the filesystem adapter:

```bash
pip install amfs
```

{: .note }
Requires Python 3.11 or later.

### Optional packages

```bash
pip install amfs-adapter-postgres   # Postgres adapter (full-text + vector search)
pip install amfs-adapter-s3         # S3-compatible adapter (AWS, ACS, MinIO, R2)
pip install amfs-http-server        # HTTP/REST API server
pip install amfs-cli                # CLI tools
pip install amfs-mcp-server         # MCP server for AI coding agents
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
- [Docker & Kubernetes](/amfs/guides/docker/) — run AMFS in containers
- [HTTP API Server](/amfs/guides/http-server/) — access AMFS from any language over HTTP
- [Core Concepts](/amfs/concepts/) — understand how AMFS works under the hood
