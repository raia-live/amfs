---
title: MCP Setup
layout: default
parent: Guides
nav_order: 4
description: "Set up AMFS as shared memory for AI coding agents in Cursor and Claude Code."
---

# MCP Setup
{: .no_toc }

AMFS provides a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives AI coding agents persistent, shared memory. This guide covers setup for Cursor, Claude Code, and any MCP-compatible client.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What You Get

After setup, your AI agents have 10 memory tools:

| Tool | Description |
|:-----|:------------|
| `amfs_read` | Read a memory entry by entity path and key |
| `amfs_write` | Write knowledge with automatic provenance tracking |
| `amfs_search` | Search across all entries with filters |
| `amfs_list` | List entries for an entity |
| `amfs_stats` | Get a memory overview (entry counts, outcome counts) |
| `amfs_commit_outcome` | Record outcomes with auto-causal linking |
| `amfs_record_context` | Capture external tool/API inputs in the causal chain |
| `amfs_history` | Retrieve version history of an entry with optional time range |
| `amfs_explain` | Inspect the full decision trace for the current session |
| `amfs_briefing` | Get compiled knowledge digests from the Memory Cortex |

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- AMFS repository cloned locally

---

## Step 1: Initialize AMFS

```bash
cd /path/to/your/project
uv run amfs init
```

This creates `amfs.yaml`, `.amfs/`, and updates `.gitignore`.

---

## Step 2: Configure Your IDE

### Cursor

Add to your project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/amfs",
        "amfs-mcp-server"
      ],
      "env": {}
    }
  }
}
```

Then copy the agent rules to teach Cursor when to use memory:

```bash
cp /path/to/amfs/.cursor/rules/amfs-memory.mdc \
   /path/to/your/project/.cursor/rules/
```

### Claude Code

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/amfs",
        "amfs-mcp-server"
      ],
      "env": {}
    }
  }
}
```

Copy the agent instructions to your project:

```bash
cp /path/to/amfs/CLAUDE.md /path/to/your/project/
```

---

## Step 3: Verify

Open your IDE and ask the agent to run `amfs_stats()`. If it returns a response, AMFS is connected.

---

## Streamable HTTP (Team / Remote)

For team deployments, run AMFS as an HTTP server instead of spawning a local stdio process per IDE.

### Start the Server

```bash
# Default: 0.0.0.0:8000/mcp
uv run amfs-mcp-server --transport http

# Custom settings
uv run amfs-mcp-server --transport http \
  --host 127.0.0.1 \
  --port 9000 \
  --path /amfs
```

Or via environment variables:

```bash
export AMFS_TRANSPORT=http
export AMFS_HOST=0.0.0.0
export AMFS_PORT=8000
export AMFS_PATH=/mcp
uv run amfs-mcp-server
```

### Connect Clients

Point your IDE to the HTTP URL instead of spawning a process:

```json
{
  "mcpServers": {
    "amfs": {
      "url": "http://your-server:8000/mcp"
    }
  }
}
```

### stdio vs. HTTP

| | stdio (default) | Streamable HTTP |
|:--|:----------------|:----------------|
| **Best for** | Local dev, single machine | Teams, remote servers |
| **Setup** | IDE spawns process | Run server separately |
| **Agents** | One per IDE session | Many connect to one server |
| **Network** | Local pipes | HTTP (load balancers, firewalls) |
| **Persistence** | Per-process lifetime | Server stays up independently |

---

## Using Postgres for Shared Memory

For team sharing across machines, use Postgres:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/amfs",
        "amfs-mcp-server"
      ],
      "env": {
        "AMFS_POSTGRES_DSN": "postgresql://user:pass@shared-host:5432/amfs"
      }
    }
  }
}
```

---

## Agent Identity

The MCP server auto-detects the agent's identity from the environment:

| Environment | Detected ID |
|:------------|:------------|
| Cursor | `cursor/<username>` |
| Claude Code | `claude-code/<username>` |
| Other | `agent/<username>` |

Override with `AMFS_AGENT_ID`:

```json
{
  "env": {
    "AMFS_AGENT_ID": "deploy-bot"
  }
}
```

---

## How It Works

1. **Agent starts** — MCP server launches, creates an `AgentMemory` instance with auto-detected agent ID.
2. **Agent gets briefed** — The agent calls `amfs_briefing` to get pre-compiled knowledge digests from the Memory Cortex, providing instant context about relevant entities and past agent activity.
3. **Agent searches** — For specific queries, the agent calls `amfs_search` to find individual entries.
4. **Agent gathers context** — External tool calls are captured with `amfs_record_context` so the decision trace is complete.
5. **Agent writes** — After completing tasks, decisions and risks are recorded with `amfs_write` (optionally specifying `memory_type`: `fact`, `belief`, or `experience`).
6. **Cortex compiles** — The Memory Cortex continuously processes new writes and compiles them into up-to-date knowledge digests.
7. **Outcomes propagate** — `amfs_commit_outcome` updates confidence on all entries the agent read.
8. **Agent reviews** — `amfs_history` shows how a memory evolved over time; `amfs_explain` reveals the full decision trace including external inputs.
9. **Knowledge compounds** — The next agent starts with compiled context instead of from scratch.

### Example Scenario

```
Machine A (Cursor / Bruno):
  → Reviews checkout-service PR
  → amfs_write("myapp/checkout", "risk-race-condition", "...")
  → amfs_commit_outcome("PR-456", "regression")

Machine B (Claude Code / Alice):
  → Starts working on checkout-service
  → amfs_search(entity_path="myapp/checkout")
  → Sees Bruno's risk signal with boosted confidence
  → Avoids the same issue
```

---

## Troubleshooting

**"amfs-mcp-server not found"**
Run `uv sync` in the AMFS directory and verify the path in your MCP config.

**Agent doesn't use memory tools**
Ensure the rules file (`.cursor/rules/amfs-memory.mdc` or `CLAUDE.md`) is in your project.

**Memory not persisting**
Check that `.amfs/` exists. For Postgres, verify the DSN and network connectivity.

**Multiple agents overwriting each other**
This is expected with the default `LAST_WRITE_WINS` policy. Use `ConflictPolicy.RAISE` or Postgres for stronger consistency.
