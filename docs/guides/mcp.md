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

After setup, your AI agents have tools across five categories:

**Identity & Context**

| Tool | Description |
|:-----|:------------|
| `amfs_set_identity` | Set agent identity for this conversation (e.g. `"dashboard-fixer"`) |
| `amfs_briefing` | Get compiled knowledge digests from the Memory Cortex |

**Read & Write**

| Tool | Description |
|:-----|:------------|
| `amfs_read` | Read a memory entry by entity path and key |
| `amfs_write` | Write knowledge with automatic provenance tracking |
| `amfs_search` | Search entries with filters and progressive retrieval (`depth`: 1=Hot, 2=+Warm, 3=all) |
| `amfs_retrieve` | Natural language retrieval with semantic + recency + confidence scoring |
| `amfs_list` | List entries for an entity |
| `amfs_stats` | Get a memory overview (entry counts, outcome counts) |
| `amfs_history` | Retrieve version history of an entry with optional time range |
| `amfs_graph_neighbors` | Explore the knowledge graph around an entity |

**Agent Brain (scoped to you)**

| Tool | Description |
|:-----|:------------|
| `amfs_recall` | Recall YOUR OWN memory for a specific key |
| `amfs_my_entries` | List everything YOU have written |
| `amfs_read_from` | Read from ANOTHER agent's memory (tracked knowledge transfer) |
| `amfs_cross_agent_reads` | See which other agents' memory you've read |

**Decision Traces**

| Tool | Description |
|:-----|:------------|
| `amfs_record_context` | Capture decisions, external tool results, or user choices in the causal chain |
| `amfs_commit_outcome` | Record outcomes — snapshots the full decision trace (reads, writes, contexts) |
| `amfs_explain` | Inspect the current session's decision trace |
| `amfs_list_traces` | Browse persisted decision traces from past sessions |
| `amfs_get_trace` | Retrieve a full decision trace by ID |

**Timeline**

| Tool | Description |
|:-----|:------------|
| `amfs_timeline` | Browse the git-style event timeline |

---

## AMFS Pro (SaaS) and Cursor

On **Sense Lab**, the AMFS dashboard (**Agents** page, MCP Connection card) is the source of truth. Cursor talks to hosted AMFS by running the **`amfs-mcp-server`** process locally with **stdio** (e.g. `uvx`), while the server uses **`AMFS_HTTP_URL`** (dashboard **Server URL**, e.g. `https://amfs-login.sense-lab.ai`) and **`AMFS_API_KEY`** to call the HTTP API. That URL is the **API base**, not an MCP Streamable HTTP path—there is **no `/mcp`** on it for this setup. The `/mcp` path applies only when you run the MCP server in **HTTP transport mode** as its own listening service (see [Streamable HTTP](#streamable-http-team--remote) below).

Use the official **[Cursor plugin](https://github.com/raia-live/cursor-plugin)** (same shape as the dashboard JSON) or copy the snippet from the dashboard. Set **`AMFS_API_KEY`** in your environment and reference it with [Cursor interpolation](https://cursor.com/docs/mcp.md#config-interpolation). See also [SaaS / hosted AMFS](https://raia-live.github.io/amfs/guides/saas/).

Example (matches dashboard; use `${env:AMFS_API_KEY}` in the plugin so keys are not committed):

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uvx",
      "args": ["amfs-mcp-server"],
      "env": {
        "AMFS_HTTP_URL": "https://amfs-login.sense-lab.ai",
        "AMFS_API_KEY": "${env:AMFS_API_KEY}"
      }
    }
  }
}
```

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

## Connecting to AMFS SaaS

When using AMFS as a hosted service (SaaS), set `AMFS_HTTP_URL` and `AMFS_API_KEY` in the MCP server environment. This routes all memory operations through the authenticated HTTP API with full tenant isolation.

### Cursor / Claude Code (SaaS)

No local code needed — install directly from PyPI with `uvx`:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uvx",
      "args": ["amfs-mcp-server@latest"],
      "env": {
        "AMFS_HTTP_URL": "https://amfs-login.sense-lab.ai",
        "AMFS_API_KEY": "<your-api-key>"
      }
    }
  }
}
```

Get your API key from the AMFS dashboard at **Settings → API Keys**.

### Finding Your Credentials

Go to the **Agents** page in the AMFS dashboard. The **MCP Connection Card** at the top shows your API URL and token, with a ready-to-copy JSON snippet for your `.cursor/mcp.json`.

{: .warning }
Never use `AMFS_POSTGRES_DSN` for external agents in multi-tenant mode. Always use `AMFS_HTTP_URL` + `AMFS_API_KEY`.

See the [SaaS Connection Guide](/amfs/guides/saas/) for full details.

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

Every Cursor chat, Claude Code conversation, or agent session **must** call `amfs_set_identity` at the start. Without it, all work is attributed to a generic default and the agent won't appear as a distinct entry on the AMFS dashboard.

```
amfs_set_identity("dashboard-agent", "Fixing entity detail pages and auth headers")
```

This sets the agent identity for the current session — all memories, traces, and cross-agent reads are attributed to this name. The description is persisted so the dashboard can show what each agent is working on.

### Naming Conventions

Use **kebab-case role/domain names** that persist across conversations about the same topic:

| Good | Bad |
|:-----|:----|
| `dashboard-agent` | `fix-button-color` (too specific, won't be reused) |
| `stripe-agent` | `agent-1` (meaningless) |
| `api-agent` | `session-42` (ephemeral) |
| `infra-agent` | `amfs-server` (too generic) |

If continuing work a previous agent started, **use the same name** to build on their knowledge.

The agent rules (`amfs-memory.mdc` / `CLAUDE.md`) already instruct agents to do this automatically.

### Default Identity (Auto-Detection)

If `amfs_set_identity` is not called, the MCP server auto-detects from the environment. IDE platform signals take priority:

| Priority | Environment | Detected ID |
|:---------|:------------|:------------|
| 1 | Cursor (`CURSOR_SESSION_ID` or `VSCODE_PID` set) | `cursor/<username>` |
| 2 | Claude Code (`CLAUDE_CODE_SESSION` set) | `claude-code/<username>` |
| 3 | `AMFS_AGENT_ID` env var (servers, CI, scripts) | value of `AMFS_AGENT_ID` |
| 4 | Fallback | `agent/<username>` |

{: .note }
IDE signals are checked **before** `AMFS_AGENT_ID`. This prevents server-side env vars (e.g. `AMFS_AGENT_ID=amfs-server` on Cloud Run) from leaking into local MCP sessions when inherited by the IDE's shell. `AMFS_AGENT_ID` is still honoured in headless, CI, and server contexts where no IDE signals are present.

For non-IDE environments (CI bots, deploy scripts), set `AMFS_AGENT_ID`:

```json
{
  "env": {
    "AMFS_AGENT_ID": "deploy-bot"
  }
}
```

---

## How It Works

1. **Agent identifies** — `amfs_set_identity("dashboard-fixer")` gives this conversation a unique, human-readable name.
2. **Agent gets briefed** — `amfs_briefing` returns pre-compiled knowledge digests from the Memory Cortex.
3. **Agent retrieves** — `amfs_retrieve` for natural language queries, `amfs_search` for structured filters, `amfs_recall` for own memories.
4. **Agent explores** — `amfs_graph_neighbors` traverses the knowledge graph to discover related entities and agents.
5. **Decisions are captured** — `amfs_record_context` captures decisions, user choices, and external tool results as they happen.
6. **Agent writes** — `amfs_write` records knowledge with automatic provenance. `pattern_refs` create graph edges.
7. **Cortex compiles** — The Memory Cortex continuously builds up-to-date knowledge digests.
8. **Outcome committed** — `amfs_commit_outcome` snapshots the full decision trace (all reads, writes, and contexts) and back-propagates confidence.
9. **Traces persist** — `amfs_list_traces` and `amfs_get_trace` let future agents learn from past decisions.
10. **Knowledge compounds** — The next agent starts with compiled context, cross-agent reads, and full decision history.

### Example Scenario

```
Machine A (Cursor / Bruno):
  → Reviews checkout-service PR
  → amfs_write("myapp/checkout", "risk-race-condition", "...")
  → amfs_commit_outcome("PR-456", "minor_failure")

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
