# AMFS MCP Server — Setup Guide

This guide walks you through setting up AMFS as shared memory for AI coding agents in Cursor and Claude Code.

## What You Get

After setup, your AI coding agents will have 9 memory tools available:

| Tool | Description |
|------|-------------|
| `amfs_read` | Read a memory entry by entity path and key |
| `amfs_write` | Write knowledge with automatic provenance (supports `memory_type`: `fact`, `belief`, `experience`) |
| `amfs_search` | Search across all entries with filters |
| `amfs_list` | List entries for an entity |
| `amfs_stats` | Memory overview |
| `amfs_commit_outcome` | Record outcomes, auto-links to read log |
| `amfs_record_context` | Capture external tool/API inputs in the decision trace |
| `amfs_history` | Retrieve version history of an entry with optional time range |
| `amfs_explain` | Inspect the full decision trace (AMFS reads + external contexts) |

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- AMFS repository cloned locally

## Step 1: Initialize AMFS

```bash
cd /path/to/your/project
uv run amfs init
```

This creates:
- `amfs.yaml` — configuration file
- `.amfs/` — local data directory
- Updates `.gitignore` to exclude `.amfs/`

## Step 2: Configure Your IDE

### Cursor

Add to your project's `.cursor/mcp.json` (create the file if it doesn't exist):

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/amfs", "amfs-mcp-server"],
      "env": {}
    }
  }
}
```

Replace `/absolute/path/to/amfs` with the actual path to your AMFS clone.

Copy the agent rules file to teach Cursor when to use memory:

```bash
cp /path/to/amfs/.cursor/rules/amfs-memory.mdc /path/to/your/project/.cursor/rules/
```

### Claude Code

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/amfs", "amfs-mcp-server"],
      "env": {}
    }
  }
}
```

Copy `CLAUDE.md` to your project root:

```bash
cp /path/to/amfs/CLAUDE.md /path/to/your/project/
```

## Step 3: Team Sharing (Optional)

### Local development (default)

By default, AMFS uses the filesystem adapter. Memory is stored in `.amfs/` in your project directory. This works for single-machine use.

### Postgres (shared across machines)

For team sharing, set the `AMFS_POSTGRES_DSN` environment variable:

```bash
export AMFS_POSTGRES_DSN="postgresql://user:pass@shared-host:5432/amfs"
```

Add it to your MCP config:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/amfs", "amfs-mcp-server"],
      "env": {
        "AMFS_POSTGRES_DSN": "postgresql://user:pass@shared-host:5432/amfs"
      }
    }
  }
}
```

### Custom data directory

To store filesystem data in a specific location:

```bash
export AMFS_DATA_DIR="/shared/mount/amfs-data"
```

## Step 4: Streamable HTTP (Remote / Team Server)

For team deployments, you can run AMFS as an HTTP server that multiple agents connect to remotely — instead of each IDE spawning its own local stdio process.

### Start the HTTP server

```bash
# Default: listens on 0.0.0.0:8000/mcp
uv run amfs-mcp-server --transport http

# Custom host/port/path
uv run amfs-mcp-server --transport http --host 127.0.0.1 --port 9000 --path /amfs
```

Or use environment variables:

```bash
export AMFS_TRANSPORT=http
export AMFS_HOST=0.0.0.0
export AMFS_PORT=8000
export AMFS_PATH=/mcp
uv run amfs-mcp-server
```

### Connect from Cursor (Streamable HTTP)

```json
{
  "mcpServers": {
    "amfs": {
      "url": "http://your-server:8000/mcp"
    }
  }
}
```

### Connect from Claude Code (Streamable HTTP)

```json
{
  "mcpServers": {
    "amfs": {
      "url": "http://your-server:8000/mcp"
    }
  }
}
```

### When to use Streamable HTTP vs stdio

| | stdio (default) | Streamable HTTP |
|---|---|---|
| **Use when** | Local dev, single machine | Team sharing, remote server |
| **Setup** | MCP config spawns process | Run server separately, point clients to URL |
| **Agents** | One per IDE session | Many agents connect to one server |
| **Network** | None (local pipes) | HTTP (supports load balancers, firewalls) |
| **Persistence** | Per-process lifetime | Server stays up independently |

## How It Works

1. **Agent starts a session** — The MCP server launches, creates an `AgentMemory` instance with auto-detected `agent_id` (e.g., `cursor/bruno` or `claude-code/alice`).

2. **Agent searches before working** — Guided by the rules, the agent calls `amfs_search` to check for existing context.

3. **Agent writes after completing tasks** — Decisions, patterns, and risks are recorded with `amfs_write`.

4. **Outcomes back-propagate** — When a deploy succeeds or an incident occurs, `amfs_commit_outcome` updates confidence scores on all related entries.

5. **Knowledge compounds** — The next agent on any machine reads what previous agents learned, starting with context instead of from scratch.

## Example Scenario

```
Machine A (Cursor/Bruno):
  → Reviews checkout-service PR
  → amfs_write("myapp/checkout", "risk-race-condition", "Race condition in order processing under load")
  → amfs_commit_outcome("PR-456", "minor_failure")

Machine B (Claude Code/Alice):
  → Starts working on checkout-service
  → amfs_search(entity_path="myapp/checkout")
  → Sees Bruno's risk signal with boosted confidence
  → Avoids the same issue, writes her own findings
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AMFS_AGENT_ID` | Override auto-detected agent identity |
| `AMFS_POSTGRES_DSN` | Postgres connection string for shared memory |
| `AMFS_DATA_DIR` | Custom filesystem data directory |
| `AMFS_TRANSPORT` | Transport: `stdio` (default) or `http` |
| `AMFS_HOST` | HTTP bind host (default: `0.0.0.0`) |
| `AMFS_PORT` | HTTP bind port (default: `8000`) |
| `AMFS_PATH` | HTTP URL path (default: `/mcp`) |
| `AMFS_TTL_SWEEP_INTERVAL` | Seconds between TTL sweep runs |
| `CURSOR_SESSION_ID` | Auto-set by Cursor (used for detection) |
| `VSCODE_PID` | Auto-set by VS Code/Cursor (used for detection) |
| `CLAUDE_CODE_SESSION` | Auto-set by Claude Code (used for detection) |

## Troubleshooting

**"amfs-mcp-server not found"**
Make sure you've run `uv sync` in the AMFS directory and the path in your MCP config is correct.

**Agent doesn't use memory tools**
Ensure the agent rules file (`.cursor/rules/amfs-memory.mdc` or `CLAUDE.md`) is in your project.

**Memory not persisting between sessions**
Check that your `.amfs/` directory exists and isn't being cleared. For team sharing, verify Postgres connectivity.
