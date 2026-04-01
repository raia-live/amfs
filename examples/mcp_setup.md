# AMFS MCP Server — Setup Guide

This guide walks you through setting up AMFS as shared memory for AI coding agents in Cursor and Claude Code.

## What You Get

After setup, your AI coding agents will have 6 memory tools available:

| Tool | Description |
|------|-------------|
| `amfs_read` | Read a memory entry by entity path and key |
| `amfs_write` | Write knowledge with automatic provenance |
| `amfs_search` | Search across all entries with filters |
| `amfs_list` | List entries for an entity |
| `amfs_stats` | Memory overview |
| `amfs_commit_outcome` | Record outcomes, auto-links to read log |

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
  → amfs_commit_outcome("PR-456", "regression")

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
