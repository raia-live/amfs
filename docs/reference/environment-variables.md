---
title: Environment Variables
layout: default
parent: Reference
nav_order: 2
description: "All environment variables supported by AMFS."
---

# Environment Variables

AMFS supports the following environment variables. They override values set in `amfs.yaml`.

---

## Core

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_AGENT_ID` | Override auto-detected agent identity | Auto-detected from environment |
| `AMFS_DATA_DIR` | Custom filesystem data directory path | `.amfs` |
| `AMFS_POSTGRES_DSN` | Postgres connection string; switches adapter to Postgres | — |

---

## MCP Server

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_TRANSPORT` | Transport protocol: `stdio` or `http` | `stdio` |
| `AMFS_HOST` | HTTP server bind host | `0.0.0.0` |
| `AMFS_PORT` | HTTP server bind port | `8000` |
| `AMFS_PATH` | HTTP server URL path | `/mcp` |
| `AMFS_TTL_SWEEP_INTERVAL` | Seconds between TTL sweep runs (set to enable automatic expiry) | — |

---

## Auto-Detection (Read-Only)

These are set by IDEs and read by AMFS for agent identity detection. You don't set these yourself.

| Variable | Set By | Used For |
|:---------|:-------|:---------|
| `CURSOR_SESSION_ID` | Cursor | Detecting Cursor environment |
| `VSCODE_PID` | VS Code / Cursor | Detecting VS Code/Cursor environment |
| `CLAUDE_CODE_SESSION` | Claude Code | Detecting Claude Code environment |

---

## Testing

| Variable | Description |
|:---------|:------------|
| `AMFS_TEST_PG_DSN` | Postgres DSN for running integration tests |

---

## Precedence

1. **Environment variables** take highest priority
2. **YAML config file** (`amfs.yaml`) is next
3. **SDK defaults** are used as fallback (filesystem adapter, `.amfs/` root, `default` namespace)
