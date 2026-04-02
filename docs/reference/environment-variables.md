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
| `AMFS_NAMESPACE` | Memory namespace for isolation | `default` |
| `AMFS_DATA_DIR` | Custom filesystem data directory path | `.amfs` |
| `AMFS_POSTGRES_DSN` | Postgres connection string; switches adapter to Postgres | — |

---

## S3 Adapter

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_S3_BUCKET` | S3 bucket name; switches adapter to S3 | — |
| `AMFS_S3_PREFIX` | Object key prefix in the bucket | `amfs/` |
| `AMFS_S3_ENDPOINT` | Custom S3 endpoint URL (for ACS, MinIO, R2) | — |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | AWS access key | — |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — |

---

## HTTP API Server

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_HTTP_HOST` | HTTP server bind host | `0.0.0.0` |
| `AMFS_HTTP_PORT` | HTTP server bind port | `8080` |
| `AMFS_API_KEYS` | Comma-separated API keys for authentication (empty = no auth) | — |
| `AMFS_CORS_ORIGINS` | Comma-separated CORS allowed origins | `*` |

---

## MCP Server

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_TRANSPORT` | Transport protocol: `stdio` or `http` | `stdio` |
| `AMFS_HOST` | MCP HTTP server bind host | `0.0.0.0` |
| `AMFS_PORT` | MCP HTTP server bind port | `8000` |
| `AMFS_PATH` | MCP HTTP server URL path | `/mcp` |
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
