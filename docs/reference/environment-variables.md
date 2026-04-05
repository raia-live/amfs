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

## Memory Cortex

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_CORTEX_DEBOUNCE_MS` | Debounce interval for digest recompilation (milliseconds) | `3000` |
| `AMFS_CONNECTOR_{NAME}_SECRET` | HMAC secret for webhook connector signature verification (e.g. `AMFS_CONNECTOR_PAGERDUTY_SECRET`) | — |
| `AMFS_CORTEX_LLM_MODEL` | LLM model for narrative digest compilation (Pro) | `gpt-4o-mini` |
| `AMFS_PRO_URL` | Pro SaaS API URL — enables hot context, anticipation ranking, and LLM features without local Pro packages | — |
| `AMFS_PRO_API_KEY` | API key for authenticating with the Pro SaaS API | — |

---

## Dashboard (Pro)

These are used by the AMFS Pro Next.js dashboard to connect to the HTTP API server.

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_API_URL` | AMFS HTTP server URL for server-side code (API routes, server components) | `http://localhost:8080` |
| `NEXT_PUBLIC_AMFS_API_URL` | AMFS HTTP server URL for client-side code (SSE live status, Pro tool panels). The `NEXT_PUBLIC_` prefix is required by Next.js to expose the variable to the browser. | — |
| `AMFS_API_KEY` | API key for authenticating with the AMFS HTTP server | — |
| `AMFS_ROOT` | Path to the `.amfs` directory for direct filesystem reads | — |
| `AMFS_DEMO` | Force demo mode with synthetic data (`true`/`false`) | `false` |

Both `AMFS_API_URL` and `NEXT_PUBLIC_AMFS_API_URL` should be set to the same value. The former is used by Next.js server-side rendering; the latter is inlined into the browser bundle at build time.

---

## Decision Traces (Pro)

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_TRACE_SIGNING_KEY` | HMAC signing key for immutable trace integrity. When set, all immutable traces are cryptographically signed and linked via Merkle chains. | — |

---

## OpenTelemetry Export (Pro)

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_OTEL_ENDPOINT` | OpenTelemetry collector endpoint (gRPC) | `http://localhost:4317` |
| `AMFS_OTEL_SERVICE_NAME` | Service name for OTel spans | `amfs` |

---

## Auto Entity Extraction (Pro)

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_AUTO_EXTRACT` | Enable automatic entity/relationship extraction from traces (`true`/`false`) | `false` |
| `AMFS_EXTRACTION_MODEL` | LLM model to use for extraction | `gpt-4o-mini` |
| `AMFS_EXTRACTION_PROVIDER` | LLM provider for extraction (`openai`, `anthropic`) | `openai` |

---

## ML Layer (Pro)

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_ML_MODEL_DIR` | Directory for learned ranking model checkpoints | — |

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
