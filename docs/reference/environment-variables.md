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
| `AMFS_ENTITY_PATH` | Home entity path for this environment. When set, the MCP server defaults `amfs_briefing` to it, advertises it in the server instructions and the `amfs_set_identity` response, and adds it to the agent's `auto_context_paths`. Lets a fresh, disposable environment (e.g. an ephemeral sandbox or CI job) auto-hydrate the right memory without the agent guessing a path. | — |
| `AMFS_NAMESPACE` | Memory namespace for isolation | `default` |
| `AMFS_DATA_DIR` | Custom filesystem data directory path | `.amfs` |
| `AMFS_POSTGRES_DSN` | Postgres connection string; switches adapter to Postgres | — |

---

## Postgres tuning

Only relevant when `AMFS_POSTGRES_DSN` is set. The defaults are fine for a single
process; these exist for deployments that run many of them.

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_POSTGRES_POOL_MAX` | Most connections one adapter may hold | `10` |
| `AMFS_POSTGRES_POOL_MIN` | Connections kept open when idle | `2` |
| `AMFS_POSTGRES_STATEMENT_TIMEOUT` | Ceiling on any one statement, as Postgres accepts it (`300s`, `5min`) | none |

{: .important }
Each process holding an adapter holds its own pool, so the connections you need
are `pool max × processes`, and Postgres has a fixed `max_connections`. If you
run the API behind an autoscaler, budget for the maximum number of instances,
not the usual one — exhausting `max_connections` refuses new connections for
everything sharing that database, including your own psql session.

Set `AMFS_POSTGRES_STATEMENT_TIMEOUT` on anything serving HTTP, at or near the
request timeout in front of it. Without one, a query whose caller has already
given up keeps running and keeps its locks, and other statements queue behind
it — a slow read becomes an outage rather than a slow read.

---

## HTTP Adapter (SaaS / Multi-Tenant)

When connecting to AMFS as a hosted service (SaaS), set these variables to route all memory operations through the authenticated HTTP API instead of a direct database connection.

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_HTTP_URL` | Base URL of the AMFS HTTP API (e.g. `https://amfs-login.sense-lab.ai`). When set, all memory operations are routed through the HTTP API. **Highest priority** — takes precedence over `AMFS_POSTGRES_DSN` and `AMFS_DATA_DIR`. | — |
| `AMFS_API_KEY` | API key for authenticating with the AMFS HTTP API. Created via the Dashboard (Agents page or Settings > API Keys). Sent as `X-AMFS-API-Key` header. | — |

{: .warning }
When `AMFS_HTTP_URL` is set, `AMFS_POSTGRES_DSN` and `AMFS_DATA_DIR` are ignored. A log warning is emitted if both `AMFS_HTTP_URL` and `AMFS_POSTGRES_DSN` are set.

{: .important }
Never share `AMFS_POSTGRES_DSN` with external agents in multi-tenant deployments. Always use `AMFS_HTTP_URL` + `AMFS_API_KEY` for external access — this ensures tenant isolation, scope enforcement, and audit logging.

### Adapter Precedence

1. **`AMFS_HTTP_URL`** — HTTP adapter via authenticated API (SaaS / multi-tenant)
2. **`AMFS_POSTGRES_DSN`** — direct Postgres adapter (self-hosted)
3. **`AMFS_DATA_DIR`** — filesystem adapter at custom path
4. **`amfs.yaml`** — configuration file discovery
5. **Default** — filesystem adapter at `.amfs/`

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

1. **`AMFS_HTTP_URL`** — routes all operations through the authenticated HTTP API (required for SaaS)
2. **Other environment variables** (`AMFS_POSTGRES_DSN`, `AMFS_DATA_DIR`, etc.)
3. **YAML config file** (`amfs.yaml`)
4. **SDK defaults** (filesystem adapter, `.amfs/` root, `default` namespace)
