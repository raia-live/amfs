---
title: HTTP API Server
layout: default
parent: Guides
nav_order: 5
description: "Run the AMFS HTTP/REST API server to access agent memory from any language over HTTP."
---

# HTTP API Server
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The AMFS HTTP server exposes the full `AgentMemory` API over REST, making it accessible from any language, service, or frontend. It includes:

- Full REST endpoints covering all AMFS operations
- Server-Sent Events (SSE) for real-time streaming
- API key authentication
- CORS support for browser-based dashboards
- OpenAPI docs at `/docs`

---

## Installation

```bash
pip install amfs-http-server
```

Or run with Docker:

```bash
docker run -p 8080:8080 ghcr.io/raia-live/amfs
```

---

## Quick Start

```bash
# Start with filesystem storage
amfs-http --port 8080

# Start with Postgres
AMFS_POSTGRES_DSN=postgresql://user:pass@localhost:5432/amfs amfs-http --port 8080

# Start with API key auth
AMFS_API_KEYS=key1,key2 amfs-http --port 8080
```

The server is available at `http://localhost:8080` with interactive API docs at `http://localhost:8080/docs`.

---

## Endpoints

### Entries

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/entries/{entity_path}/{key}` | Read the current version of an entry |
| `POST` | `/api/v1/entries` | Write a new entry (CoW) |
| `GET` | `/api/v1/entries` | List entries, optionally filtered by entity |
| `GET` | `/api/v1/entries/{entity_path}/{key}/history` | Get version history |
| `GET` | `/api/v1/search` | Search entries with filters |

### Outcomes

| Method | Path | Description |
|:-------|:-----|:------------|
| `POST` | `/api/v1/outcomes` | Commit an outcome and back-propagate confidence |
| `GET` | `/api/v1/outcomes` | List all outcomes |

### Patterns

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/patterns` | List unique pattern_refs with usage counts |

### Observability

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/stats` | Memory statistics |
| `POST` | `/api/v1/context` | Record external context in the causal chain |
| `GET` | `/api/v1/explain` | Get the causal trace for the current session |
| `GET` | `/api/v1/stream` | SSE stream of real-time memory events |
| `GET` | `/health` | Health check |

### Admin — Teams (Pro)

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/admin/teams` | List all teams |
| `POST` | `/api/v1/admin/teams` | Create a team |
| `PATCH` | `/api/v1/admin/teams/{id}` | Update a team |
| `DELETE` | `/api/v1/admin/teams/{id}` | Delete a team |
| `GET` | `/api/v1/admin/teams/{id}/members` | List team members |
| `POST` | `/api/v1/admin/teams/{id}/members` | Add a team member |
| `PATCH` | `/api/v1/admin/teams/{id}/members/{mid}` | Update member role |
| `DELETE` | `/api/v1/admin/teams/{id}/members/{mid}` | Remove a team member |

### Admin — Pattern Detection (Pro)

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/api/v1/admin/patterns` | List detected patterns |
| `POST` | `/api/v1/admin/patterns/scan` | Run pattern detection scan |
| `PATCH` | `/api/v1/admin/patterns/{id}/resolve` | Mark a pattern as resolved |

---

## Authentication

Set the `AMFS_API_KEYS` environment variable with a comma-separated list of valid API keys:

```bash
export AMFS_API_KEYS=amfs_prod_abc123,amfs_dev_xyz456
```

Clients must include the key in the `X-AMFS-API-Key` header:

```bash
curl -H "X-AMFS-API-Key: amfs_prod_abc123" http://localhost:8080/api/v1/stats
```

If `AMFS_API_KEYS` is not set, authentication is disabled (suitable for local development).

---

## Examples

### Write an entry

```bash
curl -X POST http://localhost:8080/api/v1/entries \
  -H "Content-Type: application/json" \
  -d '{
    "entity_path": "checkout-service",
    "key": "retry-pattern",
    "value": {"max_retries": 3, "backoff": "exponential"},
    "confidence": 0.85,
    "memory_type": "fact"
  }'
```

### Read an entry

```bash
curl http://localhost:8080/api/v1/entries/checkout-service/retry-pattern
```

### List entries

```bash
curl http://localhost:8080/api/v1/entries
```

Response:

```json
{
  "entries": [
    {"entity_path": "checkout-service", "key": "retry-pattern", "version": 1, ...},
    ...
  ]
}
```

### List outcomes

```bash
curl http://localhost:8080/api/v1/outcomes
```

Response:

```json
{
  "outcomes": [
    {"outcome_ref": "DEP-287", "outcome_type": "clean_deploy", ...},
    ...
  ]
}
```

### Search

```bash
curl "http://localhost:8080/api/v1/search?entity_path=checkout-service&min_confidence=0.5&limit=10"
```

### Commit an outcome

```bash
curl -X POST http://localhost:8080/api/v1/outcomes \
  -H "Content-Type: application/json" \
  -d '{
    "outcome_ref": "DEP-287",
    "outcome_type": "clean_deploy"
  }'
```

### Stream real-time events

```bash
curl -N http://localhost:8080/api/v1/stream
```

Events arrive as SSE:

```
event: memory_write
data: {"entity_path": "checkout-service", "key": "retry-pattern", "version": 2, ...}

event: outcome
data: {"outcome_ref": "DEP-287", "outcome_type": "clean_deploy", ...}
```

### Python client

```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8080",
    headers={"X-AMFS-API-Key": "your-key"},
)

# Write
client.post("/api/v1/entries", json={
    "entity_path": "myapp/auth",
    "key": "session-timeout",
    "value": "30m",
    "confidence": 0.9,
})

# Read
entry = client.get("/api/v1/entries/myapp/auth/session-timeout").json()
```

---

## Environment Variables

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_HTTP_HOST` | Bind host | `0.0.0.0` |
| `AMFS_HTTP_PORT` | Bind port | `8080` |
| `AMFS_API_KEYS` | Comma-separated API keys (empty = no auth) | — |
| `AMFS_CORS_ORIGINS` | Comma-separated allowed CORS origins | `*` |
| `AMFS_POSTGRES_DSN` | Postgres connection string (switches backend to Postgres) | — |
| `AMFS_S3_BUCKET` | S3 bucket (switches backend to S3) | — |
| `AMFS_DATA_DIR` | Filesystem data directory | `.amfs` |
| `AMFS_AGENT_ID` | Server agent identity | `amfs-http-server` |

---

## Deploying

### Docker

```bash
docker run -p 8080:8080 \
  -e AMFS_POSTGRES_DSN=postgresql://user:pass@host:5432/amfs \
  -e AMFS_API_KEYS=your_key_here \
  ghcr.io/raia-live/amfs
```

### Docker Compose

See [Docker & Kubernetes guide](/amfs/guides/docker/) for the full `docker-compose.yml`.

### Kubernetes

```bash
helm install amfs ./helm/amfs \
  --set storage.backend=postgres \
  --set amfs.apiKeys=your_key_here
```

---

## Next Steps

- [Docker & Kubernetes](/amfs/guides/docker/) — deploy AMFS in containers
- [S3 Adapter](/amfs/adapters/s3/) — use S3-compatible storage as the backend
- [MCP Setup](/amfs/guides/mcp/) — give coding agents direct memory access
