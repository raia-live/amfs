---
title: Docker & Kubernetes
layout: default
parent: Guides
nav_order: 6
description: "Run AMFS in Docker or deploy to Kubernetes with Helm — zero Python setup required."
---

# Docker & Kubernetes
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AMFS ships with a production-ready Docker image and a Helm chart for Kubernetes. You can go from zero to a running AMFS server in one command — no Python installation required.

The Docker image includes:

- The AMFS HTTP API server (FastAPI)
- All storage adapters (filesystem, Postgres, S3)
- The MCP server
- Health checks and graceful shutdown

---

## Docker

### Quick Start

```bash
# Filesystem storage (simplest — data persists in a Docker volume)
docker run -p 8080:8080 -v amfs-data:/data ghcr.io/raia-live/amfs

# Postgres backend
docker run -p 8080:8080 \
  -e AMFS_POSTGRES_DSN=postgresql://user:pass@host:5432/amfs \
  ghcr.io/raia-live/amfs

# S3 backend
docker run -p 8080:8080 \
  -e AMFS_S3_BUCKET=my-bucket \
  -e AMFS_S3_ENDPOINT=https://s3.acceleratedcloudstorage.com \
  ghcr.io/raia-live/amfs
```

### Build Locally

```bash
git clone https://github.com/raia-live/amfs.git
cd amfs
docker build -t amfs .
docker run -p 8080:8080 amfs
```

### Configuration

All [environment variables](/amfs/reference/environment-variables/) work inside the container. Common ones:

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_POSTGRES_DSN` | Use Postgres backend | — |
| `AMFS_S3_BUCKET` | Use S3 backend | — |
| `AMFS_S3_ENDPOINT` | Custom S3 endpoint (ACS, MinIO, R2) | — |
| `AMFS_API_KEYS` | Comma-separated API keys for auth | — (no auth) |
| `AMFS_DATA_DIR` | Filesystem data directory | `/data/.amfs` |
| `AMFS_AGENT_ID` | Server agent identity | `amfs-server` |

---

## Docker Compose

For local development, `docker-compose.yml` brings up AMFS + Postgres (with pgvector) in one command:

```bash
docker compose up
```

This starts:

| Service | Port | Description |
|:--------|:-----|:------------|
| `amfs` | 8080 | AMFS HTTP API server |
| `postgres` | 5432 | PostgreSQL 16 with pgvector |

The compose file lives in the repo root:

```yaml
services:
  amfs:
    build: .
    ports:
      - "8080:8080"
    environment:
      AMFS_POSTGRES_DSN: postgresql://amfs:amfs@postgres:5432/amfs
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: amfs
      POSTGRES_PASSWORD: amfs
      POSTGRES_DB: amfs
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### Verify

```bash
# Wait for startup
sleep 5

# Check health
curl http://localhost:8080/health

# Write a test entry
curl -X POST http://localhost:8080/api/v1/entries \
  -H "Content-Type: application/json" \
  -d '{"entity_path": "test", "key": "hello", "value": "world"}'

# Read it back
curl http://localhost:8080/api/v1/entries/test/hello

# List all entries
curl http://localhost:8080/api/v1/entries
# Returns: {"entries": [{...}, ...]}
```

### Connecting the Dashboard (Pro)

If you have access to the AMFS Pro dashboard, point it at the running Docker Compose stack by setting two environment variables in `dashboard/.env.local`:

```bash
# Server-side (used by Next.js API routes and server components)
AMFS_API_URL=http://localhost:8080

# Client-side (used by the browser for SSE live status and Pro tool panels)
NEXT_PUBLIC_AMFS_API_URL=http://localhost:8080
```

Both are required. `NEXT_PUBLIC_` is a Next.js convention that exposes the variable to browser-side code. After setting them, restart the dashboard dev server (`npm run dev`) for the changes to take effect.

---

## Kubernetes with Helm

### Install

```bash
# From the repo
helm install amfs ./helm/amfs

# With Postgres backend (includes a built-in Postgres StatefulSet)
helm install amfs ./helm/amfs --set storage.backend=postgres

# With external Postgres
helm install amfs ./helm/amfs \
  --set storage.backend=postgres \
  --set postgres.external=true \
  --set postgres.dsn=postgresql://user:pass@your-rds:5432/amfs

# With S3 backend
helm install amfs ./helm/amfs \
  --set storage.backend=s3 \
  --set s3.bucket=my-amfs-bucket \
  --set s3.endpoint=https://s3.acceleratedcloudstorage.com

# With API key auth
helm install amfs ./helm/amfs \
  --set amfs.apiKeys=amfs_prod_key1
```

### Helm Values

| Value | Description | Default |
|:------|:------------|:--------|
| `replicaCount` | Number of AMFS pods | `1` |
| `image.repository` | Docker image | `ghcr.io/raia-live/amfs` |
| `image.tag` | Image tag | `latest` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `8080` |
| `storage.backend` | Storage backend: `filesystem`, `postgres`, `s3` | `postgres` |
| `postgres.external` | Use an external Postgres instance | `false` |
| `postgres.dsn` | External Postgres DSN | — |
| `postgres.storage` | PVC size for built-in Postgres | `10Gi` |
| `s3.bucket` | S3 bucket name | — |
| `s3.endpoint` | Custom S3 endpoint | — |
| `amfs.apiKeys` | API keys for authentication | — |
| `amfs.namespace` | AMFS namespace | `default` |
| `ingress.enabled` | Enable Kubernetes Ingress | `false` |
| `autoscaling.enabled` | Enable HPA | `false` |
| `autoscaling.maxReplicas` | Maximum pod replicas | `5` |

### Ingress

To expose AMFS externally:

```yaml
# values.yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: amfs.yourcompany.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: amfs-tls
      hosts:
        - amfs.yourcompany.com
```

### Autoscaling

The Helm chart includes an optional HPA:

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

{: .warning }
When scaling to multiple replicas with the filesystem backend, each replica has its own isolated storage. Use Postgres or S3 for shared state across replicas.

---

## Connecting MCP Through Docker

You can point your MCP config at the Dockerized HTTP server instead of running a local process:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/amfs", "amfs-mcp-server"],
      "env": {
        "AMFS_TRANSPORT": "http",
        "AMFS_HOST": "localhost",
        "AMFS_PORT": "8080"
      }
    }
  }
}
```

Or configure the MCP server to connect to the remote AMFS HTTP API, so all your agents share the same centralized memory.

---

## Production Checklist

- [ ] Set `AMFS_API_KEYS` to enable authentication
- [ ] Use Postgres or S3 backend for durability and shared state
- [ ] Configure resource limits in Kubernetes
- [ ] Enable Ingress with TLS for external access
- [ ] Set up Postgres backups (pg_dump, WAL archiving, or managed service)
- [ ] Monitor `/health` endpoint with your observability stack
- [ ] Consider HPA for traffic-heavy deployments

---

## Next Steps

- [HTTP API Server](/amfs/guides/http-server/) — endpoint reference and usage examples
- [S3 Adapter](/amfs/adapters/s3/) — use S3-compatible storage as the backend
- [Postgres Adapter](/amfs/adapters/postgres/) — full-text + vector search with native SQL
