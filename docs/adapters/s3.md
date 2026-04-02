---
title: S3 Adapter
layout: default
parent: Adapters
nav_order: 3
description: "Store AMFS memory in any S3-compatible object storage — AWS S3, Accelerated Cloud Storage (ACS), Cloudflare R2, MinIO, and more."
---

# S3 Adapter
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The S3 adapter stores memory entries as versioned JSON objects in any S3-compatible bucket. This makes AMFS compatible with the broader cloud storage ecosystem — AWS S3, Accelerated Cloud Storage (ACS), Cloudflare R2, MinIO, DigitalOcean Spaces, and others.

Best for:

- **Distributed teams** needing shared memory across regions
- **AI training pipelines** where memory lives alongside model weights and datasets
- **Serverless environments** where a local filesystem isn't available
- **Large-scale deployments** storing millions of memory entries

---

## Installation

```bash
pip install amfs-adapter-s3
```

Requires `boto3`. AWS credentials are resolved via the standard credential chain (env vars, `~/.aws/credentials`, IAM role, etc.).

---

## Configuration

### YAML

```yaml
namespace: production
layers:
  primary:
    adapter: s3
    options:
      bucket: my-amfs-bucket
      prefix: amfs/                    # object key prefix (optional)
      endpoint_url: ""                 # custom endpoint for non-AWS (optional)
      region_name: us-east-1           # AWS region (optional)
```

### Environment Variables

| Variable | Description | Default |
|:---------|:------------|:--------|
| `AMFS_S3_BUCKET` | S3 bucket name | — (required) |
| `AMFS_S3_PREFIX` | Object key prefix | `amfs/` |
| `AMFS_S3_ENDPOINT` | Custom S3 endpoint URL (for ACS, MinIO, R2) | — |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | AWS access key | — |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — |

### Programmatic

```python
from amfs import AgentMemory
from amfs_s3 import S3Adapter

adapter = S3Adapter(
    bucket="my-amfs-bucket",
    prefix="amfs/",
    endpoint_url="https://s3.acceleratedcloudstorage.com",
    region_name="us-east-1",
)

mem = AgentMemory(agent_id="my-agent", adapter=adapter)
```

---

## Object Layout

Entries are stored as JSON objects with the following key structure:

```
{prefix}{namespace}/entries/{entity_path}/{key}/v{version}.json
{prefix}{namespace}/outcomes/{outcome_ref}.json
```

For example, with prefix `amfs/` and namespace `default`:

```
amfs/default/entries/checkout-service/retry-pattern/v1.json
amfs/default/entries/checkout-service/retry-pattern/v2.json
amfs/default/outcomes/DEP-287.json
```

Each version is a complete snapshot of the `MemoryEntry` — no deltas, no dependencies. This makes individual entries independently readable and portable.

---

## Using with ACS (Accelerated Cloud Storage)

[Accelerated Cloud Storage](https://www.acceleratedcloudstorage.com/) provides S3-compatible object storage optimized for AI workloads. To use AMFS with ACS:

```yaml
namespace: production
layers:
  primary:
    adapter: s3
    options:
      bucket: my-acs-bucket
      endpoint_url: https://s3.acceleratedcloudstorage.com
```

Or via environment variables:

```bash
export AMFS_S3_BUCKET=my-acs-bucket
export AMFS_S3_ENDPOINT=https://s3.acceleratedcloudstorage.com
export AWS_ACCESS_KEY_ID=your-acs-key
export AWS_SECRET_ACCESS_KEY=your-acs-secret
```

---

## Using with MinIO

For local development or self-hosted S3:

```bash
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

```yaml
namespace: dev
layers:
  primary:
    adapter: s3
    options:
      bucket: amfs
      endpoint_url: http://localhost:9000
```

---

## Watch (Polling)

The S3 adapter implements `watch()` via periodic polling (every 5 seconds by default), since S3 does not support native change notifications. For real-time streaming, use the [HTTP API server](/amfs/guides/http-server/) with SSE.

---

## Limitations

- **No native full-text search** — `search()` falls back to scanning all entries and filtering in memory. For high-volume search, use the [Postgres adapter](/amfs/adapters/postgres/) or route through the HTTP API.
- **No native vector search** — `semantic_search()` requires an embedder and scans entries in memory. For production-scale semantic search, combine with Postgres + pgvector.
- **Eventual consistency** — S3 provides strong read-after-write consistency for PUTs, but LIST operations may show stale results briefly.

---

## Next Steps

- [Postgres Adapter](/amfs/adapters/postgres/) — full-text + vector search with native SQL
- [HTTP API Server](/amfs/guides/http-server/) — serve AMFS over REST for any client
- [Docker & Kubernetes](/amfs/guides/docker/) — deploy AMFS with S3 backend on Kubernetes
