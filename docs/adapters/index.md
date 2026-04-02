---
title: Adapters
layout: default
nav_order: 8
has_children: true
description: "Storage backends for AMFS: filesystem, Postgres, S3-compatible, and custom adapters."
permalink: /adapters/
---

# Adapters

AMFS uses a pluggable adapter system for storage. All adapters implement the same contract, so your code works identically regardless of the backend.
{: .fs-6 .fw-300 }

---

## Available Adapters

| Adapter | Package | Best For |
|:--------|:--------|:---------|
| [Filesystem](/amfs/adapters/filesystem/) | `amfs` (included) | Local development, single machine |
| [Postgres](/amfs/adapters/postgres/) | `amfs-adapter-postgres` | Team sharing, production, full-text + vector search |
| [S3](/amfs/adapters/s3/) | `amfs-adapter-s3` | Distributed teams, AI pipelines, cloud-native |
| [Custom](/amfs/adapters/custom/) | — | Build your own adapter |

---

## Choosing an Adapter

```
Local dev, single machine         → Filesystem
Team sharing, search-heavy        → Postgres (with pgvector for semantic search)
Cloud-native, multi-region        → S3 (AWS, ACS, MinIO, R2)
Custom requirements               → Build your own with AdapterABC
```

---

## Adapter Contract

Every adapter implements these operations:

| Operation | Description |
|:----------|:------------|
| `read(entity_path, key)` | Get the current version of a key |
| `write(entry)` | Persist a new version (CoW) |
| `list(entity_path?)` | Enumerate current entries |
| `watch(entity_path, callback)` | Real-time change notifications |
| `commit_outcome(record)` | Back-propagate confidence via outcomes |
| `search(query)` | Search entries with filters |
| `stats()` | Return memory statistics |

All adapters pass the same contract test suite, ensuring identical behavior regardless of storage backend.
