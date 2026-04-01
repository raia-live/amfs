---
title: Filesystem
layout: default
parent: Adapters
nav_order: 1
description: "The default filesystem adapter — stores memory as versioned JSON files."
---

# Filesystem Adapter
{: .no_toc }

The default adapter. Stores memory as versioned JSON files on the local filesystem. No external dependencies required.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

Included with the main `amfs` package:

```bash
pip install amfs
```

---

## Configuration

### YAML

```yaml
namespace: default
layers:
  primary:
    adapter: filesystem
    options:
      root: .amfs
```

### Programmatic

```python
from amfs_filesystem import FilesystemAdapter
from pathlib import Path

adapter = FilesystemAdapter(
    root=Path(".amfs"),
    namespace="default",
)
```

### Environment Variable

```bash
export AMFS_DATA_DIR=/data/shared-memory
```

---

## On-Disk Layout

Entries are stored as versioned JSON files organized by namespace, entity, and key:

```
.amfs/
└── default/                          ← namespace
    └── checkout-service/             ← entity_path
        └── retry-pattern/            ← key
            ├── v001_superseded.json   ← old version
            ├── v002_superseded.json   ← old version
            └── v003_current.json      ← active version
```

### JSON File Contents

Each file contains a complete `MemoryEntry`:

```json
{
  "amfs_version": "0.1.0",
  "entity_path": "checkout-service",
  "key": "retry-pattern",
  "version": 3,
  "value": {"pattern": "exponential-backoff", "max_retries": 5},
  "provenance": {
    "agent_id": "review-agent",
    "session_id": "abc-123",
    "written_at": "2025-06-15T10:30:00+00:00",
    "pattern_refs": ["retry-logic"]
  },
  "confidence": 0.85,
  "outcome_count": 0,
  "ttl_at": null,
  "embedding": null
}
```

---

## Write Safety

The filesystem adapter uses two mechanisms for safe concurrent writes:

1. **Atomic rename** — Writes go to a temporary `.tmp` file, then `os.rename()` atomically moves it into place. This is safe on POSIX systems even during crashes.

2. **Advisory file locking** — `flock()` on a per-key lock file prevents concurrent writes from interleaving.

---

## Watch

The filesystem adapter watches entity directories for changes and calls your callback when new versions appear.

```python
def on_change(entry):
    print(f"{entry.key} changed")

handle = mem.watch("checkout-service", on_change)
```

---

## When to Use

- Local development
- Single-machine setups
- Quick prototyping
- CI/CD pipelines (ephemeral memory)

For team sharing or multi-machine setups, use the [Postgres adapter](/amfs/adapters/postgres/).
