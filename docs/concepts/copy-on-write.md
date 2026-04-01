---
title: Copy-on-Write
layout: default
parent: Core Concepts
nav_order: 2
description: "How AMFS preserves full version history through copy-on-write semantics."
---

# Copy-on-Write (CoW)

Every write in AMFS creates a new version of an entry. The previous version is preserved as **superseded**. This means you never lose history — every change is traceable.

---

## How It Works

When you write to a key that already exists:

1. The current version is marked as **superseded**
2. A new version is created with an incremented version number
3. The new version becomes the **current** version

```python
# Version 1 is created
mem.write("svc", "config", {"pool_size": 5})

# Version 2 is created; version 1 is superseded
mem.write("svc", "config", {"pool_size": 10})

# Version 3 is created; version 2 is superseded
mem.write("svc", "config", {"pool_size": 20})
```

Reading always returns the **current** (latest) version:

```python
entry = mem.read("svc", "config")
print(entry.version)  # 3
print(entry.value)    # {"pool_size": 20}
```

---

## Viewing History

Use `list()` with `include_superseded=True` to see all versions:

```python
all_versions = mem.list("svc", include_superseded=True)
for entry in all_versions:
    print(f"v{entry.version}: {entry.value}")
```

Or use the CLI:

```bash
amfs inspect diff svc config
```

---

## On-Disk Layout (Filesystem Adapter)

The filesystem adapter stores each version as a separate JSON file:

```
.amfs/default/checkout-service/retry-pattern/
├── v001_superseded.json
├── v002_superseded.json
└── v003_current.json
```

- `_current.json` — the active version
- `_superseded.json` — previous versions, kept for history

Writes use atomic rename (`os.rename()`) with advisory file locking (`flock()`) to prevent corruption from concurrent writes.

---

## Conflict Handling

When two agents write to the same key concurrently, the `conflict_policy` determines behavior:

| Policy | Behavior |
|:-------|:---------|
| `LAST_WRITE_WINS` | Default. The latest write takes precedence. |
| `RAISE` | Raises `StaleWriteError` if another agent advanced the version since your last read. |

```python
from amfs_core.models import ConflictPolicy

mem = AgentMemory(
    agent_id="my-agent",
    conflict_policy=ConflictPolicy.RAISE,
)
```

You can also provide a custom conflict resolver:

```python
def resolve(existing, incoming, value):
    # Merge logic here
    return {**existing.value, **value}

mem = AgentMemory(
    agent_id="my-agent",
    on_conflict=resolve,
)
```

---

## Why CoW?

- **Auditability** — Every change is recorded. You can always trace how knowledge evolved.
- **Safety** — No data is lost. Mistakes can be identified by comparing versions.
- **Outcome tracking** — Outcomes create new versions with updated confidence scores, preserving the score history.
