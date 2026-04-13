---
title: Rich Diff & Patch
layout: default
parent: Core Concepts
nav_order: 12
description: "Structural JSON diffing, field-level changes, and serializable patches."
---

# Rich Diff & Patch

AMFS provides **structural JSON diffing** that shows exactly which fields changed between entry versions, along with **serializable patches** that can be transmitted and applied.

## Structural Diff

Instead of opaque "old value vs new value" comparisons, AMFS computes field-level changes using RFC 6901 JSON Pointer paths:

```python
result = mem.diff("myapp/auth", "session-config")
# {
#   "diff_type": "modified",
#   "field_changes": [
#     {"path": "/timeout", "operation": "replace", "old_value": 30, "new_value": 60},
#     {"path": "/max_retries", "operation": "add", "new_value": 5}
#   ]
# }
```

### Change Operations

| Operation | Description |
|:----------|:------------|
| `add` | A new field was added |
| `remove` | A field was deleted |
| `replace` | A field's value changed |

### Diff Types

| Type | Description |
|:-----|:------------|
| `added` | Entry is new (no previous version) |
| `modified` | Entry exists in both versions with changes |
| `deleted` | Entry was removed |

## Memory Patches

Patches are serializable diffs that can be stored, transmitted, and applied:

```python
patch = mem.create_patch("myapp/auth", "session-config")
# {
#   "entity_path": "myapp/auth",
#   "key": "session-config",
#   "changes": [...],
#   "source_version": 1,
#   "target_version": 2
# }
```

Patches can be applied to transform values:

```python
from amfs_core.diff import apply_patch
new_value = apply_patch(old_value, patch)
```

## MCP Tool

```
amfs_diff(entity_path="myapp/auth", key="session-config")
```

## HTTP Endpoints

- `POST /api/v1/diff` — Compute a structural diff
- `POST /api/v1/patches` — Create a serializable patch

## OSS vs Pro

| Feature | OSS | Pro |
|:--------|:---:|:---:|
| Structural JSON diff | Yes | Yes |
| Memory patches (create + apply) | Yes | Yes |
| `amfs_diff` MCP tool | Yes | Yes |
| HTTP diff/patch endpoints | Yes | Yes |
| Rich diffs in dashboard Branches/PRs | — | Yes |
| Semantic diff annotations | — | Yes |
