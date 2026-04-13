---
title: DAG Versioning
layout: default
parent: Core Concepts
nav_order: 10
description: "Directed acyclic graph commit history with ancestry and merge-base."
---

# DAG Versioning

AMFS commits form a **directed acyclic graph (DAG)**, similar to git. Each commit records its parent commit(s), enabling ancestry queries, merge-base computation, and branch history tracking.

## Commit Ancestry

When a new commit is created (via `transaction()` or `amfs_commit_batch`), AMFS automatically links it to the latest commit on the same branch via `parent_ids`. Merge commits can have multiple parents.

```
c1 ← c2 ← c3 ← c4 (main)
           ↖
            c5 ← c6 (feature branch)
```

## Common Ancestor

To merge or compare branches, you often need to find the most recent common ancestor — the point where two lineages diverged.

```python
ancestor_id = mem.common_ancestor("c4", "c6")
# Returns "c2" — the fork point
```

The algorithm uses **parallel BFS** from both commits, returning the first commit that appears in both traversals.

## Branch HEAD Tracking

Each branch tracks its `head_commit_id` (the latest commit) and `base_commit_id` (where it forked from its parent branch). This enables efficient diff and merge operations.

## API Surface

| Method | Description |
|:-------|:------------|
| `mem.common_ancestor(a, b)` | Python SDK |
| `mem.commonAncestor(a, b)` | TypeScript SDK |
| `amfs_merge_base(commit_a, commit_b)` | MCP tool |
| `POST /api/v1/merge-base` | HTTP endpoint |

## OSS vs Pro

| Feature | OSS | Pro |
|:--------|:---:|:---:|
| Parent commit linking | Yes | Yes |
| Common ancestor computation | Yes | Yes |
| Branch HEAD tracking | Yes | Yes |
| Cherry-pick across branches | — | Yes |
| Sacred Timeline DAG visualization | — | Yes |
