---
title: Atomic Commits
layout: default
parent: Core Concepts
nav_order: 8
description: "Group multiple memory writes into a single atomic commit."
---

# Atomic Commits

AMFS supports **atomic multi-key commits** — grouping multiple writes into a single, all-or-nothing operation with a commit message, author, and unique ID.

## Why Commits?

When an agent updates related keys (e.g., a service's config and its validation rules), individual writes can leave the store in an inconsistent intermediate state. Commits ensure all changes land together or not at all.

## Using Transactions

### Python SDK

```python
with mem.transaction("Update auth config and token settings") as tx:
    tx.write("myapp/auth", "session-config", {"timeout": 3600})
    tx.write("myapp/auth", "token-ttl", 7200)
    tx.write("myapp/auth", "refresh-policy", "sliding")
# All three writes commit atomically on exit
```

### MCP Tool

```
amfs_commit_batch(
    writes=[
        {"entity_path": "myapp/auth", "key": "session-config", "value": "{}"},
        {"entity_path": "myapp/auth", "key": "token-ttl", "value": "7200"}
    ],
    message="Update auth configuration"
)
```

## Commit Structure

Each commit contains:

- **ID** — Unique 12-character hex identifier
- **Message** — Human-readable description
- **Author** — The agent that created the commit
- **Entries** — List of (entity_path, key, version, content_hash) tuples
- **Tree Hash** — SHA-256 of all entry content hashes (integrity seal)
- **Parent IDs** — Links to predecessor commits (DAG ancestry)
- **Timestamp** — When the commit was created

## Viewing the Commit Log

```bash
amfs log --limit 10
```

```python
commits = mem.commit_log(limit=20)
for c in commits:
    print(f"{c.id} | {c.author_agent_id} | {c.message}")
```

## Content Integrity

Every entry in a commit includes a `content_hash` (SHA-256 of the value) and an `integrity_chain` (hash link to the previous version). The commit's `tree_hash` seals the entire batch. Use `mem.verify()` to check for corruption.

## OSS vs Pro

| Feature | OSS | Pro |
|:--------|:---:|:---:|
| Transaction API | Yes | Yes |
| Commit model and log | Yes | Yes |
| All adapter implementations | Yes | Yes |
| Commit revert | — | Yes |
| Dashboard commit grouping in Event Log | — | Yes |
