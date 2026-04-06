---
title: Git-like Timeline
layout: default
parent: Core Concepts
nav_order: 6
description: "How AMFS treats each agent's memory as a Git repository with branches, commits, and history."
---

# Git-like Timeline
{: .no_toc }

AMFS models every agent's memory as a **Git repository**. Each write, outcome, cross-agent read, and brain brief is recorded as an event on the agent's timeline — like commits in Git. This gives you full auditability, rollback capability, and branch-based collaboration.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## The Mental Model

When an agent is created in AMFS, it implicitly gets a **repository** with a `main` branch. Every operation the agent performs is logged as an **event** on that branch:

```
Agent A creates its first memory
  └── main branch is created automatically

Agent A writes "checkout-service/retry-pattern"
  └── Event: write (entity_path, key, version, confidence)

Agent B reads Agent A's shared memory
  └── Event: cross_agent_read (reader_agent, entity, key)

Brain brief compiles
  └── Event: brief_compiled (digest_type, entry_count, entities)

Agent A commits an outcome
  └── Event: outcome (outcome_ref, outcome_type, causal_confidence)
```

---

## Event Types

Every operation produces a typed event on the agent's timeline:

| Event Type | Description | Triggered By |
|:-----------|:------------|:-------------|
| `write` | Memory entry created or updated | `mem.write()` |
| `outcome` | Outcome committed, confidence back-propagated | `mem.commit_outcome()` |
| `cross_agent_read` | Another agent read this agent's shared memory | `mem.read()` / `mem.read_from()` |
| `brief_compiled` | Brain brief / digest compiled by the Cortex | Cortex worker |
| `webhook_ingested` | External context ingested via webhook | Webhook endpoint |
| `branch_created` | New branch created from this agent's memory | Pro: `create_branch()` |
| `branch_merged` | Branch merged back into main | Pro: `merge_branch()` |
| `branch_closed` | Branch closed without merging | Pro: `close_branch()` |
| `access_granted` | Branch access granted to a user/team/API key | Pro: `grant_branch_access()` |
| `access_revoked` | Branch access revoked | Pro: `revoke_branch_access()` |
| `tag_created` | Snapshot tag created on a branch | Pro: `create_tag()` |
| `rollback` | Memory rolled back to a point in time | Pro: `rollback()` |
| `cherry_pick` | Entries cherry-picked from a branch to main | Pro: `cherry_pick()` |
| `fork` | Agent memory forked to a new agent | Pro: `fork()` |

---

## Querying the Timeline

### Python SDK

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="deploy-agent")

events = mem.timeline(limit=50)
for event in events:
    print(f"[{event.created_at}] {event.event_type}: {event.summary}")
```

### HTTP API

```bash
curl "http://localhost:8080/api/v1/agents/deploy-agent/timeline?limit=50"
```

Response:

```json
{
  "agentId": "deploy-agent",
  "events": [
    {
      "event_type": "write",
      "branch": "main",
      "summary": "Wrote checkout-service/retry-pattern v2",
      "details": {
        "entity_path": "checkout-service",
        "key": "retry-pattern",
        "version": 2,
        "confidence": 0.85
      },
      "created_at": "2026-04-03T14:30:00Z"
    }
  ],
  "count": 1
}
```

### MCP

```
amfs_timeline(agent_id="deploy-agent", limit=50)
```

---

## Branches (Pro)

With AMFS Pro, you can create branches from an agent's main timeline — just like Git branches. Branches let you:

- **Experiment safely** — write to a branch without affecting main memory
- **Share selectively** — grant read or read/write access to specific users, teams, or API keys
- **Review changes** — see the diff between a branch and main before merging
- **Merge or discard** — bring branch changes into main, or close the branch

### Creating a Branch

```python
from amfs_branching.sdk import extend_with_branching

mem = AgentMemory(agent_id="deploy-agent")
extend_with_branching(mem)

mem.create_branch("experiment/retry-v3", description="Testing aggressive retry")
```

### Writing to a Branch

All read/write operations accept a `branch` parameter:

```python
mem.write(
    "checkout-service",
    "retry-pattern",
    {"max_retries": 5, "backoff": "linear"},
    confidence=0.7,
    branch="experiment/retry-v3",
)
```

Or via the HTTP API:

```bash
curl -X POST http://localhost:8080/api/v1/entries \
  -H "Content-Type: application/json" \
  -d '{
    "entity_path": "checkout-service",
    "key": "retry-pattern",
    "value": {"max_retries": 5, "backoff": "linear"},
    "confidence": 0.7,
    "branch": "experiment/retry-v3"
  }'
```

### Reading from a Branch

```bash
curl "http://localhost:8080/api/v1/entries/checkout-service/retry-pattern?branch=experiment/retry-v3"
```

### Access Control

Branch access is enforced at the API level. When a request targets a non-`main` branch, the caller's API key is checked against the branch access grants:

```python
mem.grant_branch_access(
    "experiment/retry-v3",
    grantee_type="api_key",
    grantee_id="amfs_sk_agent_b_key",
    permission="read_write",
)
```

| Permission | Can Read | Can Write |
|:-----------|:---------|:----------|
| `read` | Yes | No |
| `read_write` | Yes | Yes |

Requests without access receive a `403 Forbidden` response.

### Viewing the Diff

```python
diffs = mem.diff_branch("experiment/retry-v3")
for d in diffs:
    print(f"{d.diff_type}: {d.entity_path}/{d.key}")
```

### Merging

```python
from amfs_core.models import MergeStrategy

result = mem.merge_branch("experiment/retry-v3", strategy=MergeStrategy.FAST_FORWARD)
print(f"Merged {result.merged_entries} entries")
```

### Pull Requests

For team workflows, create a PR for review before merging:

```python
pr = mem.create_pull_request(
    "Aggressive retry strategy",
    source_branch="experiment/retry-v3",
    description="Testing 5x retries with linear backoff",
)

mem.review_pull_request(pr.id, "approved", comment="Looks good")
mem.merge_pull_request(pr.id)
```

---

## Tags and Rollback (Pro)

### Tags

Create named snapshots of your agent's memory at a point in time:

```python
mem.create_tag("v1.0-stable", description="Pre-migration baseline")
```

### Rollback

Restore memory to a previous state:

```python
mem.rollback(tag_name="v1.0-stable")
```

Or roll back to a specific timestamp:

```python
from datetime import datetime, timezone

mem.rollback(timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc))
```

---

## Fork (Pro)

Clone an agent's entire memory into a new agent:

```python
count = mem.fork("deploy-agent-v2")
print(f"Forked {count} entries to deploy-agent-v2")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Agent Brain                       │
│                                                     │
│  main ──●──●──●──●──●──●──●──●──●─── (current)    │
│              │                                      │
│              └── experiment/retry-v3 ──●──●──●      │
│                                                     │
│  Events: write, outcome, read, brief, webhook       │
│  Each event = a "commit" on the timeline            │
└─────────────────────────────────────────────────────┘
```

The timeline is always available in OSS (main branch only). Branching, merging, tags, rollback, and access control require AMFS Pro.
