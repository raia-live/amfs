---
name: amfs-memory
description: >-
  Guide agents on effective use of AMFS persistent memory — when to save,
  what to save, cost-conscious patterns, and session lifecycle. Use when
  agents interact with AMFS tools, persist knowledge, write memories, read
  briefings, or commit decision traces.
---

# AMFS Agent Memory Guide

Persistent memory that survives across sessions, agents, and machines. This guide covers decision rules and conventions — not tool syntax (the MCP server already provides that).

## Cost Model

| Operation | Cost | Examples |
|:----------|:----:|:--------|
| Read | 1 op | `amfs_read`, `amfs_search`, `amfs_list`, `amfs_recall`, `amfs_briefing`, `amfs_retrieve` |
| Write | 2 ops | `amfs_write`, `amfs_record_context` |
| Commit | FREE | `amfs_commit_outcome` — always commit, never skip |

## Session Lifecycle

Every session follows four phases:

1. **Identify** — `amfs_set_identity` with a stable kebab-case role name (e.g. `api-agent`, `infra-agent`). Reuse the same name across conversations about the same domain. Don't use task-specific names like `fix-button-color`.

2. **Get Briefed** — `amfs_briefing` returns compiled knowledge from the Cortex. One briefing (1 op) replaces many individual reads. Always start here.

3. **Work** — read/write/record as you go. Write after meaningful work, not after every edit. Record decisions as they happen, not at the end.

4. **Commit** — `amfs_commit_outcome` snapshots the full decision trace. This is free. Without it, the trace is lost when the session ends.

## What to Save (worth 2 ops)

- **Decisions with rationale** — why X over Y, trade-offs considered
- **Discovered patterns** — reusable across sessions, link with `pattern_refs`
- **Risks and bugs** — gotchas that would trip up future agents
- **Task summaries** — what was done and key outcomes after significant work
- **External tool outputs** — API results, metrics, incident data that informed decisions

## What NOT to Save

- **Trivial edits** — "renamed variable", "added comment" — VCS tracks these
- **File-level changes** — save the *decision*, not the diff
- **Recomputable info** — anything derivable from the codebase in under 5 seconds
- **Transient debug output** — stack traces, test logs
- **One-sentence entries** — too low signal for 2 ops; batch into richer entries

**The test:** "Would a future agent benefit from knowing this, or could they figure it out from the code?"

## Cost-Conscious Patterns

- **Briefing first** — `amfs_briefing` gives compiled context in 1 op vs. many individual reads
- **Batch writes** — one rich entry beats three thin ones (2 ops vs. 6 ops)
- **Search before writing** — avoid duplicating existing entries
- **Record context selectively** — `amfs_record_context` costs 2 ops; use for meaningful decisions, not micro-steps
- **Always commit** — `amfs_commit_outcome` is free and preserves the decision trace

## Entity Paths

Use `{repo}/{module}` paths. Avoid generic paths like `project` or `code`.

```
myapp/auth, myapp/checkout, myapp/api, acme/infrastructure
```

## Key Prefixes

| Prefix | Use case |
|:-------|:---------|
| `pattern-` | Reusable patterns |
| `risk-` | Known risks or bugs |
| `decision-` | Architectural decisions |
| `task-summary-` | What was done and why |
| `action-` | Actions taken (experience log) |

## Memory Types

| Type | Use when | Decay |
|:-----|:---------|:------|
| `fact` (default) | Stable knowledge — patterns, configs, verified decisions | Normal |
| `belief` | Hypotheses, unverified observations (confidence < 0.9) | 2x faster |
| `experience` | Actions taken, deployment logs | 1.5x slower |

## Confidence Scale

| Score | Meaning |
|:------|:--------|
| 1.0 | Verified fact, tested in production |
| 0.7–0.9 | High confidence, not yet production-tested |
| 0.4–0.6 | Hypothesis, needs validation |
| < 0.4 | Speculative, early signal |

Beliefs should always be < 0.9. If you're sure enough for 0.9+, it's a fact.

## Anti-Patterns

| Don't | Do instead |
|:------|:-----------|
| Write every file edit | Write the *decision*, not the change |
| Forget `amfs_commit_outcome` | Always commit — it's free |
| Use generic entity paths | Use `{repo}/{module}` |
| Set confidence=1.0 on beliefs | Beliefs are hypotheses — keep < 0.9 |
| Skip `amfs_briefing` | One briefing replaces many reads |
| Write one-sentence entries | Batch into richer entries |
| Record every micro-step | Record meaningful decisions only |
