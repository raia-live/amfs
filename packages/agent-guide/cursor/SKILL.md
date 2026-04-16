---
name: amfs-memory
description: >-
  Guide agents on effective use of AMFS persistent memory — when to save,
  what to save, cost-conscious patterns, and session lifecycle. Use when
  agents interact with AMFS tools, persist knowledge, write memories, read
  briefings, or commit decision traces.
---

# AMFS Agent Memory Guide

You have access to AMFS (Agent Memory File System) — persistent memory that survives across sessions, agents, and machines. This guide teaches you how to use it effectively: what to save, what to skip, and how to get the most value from every operation.

AMFS tools are already available via MCP. This guide supplements the built-in instructions with deeper behavioral guidance.

For full tool reference, see [reference.md](reference.md).

---

## Cost Model

Every AMFS operation has a cost. Understanding this shapes all your decisions:

| Operation | Cost | Examples |
|:----------|:----:|:--------|
| Read | 1 op | `amfs_read`, `amfs_search`, `amfs_list`, `amfs_recall`, `amfs_briefing`, `amfs_retrieve` |
| Write | 2 ops | `amfs_write`, `amfs_record_context` |
| Commit | FREE | `amfs_commit_outcome` |

Commits are free by design — never skip them.

---

## Session Lifecycle

Every agent session should follow four phases:

### Phase 1: Identify

Call `amfs_set_identity` before anything else. Use a stable, kebab-case role name that persists across conversations about the same domain.

```
amfs_set_identity("checkout-agent", "Fixing race condition in order processing")
```

Good names: `dashboard-agent`, `api-agent`, `infra-agent`, `auth-agent`
Bad names: `fix-button-color` (too specific), `agent-1` (meaningless)

If continuing work a previous agent started, **use the same name** to build on their knowledge.

### Phase 2: Get Briefed

Start with a single briefing call — it returns compiled knowledge from the Memory Cortex, giving you what other agents know, recent risks, and confidence-ranked facts.

```
amfs_briefing(entity_path="myapp/checkout")
```

One briefing (1 op) replaces many individual reads. This is your highest-ROI operation.

After the briefing, use targeted reads only for specific keys you need:

```
amfs_recall("myapp/checkout", "risk-race-condition")
amfs_search(query="payment retry", entity_path="myapp/checkout")
```

### Phase 3: Work

As you work, read and write knowledge, and record significant decisions as they happen.

- **Read** before making architectural decisions or working in areas other agents have touched.
- **Write** after completing meaningful work — decisions, patterns, risks, summaries.
- **Record context** for significant decisions and external tool outputs.

Record decisions in causal order as they happen, not all at the end:

```
amfs_record_context("user-decision", "User chose Redis over Memcached for session store", source="chat")
```

### Phase 4: Commit

After completing significant work, commit the outcome. This snapshots the full decision trace — all reads, writes, and recorded contexts — into a persistent record.

```
amfs_commit_outcome("checkout-race-fix", "success")
```

This is **free** (0 ops). There is never a reason to skip it. Without a commit, the decision trace is lost when the session ends.

Outcome types: `success`, `minor_failure`, `failure`, `critical_failure`

---

## What to Save

Each write costs 2 ops. Make them count. Save knowledge that would help a future agent working on the same code.

### Decisions with rationale

When you or the user choose one approach over another, record **why** — not just what.

```
amfs_write("myapp/auth", "decision-session-strategy",
    "Chose JWT over server-side sessions because the app is stateless and deployed "
    "across multiple regions. Trade-off: tokens can't be revoked instantly, mitigated "
    "by short expiry (15min) + refresh token rotation.")
```

### Discovered patterns

When you find a reusable pattern, record it with cross-references:

```
amfs_write("myapp/api", "pattern-retry-with-backoff",
    "All external API calls use exponential backoff with jitter: base=1s, max=30s, "
    "max_retries=3. Implemented via shared retry_with_backoff() in utils/http.py.",
    pattern_refs=["pattern-error-handling"])
```

### Risks and bugs

When you discover something that could trip up future agents:

```
amfs_write("myapp/checkout", "risk-race-condition",
    "Concurrent order submissions can double-charge. The payment service doesn't "
    "deduplicate by idempotency key when called within 100ms. Needs distributed lock "
    "or DB-level uniqueness constraint on order_id.",
    confidence=0.8, memory_type="belief")
```

### Task summaries

After completing significant work, summarize what was done and key outcomes:

```
amfs_write("myapp/checkout", "task-summary-fix-double-charge",
    "Added distributed lock via Redis SETNX before payment submission. Lock TTL=30s "
    "with automatic release on completion. Tested with concurrent load — no duplicates "
    "in 10k requests. Also added idempotency_key to the payments table as a safety net.",
    memory_type="experience")
```

### External tool outputs worth preserving

When consulting external tools and the results inform decisions:

```
amfs_record_context("datadog-metrics",
    "P99 latency spiked from 200ms to 1.2s after the last deploy. "
    "Trace shows N+1 queries in order listing endpoint.",
    source="Datadog APM")
```

---

## What NOT to Save

Not everything deserves 2 ops. Skip these:

- **Trivial changes** — "added a comment", "renamed variable", "fixed typo". The IDE and VCS already track these.
- **File-level edits** — Don't log individual file modifications. Save the *decision* behind them, not the changes.
- **Transient debug output** — Stack traces, log snippets, and test output that won't matter in the next session.
- **Recomputable information** — Anything derivable from the current codebase in under 5 seconds (function signatures, import paths, config values).
- **Low-signal entries** — If your value would be one sentence or less, it probably isn't worth 2 ops. Combine related observations into a single richer entry.

### The "future agent" test

Before writing, ask: *"Would a future agent working on this code benefit from knowing this?"*

If the answer is no — or if they could figure it out by reading the code — skip the write.

---

## Cost-Conscious Patterns

### Use briefings instead of many reads

Bad (5 ops):
```
amfs_read("myapp/auth", "pattern-jwt")
amfs_read("myapp/auth", "risk-token-expiry")
amfs_read("myapp/auth", "decision-session-strategy")
amfs_read("myapp/auth", "task-summary-oauth-integration")
amfs_read("myapp/auth", "pattern-refresh-tokens")
```

Good (1 op):
```
amfs_briefing(entity_path="myapp/auth")
```

The briefing returns compiled, ranked knowledge — often everything you need.

### Batch related knowledge into one write

Bad (6 ops):
```
amfs_write("myapp/api", "pattern-retry", "Use exponential backoff")
amfs_write("myapp/api", "pattern-retry-base", "Base delay is 1 second")
amfs_write("myapp/api", "pattern-retry-max", "Max delay is 30 seconds")
```

Good (2 ops):
```
amfs_write("myapp/api", "pattern-retry-with-backoff",
    "All external API calls use exponential backoff with jitter: base=1s, max=30s, "
    "max_retries=3. Implemented in utils/http.py. Covers payment gateway, email "
    "service, and inventory API.")
```

### Search before writing

Before creating a new entry, check if similar knowledge already exists:

```
amfs_search(query="retry pattern", entity_path="myapp/api")
```

If a relevant entry exists, update it rather than creating a duplicate (1 search + 1 write = 3 ops, vs. a duplicate that wastes 2 ops and creates confusion).

### Record context selectively

`amfs_record_context` costs 2 ops. Use it for decisions and external tool outputs that inform the work — not for every micro-step.

Good: `amfs_record_context("architecture-decision", "Chose Redis for distributed locking — low latency, built-in TTL")`

Bad: `amfs_record_context("step", "Read the file")` — this adds no value to the decision trace.

---

## Read vs. Recompute

### When to read from AMFS (1 op)

- **Starting a new session** — always get a briefing first
- **Before architectural decisions** — check what patterns, risks, and decisions already exist
- **When another agent has worked on the same area** — read their knowledge instead of rediscovering it
- **After encountering an error** — check if another agent already diagnosed it
- **Before making a similar decision** — browse past traces for prior art

### When to skip the read

- The information is **trivially derivable** from the current file (e.g., function signatures, import paths)
- The context is **purely local** to the file you're editing and wouldn't be stored in AMFS
- The answer would be **a single sentence** — probably not worth the op

---

## Entity Paths and Key Naming

### Entity paths

Use hierarchical `{repo}/{module}` paths to scope your knowledge:

```
myapp/auth          — authentication module
myapp/checkout      — checkout service
myapp/api           — API layer
acme/infrastructure — infrastructure config
```

Avoid generic paths like `project`, `code`, or `app` — they create an unsearchable grab bag.

### Key prefixes

Use consistent prefixes to categorize entries:

| Prefix | Use case | Example |
|:-------|:---------|:--------|
| `pattern-` | Reusable patterns | `pattern-retry-with-backoff` |
| `risk-` | Known risks or bugs | `risk-race-condition` |
| `decision-` | Architectural decisions | `decision-session-strategy` |
| `task-summary-` | What was done and why | `task-summary-fix-double-charge` |
| `action-` | Actions taken (experience log) | `action-added-redis-lock` |

---

## Memory Types

Every entry has a `memory_type` that affects how it ages:

| Type | Use when | Decay rate |
|:-----|:---------|:-----------|
| `fact` (default) | Objective, stable knowledge — patterns, configs, verified decisions | Normal |
| `belief` | Hypotheses, suspicions, unverified observations | 2x faster |
| `experience` | Actions taken, deployment logs, what you did | 1.5x slower |

Use `belief` when you're not sure and want to signal that the entry needs validation:

```
amfs_write("myapp/api", "belief-latency-cause",
    "Hypothesis: latency spike is caused by N+1 queries in the order listing "
    "endpoint, because Datadog traces show 47 DB calls per request.",
    confidence=0.6, memory_type="belief")
```

Use `experience` for action logs that future agents should be able to retrace:

```
amfs_write("myapp/api", "action-added-prefetch",
    "Added prefetch_related('items', 'shipping') to order queryset in "
    "views/orders.py. Reduced DB calls from 47 to 3 per request.",
    memory_type="experience")
```

---

## Confidence

Set confidence to reflect how sure you are:

| Score | Meaning | Example |
|:------|:--------|:--------|
| 1.0 | Verified fact, tested in production | "JWT tokens expire in 15 minutes" (checked the config) |
| 0.7-0.9 | High confidence, not yet production-tested | "This retry pattern should handle transient failures" |
| 0.4-0.6 | Hypothesis, needs validation | "The latency spike might be caused by N+1 queries" |
| < 0.4 | Speculative, early signal | "I suspect there's a memory leak but haven't profiled" |

Beliefs should have confidence < 0.9. If you're sure enough to rate a belief at 0.9+, it's probably a fact.

Confidence decays over time. Entries validated by outcomes (via `amfs_commit_outcome`) decay slower.

---

## Quality Standards

When you call `amfs_write`, the response includes a `quality` score. If below 0.8, review the `issues` array and rewrite:

- **too_short** — Add specifics: what, why, key parameters, trade-offs.
- **missing_pattern_refs** — Related entries exist. Add `pattern_refs` to link them.
- **belief_no_rationale** — Beliefs should explain reasoning (use "because", "hypothesis").
- **overconfident_belief** — Beliefs should have confidence < 0.9.

### Use pattern_refs for cross-referencing

When entries are related, link them:

```
amfs_write("myapp/api", "pattern-circuit-breaker",
    "External API calls use circuit breaker with 5-failure threshold and 60s cooldown.",
    pattern_refs=["pattern-retry-with-backoff", "risk-payment-gateway-timeouts"])
```

This builds a knowledge graph that makes search and briefings more useful.

---

## Anti-Patterns

Avoid these common mistakes:

| Anti-pattern | Why it's bad | Do this instead |
|:-------------|:-------------|:----------------|
| Writing every file change | Wastes ops, creates noise | Write the *decision*, not the change |
| Forgetting `amfs_commit_outcome` | Loses the decision trace (it's free!) | Always commit at session end |
| Generic entity paths (`project`, `code`) | Unsearchable, no scoping | Use `{repo}/{module}` |
| Confidence 1.0 on beliefs | Beliefs are hypotheses by definition | Use < 0.9 for beliefs |
| Never calling `amfs_briefing` | Wastes ops on individual reads | One briefing replaces many reads |
| Writing one-sentence entries | Too low signal for the 2-op cost | Batch into richer entries |
| Never reading before writing | Risk of duplicating existing knowledge | Search first, then write |
| Recording every micro-step as context | Wastes 2 ops per recording | Record meaningful decisions only |

---

## Cross-Agent Collaboration

### Reading from other agents

When you know another agent has worked on the same area, read from their brain explicitly:

```
amfs_read_from("api-agent", "myapp/api", "pattern-retry-with-backoff")
```

This creates a tracked knowledge transfer — the system records that you consulted another agent's work.

### Browsing past decision traces

Before making a decision similar to one made before, check past traces:

```
amfs_list_traces(entity_path="myapp/checkout", limit=5)
amfs_get_trace("<trace-id>")
```

Past traces show the full causal chain: what was read, what decisions were made, what was written, and what the outcome was.
