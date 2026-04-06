---
title: Quick Start
layout: default
parent: Getting Started
nav_order: 2
description: "Give your agent a persistent brain in 5 minutes."
---

# Quick Start

AMFS gives each agent a **persistent brain** that works like a Git repository. When an agent writes, it's forming a memory — logged as an event on the agent's timeline. When it recalls, it's accessing its own experience. When it reads shared knowledge, it benefits from what other agents have learned. Every interaction is tracked so you can always answer: *what did I know, when did I know it, and who told me?*

---

## How Memory Sharing Works

All agents write to the **same storage backend**. The `agent_id` acts as a provenance tag — it marks *who* wrote each entry, not *where* the entry lives.

By default, every memory is **shared**: any agent can read it. This enables collective learning — Agent B can read what Agent A learned. But agents can also write **private** memories that only they can access. This gives each agent control over what it shares.

```
┌──────────────────────────────────────────┐
│           Shared Memory Pool             │
│                                          │
│  ┌──────────┐  ┌──────────┐              │
│  │ Agent A's │  │ Agent B's │             │
│  │  shared   │  │  shared   │  ← visible │
│  │ memories  │  │ memories  │    to all   │
│  └──────────┘  └──────────┘              │
│                                          │
│  ┌──────────┐  ┌──────────┐              │
│  │ Agent A's │  │ Agent B's │             │
│  │  private  │  │  private  │  ← visible │
│  │ memories  │  │ memories  │    to owner │
│  └──────────┘  └──────────┘    only      │
└──────────────────────────────────────────┘
```

---

## 1. Create Your Brain

Every agent gets its own brain via `AgentMemory`:

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="review-agent")
```

The `agent_id` is the agent's identity. Everything it writes is tagged with this ID, and it can later recall only its own memories.

---

## 2. Form a Memory

When your agent learns something, write it to memory:

```python
entry = mem.write(
    "checkout-service",       # entity_path — what this knowledge is about
    "retry-pattern",          # key — name for this piece of knowledge
    {                         # value — any JSON-serializable data
        "pattern": "exponential-backoff",
        "max_retries": 3,
        "base_delay": "200ms",
    },
    confidence=0.85,
)

print(entry.version)                 # 1
print(entry.provenance.agent_id)     # "review-agent"
print(entry.shared)                  # True (default — visible to all agents)
```

Every write creates an immutable copy-on-write version. Writing the same key again creates version 2, preserving the full history.

---

## 3. Keep Things Private

Not everything should be shared. Use `shared=False` for internal reasoning, scratchpad notes, or sensitive context:

```python
# Private memory — only this agent can see it
mem.write(
    "checkout-service",
    "internal-analysis",
    {"risk_score": 0.92, "reasoning": "Error rate doubled after last deploy"},
    shared=False,
)

# Shared memory — the conclusion that other agents should know
mem.write(
    "checkout-service",
    "risk-assessment",
    "High risk: error rate trending up since v2.3.1",
    confidence=0.9,
)
```

Private entries are invisible to other agents across all methods — `read()`, `search()`, `list()`, and `read_from()` all skip them. Only the owning agent can access its private entries via `recall()` and `my_entries()`.

---

## 4. Recall Your Memory

Ask your brain: *"What do I know about this?"*

```python
entry = mem.recall("checkout-service", "retry-pattern")

print(entry.value)       # {"pattern": "exponential-backoff", ...}
print(entry.confidence)  # 0.85
```

`recall()` returns only entries **written by this agent**, including private ones. If another agent wrote a different version, `recall()` ignores it — it's this brain's direct experience.

---

## 5. Read Shared Knowledge

Ask the shared pool: *"What does anyone know about this?"*

```python
entry = mem.read("checkout-service", "retry-pattern")
```

`read()` returns the latest **shared** version by any agent. Private entries from other agents are never returned. Both `read()` and `recall()` return `None` if no matching entry exists.

---

## 6. Learn from Another Agent

Explicitly pull knowledge from a specific agent's brain:

```python
entry = mem.read_from("deploy-agent", "checkout-service", "deploy-config")

if entry:
    print(f"Learned from deploy-agent: {entry.value}")
```

`read_from()` makes cross-agent knowledge transfer explicit and trackable. It only returns **shared** entries — you cannot read another agent's private memories. The read is logged in the causal chain so you can always trace where knowledge came from.

---

## 7. See What's in Your Brain

List everything this agent has written:

```python
my_memories = mem.my_entries()
for e in my_memories:
    print(f"{e.entity_path}/{e.key} (v{e.version}, shared={e.shared})")

# Filter to a specific entity
checkout_memories = mem.my_entries("checkout-service")
```

`my_entries()` returns both shared and private entries — it's your complete brain.

---

## 8. Learn from Experience

When something significant happens, record the outcome. AMFS automatically adjusts confidence scores on related entries:

```python
from amfs import OutcomeType

# An incident related to the retry pattern — confidence increases
updated = mem.commit_outcome(
    outcome_ref="INC-1042",
    outcome_type=OutcomeType.CRITICAL_FAILURE,
)
```

{: .tip }
If you don't pass `causal_entry_keys`, AMFS uses **auto-causal linking** — it applies the outcome to every entry the agent read during the current session.

---

## 9. Know Who You've Learned From

Track inter-agent memory relationships:

```python
# Which agents have I read from?
reads = mem.cross_agent_reads()
# {'deploy-agent': [{'entity_path': 'checkout-service', 'key': 'deploy-config', 'read_count': 2}]}

# Just the agent IDs
agents = mem.agents_i_read_from()
# ['deploy-agent']
```

---

## 10. Watch for Changes

Get notified in real-time when knowledge changes:

```python
def on_change(entry):
    print(f"Updated: {entry.entity_path}/{entry.key} v{entry.version}")

handle = mem.watch("checkout-service", on_change)

# ... later, stop watching
handle.cancel()
```

---

## 11. Context Manager

Use `AgentMemory` as a context manager for clean shutdown:

```python
with AgentMemory(agent_id="review-agent") as mem:
    mem.write("svc", "key", "value")
# Background threads cleaned up automatically
```

---

## The Mental Model

| What you want to do          | Method                  | Sees private?          | Who wrote it?       |
| ---------------------------- | ----------------------- | ---------------------- | ------------------- |
| Form a shared memory         | `write()`               | —                      | You (this agent)    |
| Form a private memory        | `write(shared=False)`   | —                      | You (this agent)    |
| Recall your own knowledge    | `recall()`              | Yes (yours only)       | Only you            |
| Read shared knowledge        | `read()`                | No                     | Anyone (latest)     |
| Learn from a specific agent  | `read_from(agent_id)`   | No                     | That specific agent |
| See all your memories        | `my_entries()`          | Yes (yours only)       | Only you            |
| Know who taught you          | `cross_agent_reads()`   | —                      | Other agents        |
| See your memory timeline     | `timeline()`            | —                      | You (events log)    |

---

## 12. View Your Timeline

Every operation is recorded on your agent's git-like timeline — like commits in a repo:

```python
events = mem.timeline(limit=10)
for event in events:
    print(f"[{event.event_type}] {event.summary}")
```

With AMFS Pro, you can create branches, merge changes, and share memory selectively. See [Git-like Timeline](/amfs/concepts/git-timeline/) for details.

---

## Next Steps

- [Configuration](/amfs/getting-started/configuration/) — YAML config, adapters, and environment variables
- [Core Concepts](/amfs/concepts/) — understand CoW, confidence, and outcome propagation
- [Git-like Timeline](/amfs/concepts/git-timeline/) — how agent memory works like Git
- [Python SDK Guide](/amfs/guides/python/) — full SDK reference with advanced features
