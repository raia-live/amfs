---
title: Provenance
layout: default
parent: Core Concepts
nav_order: 4
description: "How AMFS tracks who wrote what, and when."
---

# Provenance

Every memory entry in AMFS carries **provenance** — metadata about who wrote it and when. This creates an audit trail and enables agents to assess the trustworthiness of information.

---

## Provenance Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `agent_id` | `str` | The agent that wrote the entry (e.g., `"review-agent"`) |
| `session_id` | `str` | The session in which it was written |
| `written_at` | `datetime` | UTC timestamp of when the entry was created |
| `pattern_refs` | `list[str]` | Cross-reference tags linking to related entries |

---

## Automatic Provenance

Provenance is recorded automatically. You don't need to set it manually:

```python
mem = AgentMemory(agent_id="review-agent")
entry = mem.write("svc", "finding", "potential memory leak")

print(entry.provenance.agent_id)    # "review-agent"
print(entry.provenance.session_id)  # auto-generated UUID
print(entry.provenance.written_at)  # 2025-06-15T10:30:00+00:00
```

---

## Pattern References

Use `pattern_refs` to cross-reference related entries:

```python
mem.write(
    "checkout-service",
    "risk-timeout",
    "HTTP client has no timeout configured",
    pattern_refs=["timeout-patterns", "checkout-service/risk-latency"],
)
```

Pattern refs are free-form strings. Use them to build a web of related knowledge that agents can follow.

---

## Agent Identity

In the MCP server, agent identity is auto-detected from the environment:

| Environment | Agent ID Format |
|:------------|:----------------|
| Cursor | `cursor/<username>` |
| Claude Code | `claude-code/<username>` |
| Other | `agent/<username>` |

Override with the `AMFS_AGENT_ID` environment variable:

```bash
export AMFS_AGENT_ID="deploy-bot"
```

---

## Querying by Provenance

Search for entries written by a specific agent:

```python
results = mem.search(agent_id="review-agent")
for entry in results:
    print(f"{entry.key} by {entry.provenance.agent_id}")
```
