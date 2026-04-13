---
title: Agent-Memory Binding
layout: default
parent: Core Concepts
nav_order: 11
description: "Agent profiles, capabilities, memory contracts, and agent discovery."
---

# Agent-Memory Binding

AMFS goes beyond simple key-value storage by deeply binding agents to their memory through **profiles**, **capabilities**, **contracts**, and **discovery**.

## Agent Profiles

Every agent can declare a profile describing its role and defaults:

```python
mem.set_profile(
    description="Manages authentication and session configuration",
    tags=["auth", "security", "backend"],
    auto_context_paths=["myapp/auth", "myapp/sessions"],
)
```

Profiles include:
- **Description** — What this agent does
- **Tags** — Searchable labels for discovery
- **Auto-context paths** — Entity paths to automatically load at session start
- **Default branch** — Which branch the agent works on by default

## Agent Capabilities

Agents declare what they know about — their domains of expertise:

```python
mem.declare_capability(
    "database-migrations",
    description="Manages schema changes and data migrations",
    entity_paths=["myapp/db", "myapp/migrations"],
)
```

## Memory Contracts

Contracts define quality and structure expectations for entries:

```python
mem.set_contracts([{
    "entity_path": "myapp/auth",
    "key_pattern": "config-*",
    "min_confidence": 0.8,
    "required_fields": ["timeout", "max_retries"],
    "ttl_required": True,
    "description": "Auth configs must have timeout and retries",
}])
```

Contracts enforce:
- **Confidence bounds** — min/max confidence thresholds
- **Required fields** — Fields that must be present in dict values
- **TTL requirements** — Whether entries must have an expiration
- **Key patterns** — Glob patterns for which keys the contract applies to

## Agent Discovery

Find agents by what they know or where they work:

```python
# Find agents that know about database migrations
agents = mem.discover_agents(capability="database-migrations")

# Find agents that work on the auth module
agents = mem.discover_agents(entity_path="myapp/auth")
```

## MCP Tools

| Tool | Description |
|:-----|:------------|
| `amfs_set_profile` | Set agent description, tags, auto-context |
| `amfs_declare_capability` | Declare a domain of expertise |
| `amfs_set_contract` | Define quality expectations for entries |
| `amfs_discover_agents` | Find agents by capability or entity path |

## OSS vs Pro

| Feature | OSS | Pro |
|:--------|:---:|:---:|
| Profiles, capabilities, contracts | Yes | Yes |
| Agent discovery | Yes | Yes |
| Contract validation at write time | Yes | Yes |
| Auto-computed capabilities | — | Yes |
| Contract violation alerts | — | Yes |
| Enhanced dashboard Agent Brain section | — | Yes |
