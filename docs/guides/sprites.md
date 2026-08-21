---
title: Fly.io Sprites Integration
layout: default
parent: Guides
nav_order: 14
description: "Give Fly.io Sprites persistent agent memory with SenseLab — a freshly spun-up microVM auto-hydrates everything prior sessions learned."
---

# Fly.io Sprites Integration
{: .no_toc }

[Fly.io Sprites](https://fly.io/sprites) are hardware-isolated, disposable Linux microVMs for agent workloads. SenseLab gives them **memory that survives the microVM**, so a freshly spun-up Sprite boots already knowing what earlier sessions learned.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## The problem

A Sprite's filesystem is per-instance. When the fleet recycles a Sprite — or spins up a new one for the next task — anything the agent wrote to local disk is gone. The agent starts every task from zero: re-deriving the same decisions, repeating the same mistakes, unaware of what a sibling Sprite discovered five minutes ago.

Checkpoints solve *resuming the same VM*. They don't solve *sharing knowledge across VMs, agents, and time*. That's what SenseLab adds.

## The model

SenseLab stores memory in a shared backend keyed by a durable **`entity_path`** (e.g. `sprites/acme/checkout`). The Sprite is disposable; the `entity_path` is forever. Bind a Sprite to an `entity_path` and it reads/writes the same memory every other session on that path uses.

```
   Sprite A (spun up 10:00)  ─┐
   Sprite B (spun up 10:05)  ─┼─▶  entity_path: sprites/acme/checkout  ─▶  hosted SenseLab
   Sprite C (spun up 12:30)  ─┘         (durable, shared, versioned)
```

Two integration surfaces, depending on where the agent runs:

1. **Agent runs *inside* the Sprite and speaks MCP** (Cursor, Claude Code, Codex, Gemini CLI — all preinstalled on Sprites). Bake the MCP config into the base image; the agent calls `amfs_briefing`, `amfs_write`, `amfs_commit_outcome` as tools.
2. **A Node/Python orchestrator drives Sprites over the SDK.** Use `amfs-sprites` (Python) or `@senselab-ai/amfs` (TypeScript) to hydrate a prompt and commit outcomes programmatically.

---

## Quickstart: MCP agents on the Sprite

### 1. Bake config into the base image (once)

Run the bootstrap while building the Sprite base image. It configures every preinstalled agent (Cursor, Claude Code, Codex, Gemini) non-interactively:

```dockerfile
# In your Sprite base image
RUN curl -sSL https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh \
      | bash -s -- --yes --client all \
        --api-url https://amfs-login.sense-lab.ai
```

This bakes the MCP server config and the recall-first instruction files. It does **not** bake secrets or the entity binding — those are injected at runtime.

### 2. Inject two env vars at runtime

When you launch a Sprite for a specific workload, provide:

```bash
export AMFS_API_KEY=amfs_sk_...            # the SenseLab API key (never bake this)
export AMFS_ENTITY_PATH=sprites/acme/checkout   # this Sprite's home entity
```

`AMFS_ENTITY_PATH` is the headless binding. With it set, the MCP server:

- defaults `amfs_briefing()` (no argument) to that entity,
- advertises the binding in its instructions and in the `amfs_set_identity` response, and
- adds the path to the agent's `auto_context_paths`.

So the agent's first `amfs_briefing` call loads exactly the right memory with zero guessing.

{: .note }
Injecting `AMFS_API_KEY` and `AMFS_ENTITY_PATH` at runtime (instead of baking them) keeps the key out of the checkpoint and lets one base image serve many workloads and tenants.

### Alternative: one-shot bootstrap script

`amfs-sprites` ships a headless wrapper that reads the same env vars:

```bash
AMFS_API_KEY=amfs_sk_... AMFS_ENTITY_PATH=sprites/acme/checkout \
  bash packages/integrations/sprites/sprite-bootstrap.sh
```

---

## Quickstart: orchestrator over the SDK

### Python

```bash
pip install "amfs-sprites[http]"
```

```python
from amfs_sprites import provision_memory, derive_entity_path

# On a real Sprite the base image exports AMFS_ENTITY_PATH / AMFS_HTTP_URL /
# AMFS_API_KEY, so provision_memory() takes no arguments. Shown explicitly here:
session = provision_memory(
    entity_path=derive_entity_path("acme", "checkout"),
    agent_id="checkout-agent",
    api_url="https://amfs-login.sense-lab.ai",
    api_key="amfs_sk_...",
)

# Boot the agent already briefed — inject this into your system prompt.
system_prompt = session.hydrate_prompt()

# ... run the agent ...

# Persist the decision trace + a durable note for the next Sprite.
session.commit_outcome(
    "checkout-hardening",
    "success",
    task_input="make checkout resilient to retries",
    summary="Added idempotency keys on charge calls.",
)
```

Runnable end-to-end demo (simulates two disposable microVMs sharing one entity):

```bash
uv run python examples/sprites_integration.py
```

### TypeScript

```bash
npm install @senselab-ai/amfs
```

```ts
import { AgentMemory, HttpAdapter, OutcomeType } from "@senselab-ai/amfs";

const adapter = new HttpAdapter({
  url: process.env.AMFS_HTTP_URL!,
  apiKey: process.env.AMFS_API_KEY!,
  agentId: "checkout-agent",
});
const memory = new AgentMemory("checkout-agent", { adapter });
const entityPath = process.env.AMFS_ENTITY_PATH ?? "sprites/acme/checkout";

// Load prior context (the async methods route through the HTTP adapter).
const top = await memory.retrieveAsync(`context for ${entityPath}`, { entityPath, limit: 6 });
const systemPrompt = top.map(({ entry }) => `- ${(entry as any).key}: ${(entry as any).value}`).join("\n");

// ... run the agent ...

await memory.writeAsync(entityPath, "decision-idempotency", "Idempotency keys on charges.", { confidence: 0.9 });
await memory.commitOutcomeAsync("checkout-hardening", OutcomeType.SUCCESS, { entityPath });
```

Full example: [`examples/typescript/sprites_orchestrator.ts`](https://github.com/raia-live/amfs/blob/main/examples/typescript/sprites_orchestrator.ts).

---

## Choosing an `entity_path`

The `entity_path` is the join key for shared memory, so derive it from **stable** inputs — never from the ephemeral Sprite or machine id, which changes on every spin-up and would scatter memory across single-use paths.

| Good (stable, shared) | Bad (ephemeral, isolates memory) |
|:----------------------|:---------------------------------|
| `sprites/acme/checkout` | `sprites/<machine-id>` |
| `sprites/acme/support-triage` | `sprites/run-8f3a1c` |
| `sprites/<tenant>/<project>` | `sprites/<timestamp>` |

`derive_entity_path("acme", "checkout")` slugifies and joins parts into `sprites/acme/checkout`. Use a consistent `agent_id` (role, kebab-case) across Sprites too, so one role's brain accumulates over time.

---

## Security

- **Bake config, inject secrets.** Bake the MCP config and instruction files into the base image; inject `AMFS_API_KEY` at runtime so it never lands in a checkpoint or image layer.
- **Scope keys per tenant.** Use a distinct SenseLab API key per tenant/customer so memory isolation is enforced server-side.
- **Outcomes are secret-scanned.** `task_input` and summaries are scanned and redacted before storage.

## Graceful degradation

If SenseLab is unreachable, the agent should still run. `hydrate_prompt` returns a "fresh start" note rather than raising when it can't reach the backend, and the SDK's read paths fail soft (returning empty results). Treat memory as an enhancement to context, not a hard dependency of the request path — wrap `provision_memory` in a try/except in your orchestrator if you want a hard guarantee.

## Checkpoints vs. SenseLab

They're complementary:

| | Fly.io checkpoints | SenseLab memory |
|:--|:--|:--|
| Scope | One VM's filesystem/process state | Knowledge across all VMs, agents, sessions |
| Survives fleet recycle | The checkpoint, yes; a *new* VM, no | Yes — it's external |
| Shape | Opaque snapshot | Structured, versioned, confidence-ranked, queryable |
| Cross-agent | No | Yes — agents read each other's memory |

Use checkpoints to resume a paused VM fast; use SenseLab so every VM starts smart.

---

## See also

- [MCP Server guide](/amfs/guides/mcp/) — installer flags, including `--entity-path`
- [SaaS Connection guide](/amfs/guides/saas/) — hosted SenseLab setup
- [Environment Variables](/amfs/reference/environment-variables/) — `AMFS_ENTITY_PATH`, `AMFS_HTTP_URL`, `AMFS_API_KEY`
