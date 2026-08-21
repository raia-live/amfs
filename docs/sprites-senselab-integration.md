---
title: Fly.io Sprites × SenseLab — Integration Proposal
layout: default
nav_exclude: true
description: "A partnership and technical proposal to give Fly.io Sprites persistent agent memory via SenseLab."
---

# Fly.io Sprites × SenseLab
## Persistent memory for disposable agent microVMs

*A partnership and technical integration proposal for the Fly.io team.*

---

## The one-line pitch

Sprites give agents a clean, isolated, disposable machine. SenseLab gives those agents a **memory that outlives the machine** — so every Sprite you spin up boots already knowing what the last one learned, without your customers wiring up a database.

## Why now

Fly's team told us memory is the next thing Sprites need: when a new sandbox spins up, the agent should be able to read what came before and keep going. That's exactly the gap SenseLab was built for. A Sprite's filesystem is per-instance; checkpoints resume *one* VM, but nothing today lets knowledge flow **across** VMs, agents, and time. Agents on Sprites re-derive the same decisions and repeat the same mistakes on every cold start.

This is a natural, high-leverage partnership: Fly owns the compute substrate and the agent runtime; SenseLab owns durable, structured, cross-agent memory. Neither has to build the other's hard part.

## What the customer experience looks like

The bar is **zero-config memory**. A developer running an agent on Sprites should get continuity for free.

1. They pick a stable name for the work — an `entity_path` like `sprites/acme/checkout`. (Fly can derive this automatically from app + machine group, so most users never think about it.)
2. They set one secret (`AMFS_API_KEY`) once.
3. Every Sprite they launch for that workload boots already briefed: the agent's first move is to load the compiled memory for its entity.

No schema, no vector DB to run, no glue code. The "magic moment" is the second Sprite recalling a decision the first one made and it never made itself.

---

## Technical integration

The integration is built and working in this repository. It has four pieces; Fly only needs to adopt two env vars.

### 1. Headless entity binding (the core mechanism)

We added a single environment variable, **`AMFS_ENTITY_PATH`**, that binds a disposable environment to its home memory. When set, the SenseLab MCP server:

- defaults `amfs_briefing()` (no argument) to that entity,
- advertises the binding in its instructions and in the `amfs_set_identity` response, and
- records it on the agent's `auto_context_paths`.

This is what makes hydration *automatic* rather than something the agent has to be told to do. A fresh Sprite's very first briefing call loads exactly the right memory, with no path-guessing.

### 2. Bake once, inject at runtime

For agents that run **inside** the Sprite and speak MCP (Cursor, Claude Code, Codex, and Gemini CLI — all preinstalled on Sprites), we bake the MCP config and instruction files into the **base image** once:

```dockerfile
RUN curl -sSL https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh \
      | bash -s -- --yes --client all --api-url https://amfs-login.sense-lab.ai
```

At runtime, only two env vars are injected per workload:

```bash
AMFS_API_KEY=amfs_sk_...              # secret — never baked into an image/checkpoint
AMFS_ENTITY_PATH=sprites/acme/checkout   # the workload's home entity
```

This keeps cold-start fast (no per-boot install), keeps the API key out of checkpoints, and lets one base image serve many workloads and tenants. The installer now supports all four preinstalled agents (we added **Gemini CLI**) and an `--entity-path` flag that templates the concrete path into each agent's instructions.

### 3. SDK path for orchestrators

For teams that drive Sprites from a **Node or Python orchestrator**, memory is available programmatically:

- **Python:** `amfs-sprites` — `provision_memory()`, `session.hydrate_prompt()`, `session.commit_outcome()`.
- **TypeScript:** `@senselab-ai/amfs` — an async `HttpAdapter` bridge (`briefingAsync`, `retrieveAsync`, `writeAsync`, `commitOutcomeAsync`, …) so a Node orchestrator can hydrate a prompt and commit outcomes over HTTP. (Fly's ecosystem is TypeScript-first, so this path is first-class.)

`hydrate_prompt()` turns durable memory into a system-prompt block, so continuity works for **any** agent — MCP-native or not.

### 4. Where memory lives

Hosted SenseLab (`https://amfs-login.sense-lab.ai`) — nothing for Fly or the customer to operate. Memory is structured, versioned, confidence-ranked, and queryable; agents can read each other's memory. (A self-hosted option exists for customers who need data residency, but hosted is the default and fastest to ship.)

---

## How this composes with Sprites features

| | Fly.io checkpoints | SenseLab memory |
|:--|:--|:--|
| Scope | One VM's filesystem/process state | Knowledge across all VMs, agents, sessions |
| Survives a *new* VM | No | Yes — it's external |
| Shape | Opaque snapshot | Structured, versioned, confidence-ranked, queryable |
| Cross-agent | No | Yes |

They're complementary: **checkpoints resume a VM fast; SenseLab makes every VM start smart.** A great combined story: checkpoint to pause/resume a session cheaply, SenseLab to carry the durable knowledge forward across the fleet.

## Security & reliability

- **Secrets never baked.** `AMFS_API_KEY` is injected at runtime, so it never lands in an image layer or checkpoint.
- **Tenant isolation.** A distinct API key per tenant enforces memory isolation server-side. Outcome inputs are secret-scanned and redacted before storage.
- **Graceful degradation.** If SenseLab is unreachable, hydration returns a "fresh start" note instead of failing, and read paths fail soft. Memory enhances context; it isn't a hard dependency of the request path.

## Proposed rollout

- **Phase 1 — Guide + template (now).** Ship the [Sprites integration guide](/amfs/guides/sprites/), the base-image snippet, and the runnable examples in this repo. Fly links to it; early customers opt in with two env vars.
- **Phase 2 — First-class base image.** Fly offers a Sprite base image with the MCP config baked in, and surfaces `AMFS_ENTITY_PATH` / `AMFS_API_KEY` as documented, blessed knobs (auto-deriving the entity from app/machine-group metadata).
- **Phase 3 — Native UX.** Memory continuity is a checkbox in the Sprites launch flow; Fly provisions the SenseLab connection and entity binding on the customer's behalf.

## What each side owns

- **SenseLab provides:** the hosted memory backend, the MCP server + installer (with headless binding and Gemini support), the Python/TypeScript SDKs, the `amfs-sprites` package, docs, and examples — all in this repository today.
- **Fly provides:** the base image integration point, the two runtime env vars in the Sprites launch path, and (ideally) auto-derivation of `AMFS_ENTITY_PATH` from workload metadata.

## Artifacts in this repository

- `docs/guides/sprites.md` — developer guide (MCP path + SDK path)
- `packages/integrations/sprites/` — the `amfs-sprites` package + `sprite-bootstrap.sh`
- `examples/sprites_integration.py` — Python magic-moment demo
- `examples/typescript/sprites_orchestrator.ts` — TypeScript orchestrator demo
- `install-mcp.sh` — `--entity-path` flag + Gemini CLI support
- `docs/reference/environment-variables.md` — `AMFS_ENTITY_PATH`

---

*Let's make "spin up a Sprite" mean "spin up an agent that already knows the context." We're ready to co-develop the base image and the launch-flow UX whenever Fly is.*
