---
title: SaaS Connection
layout: default
parent: Guides
nav_order: 9
description: "Connect to AMFS SaaS — authenticate agents, use scoped API keys, and route memory through the HTTP API."
---

# Connecting to AMFS SaaS
{: .no_toc }

When using AMFS as a hosted service, all agents connect through the HTTP API with an API key. This guide covers credentials, client configuration, and security best practices.

For **ops metering, quotas, and the control-plane API** used on SenseLab Cloud, see **[Hosted billing & metering]({{ site.baseurl }}/guides/saas-billing-metering/)**.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AMFS SaaS runs as a managed, multi-tenant service. Each account is isolated via Postgres Row-Level Security, and every request is authenticated, scoped, and audited. Agents never access the database directly — all memory operations flow through the HTTP API.

```
┌────────────────────────────────────┐
│         Your Agents / Tools        │
│                                    │
│  MCP Client  Python SDK  REST API  │
└───────┬──────────┬──────────┬──────┘
        │          │          │
        │   AMFS_HTTP_URL + AMFS_API_KEY
        │          │          │
        ▼          ▼          ▼
┌────────────────────────────────────┐
│         AMFS HTTP API              │
│  ┌──────────┐  ┌────────────────┐  │
│  │ Auth +   │  │ Scope          │  │
│  │ Tenant   │──▶ Enforcement    │  │
│  │ Resolver │  │ (entity-path)  │  │
│  └──────────┘  └────────┬───────┘  │
│                         │          │
│  ┌──────────────────────▼───────┐  │
│  │ Postgres (RLS isolation)     │  │
│  │ SET amfs.current_account_id  │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
```

---

## Getting Your Credentials

### From the Dashboard

1. Go to the **Agents** page in the AMFS dashboard
2. The **MCP Connection Card** at the top shows your:
   - **MCP URL** — the HTTP API endpoint
   - **API Token** — your account's API key (click the eye icon to reveal, clipboard icon to copy)
3. If you don't have an API key yet, the card will prompt you to create one

### From Settings

Go to **Settings > API Keys** to:
- Create additional keys with specific scopes
- Set entity-path restrictions (e.g., `checkout-service/**` for read/write)
- Manage expiration and rate limits
- Revoke compromised keys

---

## Two Environment Variables

Every client — MCP, SDK, CLI, REST — uses the same two environment variables:

| Variable | Description |
|:---------|:------------|
| `AMFS_HTTP_URL` | Base URL of the AMFS API (e.g. `https://amfs-login.sense-lab.ai`) |
| `AMFS_API_KEY` | Your API key (starts with `amfs_sk_`) |

When `AMFS_HTTP_URL` is set, all other adapter configuration (`AMFS_POSTGRES_DSN`, `AMFS_DATA_DIR`, `amfs.yaml`) is ignored.

---

## Connecting MCP Clients

### Cursor

Add to your project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/amfs-internal",
        "amfs-mcp-server-pro"
      ],
      "env": {
        "AMFS_HTTP_URL": "https://amfs-login.sense-lab.ai",
        "AMFS_API_KEY": "amfs_sk_your_key_here"
      }
    }
  }
}
```

Or with the OSS MCP server (when `amfs-adapter-http` is installed):

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/amfs",
        "amfs-mcp-server"
      ],
      "env": {
        "AMFS_HTTP_URL": "https://amfs-login.sense-lab.ai",
        "AMFS_API_KEY": "amfs_sk_your_key_here"
      }
    }
  }
}
```

Or without a local process at all, using the hosted Streamable HTTP endpoint:

```json
{
  "mcpServers": {
    "senselab": {
      "url": "https://mcp.sense-lab.ai/mcp",
      "headers": { "Authorization": "Bearer amfs_sk_your_key_here" }
    }
  }
}
```

The hosted endpoint exposes a focused set of memory tools rather than the full
stdio surface, and needs nothing installed. Use a local process when you want
every tool.

### Claude Code

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/amfs",
        "amfs-mcp-server"
      ],
      "env": {
        "AMFS_HTTP_URL": "https://amfs-login.sense-lab.ai",
        "AMFS_API_KEY": "amfs_sk_your_key_here"
      }
    }
  }
}
```

### Claude Desktop

Same configuration format as Claude Code. Add to your `claude_desktop_config.json`.

### Other MCP Clients

Any MCP-compatible client that can set environment variables on the MCP server process will work. Set `AMFS_HTTP_URL` and `AMFS_API_KEY` in the environment.

---

## Connecting via REST API

Make direct HTTP calls with the `X-AMFS-API-Key` header:

```bash
# Write a memory entry
curl -X POST https://amfs-login.sense-lab.ai/api/v1/entries \
  -H "Content-Type: application/json" \
  -H "X-AMFS-API-Key: amfs_sk_your_key_here" \
  -d '{
    "entity_path": "checkout-service",
    "key": "retry-pattern",
    "value": {"max_retries": 3},
    "confidence": 0.85
  }'

# Read a memory entry
curl https://amfs-login.sense-lab.ai/api/v1/entries/checkout-service/retry-pattern \
  -H "X-AMFS-API-Key: amfs_sk_your_key_here"

# Search
curl -X POST https://amfs-login.sense-lab.ai/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-AMFS-API-Key: amfs_sk_your_key_here" \
  -d '{"entity_path": "checkout-service", "min_confidence": 0.5}'
```

See the [API Reference](/amfs/reference/api/) for the full endpoint list.

---

## Connecting via Python SDK

Set the environment variables and the SDK auto-detects the HTTP adapter:

```bash
export AMFS_HTTP_URL="https://amfs-login.sense-lab.ai"
export AMFS_API_KEY="amfs_sk_your_key_here"
```

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="my-agent")
mem.write("checkout-service", "retry-pattern", {"max_retries": 3})
```

Or configure explicitly with the `HttpAdapter`:

```python
from amfs import AgentMemory
from amfs_adapter_http import HttpAdapter

adapter = HttpAdapter(
    base_url="https://amfs-login.sense-lab.ai",
    api_key="amfs_sk_your_key_here",
)
mem = AgentMemory(agent_id="my-agent", adapter=adapter)
```

{: .note }
The `amfs-adapter-http` package must be installed for the HTTP adapter. Install with `pip install amfs-adapter-http`.

---

## Security Model

### Account Isolation

Every API key belongs to an account. Postgres Row-Level Security ensures that requests can only see and modify data belonging to their account — even if there's a bug in application code.

### Scoped API Keys

API keys can be restricted to specific entity paths and permissions:

```
amfs_sk_live_abc123  →  checkout-service/**  [READ_WRITE]
                        shared/patterns/*     [READ]
```

The scope is checked on every request:
- **READ** operations (read, list, search, history) require `READ` or `READ_WRITE` permission on the entity path
- **WRITE** operations (write, commit_outcome) require `WRITE` or `READ_WRITE` permission on the entity path

Scopes use glob patterns:
- `checkout-service/*` — direct children only
- `checkout-service/**` — recursive match
- `*` — full access within the account

### Audit Logging

Every state-changing operation is recorded with actor, action, resource, and IP address. View the audit log in the Dashboard.

---

## What NOT to Do

{: .warning }
**Never share `AMFS_POSTGRES_DSN` with external agents.** Direct database access bypasses the HTTP API's tenant middleware, scope enforcement, rate limiting, and audit logging. In multi-tenant deployments, this is a security risk.

{: .warning }
**Never hardcode API keys in source code.** Use environment variables or a secrets manager. Rotate keys regularly via the Dashboard.

**Anti-patterns:**

| Don't | Do Instead |
|:------|:-----------|
| Set `AMFS_POSTGRES_DSN` in agent environments | Set `AMFS_HTTP_URL` + `AMFS_API_KEY` |
| Share one API key across all agents | Create scoped keys per agent/service |
| Give agents admin keys | Use the narrowest scope that works |
| Hardcode keys in `.cursor/mcp.json` committed to git | Use environment variable references or `.env` files |

---

## Troubleshooting

**"amfs-adapter-http is not installed"**
Install the HTTP adapter package: `pip install amfs-adapter-http`

**401 Unauthorized**
Check that your API key is correct, active, and not expired. Verify in Dashboard > Settings > API Keys.

**403 Scope Denied**
Your API key doesn't have permission for the requested entity path or operation. Check the key's scopes in the Dashboard.

**Both AMFS_HTTP_URL and AMFS_POSTGRES_DSN are set**
`AMFS_HTTP_URL` takes precedence. A warning is logged. Remove `AMFS_POSTGRES_DSN` from the agent's environment to avoid confusion.

---

## Next Steps

- [Environment Variables Reference](/amfs/reference/environment-variables/) — full list with precedence rules
- [API Reference](/amfs/reference/api/) — REST endpoint documentation
- [MCP Setup](/amfs/guides/mcp/) — local and remote MCP configuration
- [OSS vs Pro](/amfs/editions/) — feature comparison
