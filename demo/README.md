# AMFS Demo — E-commerce Performance Marketing

A 40-minute demo showing how AMFS solves skill observability, agent-to-agent
code reuse, and pattern coordination for a customer with ~200 agent skills.

## Quick Start (local, in-memory)

```bash
cd demo
npm install
npx tsx run-demo.ts
```

This runs the full 4-act demo using the in-memory adapter (no server needed).
All agents share a single in-memory store, simulating a shared AMFS server.
Good for rehearsing the demo flow and showing code concepts.

## What it shows

### Act 1 — "Your agents start remembering"
- `facebook-ads-agent` generates 200 lines of code and stores it in AMFS
- `reporting-agent` searches by `patternRef: "facebook-ads-api"`, finds the code, reuses it
- Shows three lookup strategies: pattern tag, full-text query, `readFrom()`
- Demonstrates confidence evolution when code breaks and gets updated

### Act 2 — "An agent audits your other agents"
- `ops-review-agent` inventories all agents and their activity
- Detects redundant patterns (multiple agents generating the same code)
- Flags low-confidence entries
- Writes a structured health report back to AMFS

### Act 3 — "Your agents coordinate" (Rooms)
- Shows the SDK API for Entity Rooms (Pro feature)
- Agents join a room, discuss patterns, negotiate on a shared function signature
- Demonstrates `roomDiscuss`, `negotiateCreate`, `negotiatePropose`, `negotiateRespond`

### Act 4 — "From patterns to libraries"
- Knowledge graph reveals connections between agents and patterns
- Cortex briefing compiles the extraction picture
- Frames the "last mile" roadmap: clustering, extraction, skill rewriting

## Running against AMFS SaaS (full Pro)

The SaaS at https://amfs-login.sense-lab.ai has the full Pro stack:
Rooms, pattern detection, cortex, knowledge graph, dashboard, and all
MCP tools. Two env vars are all you need:

```bash
export AMFS_HTTP_URL="https://amfs-login.sense-lab.ai"
export AMFS_API_KEY="amfs_sk_your_key_here"
```

Get your API key from the AMFS Dashboard > Agents page (MCP Connection Card)
or Settings > API Keys.

### MCP setup for Cursor (customer's agents)

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "amfs": {
      "command": "uvx",
      "args": ["amfs-mcp-server"],
      "env": {
        "AMFS_HTTP_URL": "https://amfs-login.sense-lab.ai",
        "AMFS_API_KEY": "amfs_sk_your_key_here"
      }
    }
  }
}
```

### Python SDK (customer's agents)

```bash
pip install amfs amfs-adapter-http
export AMFS_HTTP_URL="https://amfs-login.sense-lab.ai"
export AMFS_API_KEY="amfs_sk_your_key_here"
```

```python
from amfs import AgentMemory
mem = AgentMemory(agent_id="facebook-ads-agent")
# Automatically uses HTTP adapter when AMFS_HTTP_URL is set
```

### TypeScript SDK (customer's agents)

```typescript
import { AgentMemory, HttpAdapter } from "@senselab-ai/amfs";
const mem = new AgentMemory("facebook-ads-agent", {
  adapter: new HttpAdapter({
    url: process.env.AMFS_HTTP_URL!,
    apiKey: process.env.AMFS_API_KEY,
  }),
});
```

## File structure

```
demo/
  run-demo.ts        # Full 4-act demo (single file, runs everything)
  package.json       # Dependencies
  tsconfig.json      # TypeScript config
  README.md          # This file
```
