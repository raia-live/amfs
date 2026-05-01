<p align="center">
  <img src="docs/assets/amfs-architecture.png" alt="SenseLab AMFS — Agent Management File System" width="700" />
</p>

<h1 align="center">SenseLab AMFS — Agent Management File System</h1>

<p align="center">
  <strong>Git for agent memory. Every agent gets a brain — version-controlled, diffable, reviewable.</strong>
</p>

<p align="center">
  <a href="https://github.com/raia-live/amfs/actions/workflows/test-python.yml"><img src="https://github.com/raia-live/amfs/actions/workflows/test-python.yml/badge.svg" alt="Python Tests" /></a>
  <a href="https://github.com/raia-live/amfs/actions/workflows/test-typescript.yml"><img src="https://github.com/raia-live/amfs/actions/workflows/test-typescript.yml/badge.svg" alt="TypeScript Tests" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0" /></a>
  <a href="https://pypi.org/project/amfs/"><img src="https://img.shields.io/pypi/v/amfs?color=green" alt="PyPI" /></a>
  <a href="https://www.npmjs.com/package/@senselab-ai/amfs"><img src="https://img.shields.io/npm/v/@senselab-ai/amfs?color=green" alt="npm" /></a>
</p>

<p align="center">
  <a href="https://raia-live.github.io/amfs/">Documentation</a> · <a href="https://raia-live.github.io/amfs/getting-started/quickstart/">Quick Start</a> · <a href="https://github.com/orgs/raia-live/projects/2">Roadmap</a> · <a href="https://raia-live.github.io/amfs/contributing/">Contributing</a>
</p>

---

## Why AMFS?

When agents share memory today, it's chaos — last write wins, no branching, no review, no rollback. It's like coding without Git.

The solution isn't "better permissions" or "smarter retrieval." It's giving agents the same collaboration model developers already live in: **version control for knowledge**.

```
You already know this:               AMFS does the same for agent memory:

  repo                                 agent brain
  ├── main branch                      ├── main (what the agent knows)
  ├── feature branch                   ├── experiment branch (isolated changes)
  ├── pull request                     ├── pull request (review before merge)
  ├── code review                      ├── diff (what changed in the branch)
  ├── merge                            ├── merge (accept changes into main)
  ├── git log                          ├── timeline (every operation logged)
  └── git revert                       └── rollback (restore to any point)
```

Every write is a versioned commit. Every agent has provenance. Changes stay isolated on branches until reviewed. Roll back to any point. Fork an entire brain to a new agent.

## Quick Start

```bash
pip install amfs
```

```python
from amfs import AgentMemory, OutcomeType

mem = AgentMemory(agent_id="review-agent")

# Agent discovers a pattern and commits it to memory
mem.write("checkout-service", "retry-pattern",
          {"max_retries": 3, "strategy": "exponential-backoff"},
          confidence=0.85)

# Another agent reads it — the read is tracked automatically
entry = mem.read("checkout-service", "retry-pattern")

# When the deploy fails, confidence on related entries adjusts
mem.commit_outcome("INC-1042", OutcomeType.CRITICAL_FAILURE)
```

> **[Full quick start guide →](https://raia-live.github.io/amfs/getting-started/quickstart/)**

## Installation

```bash
pip install amfs                    # Python SDK
npm install @senselab-ai/amfs        # TypeScript SDK
pip install amfs-http-server        # REST API server
pip install amfs-adapter-postgres   # Postgres backend
pip install amfs-adapter-s3         # S3 backend
pip install amfs-cli                # CLI tools
pip install amfs-strands            # Strands Agents plugin
```

Or run with Docker:

```bash
docker run -p 8080:8080 -v amfs-data:/data ghcr.io/raia-live/amfs
```

## Features

| Feature | Description |
|:--------|:------------|
| **Git-like Timeline** | Every write, outcome, and cross-agent read is logged. Full history, always. |
| **Branching & PRs** | Create branches, diff changes, open pull requests, merge or discard. |
| **Rollback & Tags** | Named snapshots. Restore to any tag or event. |
| **Access Control** | Grant read or read/write per branch, user, team, or API key. |
| **Versioned Knowledge** | Copy-on-Write — every write creates a new version. Nothing is lost. |
| **Confidence & Outcomes** | Entries carry trust scores that evolve when deploys succeed or incidents happen. |
| **Causal Explainability** | `explain()` shows exactly which memories and contexts drove a decision. |
| **Knowledge Graph** | Relationships auto-materialize from normal operations. |
| **Hybrid Search** | Full-text + semantic + recency + confidence in a single ranked result set. |
| **MCP Server** | First-class support for Cursor, Claude Desktop, Claude Code, and any MCP client. |
| **Connectors** | PagerDuty, GitHub, Slack, Jira — or [build your own](https://raia-live.github.io/amfs/guides/connectors/). |
| **Python & TypeScript** | Same API in both languages. Plus [Strands](https://raia-live.github.io/amfs/guides/strands/), [CrewAI](https://raia-live.github.io/amfs/guides/crewai/), LangGraph, LangChain, AutoGen. |

## MCP Integration

One command to give any MCP-compatible client (Cursor, Claude Desktop, Claude Code) persistent agent memory:

```bash
curl -sSL https://raw.githubusercontent.com/raia-live/amfs/main/install-mcp.sh | bash
```

> **[MCP setup guide →](https://raia-live.github.io/amfs/guides/mcp/)** · **[Cursor plugin →](https://github.com/raia-live/cursor-plugin)**

## OSS vs Pro

AMFS is open source under [Apache 2.0](LICENSE). The OSS edition gives you the full memory engine — versioned writes, confidence scoring, outcome feedback, causal traces, knowledge graph, hybrid search, git-like timeline, SDKs, adapters, HTTP API, MCP server, and CLI.

**[AMFS Pro](https://raia-live.github.io/amfs/editions/)** unlocks the full Git model: branching, merge, pull requests, access control, tags, rollback, cherry-pick, fork, multi-tenant isolation, immutable decision traces, automated pattern detection, an intelligence layer, and a web dashboard.

> OSS = single-branch repo with full history. Pro = GitHub.

**[Full comparison →](https://raia-live.github.io/amfs/editions/)**

## Development

```bash
git clone https://github.com/raia-live/amfs.git && cd amfs
uv pip install -e packages/core -e packages/adapters/filesystem -e packages/sdk-python -e packages/cli -e packages/http-server
uv run pytest tests/ -v
```

> **[Contributing guide →](https://raia-live.github.io/amfs/contributing/)**

## Community

- [GitHub Discussions](https://github.com/raia-live/amfs/discussions) — questions, ideas, show & tell
- [Roadmap](https://github.com/orgs/raia-live/projects/2) — what's shipped and what's next
- [Issues](https://github.com/raia-live/amfs/issues) — bug reports and feature requests

## License

[Apache License 2.0](LICENSE) — free for commercial use, modification, and distribution.
