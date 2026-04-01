---
title: Contributing
layout: default
nav_order: 11
description: "Set up a development environment and contribute to AMFS."
permalink: /contributing/
---

# Contributing
{: .no_toc }

We welcome contributions! This guide covers how to set up a development environment, run tests, and submit changes.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

- Python 3.11+
- Node.js 18+ (for TypeScript SDK)
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Docker (optional, for Postgres adapter tests)

---

## Setup

```bash
git clone https://github.com/raia-live/amfs.git
cd amfs

# Install all Python packages in editable mode
uv pip install \
  -e packages/core \
  -e packages/adapters/filesystem \
  -e packages/sdk-python \
  -e packages/cli \
  -e packages/mcp-server
```

For the Postgres adapter:

```bash
uv pip install -e packages/adapters/postgres
```

For the TypeScript SDK:

```bash
cd packages/sdk-typescript
npm install
```

---

## Project Structure

```
amfs/
├── packages/
│   ├── core/                    # Models, ABC, engine, lifecycle, outcome
│   ├── adapters/
│   │   ├── filesystem/          # CoW via atomic rename
│   │   └── postgres/            # psycopg3 + triggers
│   ├── sdk-python/              # AgentMemory, config, factory
│   ├── sdk-typescript/          # @amfs/sdk (TypeScript port)
│   ├── integrations/
│   │   ├── crewai/
│   │   ├── langgraph/
│   │   ├── autogen/
│   │   └── langchain/
│   ├── cli/                     # Typer CLI
│   └── mcp-server/              # FastMCP server
└── tests/
    ├── unit/                    # Unit tests
    └── integration/             # Adapter contract tests
```

### Package Relationships

```
amfs-core          ← models, ABCs, engine (no I/O)
  ↑
amfs-filesystem    ← filesystem adapter
amfs-postgres      ← Postgres adapter
  ↑
amfs (sdk-python)  ← AgentMemory, config, factory
  ↑
amfs-cli           ← CLI commands
amfs-mcp-server    ← MCP tools
```

---

## Running Tests

### Unit Tests

```bash
uv run pytest tests/unit/ -v
```

### All Tests (excluding Postgres)

```bash
uv run pytest tests/ -v
```

### Postgres Integration Tests

```bash
# Start Postgres
docker run -d \
  --name amfs-pg \
  -e POSTGRES_DB=amfs_test \
  -e POSTGRES_PASSWORD=test \
  -p 5432:5432 \
  postgres:16

# Run tests
AMFS_TEST_PG_DSN=postgresql://postgres:test@localhost/amfs_test \
  uv run pytest tests/integration/test_postgres_adapter.py -v
```

### TypeScript Tests

```bash
cd packages/sdk-typescript
npm test
```

---

## Code Quality

### Linting

```bash
uv run ruff check .
uv run ruff format --check .
```

### Type Checking

```bash
uv run mypy packages/
```

### Configuration

The project uses strict settings for code quality:

| Tool | Config |
|:-----|:-------|
| **ruff** | Python 3.11, 100 char line length, E/F/I/N/W/UP rules |
| **mypy** | Strict mode |
| **pytest** | Auto asyncio mode |

---

## Adapter Contract Tests

If you're building a new adapter, it must pass the shared contract test suite:

```python
from tests.integration.adapter_contract import AdapterContractMixin

class TestMyAdapter(AdapterContractMixin):
    def create_adapter(self):
        return MyAdapter(...)
```

The contract tests verify that all adapters behave identically for read, write, list, watch, commit_outcome, search, and stats operations.

---

## CI

GitHub Actions runs on every push and pull request:

- Python 3.11 and 3.12
- Unit tests + filesystem integration tests
- Postgres integration tests (with a Postgres service container)

See `.github/workflows/test-python.yml` for the full pipeline.

---

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Ensure tests pass: `uv run pytest tests/ -v`
5. Ensure code quality: `uv run ruff check . && uv run mypy packages/`
6. Submit a pull request

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](https://github.com/raia-live/amfs/blob/main/LICENSE).
