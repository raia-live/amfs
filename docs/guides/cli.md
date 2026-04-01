---
title: CLI
layout: default
parent: Guides
nav_order: 3
description: "Inspect, diff, and manage AMFS memory from the command line."
---

# CLI
{: .no_toc }

The AMFS CLI provides commands for inspecting, diffing, and snapshotting memory from the terminal.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

```bash
pip install amfs-cli
```

---

## Initialize a Project

```bash
amfs init
```

Creates `amfs.yaml`, `.amfs/`, and updates `.gitignore`. Run this once per project.

---

## Inspect Memory

### List Entries

```bash
# List all entries
amfs inspect list

# List entries for a specific entity
amfs inspect list checkout-service

# Include superseded versions
amfs inspect list checkout-service --superseded
```

### Read an Entry

```bash
amfs inspect read checkout-service retry-pattern
```

### Diff Version History

See how an entry changed across versions:

```bash
amfs inspect diff checkout-service retry-pattern
```

---

## Snapshots

### Export

Export all memory to a JSON file:

```bash
amfs snapshot export backup.json
```

Export a single entity:

```bash
amfs snapshot export backup.json --entity checkout-service
```

Include superseded versions:

```bash
amfs snapshot export backup.json --superseded
```

### Restore

Restore memory from a snapshot:

```bash
amfs snapshot restore backup.json
```

{: .warning }
Restore writes entries into the current adapter. Existing entries with the same keys will get new versions via CoW — nothing is overwritten destructively.

---

## Custom Config Path

All commands accept a `-c` / `--config` flag:

```bash
amfs inspect list -c /path/to/amfs.yaml
```

---

## MCP Server

The MCP server has its own executable:

```bash
# Start with stdio transport (default)
amfs-mcp-server

# Start with HTTP transport
amfs-mcp-server --transport http

# Custom host, port, and path
amfs-mcp-server --transport http --host 127.0.0.1 --port 9000 --path /amfs
```

See the [MCP Setup guide](/amfs/guides/mcp/) for full details.
