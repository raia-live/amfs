---
title: TypeScript SDK
layout: default
parent: Guides
nav_order: 2
description: "Using AMFS from TypeScript and Node.js applications."
---

# TypeScript SDK
{: .no_toc }

The TypeScript SDK provides the same conceptual API as the Python SDK for Node.js and TypeScript applications.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

```bash
npm install @amfs/sdk
```

---

## Quick Start

```typescript
import { AgentMemory, OutcomeType } from "@amfs/sdk";

const mem = new AgentMemory("review-agent");

// Write
mem.write("checkout-service", "retry-pattern", {
  pattern: "exponential-backoff",
  maxRetries: 3,
});

// Read
const entry = mem.read("checkout-service", "retry-pattern");
console.log(entry?.value);    // { pattern: "exponential-backoff", ... }
console.log(entry?.version);  // 1

// List
const entries = mem.list("checkout-service");

// Outcome (requires explicit causal keys)
const updated = mem.commitOutcome(
  "INC-1042",
  OutcomeType.P1_INCIDENT,
  ["checkout-service/retry-pattern"],
);
```

---

## Key Differences from Python SDK

| Feature | Python | TypeScript |
|:--------|:-------|:-----------|
| Default adapter | Filesystem (`.amfs/`) | In-memory |
| Auto-causal linking | Yes (via ReadTracker) | No — must pass `causalEntryKeys` explicitly |
| Config file discovery | Yes (`amfs.yaml`) | Not yet implemented |
| Async API | Sync (with async adapters) | Sync |

{: .note }
The TypeScript SDK uses an in-memory adapter by default. For persistent storage, use the Python SDK or MCP server.

---

## API Reference

### Constructor

```typescript
const mem = new AgentMemory(agentId: string);
```

### write

```typescript
mem.write(entityPath: string, key: string, value: any, options?: {
  confidence?: number;
  patternRefs?: string[];
}): MemoryEntry;
```

### read

```typescript
mem.read(entityPath: string, key: string): MemoryEntry | undefined;
```

### list

```typescript
mem.list(entityPath?: string): MemoryEntry[];
```

### commitOutcome

```typescript
mem.commitOutcome(
  outcomeRef: string,
  outcomeType: OutcomeType,
  causalEntryKeys: string[],
): MemoryEntry[];
```
