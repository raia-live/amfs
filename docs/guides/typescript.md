---
title: TypeScript SDK
layout: default
parent: Guides
nav_order: 2
description: "Using AMFS from TypeScript and Node.js applications."
---

# TypeScript SDK
{: .no_toc }

The TypeScript SDK (`@amfs/sdk`) provides full parity with the Python SDK for Node.js and TypeScript applications — including ReadTracker, auto-causal linking, search, history, stats, and explainability.

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

// Read — automatically tracked by ReadTracker
const entry = mem.read("checkout-service", "retry-pattern");
console.log(entry?.value);    // { pattern: "exponential-backoff", ... }
console.log(entry?.version);  // 1

// List all entries for an entity
const entries = mem.list("checkout-service");

// Search with filters
const results = mem.search({
  entityPath: "checkout-service",
  minConfidence: 0.7,
  sortBy: "confidence",
  limit: 10,
});

// Auto-causal outcome — uses ReadTracker (no explicit keys needed)
const updated = mem.commitOutcome("DEP-500", OutcomeType.CLEAN_DEPLOY);
```

---

## ReadTracker & Auto-Causal Linking

The TypeScript SDK includes a `ReadTracker` that automatically logs every `read()` call and external context. When you call `commitOutcome()` without explicit causal keys, the SDK uses the read log to build the causal chain — identical to the Python SDK behavior.

```typescript
// These reads are automatically tracked
mem.read("checkout-service", "retry-pattern");
mem.read("checkout-service", "pool-config");

// Record external context
mem.recordContext("pagerduty-status", "No active incidents", "PagerDuty API");

// Commit outcome — auto-links to everything read in this session
const updated = mem.commitOutcome("DEP-500", OutcomeType.CLEAN_DEPLOY);

// Explain the causal chain
const chain = mem.explain();
// → causalEntries: [retry-pattern, pool-config]
// → externalContexts: [{ label: "pagerduty-status", ... }]

// Clear the read log between tasks
mem.clearReadLog();
```

---

## Version History

```typescript
const versions = mem.history("checkout-service", "retry-pattern");
// Returns all versions, newest first

// Filter by time range
const recent = mem.history("checkout-service", "retry-pattern", {
  since: new Date("2026-03-01"),
  until: new Date("2026-04-01"),
});
```

---

## Search

```typescript
import { SearchOptions } from "@amfs/sdk";

const results = mem.search({
  entityPath: "checkout-service",  // filter by entity
  agentId: "review-agent",         // filter by author
  minConfidence: 0.5,              // minimum confidence threshold
  sortBy: "confidence",            // "confidence" or "recency"
  limit: 20,                       // max results
});
```

---

## Stats

```typescript
const stats = mem.stats();
console.log(stats.totalEntries);
console.log(stats.totalEntities);
console.log(stats.avgConfidence);
```

---

## Full API Reference

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

### search

```typescript
mem.search(options?: SearchOptions): MemoryEntry[];
```

### history

```typescript
mem.history(entityPath: string, key: string, options?: {
  since?: Date;
  until?: Date;
}): MemoryEntry[];
```

### stats

```typescript
mem.stats(): MemoryStats;
```

### commitOutcome

```typescript
// Auto-causal (uses ReadTracker)
mem.commitOutcome(outcomeRef: string, outcomeType: OutcomeType): MemoryEntry[];

// Explicit causal keys
mem.commitOutcome(
  outcomeRef: string,
  outcomeType: OutcomeType,
  causalEntryKeys: string[],
): MemoryEntry[];
```

### recordContext

```typescript
mem.recordContext(label: string, summary: string, source?: string): void;
```

### explain

```typescript
mem.explain(): { causalEntries: object[]; externalContexts: object[] };
```

### clearReadLog

```typescript
mem.clearReadLog(): void;
```

---

## Exports

```typescript
import {
  AgentMemory,
  CoWEngine,
  ReadTracker,
  ExternalContext,
  MemoryEntry,
  OutcomeType,
  MemoryType,
  SearchOptions,
  MemoryStats,
  ArtifactRef,
} from "@amfs/sdk";
```

---

## Notes

{: .note }
The TypeScript SDK uses an in-memory adapter by default. For persistent storage, use the Python SDK, HTTP API, or MCP server with a Postgres or S3 backend.
