/**
 * AgentMemory — the main SDK entry point for TypeScript agents.
 */

import type { AmfsAdapter, WatchHandle } from "./adapter.js";
import { InMemoryAdapter } from "./adapters/filesystem.js";
import { defaultConfig } from "./config.js";
import { CausalTagger, CoWEngine } from "./engine.js";
import type { AMFSConfig, MemoryEntry } from "./models.js";
import { OutcomeType } from "./models.js";
import { OutcomeBackPropagator } from "./outcome.js";
import { ReadTracker } from "./tracker.js";

export interface AgentMemoryOptions {
  sessionId?: string;
  adapter?: AmfsAdapter;
  config?: AMFSConfig;
}

export interface SearchOptions {
  entityPath?: string;
  minConfidence?: number;
  agentId?: string;
  limit?: number;
  sortBy?: "confidence" | "recency" | "version";
}

export interface MemoryStats {
  totalEntries: number;
  totalEntities: number;
  totalAgents: number;
  agents: Record<string, number>;
  entities: Record<string, number>;
  confidenceAvg: number;
  outcomeLinkedCount: number;
}

export class AgentMemory {
  readonly agentId: string;
  readonly sessionId: string;
  readonly namespace: string;
  readonly adapter: AmfsAdapter;
  readonly readTracker: ReadTracker;

  private engine: CoWEngine;
  private propagator: OutcomeBackPropagator;

  constructor(agentId: string, options?: AgentMemoryOptions) {
    const config = options?.config ?? defaultConfig();
    this.agentId = agentId;
    this.adapter = options?.adapter ?? new InMemoryAdapter();
    this.namespace = config.namespace;
    this.readTracker = new ReadTracker();

    const tagger = new CausalTagger(agentId, options?.sessionId);
    this.sessionId = tagger.sessionId;
    this.engine = new CoWEngine(this.adapter, tagger, this.readTracker);
    this.propagator = new OutcomeBackPropagator(this.adapter);
  }

  read(
    entityPath: string,
    key: string,
    options?: { minConfidence?: number }
  ): MemoryEntry | null {
    return this.engine.read(entityPath, key, options);
  }

  write(
    entityPath: string,
    key: string,
    value: unknown,
    options?: {
      confidence?: number;
      ttlAt?: string | null;
      patternRefs?: string[];
    }
  ): MemoryEntry {
    return this.engine.write(entityPath, key, value, options);
  }

  list(
    entityPath?: string,
    options?: { includeSuperseded?: boolean }
  ): MemoryEntry[] {
    return this.engine.list(entityPath, options);
  }

  watch(
    entityPath: string,
    callback: (entry: MemoryEntry) => void
  ): WatchHandle {
    return this.adapter.watch(entityPath, callback);
  }

  /** Return the full version history of a key, ordered by version. */
  history(
    entityPath: string,
    key: string,
    options?: { since?: string; until?: string }
  ): MemoryEntry[] {
    return this.engine.history(entityPath, key, options);
  }

  /** Search entries with filters. */
  search(options?: SearchOptions): MemoryEntry[] {
    let entries = this.engine.list(options?.entityPath);

    if (options?.minConfidence) {
      entries = entries.filter((e) => e.confidence >= options.minConfidence!);
    }
    if (options?.agentId) {
      entries = entries.filter(
        (e) => e.provenance.agentId === options.agentId
      );
    }

    const sortBy = options?.sortBy ?? "confidence";
    if (sortBy === "confidence") {
      entries.sort((a, b) => b.confidence - a.confidence);
    } else if (sortBy === "recency") {
      entries.sort((a, b) =>
        b.provenance.writtenAt.localeCompare(a.provenance.writtenAt)
      );
    } else if (sortBy === "version") {
      entries.sort((a, b) => b.version - a.version);
    }

    return entries.slice(0, options?.limit ?? 100);
  }

  /** Aggregate statistics about current memory state. */
  stats(): MemoryStats {
    const entries = this.engine.list();
    const agents: Record<string, number> = {};
    const entities: Record<string, number> = {};
    let totalConfidence = 0;
    let outcomeLinked = 0;

    for (const e of entries) {
      agents[e.provenance.agentId] = (agents[e.provenance.agentId] ?? 0) + 1;
      entities[e.entityPath] = (entities[e.entityPath] ?? 0) + 1;
      totalConfidence += e.confidence;
      if (e.outcomeCount > 0) outcomeLinked++;
    }

    return {
      totalEntries: entries.length,
      totalEntities: Object.keys(entities).length,
      totalAgents: Object.keys(agents).length,
      agents,
      entities,
      confidenceAvg: entries.length > 0 ? totalConfidence / entries.length : 0,
      outcomeLinkedCount: outcomeLinked,
    };
  }

  /**
   * Record external context in the causal chain without writing to storage.
   * Call after consulting external tools/APIs so explain() is complete.
   */
  recordContext(
    label: string,
    summary: string,
    options?: { source?: string }
  ): void {
    this.readTracker.recordContext(label, summary, options);
  }

  /**
   * Return the causal chain for the current session.
   * Shows which memories were read, external contexts, and their order.
   */
  explain(outcomeRef?: string): Record<string, unknown> {
    const causalKeys = this.readTracker.causalKeys;
    const entries: Record<string, unknown>[] = [];

    for (const ek of causalKeys) {
      const lastSlash = ek.lastIndexOf("/");
      if (lastSlash === -1) continue;
      const ep = ek.substring(0, lastSlash);
      const k = ek.substring(lastSlash + 1);
      const entry = this.adapter.read(ep, k);
      if (entry) {
        entries.push({
          ...entry,
          readVersion: this.readTracker.readVersion(ek),
        });
      }
    }

    return {
      outcomeRef: outcomeRef ?? null,
      agentId: this.agentId,
      sessionId: this.sessionId,
      causalChainLength: causalKeys.length,
      causalEntries: entries,
      externalContexts: this.readTracker.externalContexts,
    };
  }

  /**
   * Commit an outcome and back-propagate confidence changes.
   * If causalEntryKeys is omitted, uses the session's read log automatically.
   */
  commitOutcome(
    outcomeRef: string,
    outcomeType: OutcomeType,
    causalEntryKeys?: string[],
    options?: { causalConfidence?: number }
  ): MemoryEntry[] {
    const keys = causalEntryKeys ?? this.readTracker.causalKeys;
    const record = OutcomeBackPropagator.makeRecord(
      outcomeRef,
      outcomeType,
      keys,
      this.agentId,
      { causalConfidence: options?.causalConfidence }
    );
    return this.propagator.propagate(record);
  }

  /** Reset the session read log. */
  clearReadLog(): void {
    this.readTracker.clear();
  }
}
