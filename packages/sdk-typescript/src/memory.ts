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

export interface AgentMemoryOptions {
  sessionId?: string;
  adapter?: AmfsAdapter;
  config?: AMFSConfig;
}

export class AgentMemory {
  readonly agentId: string;
  readonly sessionId: string;
  readonly namespace: string;
  readonly adapter: AmfsAdapter;

  private engine: CoWEngine;
  private propagator: OutcomeBackPropagator;

  constructor(agentId: string, options?: AgentMemoryOptions) {
    const config = options?.config ?? defaultConfig();
    this.agentId = agentId;
    this.adapter = options?.adapter ?? new InMemoryAdapter();
    this.namespace = config.namespace;

    const tagger = new CausalTagger(agentId, options?.sessionId);
    this.sessionId = tagger.sessionId;
    this.engine = new CoWEngine(this.adapter, tagger);
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

  commitOutcome(
    outcomeRef: string,
    outcomeType: OutcomeType,
    causalEntryKeys: string[],
    options?: { causalConfidence?: number }
  ): MemoryEntry[] {
    const record = OutcomeBackPropagator.makeRecord(
      outcomeRef,
      outcomeType,
      causalEntryKeys,
      this.agentId,
      { causalConfidence: options?.causalConfidence }
    );
    return this.propagator.propagate(record);
  }
}
