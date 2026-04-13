/**
 * CoWEngine and CausalTagger — TypeScript port.
 */

import type { AmfsAdapter } from "./adapter.js";
import type { MemoryEntry, Provenance } from "./models.js";
import { ReadTracker } from "./tracker.js";

export class CausalTagger {
  readonly agentId: string;
  readonly sessionId: string;

  constructor(agentId: string, sessionId?: string) {
    this.agentId = agentId;
    this.sessionId =
      sessionId ?? `sess-${Math.random().toString(36).substring(2, 10)}`;
  }

  tag(patternRefs: string[] = []): Provenance {
    return {
      agentId: this.agentId,
      sessionId: this.sessionId,
      writtenAt: new Date().toISOString(),
      patternRefs,
    };
  }
}

export class CoWEngine {
  readonly adapter: AmfsAdapter;
  readonly tagger: CausalTagger;
  readonly readTracker: ReadTracker | null;

  constructor(
    adapter: AmfsAdapter,
    tagger: CausalTagger,
    readTracker?: ReadTracker
  ) {
    this.adapter = adapter;
    this.tagger = tagger;
    this.readTracker = readTracker ?? null;
  }

  read(
    entityPath: string,
    key: string,
    options?: { minConfidence?: number }
  ): MemoryEntry | null {
    const entry = this.adapter.read(entityPath, key, options);
    if (entry && this.readTracker) {
      this.readTracker.record(entry);
    }
    return entry;
  }

  write(
    entityPath: string,
    key: string,
    value: unknown,
    options?: {
      confidence?: number;
      ttlAt?: string | null;
      patternRefs?: string[];
      shared?: boolean;
    }
  ): MemoryEntry {
    const current = this.adapter.read(entityPath, key);
    const nextVersion = current ? current.version + 1 : 1;

    const entry: MemoryEntry = {
      amfsVersion: "0.1.0",
      entityPath,
      key,
      version: nextVersion,
      value,
      provenance: this.tagger.tag(options?.patternRefs),
      confidence: options?.confidence ?? 1.0,
      outcomeCount: current?.outcomeCount ?? 0,
      ttlAt: options?.ttlAt ?? null,
      shared: options?.shared ?? true,
      contentHash: null,
      integrityChain: null,
      commitId: null,
    };

    return this.adapter.write(entry);
  }

  list(
    entityPath?: string,
    options?: { includeSuperseded?: boolean }
  ): MemoryEntry[] {
    return this.adapter.list(entityPath, options);
  }

  history(
    entityPath: string,
    key: string,
    options?: { since?: string; until?: string }
  ): MemoryEntry[] {
    const all = this.adapter.list(entityPath, { includeSuperseded: true });
    let versions = all
      .filter((e) => e.key === key)
      .sort((a, b) => a.version - b.version);

    if (options?.since) {
      versions = versions.filter(
        (e) => e.provenance.writtenAt >= options.since!
      );
    }
    if (options?.until) {
      versions = versions.filter(
        (e) => e.provenance.writtenAt <= options.until!
      );
    }
    return versions;
  }
}
