/**
 * ReadTracker — automatically records every read within a session for
 * causal linking and conflict detection.
 *
 * TypeScript port of the Python ReadTracker from amfs-core.
 */

import type { MemoryEntry } from "./models.js";

export interface ExternalContext {
  label: string;
  summary: string;
  source?: string | null;
  recordedAt: string; // ISO 8601
}

export class ReadTracker {
  private reads: Map<string, string> = new Map(); // entry_key -> ISO timestamp
  private versions: Map<string, number> = new Map(); // entry_key -> version
  private contexts: ExternalContext[] = [];

  /** Record that an entry was read during this session. */
  record(entry: MemoryEntry): void {
    const entryKey = `${entry.entityPath}/${entry.key}`;
    this.reads.set(entryKey, new Date().toISOString());
    this.versions.set(entryKey, entry.version);
  }

  /**
   * Record external context that influenced decisions in this session.
   * Included in the causal chain returned by explain().
   */
  recordContext(
    label: string,
    summary: string,
    options?: { source?: string }
  ): void {
    this.contexts.push({
      label,
      summary,
      source: options?.source ?? null,
      recordedAt: new Date().toISOString(),
    });
  }

  /** All entry keys read in this session, ordered by read time. */
  get causalKeys(): string[] {
    return [...this.reads.entries()]
      .sort(([, a], [, b]) => a.localeCompare(b))
      .map(([k]) => k);
  }

  /** All external contexts recorded in this session, in order. */
  get externalContexts(): ExternalContext[] {
    return [...this.contexts];
  }

  get readCount(): number {
    return this.reads.size;
  }

  /** Return the version we last read for an entry, or undefined if never read. */
  readVersion(entryKey: string): number | undefined {
    return this.versions.get(entryKey);
  }

  /** Reset the read log (e.g. between sub-tasks within a session). */
  clear(): void {
    this.reads.clear();
    this.versions.clear();
    this.contexts = [];
  }

  contains(entryKey: string): boolean {
    return this.reads.has(entryKey);
  }
}
