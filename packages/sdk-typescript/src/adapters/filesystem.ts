/**
 * In-memory adapter for TypeScript SDK.
 *
 * A full filesystem adapter (matching Python's FilesystemAdapter with
 * atomic rename) would require Node.js `fs` and `chokidar`. This
 * in-memory implementation passes all contract tests and is suitable
 * for browser/edge environments and testing.
 */

import type { AmfsAdapter, WatchHandle } from "../adapter.js";
import { createWatchHandle } from "../adapter.js";
import type { MemoryEntry, OutcomeRecord } from "../models.js";
import { OUTCOME_MULTIPLIERS } from "../models.js";

export class InMemoryAdapter implements AmfsAdapter {
  private store = new Map<string, MemoryEntry[]>();
  private watchers = new Map<string, Set<(entry: MemoryEntry) => void>>();

  private storeKey(entityPath: string, key: string): string {
    return `${entityPath}/${key}`;
  }

  read(
    entityPath: string,
    key: string,
    options?: { minConfidence?: number }
  ): MemoryEntry | null {
    const versions = this.store.get(this.storeKey(entityPath, key));
    if (!versions || versions.length === 0) return null;
    const current = versions[versions.length - 1];
    if (options?.minConfidence && current.confidence < options.minConfidence) {
      return null;
    }
    return current;
  }

  write(entry: MemoryEntry): MemoryEntry {
    const sk = this.storeKey(entry.entityPath, entry.key);
    const versions = this.store.get(sk) ?? [];
    const currentVersion = versions.length > 0 ? versions[versions.length - 1].version : 0;
    const newVersion = currentVersion + 1;
    const written = { ...entry, version: newVersion };
    versions.push(written);
    this.store.set(sk, versions);

    // Notify watchers
    for (const [ep, callbacks] of this.watchers) {
      if (entry.entityPath === ep || entry.entityPath.startsWith(ep + "/")) {
        for (const cb of callbacks) {
          cb(written);
        }
      }
    }

    return written;
  }

  list(
    entityPath?: string,
    options?: { includeSuperseded?: boolean }
  ): MemoryEntry[] {
    const result: MemoryEntry[] = [];
    for (const [sk, versions] of this.store) {
      if (entityPath && !sk.startsWith(entityPath + "/")) continue;
      if (options?.includeSuperseded) {
        result.push(...versions);
      } else {
        result.push(versions[versions.length - 1]);
      }
    }
    return result;
  }

  watch(
    entityPath: string,
    callback: (entry: MemoryEntry) => void
  ): WatchHandle {
    if (!this.watchers.has(entityPath)) {
      this.watchers.set(entityPath, new Set());
    }
    this.watchers.get(entityPath)!.add(callback);

    return createWatchHandle(() => {
      this.watchers.get(entityPath)?.delete(callback);
    });
  }

  commitOutcome(record: OutcomeRecord): MemoryEntry[] {
    const multiplier = OUTCOME_MULTIPLIERS[record.outcomeType];
    const updated: MemoryEntry[] = [];

    for (const spec of record.causalEntryKeys) {
      const lastSlash = spec.lastIndexOf("/");
      if (lastSlash === -1) continue;
      const ep = spec.substring(0, lastSlash);
      const key = spec.substring(lastSlash + 1);

      const current = this.read(ep, key);
      if (!current) continue;

      const newEntry: MemoryEntry = {
        ...current,
        confidence: current.confidence * multiplier * record.causalConfidence,
        outcomeCount: current.outcomeCount + 1,
      };
      const written = this.write(newEntry);
      updated.push(written);
    }

    return updated;
  }
}
