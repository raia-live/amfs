/**
 * AmfsAdapter — interface that every TS adapter must implement.
 */

import type { MemoryEntry, OutcomeRecord, Commit } from "./models.js";

export interface WatchHandle {
  cancel(): void;
  readonly cancelled: boolean;
}

export interface AmfsAdapter {
  read(
    entityPath: string,
    key: string,
    options?: { minConfidence?: number }
  ): MemoryEntry | null;

  write(entry: MemoryEntry): MemoryEntry;

  list(
    entityPath?: string,
    options?: { includeSuperseded?: boolean }
  ): MemoryEntry[];

  watch(
    entityPath: string,
    callback: (entry: MemoryEntry) => void
  ): WatchHandle;

  commitOutcome(record: OutcomeRecord): MemoryEntry[];

  listCommits?(options?: { limit?: number }): Commit[];
}

export function createWatchHandle(cancelFn: () => void): WatchHandle {
  let _cancelled = false;
  return {
    cancel() {
      if (!_cancelled) {
        cancelFn();
        _cancelled = true;
      }
    },
    get cancelled() {
      return _cancelled;
    },
  };
}
