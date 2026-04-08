/**
 * HttpAdapter — connects the TS SDK to a remote AMFS HTTP API server.
 */

import type { MemoryEntry, OutcomeRecord } from "../models.js";
import type { AmfsAdapter, WatchHandle } from "../adapter.js";
import { createWatchHandle } from "../adapter.js";

export interface HttpAdapterOptions {
  url: string;
  apiKey?: string;
  headers?: Record<string, string>;
}

export class HttpAdapter implements AmfsAdapter {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;

  constructor(opts: HttpAdapterOptions) {
    this.baseUrl = opts.url.replace(/\/+$/, "");
    this.headers = {
      "Content-Type": "application/json",
      ...opts.headers,
    };
    if (opts.apiKey) {
      this.headers["X-AMFS-API-Key"] = opts.apiKey;
    }
  }

  private async fetch<T>(
    path: string,
    init?: RequestInit
  ): Promise<T> {
    const resp = await globalThis.fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { ...this.headers, ...(init?.headers as Record<string, string>) },
    });
    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      throw new Error(`AMFS API ${resp.status}: ${body}`);
    }
    return resp.json() as Promise<T>;
  }

  read(
    entityPath: string,
    key: string,
    options?: { minConfidence?: number }
  ): MemoryEntry | null {
    // Synchronous interface — return null and let callers use readAsync
    return null;
  }

  async readAsync(
    entityPath: string,
    key: string
  ): Promise<MemoryEntry | null> {
    try {
      return await this.fetch<MemoryEntry>(
        `/api/v1/entries/${encodeURIComponent(entityPath)}/${encodeURIComponent(key)}`
      );
    } catch {
      return null;
    }
  }

  write(entry: MemoryEntry): MemoryEntry {
    // Synchronous interface — callers should use writeAsync for HTTP
    return entry;
  }

  async writeAsync(entry: {
    entityPath: string;
    key: string;
    value: unknown;
    confidence?: number;
    memoryType?: string;
  }): Promise<MemoryEntry> {
    return this.fetch<MemoryEntry>("/api/v1/entries", {
      method: "POST",
      body: JSON.stringify({
        entity_path: entry.entityPath,
        key: entry.key,
        value: entry.value,
        confidence: entry.confidence ?? 1.0,
        memory_type: entry.memoryType ?? "fact",
      }),
    });
  }

  list(
    entityPath?: string,
    options?: { includeSuperseded?: boolean }
  ): MemoryEntry[] {
    // Synchronous interface — return empty, callers should use listAsync
    return [];
  }

  async listAsync(
    entityPath?: string,
    options?: { includeSuperseded?: boolean }
  ): Promise<MemoryEntry[]> {
    const params = new URLSearchParams();
    if (entityPath) params.set("entity_path", entityPath);
    if (options?.includeSuperseded) params.set("include_superseded", "true");
    const qs = params.toString();
    return this.fetch<MemoryEntry[]>(`/api/v1/entries${qs ? `?${qs}` : ""}`);
  }

  async searchAsync(query: {
    query?: string;
    entityPath?: string;
    minConfidence?: number;
    limit?: number;
  }): Promise<MemoryEntry[]> {
    return this.fetch<MemoryEntry[]>("/api/v1/search", {
      method: "POST",
      body: JSON.stringify({
        query: query.query,
        entity_path: query.entityPath,
        min_confidence: query.minConfidence ?? 0.0,
        limit: query.limit ?? 100,
      }),
    });
  }

  watch(
    entityPath: string,
    callback: (entry: MemoryEntry) => void
  ): WatchHandle {
    const controller = new AbortController();

    const connect = async () => {
      try {
        const resp = await globalThis.fetch(
          `${this.baseUrl}/api/v1/stream?entity_path=${encodeURIComponent(entityPath)}`,
          { headers: this.headers, signal: controller.signal }
        );

        if (!resp.ok || !resp.body) return;

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          while (buffer.includes("\n\n")) {
            const [msg, rest] = buffer.split("\n\n", 2);
            buffer = rest;
            for (const line of msg.split("\n")) {
              if (line.startsWith("data:")) {
                const raw = line.slice(5).trim();
                if (raw) {
                  try {
                    callback(JSON.parse(raw) as MemoryEntry);
                  } catch { /* skip malformed */ }
                }
              }
            }
          }
        }
      } catch {
        // aborted or network error
      }
    };

    connect();

    return createWatchHandle(() => controller.abort());
  }

  commitOutcome(record: OutcomeRecord): MemoryEntry[] {
    // Synchronous interface — return empty, callers should use commitOutcomeAsync
    return [];
  }

  async commitOutcomeAsync(record: {
    outcomeRef: string;
    outcomeType: string;
  }): Promise<unknown> {
    return this.fetch("/api/v1/outcomes", {
      method: "POST",
      body: JSON.stringify({
        outcome_ref: record.outcomeRef,
        outcome_type: record.outcomeType,
      }),
    });
  }

  async statsAsync(): Promise<Record<string, unknown>> {
    return this.fetch<Record<string, unknown>>("/api/v1/stats");
  }

  async healthAsync(): Promise<Record<string, unknown>> {
    return this.fetch<Record<string, unknown>>("/api/v1/health");
  }
}
