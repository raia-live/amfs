/**
 * HttpAdapter — connects the TS SDK to a remote AMFS HTTP API server.
 */

import type { MemoryEntry, OutcomeRecord } from "../models.js";
import type { AmfsAdapter, WatchHandle } from "../adapter.js";
import { createWatchHandle } from "../adapter.js";

export interface HttpAdapterOptions {
  url: string;
  apiKey?: string;
  agentId?: string;
  headers?: Record<string, string>;
}

export class HttpAdapter implements AmfsAdapter {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly agentId: string | undefined;

  constructor(opts: HttpAdapterOptions) {
    this.baseUrl = opts.url.replace(/\/+$/, "");
    this.agentId = opts.agentId;
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
    agentId?: string;
    shared?: boolean;
    patternRefs?: string[];
    branch?: string;
  }): Promise<MemoryEntry> {
    const agentId = entry.agentId ?? this.agentId;
    return this.fetch<MemoryEntry>("/api/v1/entries", {
      method: "POST",
      body: JSON.stringify({
        entity_path: entry.entityPath,
        key: entry.key,
        value: entry.value,
        confidence: entry.confidence ?? 1.0,
        memory_type: entry.memoryType ?? "fact",
        ...(agentId ? { agent_id: agentId } : {}),
        ...(entry.shared != null ? { shared: entry.shared } : {}),
        ...(entry.patternRefs?.length ? { pattern_refs: entry.patternRefs } : {}),
        ...(entry.branch ? { branch: entry.branch } : {}),
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
    agentId?: string;
    sortBy?: string;
    limit?: number;
  }): Promise<MemoryEntry[]> {
    return this.fetch<MemoryEntry[]>("/api/v1/search", {
      method: "POST",
      body: JSON.stringify({
        query: query.query,
        entity_path: query.entityPath,
        min_confidence: query.minConfidence ?? 0.0,
        limit: query.limit ?? 100,
        ...(query.agentId ? { agent_id: query.agentId } : {}),
        ...(query.sortBy ? { sort_by: query.sortBy } : {}),
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
    entityPath?: string;
    causalEntryKeys?: string[];
  }): Promise<unknown> {
    return this.fetch("/api/v1/outcomes", {
      method: "POST",
      body: JSON.stringify({
        outcome_ref: record.outcomeRef,
        outcome_type: record.outcomeType,
        ...(record.entityPath ? { entity_path: record.entityPath } : {}),
        ...(record.causalEntryKeys?.length ? { causal_entry_keys: record.causalEntryKeys } : {}),
      }),
    });
  }

  async statsAsync(): Promise<Record<string, unknown>> {
    return this.fetch<Record<string, unknown>>("/api/v1/stats");
  }

  async healthAsync(): Promise<Record<string, unknown>> {
    return this.fetch<Record<string, unknown>>("/api/v1/health");
  }

  async listTracesAsync(options?: {
    entityPath?: string;
    agentId?: string;
    outcomeType?: string;
    limit?: number;
  }): Promise<DecisionTraceSummary[]> {
    const params = new URLSearchParams();
    if (options?.entityPath) params.set("entity_path", options.entityPath);
    if (options?.agentId) params.set("agent_id", options.agentId);
    if (options?.outcomeType) params.set("outcome_type", options.outcomeType);
    if (options?.limit) params.set("limit", String(options.limit));
    const qs = params.toString();
    const data = await this.fetch<{ traces: DecisionTraceSummary[] }>(
      `/api/v1/traces${qs ? `?${qs}` : ""}`
    );
    return data.traces ?? [];
  }

  async getTraceAsync(traceId: string): Promise<DecisionTrace | null> {
    try {
      return await this.fetch<DecisionTrace>(`/api/v1/traces/${encodeURIComponent(traceId)}`);
    } catch {
      return null;
    }
  }
}

export interface DecisionTraceSummary {
  id: string;
  agent_id: string;
  outcome_ref: string;
  outcome_type: string;
  decision_summary?: string;
  causal_entries: number;
  external_contexts: number;
  session_duration_ms?: number;
  created_at: string;
}

export interface DecisionTrace {
  id: string;
  agent_id: string;
  session_id: string;
  outcome_ref: string;
  outcome_type: string;
  decision_summary?: string;
  causal_entries: Array<{
    entity_path: string;
    key: string;
    version: number;
    confidence: number;
    value?: unknown;
    memory_type?: string;
    written_by?: string;
    read_at?: string;
  }>;
  external_contexts: Array<{
    label: string;
    summary: string;
    source?: string;
    recorded_at?: string;
  }>;
  query_events: Array<{
    operation: string;
    parameters: Record<string, unknown>;
    result_count: number;
    duration_ms?: number;
    occurred_at?: string;
  }>;
  error_events: Array<{
    operation: string;
    error_type: string;
    message: string;
    stack_trace?: string;
    occurred_at?: string;
  }>;
  state_diff?: {
    entries_created: number;
    entries_updated: number;
  };
  session_started_at?: string;
  session_ended_at?: string;
  session_duration_ms?: number;
  created_at: string;
}
