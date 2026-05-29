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

function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
}

function convertKeys(obj: unknown): unknown {
  if (obj === null || obj === undefined) return obj;
  if (Array.isArray(obj)) return obj.map(convertKeys);
  if (typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      result[snakeToCamel(k)] = convertKeys(v);
    }
    return result;
  }
  return obj;
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
    const raw = await resp.json();
    return convertKeys(raw) as T;
  }

  read(
    entityPath: string,
    key: string,
    options?: { minConfidence?: number }
  ): MemoryEntry | null {
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
    const data = await this.fetch<{ entries: MemoryEntry[] } | MemoryEntry[]>(
      `/api/v1/entries${qs ? `?${qs}` : ""}`
    );
    if (Array.isArray(data)) return data;
    return data.entries ?? [];
  }

  async searchAsync(query: {
    query?: string;
    entityPath?: string;
    minConfidence?: number;
    maxConfidence?: number;
    agentId?: string;
    since?: string;
    patternRef?: string;
    sortBy?: string;
    limit?: number;
    depth?: number;
    branch?: string;
  }): Promise<MemoryEntry[]> {
    return this.fetch<MemoryEntry[]>("/api/v1/search", {
      method: "POST",
      body: JSON.stringify({
        query: query.query,
        entity_path: query.entityPath,
        min_confidence: query.minConfidence ?? 0.0,
        limit: query.limit ?? 100,
        ...(query.maxConfidence != null ? { max_confidence: query.maxConfidence } : {}),
        ...(query.agentId ? { agent_id: query.agentId } : {}),
        ...(query.since ? { since: query.since } : {}),
        ...(query.patternRef ? { pattern_ref: query.patternRef } : {}),
        ...(query.sortBy ? { sort_by: query.sortBy } : {}),
        ...(query.depth != null ? { depth: query.depth } : {}),
        ...(query.branch ? { branch: query.branch } : {}),
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
                    callback(convertKeys(JSON.parse(raw)) as MemoryEntry);
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
    return [];
  }

  async commitOutcomeAsync(record: {
    outcomeRef: string;
    outcomeType: string;
    entityPath?: string;
    causalEntryKeys?: string[];
    causalConfidence?: number;
    agentId?: string;
  }): Promise<unknown> {
    const agentId = record.agentId ?? this.agentId;
    return this.fetch("/api/v1/outcomes", {
      method: "POST",
      body: JSON.stringify({
        outcome_ref: record.outcomeRef,
        outcome_type: record.outcomeType,
        ...(record.entityPath ? { entity_path: record.entityPath } : {}),
        ...(record.causalEntryKeys?.length ? { causal_entry_keys: record.causalEntryKeys } : {}),
        ...(record.causalConfidence != null ? { causal_confidence: record.causalConfidence } : {}),
        ...(agentId ? { agent_id: agentId } : {}),
      }),
    });
  }

  async statsAsync(): Promise<Record<string, unknown>> {
    return this.fetch<Record<string, unknown>>("/api/v1/stats");
  }

  async healthAsync(): Promise<Record<string, unknown>> {
    return this.fetch<Record<string, unknown>>("/api/v1/health");
  }

  async briefingAsync(options?: {
    entityPath?: string;
    limit?: number;
  }): Promise<unknown> {
    const params = new URLSearchParams();
    if (options?.entityPath) params.set("entity_path", options.entityPath);
    if (options?.limit) params.set("limit", String(options.limit));
    const qs = params.toString();
    return this.fetch(`/api/v1/briefing${qs ? `?${qs}` : ""}`);
  }

  async consolidationStatusAsync(): Promise<unknown> {
    return this.fetch("/api/v1/cortex/consolidation/status");
  }

  async consolidationProposalsAsync(options?: {
    status?: string;
    limit?: number;
  }): Promise<unknown> {
    const params = new URLSearchParams();
    if (options?.status) params.set("status", options.status);
    if (options?.limit) params.set("limit", String(options.limit));
    const qs = params.toString();
    return this.fetch(`/api/v1/cortex/consolidation/proposals${qs ? `?${qs}` : ""}`);
  }

  async consolidationCandidatesAsync(options?: {
    limit?: number;
  }): Promise<unknown> {
    const params = new URLSearchParams();
    if (options?.limit) params.set("limit", String(options.limit));
    const qs = params.toString();
    return this.fetch(`/api/v1/cortex/consolidation/candidates${qs ? `?${qs}` : ""}`);
  }

  async consolidateAsync(options?: {
    dryRun?: boolean;
  }): Promise<unknown> {
    return this.fetch("/api/v1/cortex/consolidate", {
      method: "POST",
      body: JSON.stringify({ dry_run: options?.dryRun ?? true }),
    });
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
    const data = await this.fetch<{ traces: DecisionTraceSummary[] } | DecisionTraceSummary[]>(
      `/api/v1/traces${qs ? `?${qs}` : ""}`
    );
    if (Array.isArray(data)) return data;
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
  agentId: string;
  outcomeRef: string;
  outcomeType: string;
  decisionSummary?: string;
  causalEntries: number;
  externalContexts: number;
  sessionDurationMs?: number;
  createdAt: string;
}

export interface DecisionTrace {
  id: string;
  agentId: string;
  sessionId: string;
  outcomeRef: string;
  outcomeType: string;
  decisionSummary?: string;
  causalEntries: Array<{
    entityPath: string;
    key: string;
    version: number;
    confidence: number;
    value?: unknown;
    memoryType?: string;
    writtenBy?: string;
    readAt?: string;
  }>;
  externalContexts: Array<{
    label: string;
    summary: string;
    source?: string;
    recordedAt?: string;
  }>;
  queryEvents: Array<{
    operation: string;
    parameters: Record<string, unknown>;
    resultCount: number;
    durationMs?: number;
    occurredAt?: string;
  }>;
  errorEvents: Array<{
    operation: string;
    errorType: string;
    message: string;
    stackTrace?: string;
    occurredAt?: string;
  }>;
  stateDiff?: {
    entriesCreated: number;
    entriesUpdated: number;
  };
  sessionStartedAt?: string;
  sessionEndedAt?: string;
  sessionDurationMs?: number;
  createdAt: string;
}
