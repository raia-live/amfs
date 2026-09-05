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
    const raw = await this.fetchRaw(path, init);
    return convertKeys(raw) as T;
  }

  /** Like `fetch` but returns the server's JSON untouched (snake_case keys). */
  private async fetchRaw(path: string, init?: RequestInit): Promise<unknown> {
    const resp = await globalThis.fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { ...this.headers, ...(init?.headers as Record<string, string>) },
    });
    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      throw new Error(`AMFS API ${resp.status}: ${body}`);
    }
    return resp.json();
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

  /**
   * One page of decision traces, newest first, exactly as the server pages them.
   *
   * Mirrors GET /api/v1/traces: pass `cursor` (the previous page's `nextCursor`)
   * to continue; `offset` is honoured only when no cursor is given. `since`
   * (inclusive) and `until` (exclusive) are sent to the server and bound the
   * query itself, so a window that excludes the newest traces still returns
   * the older matches rather than an empty first page.
   */
  async listTracesPageAsync(options?: ListTracesOptions): Promise<DecisionTracePage> {
    const params = new URLSearchParams();
    if (options?.entityPath) params.set("entity_path", options.entityPath);
    if (options?.agentId) params.set("agent_id", options.agentId);
    if (options?.outcomeType) params.set("outcome_type", options.outcomeType);
    if (options?.limit) params.set("limit", String(options.limit));
    if (options?.cursor) params.set("cursor", options.cursor);
    else if (options?.offset) params.set("offset", String(options.offset));
    const since = toIsoString(options?.since);
    const until = toIsoString(options?.until);
    if (since) params.set("since", since);
    if (until) params.set("until", until);
    const qs = params.toString();
    const raw = await this.fetchRaw(`/api/v1/traces${qs ? `?${qs}` : ""}`);
    return toDecisionTracePage(raw);
  }

  /**
   * Decision traces as a flat array. Accepts the same paging options as
   * {@link listTracesPageAsync}; use that method when you need `nextCursor` /
   * `hasMore` to walk beyond one page.
   */
  async listTracesAsync(options?: ListTracesOptions): Promise<DecisionTraceSummary[]> {
    const page = await this.listTracesPageAsync(options);
    return page.traces;
  }

  async getTraceAsync(traceId: string): Promise<DecisionTrace | null> {
    try {
      const raw = await this.fetchRaw(`/api/v1/traces/${encodeURIComponent(traceId)}`);
      return toDecisionTrace(raw);
    } catch {
      return null;
    }
  }

  /** Server-side semantic retrieval (embedding + similarity + blend happen on the server). */
  async retrieveAsync(
    query: string,
    options?: {
      entityPath?: string;
      minConfidence?: number;
      limit?: number;
      semanticWeight?: number;
      recencyWeight?: number;
      confidenceWeight?: number;
      includeArtifacts?: boolean;
      branch?: string;
    }
  ): Promise<Array<{ entry: MemoryEntry; score: number }>> {
    const data = await this.fetch<{ entries?: unknown[] } | unknown[]>("/api/v1/retrieve", {
      method: "POST",
      body: JSON.stringify({
        query,
        min_confidence: options?.minConfidence ?? 0.0,
        limit: options?.limit ?? 10,
        semantic_weight: options?.semanticWeight ?? 0.5,
        recency_weight: options?.recencyWeight ?? 0.3,
        confidence_weight: options?.confidenceWeight ?? 0.2,
        include_artifacts: options?.includeArtifacts ?? true,
        branch: options?.branch ?? "main",
        ...(options?.entityPath ? { entity_path: options.entityPath } : {}),
      }),
    });
    const rows = Array.isArray(data) ? data : (data.entries ?? []);
    return rows.map((row) => {
      const r = row as Record<string, unknown>;
      // The server tags each row with a sidecar `_score`, but `fetch` has
      // already run convertKeys (snake→camel), which rewrites a leading
      // underscore key into a capitalized one — so `_score` arrives as `Score`.
      // Read that first, falling back to the raw forms for older servers.
      const rawScore = r.Score ?? r._score ?? r.score;
      const score = typeof rawScore === "number" ? rawScore : 0;
      return { entry: r as unknown as MemoryEntry, score };
    });
  }

  /** Full version history of a key, ordered by version. */
  async historyAsync(
    entityPath: string,
    key: string,
    options?: { since?: string; until?: string }
  ): Promise<MemoryEntry[]> {
    const params = new URLSearchParams();
    if (options?.since) params.set("since", options.since);
    if (options?.until) params.set("until", options.until);
    const qs = params.toString();
    const data = await this.fetch<{ versions?: MemoryEntry[] } | MemoryEntry[]>(
      `/api/v1/history/${encodeURIComponent(entityPath)}/${encodeURIComponent(key)}${qs ? `?${qs}` : ""}`
    );
    if (Array.isArray(data)) return data;
    return data.versions ?? [];
  }

  /** Recent timeline events for an agent (writes, outcomes, cross-agent reads). */
  async timelineAsync(
    agentId: string,
    options?: { eventType?: string; since?: string; limit?: number; branch?: string }
  ): Promise<unknown[]> {
    const params = new URLSearchParams();
    if (options?.eventType) params.set("event_type", options.eventType);
    if (options?.since) params.set("since", options.since);
    if (options?.branch) params.set("branch", options.branch);
    params.set("limit", String(options?.limit ?? 100));
    const data = await this.fetch<{ events?: unknown[] } | unknown[]>(
      `/api/v1/agents/${encodeURIComponent(agentId)}/timeline?${params.toString()}`
    );
    if (Array.isArray(data)) return data;
    return data.events ?? [];
  }

  /** Related entities in the knowledge graph (Pro). */
  async graphNeighborsAsync(options: {
    entity: string;
    relation?: string;
    direction?: string;
    minConfidence?: number;
    depth?: number;
    limit?: number;
  }): Promise<unknown[]> {
    const params = new URLSearchParams();
    params.set("entity", options.entity);
    if (options.relation) params.set("relation", options.relation);
    if (options.direction) params.set("direction", options.direction);
    if (options.minConfidence != null) params.set("min_confidence", String(options.minConfidence));
    if (options.depth != null) params.set("depth", String(options.depth));
    if (options.limit != null) params.set("limit", String(options.limit));
    const data = await this.fetch<{ edges?: unknown[] } | unknown[]>(
      `/api/v1/pro/graph/neighbors?${params.toString()}`
    );
    if (Array.isArray(data)) return data;
    return data.edges ?? [];
  }

  /** Export outcome-validated knowledge as a fine-tuning dataset (Pro). */
  async exportTrainingDataAsync(options?: {
    entityPath?: string;
    minConfidence?: number;
    format?: "sft" | "dpo" | "reward_model";
    limit?: number;
  }): Promise<unknown> {
    const params = new URLSearchParams();
    params.set("format", options?.format ?? "sft");
    params.set("min_confidence", String(options?.minConfidence ?? 0.7));
    params.set("limit", String(options?.limit ?? 100));
    if (options?.entityPath) params.set("entity_path", options.entityPath);
    return this.fetch(`/api/v1/pro/export?${params.toString()}`);
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

/** Filters and paging for GET /api/v1/traces. */
export interface ListTracesOptions {
  entityPath?: string;
  agentId?: string;
  outcomeType?: string;
  limit?: number;
  /** Rows to skip; honoured only when no `cursor` is given. */
  offset?: number;
  /** Opaque keyset position — the `nextCursor` of the previous page. */
  cursor?: string;
  /** Only traces created at or after this instant (ISO-8601 string or Date). */
  since?: string | Date;
  /** Only traces created before this instant (ISO-8601 string or Date). */
  until?: string | Date;
}

/** One page of GET /api/v1/traces. Follow `nextCursor` while `hasMore` is true. */
export interface DecisionTracePage {
  traces: DecisionTraceSummary[];
  nextCursor: string | null;
  hasMore: boolean;
}

/**
 * An action the agent recorded during the session (`amfs_record_action`).
 * `arguments` is passed through exactly as stored — its keys are the caller's,
 * not the API's, so they are not converted to camelCase.
 */
export interface ToolCall {
  toolName: string;
  arguments: Record<string, unknown>;
  resultSummary: string;
  resultHash: string;
  startedAt?: string;
  durationMs: number;
  source?: string | null;
  success: boolean;
}

/** Which model, client and toolset produced the session's decisions. */
export interface SessionMetadata {
  model?: string | null;
  clientName?: string | null;
  platform?: string | null;
  toolsAvailable: string[];
  mcpClientId?: string | null;
  mcpSessionId?: string | null;
}

export interface DecisionTrace {
  id: string;
  agentId: string;
  sessionId: string;
  outcomeRef: string;
  outcomeType: string;
  decisionSummary?: string;
  /** The request that started the work, as it arrived (may be redacted). */
  taskInput?: string | null;
  /** The agent's answer, when it was captured. */
  responseText?: string | null;
  /** Actions the agent recorded; empty for read/write-only sessions. */
  toolCalls: ToolCall[];
  sessionMetadata?: SessionMetadata | null;
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

// ---------------------------------------------------------------------------
// Trace mappers (snake_case wire → camelCase SDK types)
// ---------------------------------------------------------------------------

type Raw = Record<string, unknown>;

function asRecord(value: unknown): Raw {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Raw)
    : {};
}

function toToolCall(raw: unknown): ToolCall {
  const r = asRecord(raw);
  return {
    toolName: String(r.tool_name ?? ""),
    // Caller-owned keys: never camelCased.
    arguments: asRecord(r.arguments),
    resultSummary: String(r.result_summary ?? ""),
    resultHash: String(r.result_hash ?? ""),
    startedAt: typeof r.started_at === "string" ? r.started_at : undefined,
    durationMs: typeof r.duration_ms === "number" ? r.duration_ms : 0,
    source: (r.source as string | null | undefined) ?? null,
    success: r.success !== false,
  };
}

function toSessionMetadata(raw: unknown): SessionMetadata | null {
  if (raw === null || raw === undefined) return null;
  const r = asRecord(raw);
  return {
    model: (r.model as string | null | undefined) ?? null,
    clientName: (r.client_name as string | null | undefined) ?? null,
    platform: (r.platform as string | null | undefined) ?? null,
    toolsAvailable: Array.isArray(r.tools_available) ? r.tools_available.map(String) : [],
    mcpClientId: (r.mcp_client_id as string | null | undefined) ?? null,
    mcpSessionId: (r.mcp_session_id as string | null | undefined) ?? null,
  };
}

/**
 * Map a raw (snake_case) trace from the server to a `DecisionTrace`.
 *
 * The bulk of the object goes through the generic key converter, as before;
 * the fields the converter would get wrong are set explicitly afterwards:
 * `toolCalls[].arguments` must keep the caller's own keys, and
 * `taskInput` / `responseText` / `sessionMetadata` must exist (null, not
 * undefined) so consumers can distinguish "not captured" from "old server".
 */
export function toDecisionTrace(raw: unknown): DecisionTrace {
  const r = asRecord(raw);
  const converted = convertKeys(r) as Raw;
  const toolCalls = Array.isArray(r.tool_calls) ? r.tool_calls.map(toToolCall) : [];
  return {
    ...(converted as unknown as DecisionTrace),
    taskInput: (r.task_input as string | null | undefined) ?? null,
    responseText: (r.response_text as string | null | undefined) ?? null,
    toolCalls,
    sessionMetadata: toSessionMetadata(r.session_metadata),
  };
}

/** ISO-8601 for a query parameter; `undefined` when absent or unparseable. */
function toIsoString(value: string | Date | undefined): string | undefined {
  if (value === undefined) return undefined;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? undefined : value.toISOString();
  return Number.isNaN(Date.parse(value)) ? undefined : value;
}

/** Map a raw GET /api/v1/traces body (new page shape or legacy array) to a page. */
export function toDecisionTracePage(raw: unknown): DecisionTracePage {
  const rows: unknown[] = Array.isArray(raw)
    ? raw
    : Array.isArray(asRecord(raw).traces)
      ? (asRecord(raw).traces as unknown[])
      : [];
  const body = Array.isArray(raw) ? {} : asRecord(raw);
  const traces = rows.map((t) => toDecisionTrace(t) as unknown as DecisionTraceSummary);

  return {
    traces,
    nextCursor: typeof body.next_cursor === "string" ? body.next_cursor : null,
    hasMore: body.has_more === true,
  };
}
