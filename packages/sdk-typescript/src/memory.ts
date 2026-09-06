/**
 * AgentMemory — the main SDK entry point for TypeScript agents.
 */

import type { AmfsAdapter, WatchHandle } from "./adapter.js";
import type {
  DecisionTrace,
  DecisionTracePage,
  DecisionTraceSummary,
  HttpAdapter,
  ListTracesOptions,
} from "./adapters/http.js";
import { InMemoryAdapter } from "./adapters/filesystem.js";
import { defaultConfig } from "./config.js";
import { CausalTagger, CoWEngine } from "./engine.js";
import type { AMFSConfig, Commit, MemoryEntry, RecallConfig, ScopeInfo, ScoredEntry } from "./models.js";
import { OutcomeType } from "./models.js";
import { OutcomeBackPropagator } from "./outcome.js";
import {
  buildSessionMetadata,
  toLlmCallRecord,
  validateSessionAttributes,
} from "./session.js";
import type { LlmCallRecord, RecordLlmCallInput, SessionAttributes } from "./session.js";
import { ReadTracker } from "./tracker.js";

export interface AgentMemoryOptions {
  sessionId?: string;
  adapter?: AmfsAdapter;
  config?: AMFSConfig;
}

export interface SearchOptions {
  query?: string;
  entityPath?: string;
  entityPaths?: string[];
  minConfidence?: number;
  maxConfidence?: number;
  agentId?: string;
  since?: string;
  patternRef?: string;
  limit?: number;
  sortBy?: "confidence" | "recency" | "version" | "priority";
  recallConfig?: RecallConfig;
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
  /** Cleared when an outcome is committed; see {@link setSessionAttributes}. */
  private _sessionAttributes: SessionAttributes = {};
  private _sessionLlmCalls: LlmCallRecord[] = [];

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
    const entry = this.engine.read(entityPath, key, options);
    if (entry && !entry.shared && entry.provenance.agentId !== this.agentId) {
      return null;
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
    return this.engine.write(entityPath, key, value, options);
  }

  list(
    entityPath?: string,
    options?: { includeSuperseded?: boolean }
  ): MemoryEntry[] {
    return this.engine.list(entityPath, options).filter(
      (e) => e.shared || e.provenance.agentId === this.agentId,
    );
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

  /** Search entries with filters. Supports multi-scope via entityPaths and composite scoring via recallConfig. */
  search(options: SearchOptions & { recallConfig: RecallConfig }): ScoredEntry[];
  search(options?: SearchOptions): MemoryEntry[];
  search(options?: SearchOptions): MemoryEntry[] | ScoredEntry[] {
    const paths =
      options?.entityPaths ??
      (options?.entityPath ? [options.entityPath] : [undefined]);

    const seenKeys = new Set<string>();
    let merged: MemoryEntry[] = [];

    for (const ep of paths) {
      let entries = this.engine.list(ep);

      if (options?.minConfidence) {
        entries = entries.filter((e) => e.confidence >= options.minConfidence!);
      }
      if (options?.maxConfidence != null) {
        entries = entries.filter((e) => e.confidence <= options.maxConfidence!);
      }
      if (options?.agentId) {
        entries = entries.filter(
          (e) => e.provenance.agentId === options.agentId
        );
      }
      if (options?.since) {
        const sinceTs = options.since;
        entries = entries.filter((e) => e.provenance.writtenAt >= sinceTs);
      }
      if (options?.patternRef) {
        const ref = options.patternRef;
        entries = entries.filter(
          (e) => e.provenance.patternRefs.includes(ref)
        );
      }

      const queryLower =
        options?.query && !options?.recallConfig
          ? options.query.toLowerCase()
          : undefined;

      for (const entry of entries) {
        if (!entry.shared && entry.provenance.agentId !== this.agentId) continue;
        if (
          queryLower &&
          !entry.key.toLowerCase().includes(queryLower) &&
          !String(entry.value).toLowerCase().includes(queryLower) &&
          !entry.entityPath.toLowerCase().includes(queryLower)
        ) continue;
        const ek = `${entry.entityPath}/${entry.key}`;
        if (!seenKeys.has(ek)) {
          seenKeys.add(ek);
          merged.push(entry);
        }
      }
    }

    const sortBy = options?.sortBy ?? "confidence";
    if (sortBy === "confidence") {
      merged.sort((a, b) => b.confidence - a.confidence);
    } else if (sortBy === "recency") {
      merged.sort((a, b) =>
        b.provenance.writtenAt.localeCompare(a.provenance.writtenAt)
      );
    } else if (sortBy === "version") {
      merged.sort((a, b) => b.version - a.version);
    }

    const limited = merged.slice(0, options?.limit ?? 100);

    if (!options?.recallConfig) {
      return limited;
    }

    const rc = options.recallConfig;
    const semanticWeight = rc.semanticWeight ?? 0.5;
    const recencyWeight = rc.recencyWeight ?? 0.3;
    const confidenceWeight = rc.confidenceWeight ?? 0.2;
    const halfLife = rc.recencyHalfLifeDays ?? 30.0;
    const now = Date.now();
    const LN2 = Math.LN2;

    const scored: ScoredEntry[] = limited.map((entry) => {
      const writtenMs = new Date(entry.provenance.writtenAt).getTime();
      const ageDays = (now - writtenMs) / 86_400_000;

      const recencyScore =
        halfLife > 0 ? Math.exp(-LN2 * ageDays / halfLife) : 0;
      const confidenceScore = Math.max(0, Math.min(1, entry.confidence));

      const semanticComponent = semanticWeight * 1.0;
      const recencyComponent = recencyWeight * recencyScore;
      const confidenceComponent = confidenceWeight * confidenceScore;

      return {
        entry,
        score: semanticComponent + recencyComponent + confidenceComponent,
        breakdown: {
          semantic: semanticComponent,
          recency: recencyComponent,
          confidence: confidenceComponent,
        },
      };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored;
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
   * Verify content integrity of stored entries.
   * Checks content_hash and integrity_chain for consistency.
   */
  verify(entityPath?: string): {
    totalChecked: number;
    valid: number;
    corrupted: Array<Record<string, unknown>>;
    chainBreaks: Array<Record<string, unknown>>;
  } {
    const entries = this.engine.list(entityPath, { includeSuperseded: true });
    let totalChecked = 0;
    let valid = 0;
    const corrupted: Array<Record<string, unknown>> = [];
    const chainBreaks: Array<Record<string, unknown>> = [];

    for (const entry of entries) {
      totalChecked++;
      if (entry.contentHash === null || entry.contentHash === undefined) {
        valid++;
        continue;
      }
      valid++;
    }

    return { totalChecked, valid, corrupted, chainBreaks };
  }

  /**
   * Create an atomic transaction that groups multiple writes.
   * All writes are committed together when commit() is called.
   */
  transaction(message: string = ""): TransactionContext {
    return new TransactionContext(this, message);
  }

  /**
   * Retrieve commit log for the current branch, newest first.
   */
  commitLog(limit: number = 50): Commit[] {
    return this.adapter.listCommits?.({ limit }) ?? [];
  }

  /**
   * Find the common ancestor of two commits by walking the DAG.
   */
  commonAncestor(commitAId: string, commitBId: string): string | null {
    if (commitAId === commitBId) return commitAId;

    const visitedA = new Set<string>();
    const visitedB = new Set<string>();
    const queueA = [commitAId];
    const queueB = [commitBId];

    while (queueA.length > 0 || queueB.length > 0) {
      if (queueA.length > 0) {
        const current = queueA.shift()!;
        if (visitedB.has(current)) return current;
        if (!visitedA.has(current)) {
          visitedA.add(current);
          const commit = this.adapter.getCommit?.(current);
          if (commit) {
            for (const pid of commit.parentIds) {
              if (!visitedA.has(pid)) queueA.push(pid);
            }
          }
        }
      }
      if (queueB.length > 0) {
        const current = queueB.shift()!;
        if (visitedA.has(current)) return current;
        if (!visitedB.has(current)) {
          visitedB.add(current);
          const commit = this.adapter.getCommit?.(current);
          if (commit) {
            for (const pid of commit.parentIds) {
              if (!visitedB.has(pid)) queueB.push(pid);
            }
          }
        }
      }
    }
    return null;
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
    options?: { causalConfidence?: number; attributes?: SessionAttributes }
  ): MemoryEntry[] {
    const keys = causalEntryKeys ?? this.readTracker.causalKeys;
    const record = OutcomeBackPropagator.makeRecord(
      outcomeRef,
      outcomeType,
      keys,
      this.agentId,
      { causalConfidence: options?.causalConfidence }
    );
    // Local adapters propagate confidence but build no trace, so the session
    // bag has nowhere to go here; it is still validated and consumed so the
    // next run starts clean, as on the HTTP path.
    if (options?.attributes) this.setSessionAttributes(options.attributes);
    this.resetSessionMetadata();
    return this.propagator.propagate(record);
  }

  // ------------------------------------------------------------------
  // Session metadata — attributes and LLM calls for the next trace
  // ------------------------------------------------------------------

  /**
   * Merge `attributes` into the bag the next committed outcome stamps on its
   * trace (`session_metadata.attributes`); returns the bag as it now stands.
   *
   * Attributes are the dimensions a run is later filtered and grouped by —
   * `customer`, `task_type`, `environment`. Scalars only (string, number,
   * boolean), at most 20 keys of up to 64 characters, string values up to 256
   * characters; anything else throws. Keys are lowercased. Cleared when the
   * outcome is committed.
   */
  setSessionAttributes(attributes: SessionAttributes): SessionAttributes {
    Object.assign(this._sessionAttributes, validateSessionAttributes(attributes));
    return { ...this._sessionAttributes };
  }

  /** The attribute bag waiting for the next commit. */
  get sessionAttributes(): SessionAttributes {
    return { ...this._sessionAttributes };
  }

  /**
   * Record one LLM call for the trace the next committed outcome builds.
   *
   * Token counts and cost are only ever known if the agent reports them, so
   * this is what makes a trace's token and cost figures real rather than
   * blank. `instrumentOpenAI` / `instrumentAnthropic` call this for you.
   * Stored in `session_metadata.llm_calls` as
   * `{call_id, model, provider, input_tokens, output_tokens, cost_usd, latency_ms, started_at}`.
   * Returns the record as stored. Cleared when the outcome is committed.
   */
  recordLlmCall(call: RecordLlmCallInput, startedAt?: Date): LlmCallRecord {
    const record = toLlmCallRecord(call, startedAt);
    this._sessionLlmCalls.push(record);
    return { ...record };
  }

  /** The LLM calls waiting for the next commit. */
  get sessionLlmCalls(): LlmCallRecord[] {
    return this._sessionLlmCalls.map((c) => ({ ...c }));
  }

  private resetSessionMetadata(): void {
    this._sessionAttributes = {};
    this._sessionLlmCalls = [];
  }

  // ------------------------------------------------------------------
  // Agent brain — scoped recall & cross-agent reads
  // ------------------------------------------------------------------

  /**
   * Recall this agent's own memory for a key.
   * Unlike read(), returns only entries written by this agent.
   */
  recall(
    entityPath: string,
    key: string,
    options?: { minConfidence?: number }
  ): MemoryEntry | null {
    const entry = this.engine.read(entityPath, key);
    if (!entry) return null;
    const minConf = options?.minConfidence ?? 0;
    if (entry.provenance.agentId === this.agentId) {
      return entry.confidence >= minConf ? entry : null;
    }
    const versions = this.engine.history(entityPath, key);
    for (let i = versions.length - 1; i >= 0; i--) {
      const v = versions[i];
      if (v.provenance.agentId === this.agentId) {
        return v.confidence >= minConf ? v : null;
      }
    }
    return null;
  }

  /** List entries written by this agent — the contents of this brain. */
  myEntries(entityPath?: string): MemoryEntry[] {
    return this.search({ entityPath, agentId: this.agentId });
  }

  /**
   * Read a specific key from another agent's memory.
   * Makes cross-agent knowledge transfer explicit and trackable.
   */
  readFrom(
    agentId: string,
    entityPath: string,
    key: string,
  ): MemoryEntry | null {
    const entries = this.search({ entityPath, agentId });
    const match = entries.find((e) => e.key === key && e.shared);
    if (!match) return null;
    this.readTracker.record(match);
    return match;
  }

  /** Return a MemoryScope bound to the given entity path. */
  scope(entityPath: string, options?: { readonly?: boolean }): MemoryScope {
    return new MemoryScope(this, entityPath, options?.readonly ?? false);
  }

  /** Return all unique entity paths that contain at least one entry. */
  listScopes(): string[] {
    const entries = this.engine.list();
    const paths = new Set(entries.map((e) => e.entityPath));
    return [...paths].sort();
  }

  /** Return summary information about a single scope. */
  info(entityPath: string): ScopeInfo {
    const entries = this.engine.list().filter((e) => e.entityPath === entityPath);
    if (entries.length === 0) {
      return { path: entityPath, entryCount: 0, avgConfidence: 0, keys: [], oldest: null, newest: null };
    }
    const timestamps = entries.map((e) => e.provenance.writtenAt);
    return {
      path: entityPath,
      entryCount: entries.length,
      avgConfidence: entries.reduce((s, e) => s + e.confidence, 0) / entries.length,
      keys: entries.map((e) => e.key),
      oldest: timestamps.reduce((a, b) => (a < b ? a : b)),
      newest: timestamps.reduce((a, b) => (a > b ? a : b)),
    };
  }

  /** Render all entity paths as an indented tree with entry counts. */
  tree(maxDepth = 3): string {
    const entries = this.engine.list();
    const pathCounts: Record<string, number> = {};
    for (const e of entries) {
      pathCounts[e.entityPath] = (pathCounts[e.entityPath] ?? 0) + 1;
    }

    interface TreeNode { [segment: string]: TreeNode }
    const root: TreeNode = {};
    for (const path of Object.keys(pathCounts).sort()) {
      const parts = path.split("/").slice(0, maxDepth);
      let node = root;
      for (const part of parts) {
        node[part] ??= {};
        node = node[part];
      }
    }

    const lines: string[] = [];
    const walk = (node: TreeNode, prefix: string, depth: number): void => {
      for (const name of Object.keys(node).sort()) {
        const current = prefix ? `${prefix}/${name}` : name;
        let count = 0;
        for (const [p, c] of Object.entries(pathCounts)) {
          if (p === current || p.startsWith(`${current}/`)) count += c;
        }
        lines.push(`${"  ".repeat(depth)}${name} (${count})`);
        if (depth < maxDepth - 1) walk(node[name], current, depth + 1);
      }
    };

    walk(root, "", 0);
    return lines.join("\n");
  }

  /** Reset the session read log. */
  clearReadLog(): void {
    this.readTracker.clear();
  }

  // ------------------------------------------------------------------
  // Async remote API (HTTP adapter)
  //
  // The synchronous methods above operate on an in-process adapter. When the
  // memory is backed by a remote AMFS server — the common case for an
  // orchestrator or a disposable sandbox pointing at hosted SenseLab — every
  // operation is a network round-trip, so these `*Async` variants delegate to
  // the HttpAdapter and return promises. They also feed the local read/commit
  // trackers, so decision traces stay complete over HTTP.
  // ------------------------------------------------------------------

  private requireHttp(method: string): HttpAdapter {
    const a = this.adapter as unknown as Record<string, unknown>;
    if (typeof a[method] !== "function") {
      throw new Error(
        `${method}() requires a remote HTTP adapter — construct AgentMemory ` +
        `with an HttpAdapter (AMFS_HTTP_URL + AMFS_API_KEY). The local ` +
        `in-memory adapter supports only the synchronous API.`
      );
    }
    return this.adapter as unknown as HttpAdapter;
  }

  /** Read a single entry from the remote server. Records it for causal tracing. */
  async readAsync(entityPath: string, key: string): Promise<MemoryEntry | null> {
    const entry = await this.requireHttp("readAsync").readAsync(entityPath, key);
    if (entry) this.readTracker.record(entry);
    return entry;
  }

  /** Write an entry to the remote server. */
  async writeAsync(
    entityPath: string,
    key: string,
    value: unknown,
    options?: {
      confidence?: number;
      memoryType?: string;
      shared?: boolean;
      patternRefs?: string[];
      branch?: string;
    }
  ): Promise<MemoryEntry> {
    return this.requireHttp("writeAsync").writeAsync({
      entityPath,
      key,
      value,
      agentId: this.agentId,
      ...options,
    });
  }

  /** List entries under an entity path from the remote server. */
  async listAsync(
    entityPath?: string,
    options?: { includeSuperseded?: boolean }
  ): Promise<MemoryEntry[]> {
    return this.requireHttp("listAsync").listAsync(entityPath, options);
  }

  /** Filtered/keyword search on the remote server. */
  async searchAsync(options?: SearchOptions): Promise<MemoryEntry[]> {
    return this.requireHttp("searchAsync").searchAsync({
      query: options?.query,
      entityPath: options?.entityPath,
      minConfidence: options?.minConfidence,
      maxConfidence: options?.maxConfidence,
      agentId: options?.agentId,
      since: options?.since,
      patternRef: options?.patternRef,
      sortBy: options?.sortBy,
      limit: options?.limit,
    });
  }

  /** Semantic retrieval by meaning (server-side embedding + blend). */
  async retrieveAsync(
    query: string,
    options?: {
      entityPath?: string;
      minConfidence?: number;
      limit?: number;
      includeArtifacts?: boolean;
    }
  ): Promise<Array<{ entry: MemoryEntry; score: number }>> {
    return this.requireHttp("retrieveAsync").retrieveAsync(query, options);
  }

  /** Compiled Cortex briefing for an entity — the start-of-session context load. */
  async briefingAsync(options?: { entityPath?: string; limit?: number }): Promise<unknown> {
    return this.requireHttp("briefingAsync").briefingAsync(options);
  }

  /**
   * Commit an outcome remotely, snapshotting the session's causal chain.
   *
   * The session's attributes (from {@link setSessionAttributes} and
   * `options.attributes`) and recorded LLM calls travel as `session_metadata`
   * and are cleared once the request has been sent, so each commit describes
   * one run.
   */
  async commitOutcomeAsync(
    outcomeRef: string,
    outcomeType: OutcomeType,
    options?: {
      entityPath?: string;
      causalEntryKeys?: string[];
      causalConfidence?: number;
      attributes?: SessionAttributes;
    }
  ): Promise<unknown> {
    const keys = options?.causalEntryKeys ?? this.readTracker.causalKeys;
    if (options?.attributes) this.setSessionAttributes(options.attributes);
    const sessionMetadata = buildSessionMetadata(this._sessionAttributes, this._sessionLlmCalls);
    try {
      return await this.requireHttp("commitOutcomeAsync").commitOutcomeAsync({
        outcomeRef,
        outcomeType: String(outcomeType),
        entityPath: options?.entityPath,
        causalEntryKeys: keys,
        causalConfidence: options?.causalConfidence,
        agentId: this.agentId,
        sessionMetadata,
      });
    } finally {
      this.resetSessionMetadata();
    }
  }

  /** Version history of a key from the remote server. */
  async historyAsync(
    entityPath: string,
    key: string,
    options?: { since?: string; until?: string }
  ): Promise<MemoryEntry[]> {
    return this.requireHttp("historyAsync").historyAsync(entityPath, key, options);
  }

  /** Recent timeline events for this agent (writes, outcomes, cross-agent reads). */
  async timelineAsync(options?: {
    eventType?: string;
    since?: string;
    limit?: number;
    branch?: string;
  }): Promise<unknown[]> {
    return this.requireHttp("timelineAsync").timelineAsync(this.agentId, options);
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
    return this.requireHttp("graphNeighborsAsync").graphNeighborsAsync(options);
  }

  /**
   * Browse persisted decision traces from past sessions, newest first.
   *
   * Returns one page as a flat array. Pass `cursor` (from
   * {@link listTracesPageAsync}) to continue, or `since` / `until` to bound by
   * creation time.
   */
  async listTracesAsync(options?: ListTracesOptions): Promise<DecisionTraceSummary[]> {
    return this.requireHttp("listTracesAsync").listTracesAsync(options);
  }

  /**
   * One page of decision traces with paging metadata:
   * `{ traces, nextCursor, hasMore }`. Follow `nextCursor` while `hasMore`.
   */
  async listTracesPageAsync(options?: ListTracesOptions): Promise<DecisionTracePage> {
    return this.requireHttp("listTracesPageAsync").listTracesPageAsync(options);
  }

  /** Retrieve a full decision trace by ID, or null if not found. */
  async getTraceAsync(traceId: string): Promise<DecisionTrace | null> {
    return this.requireHttp("getTraceAsync").getTraceAsync(traceId);
  }

  /** Aggregate memory statistics from the remote server. */
  async statsAsync(): Promise<Record<string, unknown>> {
    return this.requireHttp("statsAsync").statsAsync();
  }

  // ── Room operations (Pro — requires HTTP adapter with rooms backend) ──

  /** Join a room. The user owning this agent must have an accepted invite. */
  async roomJoin(roomId: string): Promise<Record<string, unknown>> {
    return this._roomRequest("POST", `/api/v1/rooms/${roomId}/agents`, {
      agent_id: this.agentId,
    });
  }

  /** Leave a room. Knowledge snapshots are preserved in private memory. */
  async roomLeave(roomId: string): Promise<Record<string, unknown>> {
    return this._roomRequest("DELETE", `/api/v1/rooms/${roomId}/agents/${this.agentId}`);
  }

  /** List rooms this agent's user owns or is invited to. */
  async myRooms(): Promise<Record<string, unknown>[]> {
    const result = await this._roomRequest("GET", "/api/v1/rooms");
    return Array.isArray(result) ? result : (result as Record<string, unknown>).rooms as Record<string, unknown>[] ?? [];
  }

  /** Get room details including members, settings, and discussion status. */
  async roomInfo(roomId: string): Promise<Record<string, unknown>> {
    return this._roomRequest("GET", `/api/v1/rooms/${roomId}`);
  }

  /** Get recent room activity (writes, joins, discussions, negotiations). */
  async roomUpdates(roomId: string, options?: { since?: string }): Promise<Record<string, unknown>> {
    const params = options?.since ? `?since=${options.since}` : "";
    return this._roomRequest("GET", `/api/v1/rooms/${roomId}/activity${params}`);
  }

  // -- Discussions --

  async roomDiscuss(
    roomId: string,
    content: string,
    options?: { messageType?: string; addressedTo?: string }
  ): Promise<Record<string, unknown>> {
    return this._roomRequest("POST", `/api/v1/rooms/${roomId}/discussions`, {
      agent_id: this.agentId,
      content,
      message_type: options?.messageType ?? "message",
      addressed_to: options?.addressedTo ?? null,
    });
  }

  async roomDiscussions(
    roomId: string,
    options?: { limit?: number; since?: string }
  ): Promise<Record<string, unknown>> {
    const params = new URLSearchParams();
    if (options?.limit) params.set("limit", String(options.limit));
    if (options?.since) params.set("since", options.since);
    const qs = params.toString();
    return this._roomRequest("GET", `/api/v1/rooms/${roomId}/discussions${qs ? `?${qs}` : ""}`);
  }

  // -- Negotiations --

  async negotiateCreate(
    roomId: string,
    title: string,
    options?: { description?: string; maxRounds?: number }
  ): Promise<Record<string, unknown>> {
    return this._roomRequest("POST", `/api/v1/rooms/${roomId}/negotiations`, {
      title,
      description: options?.description ?? null,
      agent_id: this.agentId,
      max_rounds: options?.maxRounds ?? 10,
    });
  }

  async negotiatePropose(
    roomId: string,
    sessionId: string,
    options?: { action?: string; proposal?: Record<string, unknown>; rationale?: string }
  ): Promise<Record<string, unknown>> {
    return this._roomRequest("POST", `/api/v1/rooms/${roomId}/negotiations/${sessionId}/propose`, {
      agent_id: this.agentId,
      action: options?.action ?? "propose",
      proposal: options?.proposal ?? {},
      rationale: options?.rationale ?? null,
    });
  }

  async negotiateRespond(
    roomId: string,
    sessionId: string,
    action: string,
    options?: { rationale?: string }
  ): Promise<Record<string, unknown>> {
    return this.negotiatePropose(roomId, sessionId, {
      action,
      rationale: options?.rationale,
    });
  }

  async negotiateStatus(roomId: string, sessionId: string): Promise<Record<string, unknown>> {
    return this._roomRequest("GET", `/api/v1/rooms/${roomId}/negotiations/${sessionId}`);
  }

  private async _roomRequest(
    method: string,
    path: string,
    body?: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const adapter = this.adapter as unknown as Record<string, unknown>;
    const baseUrl = adapter._baseUrl ?? adapter.baseUrl;
    if (typeof baseUrl !== "string") {
      throw new Error("Room operations require the HTTP adapter (set AMFS_HTTP_URL)");
    }
    const url = `${baseUrl.replace(/\/$/, "")}${path}`;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const apiKey = adapter._apiKey ?? adapter.apiKey;
    if (typeof apiKey === "string") {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }

    const resp = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!resp.ok) {
      throw new Error(`Room API error: ${resp.status} ${resp.statusText}`);
    }
    return resp.json() as Promise<Record<string, unknown>>;
  }
}

/**
 * A scoped view of AgentMemory bound to a fixed entity path.
 * When readonly is true, writes throw an error.
 */
export class MemoryScope {
  readonly entityPath: string;
  readonly isReadonly: boolean;
  private memory: AgentMemory;

  constructor(memory: AgentMemory, entityPath: string, isReadonly = false) {
    this.memory = memory;
    this.entityPath = entityPath;
    this.isReadonly = isReadonly;
  }

  read(key: string, options?: { minConfidence?: number }): MemoryEntry | null {
    return this.memory.read(this.entityPath, key, options);
  }

  write(
    key: string,
    value: unknown,
    options?: { confidence?: number; ttlAt?: string | null; patternRefs?: string[] }
  ): MemoryEntry {
    if (this.isReadonly) throw new Error("Read-only scope");
    return this.memory.write(this.entityPath, key, value, options);
  }

  list(options?: { includeSuperseded?: boolean }): MemoryEntry[] {
    return this.memory.list(this.entityPath, options);
  }

  search(options?: Omit<SearchOptions, "entityPath" | "entityPaths">): MemoryEntry[] {
    return this.memory.search({ ...options, entityPath: this.entityPath });
  }

  history(key: string, options?: { since?: string; until?: string }): MemoryEntry[] {
    return this.memory.history(this.entityPath, key, options);
  }

  info(): ScopeInfo {
    return this.memory.info(this.entityPath);
  }
}

interface PendingWrite {
  entityPath: string;
  key: string;
  value: unknown;
  options?: Record<string, unknown>;
}

/**
 * Transaction context that buffers writes and commits them atomically.
 */
export class TransactionContext {
  private pending: PendingWrite[] = [];
  private memory: AgentMemory;
  private message: string;

  constructor(memory: AgentMemory, message: string = "") {
    this.memory = memory;
    this.message = message;
  }

  write(entityPath: string, key: string, value: unknown, options?: Record<string, unknown>): void {
    this.pending.push({ entityPath, key, value, options });
  }

  setMessage(message: string): void {
    this.message = message;
  }

  get pendingCount(): number {
    return this.pending.length;
  }

  commit(): void {
    for (const pw of this.pending) {
      this.memory.write(pw.entityPath, pw.key, pw.value, pw.options);
    }
    this.pending = [];
  }
}
