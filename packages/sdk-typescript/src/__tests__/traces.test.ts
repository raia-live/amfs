import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentMemory, HttpAdapter, toDecisionTrace, toDecisionTracePage } from "../index.js";

// Wire shape of a sealed trace, as GET /api/v1/traces/{id} returns it.
const RAW_TRACE = {
  id: "trace-1",
  agent_id: "deploy-agent",
  session_id: "sess-1",
  outcome_ref: "deploy-v41",
  outcome_type: "success",
  decision_summary: "Rolled back checkout",
  task_input: "checkout is returning 500s, roll back",
  response_text: "Rolled checkout back to v41.",
  tool_calls: [
    {
      tool_name: "deploy_rollback",
      arguments: { service: "checkout", to_version: "v41", dry_run: false },
      result_summary: "rolled back",
      result_hash: "abc",
      started_at: "2026-09-05T00:00:00Z",
      duration_ms: 1200,
      source: "deploy-cli",
      success: true,
    },
  ],
  session_metadata: {
    model: "claude-4-opus",
    client_name: "cursor",
    platform: "darwin",
    tools_available: ["amfs_write", "deploy_rollback"],
    mcp_client_id: "c1",
    mcp_session_id: null,
  },
  causal_entries: [
    { entity_path: "shop/checkout", key: "runbook", version: 3, confidence: 0.9 },
  ],
  external_contexts: [{ label: "pagerduty", summary: "SEV-1 open", source: "PagerDuty" }],
  query_events: [],
  error_events: [],
  state_diff: { entries_created: 1, entries_updated: 0 },
  session_duration_ms: 4200,
  created_at: "2026-09-05T00:01:00Z",
};

function mockFetch(responses: Array<{ status?: number; body: unknown }>) {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    const next = responses.shift() ?? { body: {} };
    const status = next.status ?? 200;
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => next.body,
      text: async () => JSON.stringify(next.body),
    } as unknown as Response;
  });
  vi.stubGlobal("fetch", fn);
  return { calls, fn };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("toDecisionTrace", () => {
  it("maps task_input, response_text, tool_calls and session_metadata", () => {
    const trace = toDecisionTrace(RAW_TRACE);
    expect(trace.id).toBe("trace-1");
    expect(trace.agentId).toBe("deploy-agent");
    expect(trace.taskInput).toBe("checkout is returning 500s, roll back");
    expect(trace.responseText).toBe("Rolled checkout back to v41.");
    expect(trace.toolCalls).toHaveLength(1);
    expect(trace.toolCalls[0]).toEqual({
      toolName: "deploy_rollback",
      // caller-owned keys survive untouched
      arguments: { service: "checkout", to_version: "v41", dry_run: false },
      resultSummary: "rolled back",
      resultHash: "abc",
      startedAt: "2026-09-05T00:00:00Z",
      durationMs: 1200,
      source: "deploy-cli",
      success: true,
    });
    expect(trace.sessionMetadata).toEqual({
      model: "claude-4-opus",
      clientName: "cursor",
      platform: "darwin",
      toolsAvailable: ["amfs_write", "deploy_rollback"],
      mcpClientId: "c1",
      mcpSessionId: null,
      attributes: {},
      llmCalls: [],
    });
    // The rest still goes through the generic converter.
    expect(trace.causalEntries[0].entityPath).toBe("shop/checkout");
    expect(trace.stateDiff).toEqual({ entriesCreated: 1, entriesUpdated: 0 });
    expect(trace.createdAt).toBe("2026-09-05T00:01:00Z");
  });

  it("fills safe defaults for traces from servers that predate the fields", () => {
    const { task_input, response_text, tool_calls, session_metadata, ...legacy } = RAW_TRACE;
    void task_input;
    void response_text;
    void tool_calls;
    void session_metadata;
    const trace = toDecisionTrace(legacy);
    expect(trace.taskInput).toBeNull();
    expect(trace.responseText).toBeNull();
    expect(trace.toolCalls).toEqual([]);
    expect(trace.sessionMetadata).toBeNull();
  });

  it("tolerates partial tool calls", () => {
    const trace = toDecisionTrace({ ...RAW_TRACE, tool_calls: [{ tool_name: "x", success: false }] });
    expect(trace.toolCalls[0]).toMatchObject({
      toolName: "x",
      arguments: {},
      durationMs: 0,
      success: false,
      source: null,
    });
  });
});

describe("toDecisionTracePage", () => {
  it("reads the paged shape", () => {
    const page = toDecisionTracePage({
      traces: [RAW_TRACE],
      next_cursor: "cur-2",
      has_more: true,
    });
    expect(page.traces).toHaveLength(1);
    expect(page.traces[0].agentId).toBe("deploy-agent");
    expect(page.nextCursor).toBe("cur-2");
    expect(page.hasMore).toBe(true);
  });

  it("accepts the legacy bare-array shape", () => {
    const page = toDecisionTracePage([RAW_TRACE, RAW_TRACE]);
    expect(page.traces).toHaveLength(2);
    expect(page.nextCursor).toBeNull();
    expect(page.hasMore).toBe(false);
  });

  it("keeps every row the server returned — the window is the server's job", () => {
    const older = { ...RAW_TRACE, id: "old", created_at: "2026-08-01T00:00:00Z" };
    const newer = { ...RAW_TRACE, id: "new", created_at: "2026-09-05T00:01:00Z" };
    const body = { traces: [newer, older], next_cursor: null, has_more: false };
    expect(toDecisionTracePage(body).traces.map((t) => t.id)).toEqual(["new", "old"]);
  });
});

describe("HttpAdapter traces", () => {
  it("sends since/until as query parameters instead of filtering the page", async () => {
    // A window that excludes the newest traces must reach the server: filtering
    // the returned page would empty it and stop a cursor walk before the older
    // matches were ever read.
    const older = { ...RAW_TRACE, id: "old", created_at: "2026-08-01T00:00:00Z" };
    const { calls } = mockFetch([{ body: { traces: [older], next_cursor: null, has_more: false } }]);
    const adapter = new HttpAdapter({ url: "https://api.test" });
    const page = await adapter.listTracesPageAsync({
      since: "2026-07-01T00:00:00Z",
      until: new Date("2026-09-01T00:00:00Z"),
    });
    expect(page.traces.map((t) => t.id)).toEqual(["old"]);
    expect(Object.fromEntries(new URL(calls[0].url).searchParams)).toEqual({
      since: "2026-07-01T00:00:00Z",
      until: "2026-09-01T00:00:00.000Z",
    });
  });

  it("listTracesPageAsync sends filters + cursor and returns page metadata", async () => {
    const { calls } = mockFetch([
      { body: { traces: [RAW_TRACE], next_cursor: "cur-2", has_more: true } },
    ]);
    const adapter = new HttpAdapter({ url: "https://api.test/", apiKey: "k" });
    const page = await adapter.listTracesPageAsync({
      agentId: "deploy-agent",
      entityPath: "shop/checkout",
      outcomeType: "success",
      limit: 25,
      cursor: "cur-1",
      offset: 99, // ignored when a cursor is given
    });
    expect(page).toMatchObject({ nextCursor: "cur-2", hasMore: true });
    expect(page.traces[0].id).toBe("trace-1");

    const url = new URL(calls[0].url);
    expect(url.pathname).toBe("/api/v1/traces");
    expect(Object.fromEntries(url.searchParams)).toEqual({
      entity_path: "shop/checkout",
      agent_id: "deploy-agent",
      outcome_type: "success",
      limit: "25",
      cursor: "cur-1",
    });
    expect((calls[0].init?.headers as Record<string, string>)["X-AMFS-API-Key"]).toBe("k");
  });

  it("uses offset only without a cursor", async () => {
    const { calls } = mockFetch([{ body: { traces: [], next_cursor: null, has_more: false } }]);
    const adapter = new HttpAdapter({ url: "https://api.test" });
    await adapter.listTracesPageAsync({ offset: 10, limit: 5 });
    expect(Object.fromEntries(new URL(calls[0].url).searchParams)).toEqual({
      limit: "5",
      offset: "10",
    });
  });

  it("listTracesAsync keeps returning a flat array and follows a cursor when given one", async () => {
    const { calls } = mockFetch([
      { body: { traces: [RAW_TRACE], next_cursor: "cur-2", has_more: true } },
      { body: { traces: [{ ...RAW_TRACE, id: "trace-2" }], next_cursor: null, has_more: false } },
    ]);
    const adapter = new HttpAdapter({ url: "https://api.test" });
    const first = await adapter.listTracesAsync({ agentId: "deploy-agent" });
    expect(Array.isArray(first)).toBe(true);
    expect(first.map((t) => t.id)).toEqual(["trace-1"]);

    const second = await adapter.listTracesAsync({ agentId: "deploy-agent", cursor: "cur-2" });
    expect(second.map((t) => t.id)).toEqual(["trace-2"]);
    expect(new URL(calls[1].url).searchParams.get("cursor")).toBe("cur-2");
  });

  it("walks every page via nextCursor / hasMore", async () => {
    mockFetch([
      { body: { traces: [{ ...RAW_TRACE, id: "a" }], next_cursor: "c1", has_more: true } },
      { body: { traces: [{ ...RAW_TRACE, id: "b" }], next_cursor: "c2", has_more: true } },
      { body: { traces: [{ ...RAW_TRACE, id: "c" }], next_cursor: null, has_more: false } },
    ]);
    const adapter = new HttpAdapter({ url: "https://api.test" });
    const seen: string[] = [];
    let cursor: string | undefined;
    for (;;) {
      const page = await adapter.listTracesPageAsync({ limit: 1, cursor });
      seen.push(...page.traces.map((t) => t.id));
      if (!page.hasMore || !page.nextCursor) break;
      cursor = page.nextCursor;
    }
    expect(seen).toEqual(["a", "b", "c"]);
  });

  it("getTraceAsync returns the mapped trace and null on 404", async () => {
    mockFetch([{ body: RAW_TRACE }, { status: 404, body: { detail: "not found" } }]);
    const adapter = new HttpAdapter({ url: "https://api.test" });
    const trace = await adapter.getTraceAsync("trace-1");
    expect(trace?.taskInput).toBe("checkout is returning 500s, roll back");
    expect(trace?.toolCalls[0].arguments).toEqual({
      service: "checkout",
      to_version: "v41",
      dry_run: false,
    });
    expect(trace?.sessionMetadata?.model).toBe("claude-4-opus");
    expect(await adapter.getTraceAsync("missing")).toBeNull();
  });
});

describe("AgentMemory trace facade", () => {
  it("delegates listTracesAsync / listTracesPageAsync / getTraceAsync to the HTTP adapter", async () => {
    mockFetch([
      { body: { traces: [RAW_TRACE], next_cursor: "cur-2", has_more: true } },
      { body: { traces: [RAW_TRACE], next_cursor: "cur-2", has_more: true } },
      { body: RAW_TRACE },
    ]);
    const memory = new AgentMemory("deploy-agent", {
      adapter: new HttpAdapter({ url: "https://api.test", apiKey: "k" }),
    });
    const flat = await memory.listTracesAsync({ since: "2026-09-01T00:00:00Z", limit: 10 });
    expect(flat).toHaveLength(1);
    const page = await memory.listTracesPageAsync({ cursor: "cur-1" });
    expect(page.nextCursor).toBe("cur-2");
    expect(page.hasMore).toBe(true);
    const trace = await memory.getTraceAsync("trace-1");
    expect(trace?.responseText).toBe("Rolled checkout back to v41.");
  });

  it("refuses without an HTTP adapter", async () => {
    const memory = new AgentMemory("local");
    await expect(memory.listTracesPageAsync()).rejects.toThrow(/requires a remote HTTP adapter/);
  });
});
