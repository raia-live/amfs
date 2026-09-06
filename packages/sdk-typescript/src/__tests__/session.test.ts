import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AgentMemory,
  HttpAdapter,
  OutcomeType,
  toDecisionTrace,
  validateSessionAttributes,
} from "../index.js";

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

function body(call: { init?: RequestInit }): Record<string, unknown> {
  return JSON.parse(String(call.init?.body)) as Record<string, unknown>;
}

afterEach(() => vi.unstubAllGlobals());

describe("validateSessionAttributes", () => {
  it("lowercases and trims keys and keeps scalars", () => {
    expect(validateSessionAttributes({ " Customer ": "acme", N: 3, ok: true, f: 1.5 })).toEqual({
      customer: "acme",
      n: 3,
      ok: true,
      f: 1.5,
    });
    expect(validateSessionAttributes(undefined)).toEqual({});
    expect(validateSessionAttributes(null)).toEqual({});
  });

  it("rejects rather than trims", () => {
    expect(() => validateSessionAttributes([] as unknown)).toThrow(TypeError);
    expect(() => validateSessionAttributes({ nested: { a: 1 } })).toThrow(TypeError);
    expect(() => validateSessionAttributes({ nil: null })).toThrow(TypeError);
    expect(() => validateSessionAttributes({ nan: Number.NaN })).toThrow(RangeError);
    expect(() => validateSessionAttributes({ "": "x" })).toThrow(RangeError);
    expect(() => validateSessionAttributes({ ["k".repeat(65)]: "x" })).toThrow(/64 characters/);
    expect(() => validateSessionAttributes({ v: "x".repeat(257) })).toThrow(/256 characters/);
    const many = Object.fromEntries(Array.from({ length: 21 }, (_, i) => [`k${i}`, i]));
    expect(() => validateSessionAttributes(many)).toThrow(/at most 20/);
  });
});

describe("AgentMemory session metadata", () => {
  it("setSessionAttributes merges and validates", () => {
    const mem = new AgentMemory("deploy-agent");
    expect(mem.setSessionAttributes({ Customer: "acme" })).toEqual({ customer: "acme" });
    expect(mem.setSessionAttributes({ task_type: "deploy" })).toEqual({
      customer: "acme",
      task_type: "deploy",
    });
    expect(() => mem.setSessionAttributes({ bad: [1] as unknown as string })).toThrow(TypeError);
    // A rejected bag leaves the good one untouched, and the getter is a copy.
    expect(mem.sessionAttributes).toEqual({ customer: "acme", task_type: "deploy" });
    mem.sessionAttributes.customer = "mutated";
    expect(mem.sessionAttributes.customer).toBe("acme");
  });

  it("recordLlmCall writes the wire record", () => {
    const mem = new AgentMemory("deploy-agent");
    const before = Date.now();
    const rec = mem.recordLlmCall({
      model: "gpt-4o-mini",
      inputTokens: 100,
      outputTokens: 20.9,
      costUsd: 0.00003,
      latencyMs: 412,
      provider: "openai",
    });
    expect(rec).toMatchObject({
      model: "gpt-4o-mini",
      provider: "openai",
      input_tokens: 100,
      output_tokens: 20,
      cost_usd: 0.00003,
      latency_ms: 412,
    });
    expect(typeof rec.call_id).toBe("string");
    expect(rec.call_id.length).toBeGreaterThan(8);
    expect(Date.parse(rec.started_at)).toBeGreaterThanOrEqual(before - 1);
    expect(Object.keys(rec).sort()).toEqual([
      "call_id", "cost_usd", "input_tokens", "latency_ms", "model", "output_tokens",
      "provider", "started_at",
    ]);

    const minimal = mem.recordLlmCall({ model: "claude-3-5-haiku", inputTokens: 1, outputTokens: 2 });
    expect(minimal.cost_usd).toBeNull(); // unknown, not zero
    expect(minimal.latency_ms).toBeNull();
    expect(minimal.provider).toBe("");
    expect(mem.sessionLlmCalls).toHaveLength(2);
    expect(mem.sessionLlmCalls[0].call_id).toBe(rec.call_id);
  });

  it("recordLlmCall validates", () => {
    const mem = new AgentMemory("deploy-agent");
    expect(() => mem.recordLlmCall({ model: "", inputTokens: 1, outputTokens: 1 })).toThrow(TypeError);
    expect(() =>
      mem.recordLlmCall({ model: "m", inputTokens: -1, outputTokens: 1 }),
    ).toThrow(RangeError);
    expect(() =>
      mem.recordLlmCall({ model: "m", inputTokens: "1" as unknown as number, outputTokens: 1 }),
    ).toThrow(TypeError);
    expect(() =>
      mem.recordLlmCall({ model: "m", inputTokens: 1, outputTokens: 1, costUsd: -0.1 }),
    ).toThrow(RangeError);
    expect(() =>
      mem.recordLlmCall({ model: "m", inputTokens: 1, outputTokens: 1, latencyMs: Number.NaN }),
    ).toThrow(RangeError);
    expect(mem.sessionLlmCalls).toEqual([]);
  });

  it("local commitOutcome accepts attributes and clears the session bag", () => {
    const mem = new AgentMemory("deploy-agent");
    mem.setSessionAttributes({ customer: "acme" });
    mem.recordLlmCall({ model: "m", inputTokens: 1, outputTokens: 1 });
    mem.commitOutcome("deploy-1", OutcomeType.SUCCESS, [], { attributes: { task_type: "deploy" } });
    expect(mem.sessionAttributes).toEqual({});
    expect(mem.sessionLlmCalls).toEqual([]);
  });

  it("commitOutcomeAsync sends attributes and llm_calls as session_metadata", async () => {
    const { calls } = mockFetch([{ body: { ok: true } }, { body: { ok: true } }]);
    const adapter = new HttpAdapter({ url: "http://amfs.test", apiKey: "k", agentId: "deploy-agent" });
    const mem = new AgentMemory("deploy-agent", { adapter });

    mem.setSessionAttributes({ customer: "acme" });
    const call = mem.recordLlmCall({
      model: "gpt-4o-mini", inputTokens: 10, outputTokens: 5, provider: "openai", latencyMs: 12,
    });
    await mem.commitOutcomeAsync("deploy-42", OutcomeType.SUCCESS, {
      attributes: { Task_Type: "deploy" },
    });

    expect(calls[0].url).toBe("http://amfs.test/api/v1/outcomes");
    const sent = body(calls[0]);
    expect(sent.outcome_ref).toBe("deploy-42");
    expect(sent.outcome_type).toBe("success");
    expect(sent.session_metadata).toEqual({
      attributes: { customer: "acme", task_type: "deploy" },
      llm_calls: [call],
    });

    // Consumed by the commit: the next one describes a new run.
    expect(mem.sessionAttributes).toEqual({});
    expect(mem.sessionLlmCalls).toEqual([]);
    await mem.commitOutcomeAsync("deploy-43", OutcomeType.FAILURE);
    expect(body(calls[1])).not.toHaveProperty("session_metadata");
  });

  it("commitOutcomeAsync clears the bag even when the request fails", async () => {
    mockFetch([{ status: 500, body: { detail: "boom" } }]);
    const adapter = new HttpAdapter({ url: "http://amfs.test", apiKey: "k", agentId: "a" });
    const mem = new AgentMemory("a", { adapter });
    mem.setSessionAttributes({ customer: "acme" });
    await expect(mem.commitOutcomeAsync("x", OutcomeType.SUCCESS)).rejects.toThrow();
    expect(mem.sessionAttributes).toEqual({});
  });

  it("commitOutcomeAsync rejects a bad attribute bag before sending anything", async () => {
    const { fn } = mockFetch([]);
    const adapter = new HttpAdapter({ url: "http://amfs.test", apiKey: "k", agentId: "a" });
    const mem = new AgentMemory("a", { adapter });
    await expect(
      mem.commitOutcomeAsync("x", OutcomeType.SUCCESS, { attributes: { nested: {} as never } }),
    ).rejects.toThrow(TypeError);
    expect(fn).not.toHaveBeenCalled();
  });
});

describe("toDecisionTrace session metadata", () => {
  it("surfaces attributes and llm_calls from the wire", () => {
    const trace = toDecisionTrace({
      id: "t", agent_id: "a", session_id: "s", outcome_ref: "o", outcome_type: "success",
      tool_calls: [], causal_entries: [], external_contexts: [], query_events: [], error_events: [],
      created_at: "2026-09-05T00:00:00Z",
      session_metadata: {
        model: "gpt-4o",
        attributes: { customer: "acme", n: 2, skip: { nested: true } },
        llm_calls: [{ call_id: "c1", model: "gpt-4o", input_tokens: 1, output_tokens: 2 }],
      },
    });
    expect(trace.sessionMetadata?.attributes).toEqual({ customer: "acme", n: 2 });
    expect(trace.sessionMetadata?.llmCalls).toEqual([
      { call_id: "c1", model: "gpt-4o", input_tokens: 1, output_tokens: 2 },
    ]);
  });

  it("defaults them when an older server omits them", () => {
    const trace = toDecisionTrace({
      id: "t", agent_id: "a", session_id: "s", outcome_ref: "o", outcome_type: "success",
      created_at: "2026-09-05T00:00:00Z", session_metadata: { model: "gpt-4o" },
    });
    expect(trace.sessionMetadata).toMatchObject({ model: "gpt-4o", attributes: {}, llmCalls: [] });
  });
});
