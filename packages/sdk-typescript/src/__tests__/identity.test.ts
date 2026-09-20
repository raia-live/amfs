/**
 * Who is calling, on every request the HTTP adapter makes.
 *
 * A read carries no agent or session in its query or body, so a hosted server
 * learns who asked only from headers — and needs a per-session key to keep the
 * whole session on one arm of a live repair canary. `AgentMemory` binds both
 * when it is built; an adapter used directly sends neither.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { AGENT_ID_HEADER, AgentMemory, HttpAdapter, SESSION_HEADER } from "../index.js";

const calls: Array<{ url: string; headers: Record<string, string> }> = [];

function mockFetch() {
  calls.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, headers: (init?.headers ?? {}) as Record<string, string> });
      return {
        ok: true,
        status: 200,
        json: async () => ({ status: "not_found" }),
        text: async () => "{}",
      } as unknown as Response;
    })
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("identity headers", () => {
  it("an unbound adapter sends none", async () => {
    mockFetch();
    await new HttpAdapter({ url: "http://s" }).readAsync("acme/x", "k");
    expect(calls[0].headers[AGENT_ID_HEADER]).toBeUndefined();
    expect(calls[0].headers[SESSION_HEADER]).toBeUndefined();
  });

  it("a bound adapter sends both on every request, and keeps the base headers", async () => {
    mockFetch();
    const bound = new HttpAdapter({ url: "http://s", apiKey: "key" }).bind("sre-agent", "sess-1");
    await bound.readAsync("acme/x", "k");
    await bound.listAsync("acme/x");
    expect(calls).toHaveLength(2);
    for (const call of calls) {
      expect(call.headers[AGENT_ID_HEADER]).toBe("sre-agent");
      expect(call.headers[SESSION_HEADER]).toBe("sess-1");
      expect(call.headers["X-AMFS-API-Key"]).toBe("key");
    }
    expect(bound.identityHeaders).toEqual({
      [AGENT_ID_HEADER]: "sre-agent",
      [SESSION_HEADER]: "sess-1",
    });
  });

  it("binding does not touch the adapter it was bound from", async () => {
    mockFetch();
    const base = new HttpAdapter({ url: "http://s" });
    base.bind("one", "s");
    await base.readAsync("acme/x", "k");
    expect(calls[0].headers[AGENT_ID_HEADER]).toBeUndefined();
    expect(base.identityHeaders).toEqual({});
  });

  it("drops unsafe characters rather than refusing the request", () => {
    const bound = new HttpAdapter({ url: "http://s" }).bind("équipe\r\nX-Evil: 1", undefined);
    expect(bound.identityHeaders).toEqual({ [AGENT_ID_HEADER]: "quipeX-Evil: 1" });
  });

  it("AgentMemory binds its agent and session onto an HTTP adapter", async () => {
    mockFetch();
    const mem = new AgentMemory("sre-agent", {
      adapter: new HttpAdapter({ url: "http://s" }),
      sessionId: "sess-42",
    });
    await mem.readAsync("acme/x", "k");
    expect(calls[0].headers[AGENT_ID_HEADER]).toBe("sre-agent");
    expect(calls[0].headers[SESSION_HEADER]).toBe("sess-42");
  });

  it("leaves an adapter without bind alone", () => {
    const mem = new AgentMemory("sre-agent");
    expect(typeof (mem.adapter as { bind?: unknown }).bind).toBe("undefined");
  });
});
