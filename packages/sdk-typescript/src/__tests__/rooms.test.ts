/**
 * The room methods reach the rooms API as the caller.
 *
 * They did not. `_roomRequest` built its own `fetch` and took the key from
 * `adapter.apiKey`, a property `HttpAdapter` has never had, so every room call
 * left with no API key and no agent or session header and a hosted server
 * answered 401. Nothing here called them.
 *
 * Each test stubs `fetch` and asserts what went over the wire: the method, the
 * URL, the body, and the headers that make the request the caller's.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { AGENT_ID_HEADER, AgentMemory, HttpAdapter, SESSION_HEADER } from "../index.js";

const ROOM = "11111111-2222-3333-4444-555555555555";
const NEG = "99999999-8888-7777-6666-555555555555";

interface Call {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

const calls: Call[] = [];

function mockFetch(answer: unknown = {}, status = 200) {
  calls.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({
        url,
        method: init?.method ?? "GET",
        headers: (init?.headers ?? {}) as Record<string, string>,
        body: init?.body ? JSON.parse(init.body as string) : undefined,
      });
      return {
        ok: status >= 200 && status < 300,
        status,
        json: async () => answer,
        text: async () => JSON.stringify(answer),
      } as unknown as Response;
    })
  );
}

function memory(agentId = "agent-a"): AgentMemory {
  return new AgentMemory(agentId, {
    sessionId: "sess-1",
    adapter: new HttpAdapter({ url: "http://s/", apiKey: "key-a" }),
  });
}

function only(): Call {
  expect(calls).toHaveLength(1);
  return calls[0];
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("room transport", () => {
  it("carries the API key and the caller's identity", async () => {
    mockFetch();
    await memory().roomInfo(ROOM);
    const call = only();
    expect(call.headers["X-AMFS-API-Key"]).toBe("key-a");
    expect(call.headers[AGENT_ID_HEADER]).toBe("agent-a");
    expect(call.headers[SESSION_HEADER]).toBe("sess-1");
  });

  it("returns the server's JSON untouched", async () => {
    mockFetch({ room_id: ROOM, briefing_delivered: true });
    expect(await memory().roomJoin(ROOM)).toEqual({ room_id: ROOM, briefing_delivered: true });
  });

  it("a refusal carries the status and the server's explanation", async () => {
    mockFetch({ detail: { error: "room_limit_reached", limit: 1 } }, 402);
    await expect(memory().roomCreate("acme/billing")).rejects.toThrow(/402.*room_limit_reached/);
  });

  it("the in-memory adapter says what to do", async () => {
    mockFetch();
    await expect(new AgentMemory("agent-a").myRooms()).rejects.toThrow(/fromEnv/);
    expect(calls).toHaveLength(0);
  });
});

describe("room lifecycle", () => {
  it("roomCreate", async () => {
    mockFetch({ id: ROOM });
    const room = await memory().roomCreate(["acme/billing", "acme/refunds"], {
      name: "Billing",
      description: "Shared rules",
    });
    const call = only();
    expect([call.method, call.url]).toEqual(["POST", "http://s/api/v1/rooms"]);
    expect(call.body).toEqual({
      entity_paths: ["acme/billing", "acme/refunds"],
      display_name: "Billing",
      description: "Shared rules",
      discussions_enabled: true,
      agent_id: "agent-a",
    });
    expect(room).toEqual({ id: ROOM });
  });

  it("roomCreate takes one topic as a string, and discussions can be off", async () => {
    mockFetch();
    await memory().roomCreate("acme/billing", { discussions: false });
    const body = only().body as Record<string, unknown>;
    expect(body.entity_paths).toEqual(["acme/billing"]);
    expect(body.discussions_enabled).toBe(false);
  });

  it("roomCreate without a topic is refused before any request", async () => {
    mockFetch();
    await expect(memory().roomCreate([" ", ""])).rejects.toThrow(/at least one topic/);
    expect(calls).toHaveLength(0);
  });

  it("roomShare", async () => {
    mockFetch({ token: "tok" });
    const link = await memory().roomShare(ROOM, { expiresInDays: 1, maxUses: 2, role: "viewer" });
    const call = only();
    expect([call.method, call.url]).toEqual(["POST", `http://s/api/v1/rooms/${ROOM}/join-links`]);
    expect(call.body).toEqual({
      expires_in_days: 1,
      max_uses: 2,
      role: "viewer",
      clamp_max_uses_to_plan: true,
    });
    expect(link).toEqual({ token: "tok" });
  });

  it("roomShare can ask for the refusal instead of the clamp", async () => {
    mockFetch();
    await memory().roomShare(ROOM, { clampToPlan: false });
    expect(only().body).toEqual({
      expires_in_days: null,
      max_uses: null,
      role: "collaborator",
      clamp_max_uses_to_plan: false,
    });
  });

  it.each([
    "tok_abc123",
    "https://amfs.sense-lab.ai/join/tok_abc123",
    "<https://amfs.sense-lab.ai/join/tok_abc123?utm_source=email>.",
    "  https://amfs.sense-lab.ai/j/tok_abc123/  ",
    // Wrapping in any order: the bracket behind the full stop is the ordinary
    // way a link ends a sentence in an email.
    "<https://amfs.sense-lab.ai/join/tok_abc123>.",
    "<https://amfs.sense-lab.ai/join/tok_abc123>,",
    "(https://amfs.sense-lab.ai/join/tok_abc123).",
    "[the room](https://amfs.sense-lab.ai/join/tok_abc123);",
    '"https://amfs.sense-lab.ai/join/tok_abc123".',
    "<tok_abc123>.",
    "https://amfs.sense-lab.ai/join/tok_abc123 .",
    // What copying out of an email or a document adds: non-breaking and other
    // Unicode spaces, zero-width characters, curly quotes.
    "\u00a0https://amfs.sense-lab.ai/join/tok_abc123\u00a0",
    "tok_abc123\u00a0",
    "\u2003<https://amfs.sense-lab.ai/join/tok_abc123>\u00a0.\u00a0",
    "\ufeffhttps://amfs.sense-lab.ai/join/tok_abc123\u200b",
    "\u201chttps://amfs.sense-lab.ai/join/tok_abc123\u201d.",
    "\u2018tok_abc123\u2019",
  ])("roomRedeem takes the token or the whole link: %s", async (pasted) => {
    mockFetch({ room_id: ROOM });
    expect(await memory().roomRedeem(pasted)).toEqual({ room_id: ROOM });
    const call = only();
    expect([call.method, call.url]).toEqual(["POST", "http://s/api/v1/room-join/tok_abc123"]);
  });

  it.each(["tok_abc-", "tok_abc_", "-tok_abc", "_tok-abc_"])(
    "roomRedeem keeps a token that starts or ends in a dash: %s",
    async (token) => {
      mockFetch({ room_id: ROOM });
      await memory().roomRedeem(`<https://amfs.sense-lab.ai/join/${token}>.`);
      expect(only().url).toBe(`http://s/api/v1/room-join/${token}`);
    },
  );

  it("a long run of brackets or slashes costs linear time, not quadratic", async () => {
    mockFetch({ room_id: ROOM });
    const started = Date.now();
    await memory().roomRedeem(`${"<".repeat(200_000)}x${"<".repeat(200_000)}tok_abc123`).catch(() => {});
    new HttpAdapter({ url: `http://s${"/".repeat(200_000)}x`, apiKey: "key-a" });
    expect(Date.now() - started).toBeLessThan(2_000);
  });

  it("several trailing slashes on the base URL are all dropped", async () => {
    mockFetch({ rooms: [] });
    const mem = new AgentMemory("agent-a", {
      adapter: new HttpAdapter({ url: "http://s///", apiKey: "key-a" }),
    });
    await mem.myRooms();
    expect(only().url).toBe("http://s/api/v1/rooms");
  });

  it("roomRedeem without a token is refused before any request", async () => {
    mockFetch();
    await expect(memory().roomRedeem("  ")).rejects.toThrow(/join token/);
    expect(calls).toHaveLength(0);
  });

  it("roomJoin", async () => {
    mockFetch();
    await memory().roomJoin(ROOM);
    const call = only();
    expect([call.method, call.url]).toEqual(["POST", `http://s/api/v1/rooms/${ROOM}/agents`]);
    expect(call.body).toEqual({ agent_id: "agent-a" });
  });

  it("roomLeave escapes the agent id", async () => {
    mockFetch();
    await memory("agent a?x").roomLeave(ROOM);
    const call = only();
    expect([call.method, call.url]).toEqual([
      "DELETE",
      `http://s/api/v1/rooms/${ROOM}/agents/agent%20a%3Fx`,
    ]);
    expect(call.body).toBeUndefined();
  });

  it("myRooms returns the list, or unwraps an envelope", async () => {
    const rooms = [{ id: ROOM, role: "owner" }];
    mockFetch(rooms);
    expect(await memory().myRooms()).toEqual(rooms);
    expect([only().method, only().url]).toEqual(["GET", "http://s/api/v1/rooms"]);
    mockFetch({ rooms });
    expect(await memory().myRooms()).toEqual(rooms);
  });

  it("roomInfo", async () => {
    mockFetch();
    await memory().roomInfo(ROOM);
    expect([only().method, only().url]).toEqual(["GET", `http://s/api/v1/rooms/${ROOM}`]);
  });

  it("roomBriefing", async () => {
    mockFetch();
    await memory().roomBriefing(ROOM);
    const call = only();
    expect([call.method, call.url]).toEqual(["POST", `http://s/api/v1/rooms/${ROOM}/briefing`]);
    expect(call.body).toEqual({ agent_id: "agent-a" });
  });

  it("roomUpdates without since sends no query", async () => {
    mockFetch();
    await memory().roomUpdates(ROOM);
    expect(only().url).toBe(`http://s/api/v1/rooms/${ROOM}/activity`);
  });

  it("roomUpdates sends since intact", async () => {
    // Pasted into the URL unescaped, the `+` of a UTC offset reads as a space.
    mockFetch();
    const since = "2026-05-01T00:00:00+00:00";
    await memory().roomUpdates(ROOM, { since });
    const url = new URL(only().url);
    expect(url.pathname).toBe(`/api/v1/rooms/${ROOM}/activity`);
    expect(url.searchParams.get("since")).toBe(since);
    expect(url.search).not.toContain("+");
  });

  it("roomClose", async () => {
    mockFetch();
    await memory().roomClose(ROOM);
    expect([only().method, only().url]).toEqual(["POST", `http://s/api/v1/rooms/${ROOM}/close`]);
  });

  it("roomDelete", async () => {
    mockFetch();
    await memory().roomDelete(ROOM);
    const call = only();
    expect([call.method, call.url]).toEqual(["DELETE", `http://s/api/v1/rooms/${ROOM}`]);
    expect(call.body).toBeUndefined();
  });
});

describe("room discussions", () => {
  it("roomDiscuss", async () => {
    mockFetch();
    await memory().roomDiscuss(ROOM, "Is the retry rule still right?", {
      messageType: "question",
      addressedTo: "agent-b",
    });
    const call = only();
    expect([call.method, call.url]).toEqual(["POST", `http://s/api/v1/rooms/${ROOM}/discussions`]);
    expect(call.body).toEqual({
      agent_id: "agent-a",
      content: "Is the retry rule still right?",
      message_type: "question",
      addressed_to: "agent-b",
    });
  });

  it("roomDiscuss can answer a message", async () => {
    mockFetch();
    await memory().roomDiscuss(ROOM, "Yes.", { messageType: "answer", replyTo: NEG });
    expect((only().body as Record<string, unknown>).reply_to).toBe(NEG);
  });

  it("roomDiscussions without options sends no query", async () => {
    mockFetch();
    await memory().roomDiscussions(ROOM);
    expect([only().method, only().url]).toEqual(["GET", `http://s/api/v1/rooms/${ROOM}/discussions`]);
  });

  it("roomDiscussions filters", async () => {
    mockFetch();
    const since = "2026-05-01T00:00:00+00:00";
    await memory().roomDiscussions(ROOM, { limit: 20, since, addressedTo: "agent-a" });
    const url = new URL(only().url);
    expect(Object.fromEntries(url.searchParams)).toEqual({
      limit: "20",
      since,
      addressed_to: "agent-a",
    });
  });
});

describe("room negotiations", () => {
  it("negotiateCreate", async () => {
    mockFetch();
    await memory().negotiateCreate(ROOM, "Retry budget", { description: "Pick one", maxRounds: 4 });
    const call = only();
    expect([call.method, call.url]).toEqual(["POST", `http://s/api/v1/rooms/${ROOM}/negotiations`]);
    expect(call.body).toEqual({
      title: "Retry budget",
      description: "Pick one",
      agent_id: "agent-a",
      max_rounds: 4,
    });
  });

  it("negotiatePropose", async () => {
    mockFetch();
    await memory().negotiatePropose(ROOM, NEG, { proposal: { retries: 3 }, rationale: "p99" });
    const call = only();
    expect([call.method, call.url]).toEqual([
      "POST",
      `http://s/api/v1/rooms/${ROOM}/negotiations/${NEG}/propose`,
    ]);
    expect(call.body).toEqual({
      agent_id: "agent-a",
      action: "propose",
      proposal: { retries: 3 },
      rationale: "p99",
    });
  });

  it("negotiateRespond", async () => {
    mockFetch();
    await memory().negotiateRespond(ROOM, NEG, "accept", { rationale: "agreed" });
    expect(only().body).toEqual({
      agent_id: "agent-a",
      action: "accept",
      proposal: {},
      rationale: "agreed",
    });
  });

  it("negotiateStatus", async () => {
    mockFetch();
    await memory().negotiateStatus(ROOM, NEG);
    expect([only().method, only().url]).toEqual([
      "GET",
      `http://s/api/v1/rooms/${ROOM}/negotiations/${NEG}`,
    ]);
  });
});

describe("AgentMemory.fromEnv", () => {
  it("builds an HttpAdapter from the environment", async () => {
    vi.stubEnv("AMFS_HTTP_URL", "http://s");
    vi.stubEnv("AMFS_API_KEY", "key-env");
    mockFetch([]);
    const mem = AgentMemory.fromEnv("agent-a", { sessionId: "sess-9" });
    expect(mem.adapter).toBeInstanceOf(HttpAdapter);
    expect(mem.sessionId).toBe("sess-9");
    await mem.myRooms();
    const call = only();
    expect(call.url).toBe("http://s/api/v1/rooms");
    expect(call.headers["X-AMFS-API-Key"]).toBe("key-env");
    expect(call.headers[AGENT_ID_HEADER]).toBe("agent-a");
  });

  it("arguments override the environment", async () => {
    vi.stubEnv("AMFS_HTTP_URL", "http://s");
    vi.stubEnv("AMFS_API_KEY", "key-env");
    mockFetch([]);
    await AgentMemory.fromEnv("agent-b", { apiKey: "key-other" }).myRooms();
    expect(only().headers["X-AMFS-API-Key"]).toBe("key-other");
  });

  it.each([
    [{}, ["AMFS_HTTP_URL", "AMFS_API_KEY"]],
    [{ AMFS_HTTP_URL: "http://s" }, ["AMFS_API_KEY"]],
    [{ AMFS_API_KEY: "key-env" }, ["AMFS_HTTP_URL"]],
    [{ AMFS_HTTP_URL: "   ", AMFS_API_KEY: "key-env" }, ["AMFS_HTTP_URL"]],
  ] as Array<[Record<string, string>, string[]]>)(
    "a missing value throws and names it: %j",
    (env, named) => {
      vi.stubEnv("AMFS_HTTP_URL", "");
      vi.stubEnv("AMFS_API_KEY", "");
      for (const [name, value] of Object.entries(env)) vi.stubEnv(name, value);
      let message = "";
      try {
        AgentMemory.fromEnv("agent-a");
      } catch (err) {
        message = (err as Error).message;
      }
      for (const name of named) expect(message).toContain(name);
      for (const name of ["AMFS_HTTP_URL", "AMFS_API_KEY"].filter((n) => !named.includes(n))) {
        expect(message).not.toContain(name);
      }
    }
  );

  it("the constructor still ignores the environment", () => {
    vi.stubEnv("AMFS_HTTP_URL", "http://s");
    vi.stubEnv("AMFS_API_KEY", "key-env");
    expect(new AgentMemory("agent-a").adapter).not.toBeInstanceOf(HttpAdapter);
  });
});
