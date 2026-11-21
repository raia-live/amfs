import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpAdapter } from "../adapters/http.js";
import { AgentMemory } from "../memory.js";
import type { MemoryEntry } from "../models.js";

describe("HttpAdapter.watch SSE frame parsing and AgentMemory._roomRequest headers", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("delivers all SSE events when multiple events arrive in a single stream chunk", async () => {
    const encoder = new TextEncoder();
    const e1 = {
      entity_path: "svc/auth",
      key: "k1",
      version: 1,
      value: "first",
      provenance: {
        agent_id: "agent-1",
        session_id: "sess-1",
        written_at: "2026-11-14T00:00:00Z",
        pattern_refs: [],
      },
      confidence: 1.0,
      outcome_count: 0,
      shared: true,
    };
    const e2 = { ...e1, key: "k2", value: "second" };
    const e3 = { ...e1, key: "k3", value: "third" };

    const chunk = encoder.encode(
      `data: ${JSON.stringify(e1)}\n\ndata: ${JSON.stringify(e2)}\n\ndata: ${JSON.stringify(e3)}\n\n`
    );

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(chunk);
        controller.close();
      },
    });

    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      body: stream,
    } as unknown as Response);

    const adapter = new HttpAdapter({ url: "http://localhost:8000" });
    const received: MemoryEntry[] = [];

    const handle = adapter.watch("svc/auth", (entry) => {
      received.push(entry);
    });

    await new Promise((resolve) => setTimeout(resolve, 25));
    handle.cancel();

    expect(received.map((e) => e.key)).toEqual(["k1", "k2", "k3"]);
    expect(received.map((e) => e.value)).toEqual(["first", "second", "third"]);
  });

  it("includes HttpAdapter API key and identity headers on room requests", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ rooms: [] }),
    } as unknown as Response);

    const http = new HttpAdapter({
      url: "http://localhost:8000",
      apiKey: "amfs_secret_key",
      headers: { "X-Custom-Header": "custom-val" },
    });
    const mem = new AgentMemory("agent-room", {
      adapter: http,
      sessionId: "sess-room",
    });

    await mem.myRooms();

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [, init] = fetchSpy.mock.calls[0];
    const headers = init?.headers as Record<string, string>;
    expect(headers["X-AMFS-API-Key"]).toBe("amfs_secret_key");
    expect(headers["Authorization"]).toBe("Bearer amfs_secret_key");
    expect(headers["X-AMFS-Agent-Id"]).toBe("agent-room");
    expect(headers["X-AMFS-Session"]).toBe("sess-room");
    expect(headers["X-Custom-Header"]).toBe("custom-val");
  });
});
