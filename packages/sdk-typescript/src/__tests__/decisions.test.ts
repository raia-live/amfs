import { describe, it, expect } from "vitest";
import { DecisionClient } from "../decisions.js";

describe("decision model lifecycle", () => {
  it("keeps retry keys and requires explicit compare-and-set activation", async () => {
    const seen: { url: string; init?: RequestInit }[] = [];
    const client = new DecisionClient({ baseUrl: "https://example.test", apiKey: "test", fetch: async (url, init) => {
      seen.push({ url: String(url), init }); return new Response(JSON.stringify({ state: "queued" }), { status: 202 });
    } });
    await client.enqueueTraining("test:model", { version: "v1", dataset_id: "approved" }, "same-key");
    await client.enqueueTraining("test:model", { version: "v1", dataset_id: "approved" }, "same-key");
    await client.activateVersion("test:model", "v1", null);
    await client.setMode("test:model", "observe");
    expect(new Headers(seen[0].init?.headers).get("Idempotency-Key")).toBe("same-key");
    expect(seen[0].init?.body).toEqual(seen[1].init?.body);
    expect(JSON.parse(String(seen[2].init?.body))).toEqual({ expected_active_version: null });
    expect(seen[3].init?.method).toBe("PATCH");
    // @ts-expect-error omission is prohibited for both TypeScript and JavaScript callers
    expect(() => client.activateVersion("test:model", "v1")).toThrow("required");
  });
  it("preserves opaque cursors and exposes stale activation failures", async () => {
    const urls: URL[] = [];
    const client = new DecisionClient({ baseUrl: "https://example.test", apiKey: "test", fetch: async (url) => {
      urls.push(new URL(String(url))); return new Response("{}", { status: String(url).endsWith("activate") ? 409 : 200 });
    } });
    await client.history("example", { limit: 10, cursor: "opaque+/=&x" });
    await client.usage("example", 7);
    expect(urls[0].searchParams.get("cursor")).toBe("opaque+/=&x");
    expect(urls[1].searchParams.get("days")).toBe("7");
    await expect(client.activateVersion("example", "v2", "v1")).rejects.toThrow("409");
    expect(() => client.history("example", { limit: 101 })).toThrow("limit");
    expect(() => client.enqueueTraining("example", { version: "v1", dataset_id: "approved" }, "")).toThrow("idempotencyKey");
  });
});
