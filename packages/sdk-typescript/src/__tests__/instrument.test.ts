import { describe, expect, it, vi } from "vitest";
import { AgentMemory, instrumentAnthropic, instrumentOpenAI } from "../index.js";

async function* stream<T>(chunks: T[], failAfter?: number): AsyncGenerator<T> {
  let i = 0;
  for (const c of chunks) {
    if (failAfter !== undefined && i++ >= failAfter) throw new Error("stream broke");
    yield c;
  }
}

/** A stream that also carries a method, as the SDKs' `Stream` objects do. */
function streamWithMethod<T>(chunks: T[]) {
  const inner = stream(chunks);
  return Object.assign(inner, { controller: { abort: vi.fn() } });
}

function chatResponse(model = "gpt-4o-mini", prompt = 100, completion = 20) {
  return {
    id: "chatcmpl-1",
    model,
    choices: [{ finish_reason: "stop", message: { content: "hi" } }],
    usage: { prompt_tokens: prompt, completion_tokens: completion },
  };
}

function fakeOpenAI(create: (...args: unknown[]) => unknown, responsesCreate?: (...a: unknown[]) => unknown) {
  const client: Record<string, unknown> = { chat: { completions: { create } } };
  if (responsesCreate) client.responses = { create: responsesCreate };
  return client as {
    chat: { completions: { create: (...args: unknown[]) => Promise<unknown> } };
    responses?: { create: (...args: unknown[]) => Promise<unknown> };
  };
}

describe("instrumentOpenAI", () => {
  it("records a chat completion with model, tokens, latency and provider", async () => {
    const mem = new AgentMemory("deploy-agent");
    const create = vi.fn(async (params: unknown) => chatResponse());
    const client = instrumentOpenAI(fakeOpenAI(create), mem);
    expect(instrumentOpenAI(client, mem)).toBe(client); // idempotent

    const out = (await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    })) as ReturnType<typeof chatResponse>;
    expect(out.choices[0].message.content).toBe("hi"); // untouched
    expect(create).toHaveBeenCalledTimes(1);

    const [call] = mem.sessionLlmCalls;
    expect(call).toMatchObject({
      model: "gpt-4o-mini", provider: "openai", input_tokens: 100, output_tokens: 20,
    });
    expect(call.latency_ms).toBeGreaterThanOrEqual(0);
    expect(call.cost_usd).toBeNull(); // pricing is the server's job
  });

  it("uses the requested model when the response has none", async () => {
    const mem = new AgentMemory("a");
    const client = instrumentOpenAI(fakeOpenAI(async () => ({ usage: { prompt_tokens: 1, completion_tokens: 1 } })), mem);
    await client.chat.completions.create({ model: "gpt-4o", messages: [] });
    expect(mem.sessionLlmCalls[0].model).toBe("gpt-4o");
  });

  it("records a stream when it is exhausted, from the final usage chunk", async () => {
    const mem = new AgentMemory("a");
    const chunks = [
      { id: "c", model: "gpt-4o-mini", choices: [{ delta: { content: "h" } }], usage: null },
      { id: "c", model: "gpt-4o-mini", choices: [{ delta: { content: "i" }, finish_reason: "stop" }], usage: null },
      { id: "c", model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 7, completion_tokens: 2 } },
    ];
    const s = streamWithMethod(chunks);
    const client = instrumentOpenAI(fakeOpenAI(async () => s), mem);
    const result = (await client.chat.completions.create({
      model: "gpt-4o-mini", messages: [], stream: true,
    })) as AsyncIterable<unknown> & { controller: { abort: () => void } };

    expect(mem.sessionLlmCalls).toHaveLength(0); // not yet: the stream is still open
    const seen: unknown[] = [];
    for await (const chunk of result) seen.push(chunk);
    expect(seen).toEqual(chunks);
    expect(mem.sessionLlmCalls).toHaveLength(1);
    expect(mem.sessionLlmCalls[0]).toMatchObject({ input_tokens: 7, output_tokens: 2, model: "gpt-4o-mini" });
    result.controller.abort(); // other members are forwarded
    expect(s.controller.abort).toHaveBeenCalled();
  });

  it("records a stream the consumer breaks out of, without tokens", async () => {
    const mem = new AgentMemory("a");
    const chunks = [{ model: "gpt-4o" }, { model: "gpt-4o" }, { model: "gpt-4o", usage: { prompt_tokens: 9, completion_tokens: 9 } }];
    const client = instrumentOpenAI(fakeOpenAI(async () => stream(chunks)), mem);
    const result = (await client.chat.completions.create({ model: "gpt-4o", stream: true })) as AsyncIterable<unknown>;
    for await (const _ of result) break;
    expect(mem.sessionLlmCalls).toHaveLength(1);
    expect(mem.sessionLlmCalls[0]).toMatchObject({ model: "gpt-4o", input_tokens: 0, output_tokens: 0 });
  });

  it("records a failing stream as a failed call and rethrows", async () => {
    const mem = new AgentMemory("a");
    const client = instrumentOpenAI(fakeOpenAI(async () => stream([{ model: "gpt-4o" }, { model: "gpt-4o" }], 1)), mem);
    const result = (await client.chat.completions.create({ model: "gpt-4o", stream: true })) as AsyncIterable<unknown>;
    await expect((async () => { for await (const _ of result) { /* drain */ } })()).rejects.toThrow("stream broke");
    expect(mem.sessionLlmCalls).toHaveLength(1);
    expect(mem.sessionLlmCalls[0].error).toMatch(/stream broke/);
  });

  it("records a provider error and rethrows it unchanged", async () => {
    const mem = new AgentMemory("a");
    const boom = new Error("429 rate limited");
    const client = instrumentOpenAI(fakeOpenAI(async () => { throw boom; }), mem);
    await expect(client.chat.completions.create({ model: "gpt-4o", messages: [] })).rejects.toBe(boom);
    const [call] = mem.sessionLlmCalls;
    expect(call).toMatchObject({ model: "gpt-4o", input_tokens: 0, output_tokens: 0 });
    expect(call.error).toContain("429");
  });

  it("instruments the Responses API, streaming included", async () => {
    const mem = new AgentMemory("a");
    const events = [
      { type: "response.created", response: { id: "r", model: "gpt-4o" } },
      { type: "response.output_text.delta", delta: "x" },
      { type: "response.completed", response: { id: "r", model: "gpt-4o", usage: { input_tokens: 30, output_tokens: 10 } } },
    ];
    const responsesCreate = vi.fn(async (params: { stream?: boolean }) =>
      params.stream ? stream(events) : { id: "r2", model: "gpt-4o-2024-08-06", usage: { input_tokens: 3, output_tokens: 4 } },
    );
    const client = instrumentOpenAI(fakeOpenAI(async () => chatResponse(), responsesCreate), mem);
    await client.responses!.create({ model: "gpt-4o", input: "y" });
    const s = (await client.responses!.create({ model: "gpt-4o", input: "x", stream: true })) as AsyncIterable<unknown>;
    for await (const _ of s) { /* drain */ }
    expect(mem.sessionLlmCalls.map((c) => [c.model, c.input_tokens, c.output_tokens])).toEqual([
      ["gpt-4o-2024-08-06", 3, 4],
      ["gpt-4o", 30, 10],
    ]);
  });

  it("never breaks the call when the recorder throws", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const sink = { recordLlmCall: () => { throw new Error("recorder bug"); } };
    const client = instrumentOpenAI(fakeOpenAI(async () => chatResponse()), sink);
    const out = (await client.chat.completions.create({ model: "gpt-4o-mini", messages: [] })) as { id: string };
    expect(out.id).toBe("chatcmpl-1");
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("tolerates a client without the resources", () => {
    const mem = new AgentMemory("a");
    const bare = {};
    expect(instrumentOpenAI(bare, mem)).toBe(bare);
  });

  it("preserves `this` for SDK methods that rely on it", async () => {
    const mem = new AgentMemory("a");
    class Completions {
      tag = "real";
      async create(_params: unknown) {
        return { model: `from-${this.tag}`, usage: { prompt_tokens: 1, completion_tokens: 1 } };
      }
    }
    const client = { chat: { completions: new Completions() } };
    instrumentOpenAI(client, mem);
    await client.chat.completions.create({ model: "x" });
    expect(mem.sessionLlmCalls[0].model).toBe("from-real");
  });
});

describe("instrumentAnthropic", () => {
  it("records messages.create, plain and streaming", async () => {
    const mem = new AgentMemory("a");
    const message = {
      id: "msg_1", model: "claude-3-5-haiku-20241022", stop_reason: "end_turn",
      usage: { input_tokens: 50, output_tokens: 25 }, content: [{ type: "text", text: "ok" }],
    };
    const events = [
      { type: "message_start", message: { id: "msg_2", model: "claude-3-5-haiku-20241022", usage: { input_tokens: 8, output_tokens: 1 } } },
      { type: "content_block_delta", delta: { type: "text_delta", text: "h" } },
      { type: "message_delta", delta: { stop_reason: "max_tokens" }, usage: { output_tokens: 6 } },
      { type: "message_stop" },
    ];
    const create = vi.fn(async (params: { stream?: boolean }) => (params.stream ? stream(events) : message));
    const client = instrumentAnthropic({ messages: { create } }, mem);
    expect(instrumentAnthropic(client, mem)).toBe(client);

    const out = (await client.messages.create({ model: "claude-3-5-haiku-20241022", max_tokens: 100, messages: [] })) as typeof message;
    expect(out.content[0].text).toBe("ok");
    const s = (await client.messages.create({ model: "claude-3-5-haiku-20241022", max_tokens: 8, messages: [], stream: true })) as AsyncIterable<unknown>;
    for await (const _ of s) { /* drain */ }

    expect(mem.sessionLlmCalls.map((c) => [c.provider, c.model, c.input_tokens, c.output_tokens])).toEqual([
      ["anthropic", "claude-3-5-haiku-20241022", 50, 25],
      ["anthropic", "claude-3-5-haiku-20241022", 8, 6],
    ]);
  });
});
