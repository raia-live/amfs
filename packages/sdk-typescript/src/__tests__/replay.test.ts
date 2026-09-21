import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AgentMemory,
  BRANCH_ENV,
  CASE_ID_ATTRIBUTE,
  HttpAdapter,
  MEMORY_BRANCH_ATTRIBUTE,
  OutcomeType,
  PayloadError,
  ReplayReceiver,
  SESSION_ATTRIBUTES_MAX_KEYS,
  SignatureError,
  createFetchHandler,
  parseReplayRequest,
  serveReplay,
  signReplayBody,
  validateSessionAttributes,
  verifyReplaySignature,
  type ReplayMemory,
  type ReplayRequest,
} from "../index.js";
import { DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER } from "../replay.js";

const SECRET = "whsec_ts_test";

function payload(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    event: "replay_requested",
    delivery_id: crypto.randomUUID(),
    sent_at: new Date().toISOString(),
    fix_id: "0d1f3a6c-2b4e-4f60-9a1e-7c8d9e0f1a2b",
    agent_id: "support-agent",
    branch: "repair/0d1f3a6c",
    case_id: "9b8a7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d",
    case_set_id: crypto.randomUUID(),
    task_input: "customer says the invoice email never arrived",
    expected: { action: "resolve:resend_email" },
    deadline_at: new Date(Date.now() + 6 * 3600_000).toISOString(),
    ...over,
  };
}

async function signed(
  body: Record<string, unknown>,
  secret: string | null = SECRET
): Promise<[Record<string, string>, string]> {
  const raw = JSON.stringify(body);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    [EVENT_HEADER.toLowerCase()]: String(body.event),
    [DELIVERY_HEADER.toLowerCase()]: String(body.delivery_id ?? ""),
  };
  if (secret) headers[SIGNATURE_HEADER.toLowerCase()] = await signReplayBody(secret, raw);
  return [headers, raw];
}

class FakeMemory implements ReplayMemory {
  branch: string;
  commits: Array<{ ref: string; outcome: OutcomeType; options: any }> = [];
  constructor(branch: string) {
    this.branch = branch;
  }
  checkout(branch: string): string {
    this.branch = branch;
    return branch;
  }
  async commitOutcomeAsync(ref: string, outcome: OutcomeType, options?: any): Promise<unknown> {
    this.commits.push({ ref, outcome, options });
    return { trace_id: "trace-1" };
  }
}

describe("the signature", () => {
  it("matches the sender's scheme and rejects junk", async () => {
    const body = '{"a": 1}';
    const header = await signReplayBody(SECRET, body);
    expect(header).toMatch(/^sha256=[0-9a-f]{64}$/);
    expect(await verifyReplaySignature(SECRET, body, header)).toBe(true);
    expect(await verifyReplaySignature(SECRET, body, header.toUpperCase().replace("SHA256", "sha256"))).toBe(true);
    expect(await verifyReplaySignature(SECRET, body + " ", header)).toBe(false);
    expect(await verifyReplaySignature("other", body, header)).toBe(false);
    for (const junk of [undefined, null, "", "sha256=", "md5=abc", "abc", "sha256"]) {
      expect(await verifyReplaySignature(SECRET, body, junk)).toBe(false);
    }
    // Bytes and strings sign alike.
    expect(await signReplayBody(SECRET, new TextEncoder().encode(body))).toBe(header);
  });

  it("interoperates with the Python sender: a fixed vector", async () => {
    // hmac.new(b"whsec_ts_test", b'{"x":1}', hashlib.sha256).hexdigest()
    expect(await signReplayBody(SECRET, '{"x":1}')).toBe(
      "sha256=25fee84d8236cabf9117a8953c83bd3f9b3aa01e88c1476e432b545f97067178"
    );
  });
});

describe("the parse", () => {
  it("reads every field and refuses what it cannot act on", async () => {
    const p = payload();
    const [headers, body] = await signed(p);
    const req = await parseReplayRequest(headers, body, { secret: SECRET });
    expect(req.fixId).toBe(p.fix_id);
    expect(req.caseId).toBe(p.case_id);
    expect(req.branch).toBe("repair/0d1f3a6c");
    expect(req.taskInput).toBe(p.task_input);
    expect(req.expected).toEqual({ action: "resolve:resend_email" });
    expect(req.deadlineAt).toBeInstanceOf(Date);

    await expect(parseReplayRequest(...(await signed(p, "wrong")), { secret: SECRET })).rejects.toBeInstanceOf(
      SignatureError
    );
    for (const over of [{ case_id: "" }, { branch: null }, { task_input: "" }, { branch: "main" }, { event: "x" }]) {
      await expect(parseReplayRequest(...(await signed(payload(over))), { secret: SECRET })).rejects.toBeInstanceOf(
        PayloadError
      );
    }
    // Headers arrive as a Headers instance too.
    const [h, b] = await signed(p);
    expect((await parseReplayRequest(new Headers(h), b, { secret: SECRET })).caseId).toBe(p.case_id);
    // No secret, no verification.
    expect((await parseReplayRequest({}, b, { secret: null })).caseId).toBe(p.case_id);
    // A ping.
    const [ph, pb] = await signed({ event: "ping", delivery_id: "d", agent_id: "a" });
    expect((await parseReplayRequest(ph, pb, { secret: SECRET })).event).toBe("ping");
  });
});

describe("the receiver", () => {
  function world(opts: { background?: boolean; run?: any } = {}) {
    const memories: FakeMemory[] = [];
    const runs: Array<[string, string]> = [];
    const receiver = new ReplayReceiver<FakeMemory>({
      secret: SECRET,
      background: opts.background ?? false,
      memoryFactory: (req: ReplayRequest) => {
        const m = new FakeMemory(req.branch);
        memories.push(m);
        return m;
      },
      run:
        opts.run ??
        (async (task: string, memory: FakeMemory) => {
          runs.push([task, memory.branch]);
          return `handled: ${task}`;
        }),
    });
    return { receiver, memories, runs };
  }

  it("runs inline on the branch and commits the graded attributes", async () => {
    const { receiver, memories, runs } = world();
    const p = payload();
    const { status, body } = await receiver.handle(...(await signed(p)));
    expect(status).toBe(200);
    expect(body.ran).toBe(true);
    expect(body.outcome_type).toBe("success");
    expect(body.outcome_ref).toBe("replay:0d1f3a6c:9b8a7c6d");
    expect(body.trace_id).toBe("trace-1");
    expect(runs).toEqual([[p.task_input, "repair/0d1f3a6c"]]);
    const commit = memories[0].commits[0];
    expect(commit.options.attributes[CASE_ID_ATTRIBUTE]).toBe(p.case_id);
    expect(commit.options.attributes[MEMORY_BRANCH_ATTRIBUTE]).toBe("repair/0d1f3a6c");
    expect(commit.options.attributes.fix_id).toBe(p.fix_id);
    expect(commit.options.taskInput).toBe(p.task_input);
    expect(commit.options.responseText).toBe(`handled: ${p.task_input}`);
  });

  it("acknowledges a duplicate delivery without a second run", async () => {
    const { receiver, runs } = world();
    const p = payload();
    await receiver.handle(...(await signed(p)));
    const second = await receiver.handle(...(await signed(p)));
    expect(second.status).toBe(200);
    expect(second.body.duplicate).toBe(true);
    expect(second.body.outcome_type).toBe("success");
    expect(runs).toHaveLength(1);
  });

  it("answers ping and the refusals without running", async () => {
    const { receiver, runs } = world();
    expect((await receiver.handle(...(await signed({ event: "ping", delivery_id: "p", agent_id: "a" })))).status).toBe(200);
    expect((await receiver.handle(...(await signed(payload(), "wrong")))).status).toBe(401);
    expect((await receiver.handle(...(await signed(payload({ case_id: "" }))))).status).toBe(400);
    const past = new Date(Date.now() - 60_000).toISOString();
    const gone = await receiver.handle(...(await signed(payload({ deadline_at: past }))));
    expect(gone.status).toBe(410);
    expect(gone.body.code).toBe("past_deadline");
    expect(runs).toHaveLength(0);
  });

  it("commits a runner that throws as a failure, not a missing case", async () => {
    const { receiver, memories } = world({
      run: async () => {
        throw new Error("model endpoint down");
      },
    });
    const { body } = await receiver.handle(...(await signed(payload())));
    expect(body.outcome_type).toBe("failure");
    const commit = memories[0].commits[0];
    expect(commit.outcome).toBe(OutcomeType.FAILURE);
    expect(commit.options.responseText).toContain("Error: model endpoint down");
    expect(commit.options.attributes.replay_error).toBe("Error");
    expect(commit.options.attributes[CASE_ID_ATTRIBUTE]).toBeTruthy();
  });

  it("takes a result object, and the grader's keys win over the runner's bag", async () => {
    const { receiver, memories } = world({
      run: async () => ({
        responseText: "explicit",
        outcomeType: "minor_failure",
        attributes: { model: "gpt-x", [CASE_ID_ATTRIBUTE]: "spoofed" },
      }),
    });
    const { body } = await receiver.handle(...(await signed(payload())));
    expect(body.outcome_type).toBe("minor_failure");
    const attrs = memories[0].commits[0].options.attributes;
    expect(attrs.model).toBe("gpt-x");
    expect(attrs[CASE_ID_ATTRIBUTE]).toBe("9b8a7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d");
  });

  it("fits the grader's keys ahead of a runner that filled the attribute bag", async () => {
    class BagMemory extends FakeMemory {
      sessionAttributes: Record<string, string | number | boolean> = {};
      clearSessionAttributes() {
        this.sessionAttributes = {};
      }
    }
    const memories: BagMemory[] = [];
    const full: Record<string, number> = {};
    for (let i = 0; i < SESSION_ATTRIBUTES_MAX_KEYS; i++) full[`dim${i}`] = i;
    const receiver = new ReplayReceiver<BagMemory>({
      secret: SECRET,
      background: false,
      memoryFactory: (req) => {
        const m = new BagMemory(req.branch);
        memories.push(m);
        return m;
      },
      run: async (_task, memory) => {
        memory.sessionAttributes = { ...full };
        return { responseText: "ok", attributes: { model: "gpt-x", [CASE_ID_ATTRIBUTE]: "spoofed" } };
      },
    });
    const { status, body } = await receiver.handle(...(await signed(payload())));
    expect(status).toBe(200);
    const attrs = memories[0].commits[0].options.attributes as Record<string, unknown>;
    const counted = Object.keys(attrs).filter((k) => k !== MEMORY_BRANCH_ATTRIBUTE);
    expect(counted).toHaveLength(SESSION_ATTRIBUTES_MAX_KEYS);
    expect(attrs[CASE_ID_ATTRIBUTE]).toBe("9b8a7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d");
    expect(attrs.fix_id).toBeDefined();
    expect(attrs.replay_delivery_id).toBeDefined();
    expect(attrs[MEMORY_BRANCH_ATTRIBUTE]).toBe("repair/0d1f3a6c");
    expect(attrs.dim0).toBe(0);
    expect(attrs.model).toBeUndefined();
    expect(body.dropped_attributes).toEqual(["dim17", "dim18", "dim19", "model"]);
    expect(memories[0].sessionAttributes).toEqual({});
  });

  it("forwards the runner's tool calls so the judge can grade the action taken", async () => {
    const toolCalls = [{ tool_name: "resolve", arguments: { action: "resend_email" } }];
    const { receiver, memories } = world({
      run: async () => ({ responseText: "resent", outcomeType: "success", toolCalls }),
    });
    await receiver.handle(...(await signed(payload())));
    expect(memories[0].commits[0].options.toolCalls).toEqual(toolCalls);
    // A runner that answers with a string commits none.
    const plain = world();
    await plain.receiver.handle(...(await signed(payload({ delivery_id: "d-plain" }))));
    expect(plain.memories[0].commits[0].options.toolCalls).toBeUndefined();
  });

  it("in the background: 202 first, the run after, and a redelivery meanwhile is acknowledged", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const { receiver, runs } = world({
      background: true,
      run: async (task: string, memory: FakeMemory) => {
        await gate;
        runs.push([task, memory.branch]);
        return "late";
      },
    });
    const p = payload();
    const first = await receiver.handle(...(await signed(p)));
    expect(first.status).toBe(202);
    expect(first.body.accepted).toBe(true);
    expect(runs).toHaveLength(0);
    expect((await receiver.handle(...(await signed(p)))).status).toBe(200);
    release();
    await receiver.drain();
    expect(runs).toEqual([[p.task_input, "repair/0d1f3a6c"]]);
    expect(receiver.status(String(p.delivery_id))?.ran).toBe(true);
  });

  it("mounts as a fetch handler", async () => {
    const { receiver } = world();
    const handler = createFetchHandler(receiver);
    const health = await handler(new Request("http://x/amfs/replay"));
    expect(health.status).toBe(200);
    expect((await health.json()).receiver).toBe("amfs-replay");
    expect((await handler(new Request("http://x/elsewhere", { method: "POST" }))).status).toBe(404);
    const [headers, body] = await signed(payload());
    const res = await handler(new Request("http://x/amfs/replay", { method: "POST", headers, body }));
    expect(res.status).toBe(200);
    expect((await res.json()).outcome_type).toBe("success");
    const bad = await handler(
      new Request("http://x/amfs/replay", { method: "POST", ...(await (async () => {
        const [h, b] = await signed(payload(), "wrong");
        return { headers: h, body: b };
      })()) })
    );
    expect(bad.status).toBe(401);
  });

  it("answers an unexpected error with a generic 500 and reports it to onError", async () => {
    const { receiver } = world();
    const boom = new Error("secret database dsn=postgres://user:pw@host/db");
    receiver.handle = async () => {
      throw boom;
    };
    const seen: unknown[] = [];
    const handler = createFetchHandler(receiver, { onError: (e) => seen.push(e) });
    const [headers, body] = await signed(payload());
    const res = await handler(new Request("http://x/amfs/replay", { method: "POST", headers, body }));
    expect(res.status).toBe(500);
    const answer = await res.json();
    expect(answer).toEqual({ ok: false, error: "internal error" });
    expect(JSON.stringify(answer)).not.toContain("dsn=");
    expect(seen).toEqual([boom]);
  });

  it("serves over node:http", async () => {
    const { receiver, runs } = world();
    const server = await serveReplay(receiver, { host: "127.0.0.1", port: 0 });
    try {
      const { port } = server.address() as { port: number };
      const url = `http://127.0.0.1:${port}/amfs/replay`;
      expect((await fetch(url)).status).toBe(200);
      const [headers, body] = await signed(payload());
      const res = await fetch(url, { method: "POST", headers, body });
      expect(res.status).toBe(200);
      expect((await res.json()).outcome_type).toBe("success");
      expect(runs).toHaveLength(1);
      const [h2, b2] = await signed(payload(), "wrong");
      expect((await fetch(url, { method: "POST", headers: h2, body: b2 })).status).toBe(401);
      expect((await fetch(`http://127.0.0.1:${port}/nope`, { method: "POST", body: "{}" })).status).toBe(404);
    } finally {
      server.close();
    }
  });
});

describe("the branch on AgentMemory", () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  function mockFetch() {
    calls.length = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push({ url, init });
        return {
          ok: true,
          status: 200,
          json: async () => ({ id: "t1" }),
          text: async () => "{}",
        } as unknown as Response;
      })
    );
  }
  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env[BRANCH_ENV];
  });

  it("defaults from AMFS_BRANCH, is overridden by options.branch, and checkout moves it", () => {
    process.env[BRANCH_ENV] = "canary/abc";
    const fromEnv = new AgentMemory("a");
    expect(fromEnv.branch).toBe("canary/abc");
    const explicit = new AgentMemory("a", { branch: "repair/x" });
    expect(explicit.branch).toBe("repair/x");
    expect(explicit.checkout(null)).toBe("main");
    expect(explicit.checkout("  ")).toBe("main");
    expect(explicit.checkout("repair/y")).toBe("repair/y");
    delete process.env[BRANCH_ENV];
    expect(new AgentMemory("a").branch).toBe("main");
  });

  it("sends the memory's branch on every remote read and write unless the call names one", async () => {
    mockFetch();
    const mem = new AgentMemory("a", {
      adapter: new HttpAdapter({ url: "http://s" }),
      branch: "repair/x",
    });
    await mem.readAsync("acme/x", "k");
    expect(calls.at(-1)!.url).toContain("branch=repair%2Fx");
    await mem.listAsync("acme/x");
    expect(calls.at(-1)!.url).toContain("branch=repair%2Fx");
    await mem.writeAsync("acme/x", "k", "v");
    expect(JSON.parse(String(calls.at(-1)!.init?.body)).branch).toBe("repair/x");
    await mem.searchAsync({ entityPath: "acme/x" });
    expect(JSON.parse(String(calls.at(-1)!.init?.body)).branch).toBe("repair/x");
    await mem.retrieveAsync("q", { branch: "other/branch" });
    expect(JSON.parse(String(calls.at(-1)!.init?.body)).branch).toBe("other/branch");
    await mem.briefingAsync({ entityPath: "acme/x" });
    expect(calls.at(-1)!.url).toContain("branch=repair%2Fx");

    mem.checkout("main");
    await mem.readAsync("acme/x", "k");
    expect(calls.at(-1)!.url).not.toContain("branch=");
    await mem.writeAsync("acme/x", "k", "v");
    expect(JSON.parse(String(calls.at(-1)!.init?.body)).branch).toBeUndefined();
  });

  it("stamps memory_branch on a committed outcome off main, and sends the capture", async () => {
    mockFetch();
    const mem = new AgentMemory("a", {
      adapter: new HttpAdapter({ url: "http://s" }),
      branch: "repair/x",
    });
    await mem.commitOutcomeAsync("o-1", OutcomeType.SUCCESS, {
      attributes: { model: "m" },
      taskInput: "the ask",
      responseText: "the answer",
      toolCalls: [{ tool_name: "resolve", arguments: { action: "resend_email" } }],
    });
    const sent = JSON.parse(String(calls.at(-1)!.init?.body));
    expect(sent.session_metadata.attributes).toEqual({ model: "m", [MEMORY_BRANCH_ATTRIBUTE]: "repair/x" });
    expect(sent.task_input).toBe("the ask");
    expect(sent.response_text).toBe("the answer");
    expect(sent.tool_calls[0].tool_name).toBe("resolve");

    // A caller's own value wins; main carries no stamp.
    await mem.commitOutcomeAsync("o-2", OutcomeType.SUCCESS, {
      attributes: { [MEMORY_BRANCH_ATTRIBUTE]: "canary/z" },
    });
    expect(JSON.parse(String(calls.at(-1)!.init?.body)).session_metadata.attributes[MEMORY_BRANCH_ATTRIBUTE]).toBe(
      "canary/z"
    );
    mem.checkout("main");
    await mem.commitOutcomeAsync("o-3", OutcomeType.SUCCESS);
    expect(JSON.parse(String(calls.at(-1)!.init?.body)).session_metadata).toBeUndefined();
  });

  it("the stamp does not count against the attribute cap, here or on the wire", async () => {
    mockFetch();
    const mem = new AgentMemory("a", {
      adapter: new HttpAdapter({ url: "http://s" }),
      branch: "repair/x",
    });
    const full: Record<string, number> = {};
    for (let i = 0; i < SESSION_ATTRIBUTES_MAX_KEYS; i++) full[`k${i}`] = i;
    mem.setSessionAttributes(full);
    expect(() => mem.setSessionAttributes({ k20: 20 })).toThrow(/at most 20/);
    await mem.commitOutcomeAsync("o-1", OutcomeType.SUCCESS);
    const sent = JSON.parse(String(calls.at(-1)!.init?.body)).session_metadata.attributes;
    expect(Object.keys(sent)).toHaveLength(SESSION_ATTRIBUTES_MAX_KEYS + 1);
    expect(sent[MEMORY_BRANCH_ATTRIBUTE]).toBe("repair/x");
    // The 21-key bag the SDK just sent passes the validator the server shares;
    // a 21st caller key still does not, stamp or no stamp.
    expect(Object.keys(validateSessionAttributes(sent))).toHaveLength(SESSION_ATTRIBUTES_MAX_KEYS + 1);
    expect(() => validateSessionAttributes({ ...full, k20: 20 })).toThrow(/at most 20/);
    expect(() => validateSessionAttributes({ ...full, k20: 20, [MEMORY_BRANCH_ATTRIBUTE]: "b" })).toThrow(
      /at most 20/
    );
  });
});
