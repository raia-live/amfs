/**
 * Answer a replay request: the customer's end of the repair loop's Tier 2.
 *
 * When SenseLab has drafted a fix for one of your agent's behaviours and wants
 * proof from *your* agent before it ships, it sends one signed webhook per
 * test case:
 *
 *     POST <your replay URL>
 *     X-AMFS-Event: replay_requested
 *     X-AMFS-Delivery: <uuid, unique per request>
 *     X-AMFS-Signature: sha256=<hex HMAC-SHA256(secret, raw body)>
 *
 *     {"event": "replay_requested", "fix_id": ..., "agent_id": ..., "branch":
 *      "repair/<fix-id>", "case_id": ..., "task_input": ..., "expected": {...},
 *      "deadline_at": ..., ...}
 *
 * Your side has three obligations, and this module discharges all of them:
 *
 * 1. verify the signature over the raw body ({@link verifyReplaySignature});
 * 2. run the agent on `task_input` with its memory on `branch` — the branch
 *    holds the fix; a run on `main` never read it and proves nothing;
 * 3. commit the outcome with `attributes.case_id` and
 *    `attributes.memory_branch` so the grader can find the trace.
 *
 * The sender waits ten seconds and retries a non-2xx a bounded number of
 * times, so the receiver acknowledges with `202` at once and runs the agent
 * afterwards; delivery is at-least-once, so a delivery id seen before is
 * acknowledged again without a second run. A `ping` event is answered `200`.
 *
 * Runtime-agnostic, like the rest of the SDK: signatures use Web Crypto, so
 * the receiver runs on Node 18+, Bun, Deno and edge runtimes alike.
 * {@link ReplayReceiver.handle} takes headers and the raw body and returns a
 * status and a JSON body, so it mounts in Express, Fastify or anything else;
 * {@link createFetchHandler} wraps it as a `(Request) => Response` for
 * Next.js route handlers, Hono, Bun and workers; {@link serveReplay} starts a
 * `node:http` server around it.
 *
 * @example
 * ```ts
 * const receiver = new ReplayReceiver({
 *   secret: process.env.AMFS_REPLAY_SECRET!,
 *   memoryFactory: (req) => new AgentMemory(req.agentId, { adapter: http, branch: req.branch }),
 *   run: async (taskInput, memory) => myAgent.answer(taskInput, memory),
 * });
 * app.post("/amfs/replay", express.raw({ type: "*\/*" }), async (req, res) => {
 *   const { status, body } = await receiver.handle(req.headers, req.body);
 *   res.status(status).json(body);
 * });
 * ```
 */

import { OutcomeType } from "./models.js";
import { MEMORY_BRANCH_ATTRIBUTE } from "./memory.js";
import type { SessionAttributes } from "./session.js";

export const SIGNATURE_HEADER = "X-AMFS-Signature";
export const EVENT_HEADER = "X-AMFS-Event";
export const DELIVERY_HEADER = "X-AMFS-Delivery";

export const EVENT_REPLAY_REQUESTED = "replay_requested";
export const EVENT_PING = "ping";

/** The attributes the grader reads a replayed trace through. */
export const CASE_ID_ATTRIBUTE = "case_id";
export const FIX_ID_ATTRIBUTE = "fix_id";
export const DELIVERY_ID_ATTRIBUTE = "replay_delivery_id";
export { MEMORY_BRANCH_ATTRIBUTE };

/** How many delivery ids the receiver remembers for de-duplication. */
export const SEEN_DELIVERIES = 4096;

export type HeaderBag =
  | Record<string, string | string[] | undefined>
  | Headers
  | Iterable<[string, string]>;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function toBytes(body: string | Uint8Array): Uint8Array<ArrayBuffer> {
  if (typeof body === "string") return encoder.encode(body) as Uint8Array<ArrayBuffer>;
  // A view over a plain ArrayBuffer, as Web Crypto wants; copies only when the
  // source sits on a SharedArrayBuffer or is a partial view.
  if (body.buffer instanceof ArrayBuffer && body.byteOffset === 0 && body.byteLength === body.buffer.byteLength) {
    return body as Uint8Array<ArrayBuffer>;
  }
  const copy = new Uint8Array(body.byteLength);
  copy.set(body);
  return copy;
}

async function hmacHex(secret: string, body: string | Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("Web Crypto (crypto.subtle) is not available in this runtime");
  const key = await subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await subtle.sign("HMAC", key, toBytes(body));
  return Array.from(new Uint8Array(signature), (b) => b.toString(16).padStart(2, "0")).join("");
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/** The `X-AMFS-Signature` value for `body`: `sha256=<hex>`. */
export async function signReplayBody(secret: string, body: string | Uint8Array): Promise<string> {
  return `sha256=${await hmacHex(secret, body)}`;
}

/** Whether `header` signs `body` under `secret`. Constant-time; junk is `false`. */
export async function verifyReplaySignature(
  secret: string,
  body: string | Uint8Array,
  header: string | undefined | null
): Promise<boolean> {
  if (!header || typeof header !== "string") return false;
  const eq = header.indexOf("=");
  if (eq < 0) return false;
  const scheme = header.slice(0, eq).trim().toLowerCase();
  const digest = header.slice(eq + 1).trim().toLowerCase();
  if (scheme !== "sha256" || !digest) return false;
  return constantTimeEqual(await hmacHex(secret, body), digest);
}

function headerOf(headers: HeaderBag, name: string): string | undefined {
  const wanted = name.toLowerCase();
  if (typeof (headers as Headers).get === "function" && !(Symbol.iterator in Object(headers) && Array.isArray(headers))) {
    const value = (headers as Headers).get(name);
    if (value != null) return value;
  }
  const entries: Iterable<[string, unknown]> =
    Symbol.iterator in Object(headers) && typeof (headers as Headers).get !== "function"
      ? (headers as Iterable<[string, string]>)
      : Object.entries(headers as Record<string, unknown>);
  for (const [key, value] of entries) {
    if (String(key).toLowerCase() !== wanted) continue;
    if (Array.isArray(value)) return value[0] == null ? undefined : String(value[0]);
    return value == null ? undefined : String(value);
  }
  return undefined;
}

function parseWhen(value: unknown): Date | undefined {
  if (typeof value !== "string" || !value) return undefined;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? undefined : d;
}

export class ReplayError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly code: string
  ) {
    super(message);
    this.name = "ReplayError";
  }
}

export class SignatureError extends ReplayError {
  constructor(message = "signature missing or invalid") {
    super(401, message, "bad_signature");
    this.name = "SignatureError";
  }
}

export class PayloadError extends ReplayError {
  constructor(message: string) {
    super(400, message, "bad_payload");
    this.name = "PayloadError";
  }
}

/** One verified replay request, as the sender framed it. */
export interface ReplayRequest {
  event: string;
  deliveryId: string;
  fixId: string;
  agentId: string;
  branch: string;
  caseId: string;
  caseSetId?: string;
  taskInput: string;
  expected: Record<string, unknown>;
  deadlineAt?: Date;
  sentAt?: Date;
  instructions?: string;
}

/** The outcome reference a replay commits under. */
export function replayOutcomeRef(request: ReplayRequest): string {
  return `replay:${request.fixId.slice(0, 8)}:${request.caseId.slice(0, 8)}`;
}

/** The session attributes the committed outcome must carry. */
export function replayAttributes(request: ReplayRequest): SessionAttributes {
  return {
    [CASE_ID_ATTRIBUTE]: request.caseId,
    [MEMORY_BRANCH_ATTRIBUTE]: request.branch,
    [FIX_ID_ATTRIBUTE]: request.fixId,
    [DELIVERY_ID_ATTRIBUTE]: request.deliveryId,
  };
}

/**
 * Verify and parse one delivery. Throws {@link SignatureError} when `secret`
 * is set and the signature does not match, {@link PayloadError} when the body
 * is not a request the receiver can act on. `secret: null` skips
 * verification — for a receiver behind its own authentication and for local
 * simulations.
 */
export async function parseReplayRequest(
  headers: HeaderBag,
  body: string | Uint8Array,
  options: { secret: string | null }
): Promise<ReplayRequest> {
  if (options.secret) {
    if (!(await verifyReplaySignature(options.secret, body, headerOf(headers, SIGNATURE_HEADER)))) {
      throw new SignatureError();
    }
  }
  const text = typeof body === "string" ? body : decoder.decode(body);
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch (err) {
    throw new PayloadError(`body is not JSON: ${(err as Error).message}`);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new PayloadError("body must be a JSON object");
  }
  const p = payload as Record<string, unknown>;
  const str = (v: unknown): string => (v == null ? "" : String(v)).trim();
  const event = str(p.event) || str(headerOf(headers, EVENT_HEADER));
  const deliveryId = str(p.delivery_id) || str(headerOf(headers, DELIVERY_HEADER));

  if (event === EVENT_PING) {
    return {
      event: EVENT_PING,
      deliveryId,
      fixId: "",
      agentId: str(p.agent_id),
      branch: "",
      caseId: "",
      taskInput: "",
      expected: {},
      sentAt: parseWhen(p.sent_at),
    };
  }
  if (event !== EVENT_REPLAY_REQUESTED) throw new PayloadError(`unknown event ${JSON.stringify(event)}`);

  const missing = (["fix_id", "branch", "case_id", "task_input"] as const).filter((k) => !str(p[k]));
  if (!deliveryId) missing.unshift("delivery_id" as never);
  if (missing.length) throw new PayloadError(`replay request is missing ${missing.join(", ")}`);
  const branch = str(p.branch);
  // The whole point of a replay is a run that read the fix. A request naming
  // main is not one the sender makes; refuse it rather than commit a trace
  // the grader would take for a repaired run.
  if (branch === "main") throw new PayloadError("a replay cannot run on main");
  const expected = p.expected;
  return {
    event: EVENT_REPLAY_REQUESTED,
    deliveryId,
    fixId: str(p.fix_id),
    agentId: str(p.agent_id),
    branch,
    caseId: str(p.case_id),
    caseSetId: str(p.case_set_id) || undefined,
    taskInput: String(p.task_input),
    expected: expected && typeof expected === "object" && !Array.isArray(expected)
      ? { ...(expected as Record<string, unknown>) }
      : {},
    deadlineAt: parseWhen(p.deadline_at),
    sentAt: parseWhen(p.sent_at),
    instructions: str(p.instructions) || undefined,
  };
}

/** What a run hands back. A plain string is the agent's answer, taken as a success. */
export interface ReplayResult {
  responseText?: string | null;
  outcomeType?: OutcomeType | `${OutcomeType}` | boolean;
  attributes?: SessionAttributes;
}

export type ReplayAnswer = ReplayResult | string | null | undefined;

/**
 * A memory the receiver can run a case through: reads on `branch` and a
 * remote outcome commit. `AgentMemory` over an `HttpAdapter` satisfies it.
 */
export interface ReplayMemory {
  readonly branch?: string;
  checkout?(branch: string): string;
  commitOutcomeAsync(
    outcomeRef: string,
    outcomeType: OutcomeType,
    options?: { attributes?: SessionAttributes; taskInput?: string; responseText?: string | null }
  ): Promise<unknown>;
}

export interface ReplayReceiverOptions<M extends ReplayMemory = ReplayMemory> {
  /** The shared secret registered with the webhook; `null` accepts unsigned requests. */
  secret: string | null;
  /** Builds the memory for a request — on `request.branch`. */
  memoryFactory: (request: ReplayRequest) => M | Promise<M>;
  /** Answers a case: `run(taskInput, memory)`; read through `memory` and the run reads the fix. */
  run: (taskInput: string, memory: M, request: ReplayRequest) => Promise<ReplayAnswer> | ReplayAnswer;
  /** Acknowledge with 202 and run afterwards (default). `false` runs inline and answers 200 with the outcome. */
  background?: boolean;
  /** Called with a run that could not be committed at all. */
  onError?: (error: unknown, request: ReplayRequest) => void;
  now?: () => Date;
}

export interface ReplayResponse {
  status: number;
  body: Record<string, unknown>;
}

function toOutcome(value: ReplayResult["outcomeType"]): OutcomeType {
  if (value === undefined || value === null) return OutcomeType.SUCCESS;
  if (typeof value === "boolean") return value ? OutcomeType.SUCCESS : OutcomeType.FAILURE;
  return value as OutcomeType;
}

/** The receiving end of the replay webhook. */
export class ReplayReceiver<M extends ReplayMemory = ReplayMemory> {
  private readonly seen = new Map<string, Record<string, unknown>>();
  private readonly inflight = new Set<Promise<void>>();
  private readonly options: ReplayReceiverOptions<M>;

  constructor(options: ReplayReceiverOptions<M>) {
    this.options = { background: true, now: () => new Date(), ...options };
  }

  /**
   * Answer one HTTP request.
   *
   * - `ping` → 200; bad signature → 401; a body it cannot act on → 400;
   * - past the deadline → 410; a delivery seen before → 200 with the first answer;
   * - otherwise 202 and the run starts (200 with its outcome when inline).
   */
  async handle(headers: HeaderBag, body: string | Uint8Array): Promise<ReplayResponse> {
    let request: ReplayRequest;
    try {
      request = await parseReplayRequest(headers, body, { secret: this.options.secret });
    } catch (err) {
      if (err instanceof ReplayError) {
        return { status: err.status, body: { ok: false, error: err.message, code: err.code } };
      }
      throw err;
    }
    if (request.event === EVENT_PING) {
      return { status: 200, body: { ok: true, event: EVENT_PING, agent_id: request.agentId } };
    }
    const prior = this.seen.get(request.deliveryId);
    if (prior) return { status: 200, body: { ...prior, duplicate: true } };
    if (request.deadlineAt && this.options.now!() >= request.deadlineAt) {
      return {
        status: 410,
        body: {
          ok: false,
          code: "past_deadline",
          case_id: request.caseId,
          error: `deadline ${request.deadlineAt.toISOString()} has passed`,
        },
      };
    }
    const answer: Record<string, unknown> = {
      ok: true,
      accepted: true,
      delivery_id: request.deliveryId,
      case_id: request.caseId,
      branch: request.branch,
    };
    this.remember(request.deliveryId, answer);

    if (!this.options.background) {
      const outcome = await this.replay(request);
      const full = { ...answer, accepted: false, ran: true, ...outcome };
      this.remember(request.deliveryId, full);
      return { status: 200, body: full };
    }
    const job = this.runInBackground(request);
    this.inflight.add(job);
    void job.finally(() => this.inflight.delete(job));
    return { status: 202, body: answer };
  }

  /**
   * Run one case and commit its outcome on the request's branch. A runner
   * that throws is committed as a failure with the error as the answer: a
   * graded failure tells the loop more than a case that never came back.
   */
  async replay(request: ReplayRequest): Promise<Record<string, unknown>> {
    const memory = await this.options.memoryFactory(request);
    if (typeof memory.checkout === "function" && memory.branch !== request.branch) {
      memory.checkout(request.branch);
    }
    let result: ReplayResult;
    try {
      const answer = await this.options.run(request.taskInput, memory, request);
      result =
        answer == null
          ? { responseText: null }
          : typeof answer === "string"
            ? { responseText: answer }
            : answer;
    } catch (err) {
      const e = err as Error;
      result = {
        responseText: `replay runner raised ${e?.name ?? "Error"}: ${e?.message ?? String(err)}`.slice(0, 2000),
        outcomeType: OutcomeType.FAILURE,
        attributes: { replay_error: e?.name ?? "Error" },
      };
    }
    const outcomeType = toOutcome(result.outcomeType);
    const attributes: SessionAttributes = { ...(result.attributes ?? {}), ...replayAttributes(request) };
    const outcomeRef = replayOutcomeRef(request);
    const committed = await memory.commitOutcomeAsync(outcomeRef, outcomeType, {
      attributes,
      taskInput: request.taskInput,
      responseText: result.responseText ?? null,
    });
    const traceId =
      committed && typeof committed === "object"
        ? ((committed as Record<string, unknown>).trace_id ?? (committed as Record<string, unknown>).id)
        : undefined;
    return {
      outcome_type: outcomeType,
      outcome_ref: outcomeRef,
      trace_id: traceId == null ? null : String(traceId),
    };
  }

  /** Wait for background runs to finish. */
  async drain(): Promise<void> {
    while (this.inflight.size) await Promise.allSettled([...this.inflight]);
  }

  /** What became of a delivery, if the receiver still remembers it. */
  status(deliveryId: string): Record<string, unknown> | undefined {
    const s = this.seen.get(deliveryId);
    return s ? { ...s } : undefined;
  }

  private async runInBackground(request: ReplayRequest): Promise<void> {
    try {
      const outcome = await this.replay(request);
      this.remember(request.deliveryId, { ...(this.seen.get(request.deliveryId) ?? {}), ran: true, ...outcome });
    } catch (err) {
      this.remember(request.deliveryId, {
        ...(this.seen.get(request.deliveryId) ?? {}),
        ran: false,
        error: "commit failed",
      });
      this.options.onError?.(err, request);
    }
  }

  private remember(deliveryId: string, answer: Record<string, unknown>): void {
    this.seen.delete(deliveryId);
    this.seen.set(deliveryId, answer);
    while (this.seen.size > SEEN_DELIVERIES) {
      const oldest = this.seen.keys().next().value;
      if (oldest === undefined) break;
      this.seen.delete(oldest);
    }
  }
}

/**
 * A `(Request) => Promise<Response>` handler mounting `receiver` at `path` —
 * the shape Next.js route handlers, Hono, Bun.serve, Deno.serve and workers
 * take. `GET path` answers 200 for a health check; any other path is 404.
 */
export function createFetchHandler(
  receiver: ReplayReceiver<any>,
  options?: { path?: string }
): (request: Request) => Promise<Response> {
  const path = options?.path ?? "/amfs/replay";
  const json = (status: number, body: unknown) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  return async (request: Request) => {
    const url = new URL(request.url, "http://replay.local").pathname;
    if (url !== path) return json(404, { ok: false, error: "not found" });
    if (request.method === "GET") return json(200, { ok: true, receiver: "amfs-replay", path });
    if (request.method !== "POST") return json(405, { ok: false, error: "method not allowed" });
    try {
      const body = new Uint8Array(await request.arrayBuffer());
      const { status, body: answer } = await receiver.handle(request.headers, body);
      return json(status, answer);
    } catch (err) {
      return json(500, { ok: false, error: String((err as Error)?.message ?? err) });
    }
  };
}

/**
 * Start a `node:http` server around `receiver` (Node 18+). Resolves once it
 * is listening; `server.address()` has the port when `port: 0` was asked for.
 * Typed loosely on purpose: the SDK carries no Node type definitions.
 */
export async function serveReplay(
  receiver: ReplayReceiver<any>,
  options?: { host?: string; port?: number; path?: string }
): Promise<{ close(): void; address(): unknown }> {
  const specifier = "node:http";
  const http = (await import(/* @vite-ignore */ specifier)) as {
    createServer(listener: (req: any, res: any) => void): any;
  };
  const path = options?.path ?? "/amfs/replay";
  const server = http.createServer((req: any, res: any) => {
    const send = (status: number, body: unknown) => {
      const raw = encoder.encode(JSON.stringify(body));
      res.writeHead(status, { "Content-Type": "application/json", "Content-Length": String(raw.byteLength) });
      res.end(raw);
    };
    const url = String(req.url ?? "/").split("?")[0];
    if (url !== path) return send(404, { ok: false, error: "not found" });
    if (req.method === "GET") return send(200, { ok: true, receiver: "amfs-replay", path });
    if (req.method !== "POST") return send(405, { ok: false, error: "method not allowed" });
    const chunks: Uint8Array[] = [];
    req.on("data", (c: Uint8Array) => chunks.push(c));
    req.on("end", () => {
      const total = chunks.reduce((n, c) => n + c.byteLength, 0);
      const body = new Uint8Array(total);
      let offset = 0;
      for (const c of chunks) {
        body.set(c, offset);
        offset += c.byteLength;
      }
      receiver
        .handle(req.headers as Record<string, string | string[] | undefined>, body)
        .then(({ status, body: answer }) => send(status, answer))
        .catch((err: unknown) => send(500, { ok: false, error: String((err as Error)?.message ?? err) }));
    });
  });
  await new Promise<void>((resolve) => server.listen(options?.port ?? 8787, options?.host ?? "0.0.0.0", resolve));
  return server;
}
