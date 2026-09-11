/**
 * Shared machinery for the provider instrumentations.
 *
 * Each `instrument*` function replaces a client method on the *instance* with a
 * wrapper that times the call, reads token usage off the response (or, for a
 * stream, accumulates it while the caller iterates and records when the stream
 * ends) and hands the result to `memory.recordLlmCall`. The wrapped call is
 * never altered: an instrumentation failure is logged and swallowed, an error
 * from the provider is recorded as a failed call and re-thrown unchanged.
 *
 * Responses are read by duck typing, so neither `openai` nor `@anthropic-ai/sdk`
 * is imported — they are optional peers — and tests drive the wrappers with
 * plain objects.
 */

import type { RecordLlmCallInput } from "../session.js";

/** The one method of `AgentMemory` the instrumentations need. */
export interface LlmCallSink {
  recordLlmCall(call: RecordLlmCallInput, startedAt?: Date): unknown;
}

export interface Usage {
  model?: string;
  inputTokens?: number;
  outputTokens?: number;
}

export type Extractor = (response: unknown, usage: Usage) => void;

const FLAG = "__amfsInstrumented";

export function isInstrumented(target: unknown): boolean {
  return Boolean((target as Record<string, unknown> | null)?.[FLAG]);
}

export function markInstrumented(target: unknown): void {
  try {
    Object.defineProperty(target, FLAG, { value: true, enumerable: false, configurable: true });
  } catch {
    // frozen or proxy — nothing to do
  }
}

export function get(obj: unknown, key: string): unknown {
  if (obj === null || obj === undefined) return undefined;
  return (obj as Record<string, unknown>)[key];
}

export function asInt(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  return Math.trunc(value);
}

export function resolve(target: unknown, path: string[]): unknown {
  let obj = target;
  for (const hop of path) {
    obj = get(obj, hop);
    if (obj === null || obj === undefined) return undefined;
  }
  return obj;
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return (
    value !== null &&
    typeof value === "object" &&
    typeof (value as Record<PropertyKey, unknown>)[Symbol.asyncIterator] === "function"
  );
}

function warn(message: string, err: unknown): void {
  try {
    console.warn(`[amfs] ${message}`, err);
  } catch {
    // no console — stay silent
  }
}

class Call {
  readonly startedAt = new Date();
  private readonly t0 = performance.now();
  private done = false;

  constructor(
    private readonly sink: LlmCallSink,
    private readonly provider: string,
    private readonly requestedModel: string | undefined,
  ) {}

  record(usage: Usage, error?: unknown): void {
    if (this.done) return;
    this.done = true;
    const latencyMs = performance.now() - this.t0;
    try {
      const model = usage.model || this.requestedModel || "";
      if (!model) return; // nothing to attribute the call to
      this.sink.recordLlmCall(
        {
          model,
          provider: this.provider,
          inputTokens: usage.inputTokens ?? 0,
          outputTokens: usage.outputTokens ?? 0,
          latencyMs,
          ...(error !== undefined ? { error: String(error) } : {}),
        },
        this.startedAt,
      );
    } catch (err) {
      warn(`${this.provider} instrumentation: could not record call`, err);
    }
  }
}

/**
 * Wrap an async iterable so its chunks are observed and the call is recorded
 * once the stream is exhausted, breaks, or is returned early. Every other
 * property (e.g. `controller`, `finalMessage()`) is forwarded to the original.
 */
function proxyStream(inner: AsyncIterable<unknown>, call: Call, onChunk: Extractor): unknown {
  const usage: Usage = {};
  const iterate = async function* () {
    try {
      for await (const chunk of inner) {
        try {
          onChunk(chunk, usage);
        } catch (err) {
          warn("stream chunk extraction failed", err);
        }
        yield chunk;
      }
      call.record(usage);
    } catch (err) {
      call.record(usage, err);
      throw err;
    } finally {
      call.record(usage); // early `return()` / `break` lands here
    }
  };
  return new Proxy(inner as object, {
    get(target, prop, receiver) {
      if (prop === Symbol.asyncIterator) return () => iterate();
      const value = Reflect.get(target, prop, receiver);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

export interface PatchOptions {
  provider: string;
  sink: LlmCallSink;
  /** Reads usage from a complete response. */
  extract: Extractor;
  /** Reads usage from one stream chunk, accumulating into the usage. */
  onChunk: Extractor;
}

/**
 * Replace `owner[method]` with an instrumented wrapper. Returns whether anything
 * was patched (`false` when the method is missing or already wrapped).
 *
 * The provider SDKs return a promise-like (`APIPromise`) from `create`; the
 * wrapper awaits it, so callers always get a plain `Promise`. `stream: true`
 * calls resolve to an async iterable, which is proxied.
 */
export function patchMethod(owner: unknown, method: string, options: PatchOptions): boolean {
  if (owner === null || typeof owner !== "object") return false;
  const holder = owner as Record<string, unknown>;
  const original = holder[method];
  if (typeof original !== "function" || isInstrumented(original)) return false;

  const wrapper = async function (this: unknown, ...args: unknown[]): Promise<unknown> {
    const params = (args[0] ?? {}) as Record<string, unknown>;
    const call = new Call(
      options.sink,
      options.provider,
      typeof params.model === "string" ? params.model : undefined,
    );
    let result: unknown;
    try {
      result = await (original as (...a: unknown[]) => unknown).apply(this ?? owner, args);
    } catch (err) {
      call.record({}, err);
      throw err;
    }
    try {
      if (params.stream === true && isAsyncIterable(result)) {
        return proxyStream(result, call, options.onChunk);
      }
      const usage: Usage = {};
      options.extract(result, usage);
      call.record(usage);
    } catch (err) {
      warn(`${options.provider} instrumentation failed; call unaffected`, err);
    }
    return result;
  };
  markInstrumented(wrapper);
  try {
    holder[method] = wrapper;
  } catch (err) {
    warn(`${options.provider} instrumentation: cannot patch ${method}`, err);
    return false;
  }
  return true;
}
