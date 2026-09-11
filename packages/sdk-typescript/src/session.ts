/**
 * Session metadata the SDK collects for a trace: the attribute bag a run is
 * filtered and grouped by, and the LLM calls that give it real token and cost
 * figures. Both travel in `session_metadata` when the outcome is committed —
 * `attributes` and `llm_calls`, the same shape the Python SDK and MCP server
 * write, so the server treats every producer alike.
 */

export type AttributeValue = string | number | boolean;
export type SessionAttributes = Record<string, AttributeValue>;

/** Attributes are indexed server-side, so the bag is kept small and flat. */
export const SESSION_ATTRIBUTES_MAX_KEYS = 20;
export const SESSION_ATTRIBUTE_KEY_MAX_LEN = 64;
export const SESSION_ATTRIBUTE_VALUE_MAX_LEN = 256;

/**
 * The validated form of an attribute bag, or a thrown `TypeError` / `RangeError`.
 *
 * Keys are trimmed and lowercased (the server indexes them that way, so two
 * spellings of one dimension would otherwise never group together). Values
 * must be string, finite number or boolean; at most 20 keys of up to 64
 * characters, string values up to 256 characters. Rejected rather than
 * trimmed: a silently shortened customer id is worse than an error at the
 * call site.
 */
export function validateSessionAttributes(attributes: unknown): SessionAttributes {
  if (attributes === null || attributes === undefined) return {};
  if (typeof attributes !== "object" || Array.isArray(attributes)) {
    throw new TypeError("attributes must be an object of scalar values");
  }
  const entries = Object.entries(attributes as Record<string, unknown>);
  if (entries.length > SESSION_ATTRIBUTES_MAX_KEYS) {
    throw new RangeError(
      `at most ${SESSION_ATTRIBUTES_MAX_KEYS} session attributes are allowed (got ${entries.length})`,
    );
  }
  const out: SessionAttributes = {};
  for (const [rawKey, value] of entries) {
    const key = String(rawKey).trim().toLowerCase();
    if (!key) throw new RangeError("attribute keys must be non-empty strings");
    if (key.length > SESSION_ATTRIBUTE_KEY_MAX_LEN) {
      throw new RangeError(
        `attribute key '${key.slice(0, 20)}...' exceeds ${SESSION_ATTRIBUTE_KEY_MAX_LEN} characters`,
      );
    }
    if (typeof value === "string") {
      if (value.length > SESSION_ATTRIBUTE_VALUE_MAX_LEN) {
        throw new RangeError(
          `attribute '${key}' exceeds ${SESSION_ATTRIBUTE_VALUE_MAX_LEN} characters`,
        );
      }
      out[key] = value;
    } else if (typeof value === "number") {
      if (!Number.isFinite(value)) throw new RangeError(`attribute '${key}' must be a finite number`);
      out[key] = value;
    } else if (typeof value === "boolean") {
      out[key] = value;
    } else {
      throw new TypeError(
        `attribute '${key}' must be a string, number or boolean, not ${value === null ? "null" : typeof value}`,
      );
    }
  }
  return out;
}

/** What {@link AgentMemory.recordLlmCall} takes. */
export interface RecordLlmCallInput {
  model: string;
  inputTokens: number;
  outputTokens: number;
  /** USD; omit when unknown — a blank cost is honest, a zero one is not. */
  costUsd?: number | null;
  latencyMs?: number | null;
  provider?: string | null;
  callId?: string | null;
  /** Set when the call failed; the record still counts the attempt. */
  error?: string | null;
}

/**
 * One LLM call as stored in `session_metadata.llm_calls` (wire shape, snake_case).
 * Mirrors the record the Python SDK's `record_llm_call` writes.
 */
export interface LlmCallRecord {
  call_id: string;
  model: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  latency_ms: number | null;
  started_at: string;
  [extra: string]: unknown;
}

function nonNegativeInt(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a number`);
  }
  const n = Math.trunc(value);
  if (n < 0) throw new RangeError(`${label} must be non-negative`);
  return n;
}

function optionalNonNegative(value: unknown, label: string): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative number or null`);
  }
  return value;
}

function newCallId(): string {
  const c = globalThis.crypto as { randomUUID?: () => string } | undefined;
  if (c?.randomUUID) return c.randomUUID();
  // Fallback for runtimes without WebCrypto: time + random, unique enough per session.
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 14)}`;
}

/** Validate and normalise a call into the wire record. Throws on bad input. */
export function toLlmCallRecord(input: RecordLlmCallInput, startedAt?: Date): LlmCallRecord {
  if (typeof input?.model !== "string" || !input.model) {
    throw new TypeError("model must be a non-empty string");
  }
  const record: LlmCallRecord = {
    call_id: input.callId || newCallId(),
    model: input.model,
    provider: input.provider ?? "",
    input_tokens: nonNegativeInt(input.inputTokens, "inputTokens"),
    output_tokens: nonNegativeInt(input.outputTokens, "outputTokens"),
    cost_usd: optionalNonNegative(input.costUsd, "costUsd"),
    latency_ms: optionalNonNegative(input.latencyMs, "latencyMs"),
    started_at: (startedAt ?? new Date()).toISOString(),
  };
  if (input.error) record.error = String(input.error);
  return record;
}

/**
 * The `session_metadata` object sent with an outcome: the two collections when
 * they have content, nothing otherwise (so a server that predates them sees
 * the request it always saw).
 */
export function buildSessionMetadata(
  attributes: SessionAttributes,
  llmCalls: LlmCallRecord[],
): Record<string, unknown> | undefined {
  const out: Record<string, unknown> = {};
  if (Object.keys(attributes).length) out.attributes = { ...attributes };
  if (llmCalls.length) out.llm_calls = llmCalls.map((c) => ({ ...c }));
  return Object.keys(out).length ? out : undefined;
}
