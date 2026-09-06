/**
 * `instrumentAnthropic(client, memory)` — every `messages.create` on the
 * `@anthropic-ai/sdk` client is recorded with `memory.recordLlmCall`.
 *
 * Streaming calls (`stream: true`) are recorded when the stream ends: input
 * tokens come from `message_start`, output tokens from the final
 * `message_delta`. The `messages.stream()` helper is not patched; iterate
 * `create({ stream: true })` or call `memory.recordLlmCall` by hand.
 * `@anthropic-ai/sdk` is an optional peer and is not imported here.
 */

import type { Extractor, LlmCallSink, Usage } from "./common.js";
import { asInt, get, isInstrumented, markInstrumented, patchMethod, resolve } from "./common.js";

const PROVIDER = "anthropic";

function readUsage(u: unknown, usage: Usage, outputOnly = false): void {
  if (!u) return;
  if (!outputOnly) {
    const inTok = asInt(get(u, "input_tokens"));
    if (inTok !== undefined) usage.inputTokens = inTok;
  }
  const outTok = asInt(get(u, "output_tokens"));
  if (outTok !== undefined) usage.outputTokens = outTok;
}

export const extractMessage: Extractor = (message, usage) => {
  const model = get(message, "model");
  if (typeof model === "string" && model) usage.model = model;
  readUsage(get(message, "usage"), usage);
};

export const extractMessageEvent: Extractor = (event, usage) => {
  const type = String(get(event, "type") ?? "");
  if (type === "message_start") extractMessage(get(event, "message"), usage);
  else if (type === "message_delta") readUsage(get(event, "usage"), usage, true);
};

/** Instrument an Anthropic client in place and return it. Idempotent. */
export function instrumentAnthropic<C>(client: C, memory: LlmCallSink): C {
  if (isInstrumented(client)) return client;
  const messages = resolve(client, ["messages"]);
  if (messages) {
    patchMethod(messages, "create", {
      provider: PROVIDER,
      sink: memory,
      extract: extractMessage,
      onChunk: extractMessageEvent,
    });
  }
  markInstrumented(client);
  return client;
}
