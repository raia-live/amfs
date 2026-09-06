/**
 * `instrumentOpenAI(client, memory)` — every `chat.completions.create` and
 * `responses.create` on the `openai` package's client is recorded with
 * `memory.recordLlmCall` (model, tokens, latency, provider).
 *
 * Streaming calls (`stream: true`) are recorded when the stream ends; the Chat
 * Completions API only sends token counts on the final chunk when the request
 * has `stream_options: { include_usage: true }` — without it the call is still
 * recorded, with a model and latency but zero tokens. `openai` is an optional
 * peer and is not imported here.
 */

import type { Extractor, LlmCallSink, Usage } from "./common.js";
import { asInt, get, isInstrumented, markInstrumented, patchMethod, resolve } from "./common.js";

const PROVIDER = "openai";

export const extractChatCompletion: Extractor = (response, usage) => {
  const model = get(response, "model");
  if (typeof model === "string" && model) usage.model = model;
  const u = get(response, "usage");
  if (u) {
    const inTok = asInt(get(u, "prompt_tokens"));
    const outTok = asInt(get(u, "completion_tokens"));
    if (inTok !== undefined) usage.inputTokens = inTok;
    if (outTok !== undefined) usage.outputTokens = outTok;
  }
};

export const extractResponse: Extractor = (response, usage) => {
  const model = get(response, "model");
  if (typeof model === "string" && model) usage.model = model;
  const u = get(response, "usage");
  if (u) {
    const inTok = asInt(get(u, "input_tokens"));
    const outTok = asInt(get(u, "output_tokens"));
    if (inTok !== undefined) usage.inputTokens = inTok;
    if (outTok !== undefined) usage.outputTokens = outTok;
  }
};

/** Responses API stream: the terminal event carries the whole response. */
export const extractResponseEvent: Extractor = (event, usage: Usage) => {
  const type = String(get(event, "type") ?? "");
  if (type === "response.completed" || type === "response.incomplete" || type === "response.failed") {
    extractResponse(get(event, "response"), usage);
  } else if (type === "response.created") {
    const model = get(get(event, "response"), "model");
    if (typeof model === "string" && model) usage.model = model;
  }
};

/**
 * Instrument an OpenAI (or AzureOpenAI) client in place and return it. Every
 * call made through it is recorded on `memory` until the next
 * `commitOutcomeAsync`, which ships the calls with the outcome. Idempotent.
 *
 * ```ts
 * const openai = instrumentOpenAI(new OpenAI(), memory);
 * const r = await openai.chat.completions.create({ model: "gpt-4o-mini", messages });
 * await memory.commitOutcomeAsync("deploy-42", OutcomeType.SUCCESS);
 * ```
 */
export function instrumentOpenAI<C>(client: C, memory: LlmCallSink): C {
  if (isInstrumented(client)) return client;
  const completions = resolve(client, ["chat", "completions"]);
  if (completions) {
    patchMethod(completions, "create", {
      provider: PROVIDER,
      sink: memory,
      extract: extractChatCompletion,
      onChunk: extractChatCompletion,
    });
  }
  const responses = resolve(client, ["responses"]);
  if (responses) {
    patchMethod(responses, "create", {
      provider: PROVIDER,
      sink: memory,
      extract: extractResponse,
      onChunk: extractResponseEvent,
    });
  }
  markInstrumented(client);
  return client;
}
