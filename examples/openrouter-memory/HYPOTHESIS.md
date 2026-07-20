# Hypothesis — Memory-augmenting OpenRouter proxy (shape C)

## The idea

OpenRouter routes each call to whatever model is best/cheapest/available. Provider-native
memory is locked to one provider and is lost the moment the route changes. AMFS can be the
external, provider-neutral memory layer so the model becomes swappable while what the agent
*knows* follows it.

OpenRouter is a **stateless completions gateway** with no memory hook, so a bridge is
required. Shape C is a drop-in OpenAI-compatible proxy: point `base_url` at AMFS instead of
OpenRouter, and memory injection + persistence happens transparently.

## The pain being tested

1. Provider lock-in of memory + loss of continuity under multi-model routing.
2. No audit trail of *what the agent remembered* and *which model acted on it*.

## Competitive scan (need-signal) — done before building

The generic version of shape C is already commoditized:

- **Supermemory Memory Router** — transparent OpenAI-compatible proxy; prepend a URL, zero
  code change, header-scoped (`x-sm-user-id`), auto inject/retrieve, pass-through fallback.
  Its supported-providers list includes `openrouter.ai/api/v1` as "Fully Supported".
- **mem0** (~48k stars, YC) — provider-agnostic memory layer; tagline "the memory follows the
  user, not the framework"; SOC2/HIPAA; "every read and write logged". Library/API, not a proxy.
- **Zep** — bi-temporal knowledge graph (Graphiti), leads temporal-recall benchmarks.
- **Letta** — stateful agent runtime with self-editing tiered memory.
- **OpenRouter native** — Agent SDK `StateAccessor` persists conversation state; Auto Router
  session stickiness pins the model per conversation (partially defuses mid-conversation switches).

**Conclusion:** a me-too "memory follows across models" proxy is a **no-go** — already solved.

## The narrowed hypothesis this spike tests

> AMFS's differentiator is not personalization recall (incumbents win that, incl. benchmarks).
> It is **versioned, causal, git-like memory with decision traces** — confidence, outcome
> back-propagation, integrity hashing, and model attribution. A proxy that injects memory AND
> emits an auditable trace of "answer -> which remembered facts -> which routed model" gives
> multi-model / regulated teams something Supermemory and mem0 do not.

So the benchmark measures the **audit/governance differentiator**, not generic recall.

## Go/no-go thresholds

- Technical: memory survives a forced model switch (fact stated via model A, answered via model B).
- Differentiator: we can reconstruct, after the fact, which memory entry (with version +
  confidence) and which routed model produced a given answer.
- Overhead: injection adds acceptable latency/tokens (sanity check, not a hard SLA in the spike).
