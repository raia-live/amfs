# Results — Memory-augmenting OpenRouter proxy spike

Live run against OpenRouter. Session 1 states a fact via `openai/gpt-4o-mini`; session 2 is a
fresh context forced to a different vendor (`mistralai/mistral-nemo`) and asked a question that
requires the fact.

## Does memory follow across the model switch?

| Arm | Remembered? | Session-2 answer | Latency |
|-----|-------------|------------------|---------|
| no-memory (direct OpenRouter) | No | "MySQL 5.7.31 ... deploy every Thursday" (hallucinated) | ~1386 ms |
| proxy (AMFS memory follows) | Yes | "PostgreSQL 15 ... Fridays" (correct) | ~915 ms |

The no-memory arm confirms the pain: with routing across models, a fact stated to one model is
gone when another model answers. The proxy makes memory follow across the switch. (Proxy latency
was lower here only because the answer was shorter/decisive; injection overhead is small but not
what this spike optimizes.)

## The differentiator: reconstructable decision trace

For the proxy answer, AMFS reconstructs exactly what produced it:

```
outcome_ref : req-1784402129846
routed_model: mistralai/mistral-nemo  (requested: mistralai/mistral-nemo)
cost_usd    : 1.716e-06   latency_ms: 908.8
memory used in this answer:
  - bench/run-.../obs-... v1 (conf 0.8): "our production database is PostgreSQL 15 ... Fridays"
context     : [routed-model] model=mistralai/mistral-nemo cost_usd=1.716e-06 tokens=77+11
```

Each remembered fact carries a **version, confidence, and integrity hash**, and the trace links
the answer to the exact memory entry + the routed model + cost/tokens. This "answer -> which
remembered fact -> which routed model" reconstruction is the thing generic memory proxies
(Supermemory, mem0) do not expose.

## Honest limitations observed

- **Naive write-back stores noise.** The heuristic ("not ending in `?`") stored the user's
  *question* as a memory too, because it ended in a period. Extraction quality is the real hard
  problem and was deliberately out of scope. This is the single biggest thing a real build must
  solve, and incumbents already invest heavily here.
- **No native trace persistence on the filesystem adapter.** `commit_outcome` calls
  `adapter.save_trace`, which the filesystem adapter does not implement, so we persisted the
  audit trail to `audit.jsonl` and used live `explain()`. SenseLab's immutable, signed traces are
  what would back this in production.
- **No embedder** — retrieval is lexical overlap. Fine for the demo; real retrieval needs an
  embedder wired in.
- **Single-trial, single scenario.** Enough to prove the mechanism, not to make reliability claims.
- **Critical-path exposure.** The proxy sits inline with inference; a real build inherits
  latency/reliability responsibility.

## Recommendation

**Conditional go — but not as a generic memory proxy.**

- Do **not** ship a me-too "memory follows across models" proxy. Supermemory already does exactly
  this (with OpenRouter support) and mem0 owns the mindshare. We would be catching up on their turf.
- The **only** defensible wedge is the one this spike demonstrated: an **auditable, versioned
  decision-trace proxy** — "prove which remembered fact and which routed model produced each
  answer, with confidence and integrity." That maps to SenseLab's existing trace/governance story
  and targets multi-model-routing / regulated teams, not personalization.

If we proceed, the follow-up build is: productionize this as an experimental, flag-gated
`/v1/chat/completions` router in `packages/http-server`, back the trace with SenseLab's persisted
signed traces (not JSONL), wire a real embedder, and replace naive write-back with proper
extraction. If we can't commit to the governance positioning, this is a **no-go** — the generic
space is taken.
