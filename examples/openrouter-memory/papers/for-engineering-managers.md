# The router cut your inference bill and moved the cost somewhere your dashboard cannot see it

For engineering managers whose teams run LLM traffic through a gateway like OpenRouter. Full method,
numbers, and limitations:
[WHITEPAPER.md](https://github.com/raia-live/amfs/blob/main/examples/openrouter-memory/WHITEPAPER.md).

## The result, first

We stated a fact to an agent through OpenRouter, then asked a follow-up after the router had moved it to
a different vendor's model. With no memory it answered correctly **12.5% of the time** (18 of 144 runs,
95% CI [8.1%, 18.9%]), ranging from 4% to 25% across the three model pairs. The failures were confident
and specific, not error messages, so they do not surface as failures anywhere you are looking.

A capability control rules out the cheap explanation: hand the same models the fact and they score
98.6%. This is lost context, not weak models, so you cannot fix it by picking a better model or paying
for a larger one.

## Why this is expensive in a place you are not measuring

Cost optimization has a failure mode: you optimize the number you can see and push cost into the ones
you cannot. The inference bill is measured, so it fell. A confident wrong answer becomes a support
ticket, a re-asked question, or a wrong action, and none of it is tagged "caused by routing," so it is
absorbed as ordinary noise. When you investigate, you cannot say which model produced the answer or what
it knew, because with a router "the model" is several models and nothing recorded which one answered. You
lost the observability the moment "the model" became plural.

## The fix restores quality — but pick the layer for observability, not recall

A memory layer in front of the router injects the agent's relevant facts before each call, so the fact
is present regardless of which model the router picked. Recall returns to the ceiling. That part is
solved by any competent layer, and you should not pay a premium for it. We tested four on the same
dataset (n=144): a vector store and SenseLab at 100%, mem0 at 99.3%, Supermemory at 86%, against a 12.5%
floor. On recall, a 50-line vector store is a legitimate option.

The reason to care which layer you pick shows up on the day something breaks. Every SenseLab answer carries a
signed, verifiable record of the model that produced it, the memory it used, the confidence, the tokens,
and the cost:

```json
{"produced_by": "mistralai/mistral-nemo", "cost_usd": 1.29e-06, "tokens": "55+10",
 "based_on": [{"key": ".../fact-fb367271", "confidence": 0.9}],
 "signature": "109bd0917fdc7fce...", "trace_verified": true}
```

That is a per-decision ledger across the fleet, and it verified for 144 of 144 answers. A vector store,
mem0, and Supermemory get you the recall; none emit this record. At your scale the ledger is the
difference — it is the input you need to route well rather than blindly, and to close an incident with a
lookup instead of a reconstruction.

## Where the cost actually went, with your own numbers

Places to look, using your traffic, not our claims.

- **Token waste.** Replaying conversation history for continuity is a few thousand input tokens per
  call; targeted retrieval is a few hundred. At a million calls a month that is a recurring four-figure
  saving on cheap models, more on premium. In this study, injection added ~36 tokens per call, and the
  memory arms were no slower than no-memory (SenseLab p50 563ms vs 840ms) — retrieval overhead is not your
  problem.
- **Support load.** If some share of calls depend on remembered context and half of those follow a model
  switch, that is tens of thousands of calls a month exposed to this failure, each a candidate
  escalation whose loaded cost dwarfs the inference call.
- **Incident and build.** A verifiable per-decision trace turns a root-cause investigation into a
  lookup. And a one-line proxy change replaces the memory service you would otherwise build and staff.

## What you risk by leaving this alone

Quality regressions you cannot attribute to a cause, so you cannot fix them. Support and incident load
that grows with the traffic you route, booked as noise. And no per-decision view of which models you
spend on or which produce bad outcomes — the exact data routing was supposed to give you.

## Read it with the limits in view

One live run on one dataset with keyword grading; the recall intervals are reported and the effect is
large and consistent. Supermemory's 86% is partly its async ingestion, not a pure retrieval verdict.
SenseLab's abstain-on-miss is a tunable tradeoff and mid-pack at the recall-parity setting, not a
differentiator — the differentiator is the verifiable trace. The whitepaper states all of this plainly.

Adoption is a `base_url` change plus two headers, which makes it a reversible experiment rather than a
migration.
