# A routed agent stated a fabricated fact with full confidence, and nothing recorded that it did

For risk, compliance, and security leaders whose organizations run agents behind a model router. Full
method, numbers, and limitations:
[WHITEPAPER.md](https://github.com/raia-live/amfs/blob/main/examples/openrouter-memory/WHITEPAPER.md).

## The result, first

We stated a fact to an agent through OpenRouter, then asked a follow-up after the router had moved it to
a different vendor's model. With no memory it answered correctly **12.5% of the time** (18 of 144 runs,
95% CI [8.1%, 18.9%]). Asked for a database version it had been told was PostgreSQL 15, one run
confidently returned a wrong, specific value. For an automated decision this is the worst kind of error:
indistinguishable from a correct answer unless you already know the truth, and it raises no error for
anyone to catch.

## Why it happens, and why it is a control problem

An LLM call is stateless, so when a router spreads calls across vendors, a fact stated under one model
is absent for the next, and the model fills the gap with an invented answer. A capability control rules
out the obvious alternative: hand the same models the fact and they answer correctly 98.6% of the time.
This is missing information, not a weak model.

Now hold that against the question you must answer about any output: what did the system rely on, and
can you prove it. On a router you cannot. You have the prompt and the completion in a log, and a log is
not provenance. You cannot show which of several models produced the answer, what it relied on, or
whether what it relied on was even real. The fabrication above would enter your records as a normal,
confident statement, with nothing marking it as unsupported.

## Memory alone does not fix this — verifiable memory does

Here is the finding that matters for procurement: adding a memory layer restores recall, but **recall is
not the control you need.** We tested four memory layers on the same dataset (n=144); a vector store,
mem0, and SenseLab all reach ~100% recall, Supermemory 86%. On the numbers that satisfy an auditor, they are
not equivalent at all.

An LLM memory layer helps your obligations in two ways, and only one vendor in this test delivered the
second.

**1. It decides on recorded facts, and can decline instead of guessing.** SenseLab injects only entries
above a relevance floor; below it, the agent is told to answer that it does not have the fact, so a
genuine gap surfaces as "I don't have that" rather than as a fabrication. This is a tunable control with
a documented tradeoff (raising the floor increases refusals-on-miss but lowers recall), not a switch we
can claim is free — the whitepaper's section 4.4 shows the exact curve.

**2. It captures verifiable provenance at decision time, for every output.** This is the part a log
cannot give you and the other memory layers do not produce. A real, sealed record from the run:

```json
{
  "produced_by": "mistralai/mistral-nemo", "cost_usd": 1.29e-06,
  "based_on": [{"key": ".../fact-fb367271", "confidence": 0.9, "semantic": 0.818}],
  "content_hash": "45ce6cc19207a371...",
  "signature": "109bd0917fdc7fce...", "signing_key_id": "hmac-v1",
  "sequence_number": 0, "parent_hash": null,
  "trace_verified": true
}
```

What we tested on this artifact, not asserted:

- **Coverage.** A signed, verified trace for **144 of 144** outputs (100%), each naming the responsible
  model.
- **Tamper-evidence.** We altered a sealed trace; re-verification failed with a `content_hash mismatch`
  and a signature-verification failure. The clean trace verified, the altered one did not.
- **Chain integrity.** Sequential traces link by `parent_hash` into a chain that verified end-to-end —
  an ordered, gap-detecting record, not an append-only log you have to trust.
- **Write-time validation.** The safety validator blocked a write that contradicted an existing
  high-confidence fact, and flagged a low-confidence write, before commit.

The claim you can check without trusting us: a vector store, mem0, and Supermemory return retrieved text
and a completion. **None emit a signed, hashed, model-attributed, verifiable causal trace.** In this
run, tamper-evident decision traces produced: 144 for SenseLab, 0 for the other three. That is structural,
not a configuration gap.

Be precise about scope. Hashing proves a stored value was not changed after the fact; it does not prove
the value was true when written. A false statement written at high confidence can still be stored and
later relied on. Write-side truth remains a control you own; the validator narrows the gap, it does not
close it.

## What you risk by leaving this alone

Decisions made on missing or fabricated information, with no way to detect it at the time and no way to
prove afterward what informed any given output. Audit requests you cannot satisfy, because the inputs
were never captured and cannot be reconstructed. And decision-relevant memory scattered across whichever
providers your router used, working against data-residency and vendor-exit requirements.

## What it is, and is not

This captures, retains, and cryptographically verifies decision provenance and memory integrity, and it
supports record-keeping and traceability obligations. It is not, by itself, certification against any
regime, and mapping these capabilities to your specific obligations is your work. Two scoping notes: the
signing key in this example is a development HMAC key (`signing_key_id: hmac-v1`) — production sets a
managed key, same mechanism; and the study is one live run on one dataset. The recall figures could
shift across runs; the governance results are structural and would not.

SenseLab is available as a hosted service at [amfs.sense-lab.ai](https://amfs.sense-lab.ai). Full
method, numbers, and the evaluation harness behind them are in the
[whitepaper](https://github.com/raia-live/amfs/blob/main/examples/openrouter-memory/WHITEPAPER.md).
