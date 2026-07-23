# A routed agent fabricates most of the time, and no memory layer catches it

### A head-to-head study of memory across model switches, and of what each layer can prove afterward

A reproducible study using SenseLab as a drop-in layer in front of OpenRouter,
measured against a vector store, mem0, and Supermemory on the same dataset and grader. The harness and
raw results are in the repository:
[examples/openrouter-memory](https://github.com/raia-live/amfs/tree/main/examples/openrouter-memory).

## The finding, first

We took one agent, told it an operational fact through OpenRouter, then asked a follow-up after the
router had moved the agent to a different vendor's model. Nothing but the model changed. With no memory
layer the agent answered correctly **12.5% of the time** (18 of 144 runs, 95% CI [8.1%, 18.9%]).

The 87.5% that were wrong is the part to forward to your team. The failures were not errors or
refusals. They were confident, specific, invented answers. Asked for a database version it had been
told was PostgreSQL 15, one run replied `5.7.31-0ubuntu0.18.04.1` — a real-looking MySQL string — and
named the wrong deploy day, with no hedge. Nothing in the response signals a guess, and nothing appears
in a log as an error. It looks like an answer.

Two controls make the cause unambiguous. Hand the same models the fact inside the prompt and they
answer correctly 98.6% of the time: the models are capable, they just lack the fact. Put any competent
memory layer in front of the router and recall returns to the ceiling. That last point is the one most
memory vendors stop at, and it is where this study starts, because on recall the memory layers are
interchangeable. The difference that survives is what each one can *prove* about the answer afterward.

Every claim below has a checkable form. The code, the grader, and the raw JSON are linked so you can
re-run it and disagree with any step.

## 1. The mechanism of the failure

"Agents forget" is too vague to act on. The failure decomposes into four steps, each verifiable:

1. An LLM API call is stateless. The model retains nothing between calls; continuity comes only from
   what you resend in the request.
2. A router sends call N+1 to a different provider than call N, chosen on price or availability. That
   is the purpose of a gateway like OpenRouter, not a misconfiguration.
3. Nothing carries prior state across that hop. There is no shared context window between two separate
   requests, and provider-native memory is scoped to one provider, so it does not follow.
4. When the needed fact is absent, current models do not abstain. They complete the pattern with a
   fluent, plausible value. Absence becomes fabrication rather than "I don't know."

Steps 1 through 3 are properties of the stack you can confirm from the API contracts. Step 4 is the
empirical one, and section 4 measures it at 87.5%.

Why this is dangerous and not merely wrong: a fabricated answer is indistinguishable from a correct one
without the ground truth, and it raises no exception, so it is invisible to the tools you would use to
catch a failure. It surfaces downstream as a wrong action or a customer complaint, detached from its
cause.

## 2. Who this applies to, and who it does not

This is a problem if you route one agent's traffic across multiple model providers, the agent relies on
facts established earlier, and you may have to explain or prove a decision after the fact.

It is not your problem if you run a single model from a single vendor inside one context window, or if
your agent is stateless by design. There, provider-native context is enough and a memory layer only
adds moving parts. That boundary is part of the claim, not a hedge.

## 3. Method

The honest baseline is not "does memory beat no memory" — everyone knows it does. It is "does this beat
the memory layers people already run, and does it give you something they cannot?" So we ran six
conditions against the same dataset and grader.

**Dataset.** 8 durable operational facts (production database version and deploy day; on-call engineer
and extension; region; internal billing API URL; SEV-1 acknowledgement window; key-rotation cadence;
top customer and renewal date; merge-approval rule). Each has a question and keyword-based grading.

**Model pairs**, writer then reader, always crossing vendors:
`openai/gpt-4o-mini -> mistralai/mistral-nemo`,
`mistralai/mistral-nemo -> amazon/nova-micro-v1`,
`amazon/nova-micro-v1 -> cohere/command-r7b-12-2024`.

**Trials.** 6 per (fact × pair), giving **144 evaluations per condition**, 48 per model pair.

**Conditions (6):**

- **no-memory** — reader answers with no prior context. The fabrication floor.
- **in-context** — the fact is pasted into the reader's prompt. A capability ceiling that isolates
  memory transfer as the only variable.
- **vector** — a local embedding store (`fastembed`, bge-small) with top-k cosine retrieval. The RAG
  baseline an engineer builds in an afternoon.
- **mem0** — the hosted mem0 platform, seeded and queried through its live API.
- **supermemory** — the hosted Supermemory platform, seeded and queried through its live API.
- **SenseLab** — multi-strategy retrieval (semantic + BM25 + confidence) with a confidence
  and semantic-relevance gate, write-time safety validation, and a signed, sealed decision trace per
  answer.

**Grading.** Keyword match against the known answer, the same rule for every condition. Wilson score
intervals for all proportions.

All numbers below are from one live run against the hosted services. Raw file:
[pro_results.json](https://github.com/raia-live/amfs/blob/main/examples/openrouter-memory/pro_results.json).

## 4. Results

### 4.1 Recall across the model switch (n = 144 per condition)

| Condition | Rate | 95% CI (Wilson) |
|-----------|------|-----------------|
| no-memory (fabrication floor) | 12.5% | [8.1%, 18.9%] |
| in-context (capability ceiling) | 98.6% | [95.1%, 99.6%] |
| vector store (fastembed) | 100% | [97.4%, 100%] |
| mem0 (hosted) | 99.3% | [96.2%, 99.9%] |
| Supermemory (hosted) | 86.1% | [79.5%, 90.8%] |
| **SenseLab** | **100%** | [97.4%, 100%] |

The headline is deliberately unflattering to a "we remember better" pitch: **every serious memory
layer clears the bar, and SenseLab is at the top of a tight pack, not alone.** Vector and SenseLab hit 100%,
mem0 99.3%. Supermemory landed lower here (86.1%), but read that carefully — its ingestion is
asynchronous, and in several scopes a just-written fact was not yet searchable when the question fired.
That is a real operational property worth knowing, not evidence that its retrieval is weak. The
defensible conclusion is that recall is table stakes; the memory layers are close enough that recall
alone does not separate them.

Per model pair (n = 48 each), to show failures do not cluster:

| Pair (writer → reader) | no-memory | SenseLab |
|------------------------|-----------|----------|
| gpt-4o-mini → mistral-nemo | 8.3% | 100% |
| mistral-nemo → nova-micro | 4.2% | 100% |
| nova-micro → command-r7b | 25.0% | 100% |

The recall gap between no-memory and every memory arm is large and consistent across all three pairs,
so the direction is not in doubt.

### 4.2 Overhead (p50 / p95 reader latency, mean prompt tokens)

| Condition | p50 | p95 | Mean prompt tokens |
|-----------|-----|-----|--------------------|
| no-memory | 840 ms | 2284 ms | 16.7 |
| in-context | 614 ms | 1501 ms | 35.7 |
| vector | 580 ms | 1713 ms | 51.9 |
| mem0 | 603 ms | 1637 ms | 56.9 |
| Supermemory | 624 ms | 2170 ms | 55.9 |
| SenseLab | 563 ms | 1644 ms | 52.3 |

Injecting a retrieved fact adds ~36 prompt tokens on these short prompts. Model-side latency dominates
and swamps every memory layer's own cost, which is why the memory arms are not slower than no-memory.
We do not claim a latency win; the point is that the memory layers, SenseLab included, are in the same
latency band. Retrieval overhead is not the reason to pick one.

### 4.3 Abstain-on-miss (n = 48 per condition)

We seeded the 8 facts, then asked 8 questions whose answers were **never stored** (AWS root email, WiFi
password, Redis version, and so on) and measured how often each arm declined instead of inventing an
answer.

| Condition | Declines on a miss |
|-----------|--------------------|
| no-memory | 12.5% |
| Supermemory | 37.5% |
| SenseLab (gate = 0.55) | 41.7% |
| vector | 47.9% |
| mem0 | 56.2% |

Here is the honest part: **at the gate setting that maximizes recall, SenseLab does not win abstention** —
it sits mid-pack, below mem0 and the vector store. Abstention is not a free property you get by
choosing SenseLab; it is a knob, and section 6 shows the knob's cost.

### 4.4 The gate tradeoff (deterministic, no LLM calls)

SenseLab's semantic-relevance floor (`SEM_FLOOR`) decides whether a retrieved entry is relevant enough to
inject. Because that decision is purely a function of the embeddings, we can sweep it exactly:

| SEM_FLOOR | Recall gate-pass | Miss-abstain |
|-----------|------------------|--------------|
| ≤ 0.50 | 100% | 0% |
| 0.55 | 100% | 25% |
| 0.60 | 87.5% | 62.5% |
| 0.65 | 87.5% | 75% |
| 0.70 | 87.5% | 100% |
| 0.75 | 75% | 100% |

The two curves move against each other, and the classes overlap: the hardest true question–fact pair
has similarity 0.592, while the worst miss-question's nearest stored fact scores 0.686. So **no single
global threshold cleanly separates a real question from a miss** on this embedder. You can tune SenseLab to
abstain on 100% of misses (floor 0.70) at the cost of dropping recall to 87.5%, or tune for full recall
and accept that a quarter of misses leak through. This is a genuine precision/recall tradeoff, stated
plainly. A per-entry or learned threshold, or a stronger embedder, would separate the classes better;
that is future work, not a shipped claim.

## 5. The worked example

Writer call to `openai/gpt-4o-mini`:

```
Our production database is PostgreSQL 15 and we only deploy on Fridays.
```

Reader call, routed to `mistralai/mistral-nemo`, no memory carried across:

```
Q: What version is our production database, and which day do we deploy?
A: Current Production DB Version: 5.7.31-0ubuntu0.18.04.1
   Deployment Day: Every Thursday
```

Same reader model, same question, with SenseLab carrying the fact across:

```
A: Production Database: PostgreSQL 15
   Deployment Day: Fridays
```

## 6. The differentiator, and how to check it

Recall parity is where the memory market lives. The part that is hard to get elsewhere is a per-answer
record that ties the output to specific, versioned, hashed facts and to the model that produced it, and
that can be **cryptographically verified and shown to have detected tampering.** For every one of the
144 SenseLab answers, the proxy sealed a record like this (real, from the run):

```json
{
  "outcome_ref": "req-0a83a424a0",
  "routed_model": "mistralai/mistral-nemo",
  "cost_usd": 1.29e-06,
  "prompt_tokens": 55, "completion_tokens": 10,
  "memory_used": [{"key": ".../fact-fb367271", "semantic": 0.818, "confidence": 0.9}],
  "content_hash": "45ce6cc19207a371...",
  "signature": "109bd0917fdc7fce...", "signing_key_id": "hmac-v1",
  "parent_hash": null, "sequence_number": 0,
  "trace_verified": true,
  "answer_excerpt": "PostgreSQL 15, Fridays"
}
```

What we measured on this artifact, not asserted:

- **Coverage.** A signed, verified trace was produced for **144 of 144** answers (100%), each naming the
  model that produced it.
- **Tamper-evidence.** We altered a sealed trace and re-verified. Verification failed with a
  `content_hash mismatch` and a `signature verification failed` error. The clean trace verified; the
  altered one did not.
- **Chain integrity.** Sequential traces link by `parent_hash`/`sequence_number` into a Merkle-style
  chain that verified end-to-end.
- **Write-time safety.** The `MemorySafetyValidator` flagged a write that contradicted an existing
  high-confidence entry, and flagged a low-confidence write, before either was committed.

The claim you can check without trusting us: a vector store, mem0, and Supermemory return retrieved
text and a completion. **None emit a signed, content-hashed, model-attributed, verifiable causal trace
for the answer.** In this run, the count of tamper-evident decision traces produced by the four memory
arms was 144 for SenseLab and 0 for the other three. That is a structural property of what those systems
store, not a tuning gap, and it is the only dimension on which the arms were not close.

## 7. Limitations and failure modes

Stating where this breaks is the point of this section.

- **Abstention is a tradeoff, not a win.** At the recall-parity gate, SenseLab abstains on 41.7% of misses,
  behind mem0. Section 4.4 is the honest picture: you buy abstention with recall.
- **One dataset, one grader, one run.** 8 facts, keyword grading, a single live run. The recall effect
  is large and the CIs are reported, but a second independent run and a harder, adversarially-worded
  dataset would strengthen it. The governance results are structural and would not change across runs.
- **Supermemory's 86.1% is partly ingestion latency**, not a pure retrieval verdict. We warmed each
  hosted scope before querying, and some scopes still were not searchable in time. A longer warm window
  would likely raise it. We report what a caller who writes-then-reads would actually observe.
- **Write-time poisoning is only mitigated, not solved.** The safety validator catches contradictions
  and low-confidence writes, and hashing proves a stored value was not altered afterward. Neither proves
  a value was *true* when written. A confident, false statement can still be stored and later injected.
- **The signing key here is a dev HMAC key** (`signing_key_id: hmac-v1`). Production deployments set a
  managed key; the mechanism is identical, the key management is not exercised in this example.
- **Retrieval was scoped per fact**, so this measures cross-model transfer and gating, not retrieval
  precision when hundreds of facts compete. That regime is where the embedder and extraction quality
  matter most and is not stressed here.

## 8. How to adopt

SenseLab runs as a hosted service. Change one line and add two headers; your code stays on the OpenAI
SDK, and OpenRouter routing, pricing, and failover are unchanged. The layer is designed to fail open:
if it errors, the request still reaches the router and inference continues. Get a key at
[amfs.sense-lab.ai](https://amfs.sense-lab.ai).

```python
client = OpenAI(
    base_url="https://amfs.sense-lab.ai/v1",   # was https://openrouter.ai/api/v1
    api_key=OPENROUTER_KEY,
    default_headers={"X-AMFS-Agent": "support", "X-AMFS-Entity": "acme/customer-42"},
)
```

## 9. Reproduce the measurements

Trust this over any sentence above. The evaluation harness and raw outputs behind every number in this
paper are provided. Set the three keys and run:

```bash
export OPENROUTER_API_KEY=sk-or-...
export MEM0_API_KEY=...  SUPERMEMORY_API_KEY=...   # for the hosted comparison arms
python gate_sweep.py                     # deterministic gate tradeoff -> gate_sweep.json
PRO_TRIALS=6 PRO_MISS_TRIALS=6 PRO_SEM_FLOOR=0.55 python pro_benchmark.py   # -> pro_results.json
```

Harness and raw output: `pro_benchmark.py`, `gate_sweep.py`, `pro_results.json`, `gate_sweep.json`,
and `pro_audit.jsonl` (144 sealed traces).
Product and docs: [amfs.sense-lab.ai](https://amfs.sense-lab.ai).
