# A routed agent fabricates most of the time, and none of it shows up in a stack trace

For engineers running an agent behind OpenRouter or any OpenAI-compatible gateway. Full method, numbers,
and limitations:
[WHITEPAPER.md](https://github.com/raia-live/amfs/blob/main/examples/openrouter-memory/WHITEPAPER.md).
Commands to reproduce are at the end.

## The result, first

We told an agent a fact through OpenRouter, then asked a follow-up after the router had moved it to a
different vendor's model. Only the model changed. With no memory it answered correctly **12.5% of the
time** (18 of 144 runs, 95% CI [8.1%, 18.9%]).

The 87.5% that failed is the interesting part. Not errors, not refusals. Asked for a database version
it had been told was PostgreSQL 15, one run answered `5.7.31-0ubuntu0.18.04.1` — a real MySQL version
string — and gave the wrong deploy day, with no hedge. Your monitoring sees a `200` and a fluent
completion. There is nothing to alert on. It looks like an answer.

## Why it happens, precisely

Not "models are unreliable." Four steps you can each verify:

1. An LLM call is stateless. The model keeps nothing between calls; continuity is only what you resend.
2. A router sends the next call to a different provider. That is the feature, not a bug.
3. Nothing carries state across that hop. Separate requests share no context window, and
   provider-native memory is tied to one provider, so it does not follow.
4. With the fact absent, the model does not say "I don't know." It completes the pattern with a
   plausible value.

The control settles the cause: hand the same models the fact in the prompt and they score 98.6%. The
bug is not your prompt or your model choice — it is the gap between calls, which is exactly the place
you added a router.

## The fix is boring, and every memory layer does it

Put a memory layer in front of the router. It resolves who the agent is, retrieves the agent's stored
facts, and injects the relevant one as a system message before forwarding to OpenRouter. The fact is
present at inference time again, so recall returns to the ceiling. We tested four such layers on the
same dataset and grader (n=144 each):

| Layer | Recall | 95% CI |
|-------|--------|--------|
| no-memory | 12.5% | [8.1%, 18.9%] |
| vector store (fastembed) | 100% | [97.4%, 100%] |
| mem0 (hosted) | 99.3% | [96.2%, 99.9%] |
| Supermemory (hosted) | 86.1% | [79.5%, 90.8%] |
| SenseLab | 100% | [97.4%, 100%] |

Be suspicious of anyone selling you recall. On recall these are the same tool. SenseLab is tied at the top,
but a 50-line vector store gets you there too. If recall is all you need, build the vector store and
move on — this paper is about the thing that separates them, which shows up the day something goes
wrong.

## What you actually get from SenseLab that the others cannot give you

Two things, and both are checkable from their APIs, not our marketing.

**1. A verifiable record of every answer.** For all 144 SenseLab answers, the layer sealed a signed trace:

```json
{
  "routed_model": "mistralai/mistral-nemo",
  "memory_used": [{"key": ".../fact-fb367271", "semantic": 0.818, "confidence": 0.9}],
  "content_hash": "45ce6cc19207a371...",
  "signature": "109bd0917fdc7fce...", "signing_key_id": "hmac-v1",
  "trace_verified": true,
  "cost_usd": 1.29e-06, "prompt_tokens": 55, "completion_tokens": 10,
  "answer_excerpt": "PostgreSQL 15, Fridays"
}
```

When the next "why did it say that" lands, you have the exact fact it read, its confidence, the model
that actually answered, the cost, and a signature that proves the record was not edited afterward — as
a lookup, not an archaeology project. We altered a sealed trace and re-verification failed with a
`content_hash mismatch`. A vector store, mem0, and Supermemory hand back retrieved text and a
completion; none emit a signed, hashed, model-attributed trace. In this run: 144 tamper-evident traces
from SenseLab, 0 from the other three.

**2. A gate you control, with a cost you can see.** On a question whose answer was never stored, most
layers still inject their nearest match and let the model fabricate. SenseLab can refuse to inject below a
relevance floor and instruct the model to decline. But this is a knob, not free — and here is the honest
tradeoff, computed exactly from the embeddings:

| Relevance floor | Recall | Declines on a miss |
|-----------------|--------|--------------------|
| 0.55 | 100% | 25% |
| 0.60 | 87.5% | 62.5% |
| 0.70 | 87.5% | 100% |

Tune it toward full recall and a quarter of unanswerable questions leak through; tune it toward safety
and you drop recall. At the recall-parity setting, SenseLab abstains on 41.7% of misses — mid-pack, behind
mem0's 56.2%. We are not going to tell you SenseLab abstains best; it does not. We are telling you it is the
only one that lets you set the threshold and shows you what it costs.

## What you risk by leaving this alone

Confident-wrong answers you cannot reproduce, because "the model" was several models and you kept no
record of which one answered or what it knew. The usual alternative — build your own memory, or lean on
one provider's native memory — couples your agent's knowledge to a vendor, so adopting a newer model on
OpenRouter quietly resets what your agents know.

## Where it still breaks

Read the whitepaper's limitations. One dataset, keyword grading, one live run — the recall effect is
large and the CIs are reported, but a harder dataset would strengthen it. Supermemory's 86.1% is partly
its async ingestion, not a pure retrieval verdict. And a layer that writes back what it is told will
faithfully store a false statement; SenseLab's validator flags contradictions and low confidence, but
write-side truth is still on you.

## Adopt it, and reproduce the numbers

SenseLab is a hosted service — change one line and add two headers. You stay on the OpenAI SDK;
routing, pricing, and failover are unchanged, and the layer fails open (if it errors, the request
still reaches the router). Grab a key at [amfs.sense-lab.ai](https://amfs.sense-lab.ai).

```python
client = OpenAI(
    base_url="https://amfs.sense-lab.ai/v1",   # was https://openrouter.ai/api/v1
    api_key=OPENROUTER_KEY,
    default_headers={"X-AMFS-Agent": "support", "X-AMFS-Entity": "acme/customer-42"},
)
```

To reproduce the numbers in this paper, run the evaluation harness against the three services:

```bash
export OPENROUTER_API_KEY=sk-or-...  MEM0_API_KEY=...  SUPERMEMORY_API_KEY=...
python gate_sweep.py                                                    # gate tradeoff
PRO_TRIALS=6 PRO_MISS_TRIALS=6 python pro_benchmark.py                  # pro_results.json
```
