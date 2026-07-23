# OpenRouter + AMFS memory proxy (validation spike)

A throwaway spike that tests whether a **memory-augmenting OpenRouter proxy** (shape C) is
worth building for AMFS. See [HYPOTHESIS.md](HYPOTHESIS.md) for the reasoning and competitive
scan, and [RESULTS.md](RESULTS.md) for the findings and go/no-go.

This is **not** production code and touches no AMFS packages — it reuses the `amfs` SDK and the
filesystem adapter only.

## What it does

`proxy.py` exposes an OpenAI-compatible `POST /v1/chat/completions`. Point your client's
`base_url` at it instead of OpenRouter. Per request it:

1. Reads scope from headers `X-AMFS-Agent` and `X-AMFS-Entity`.
2. Retrieves scoped memory (lexical ranking — no embedder) and injects it as a system message.
3. Forwards to OpenRouter (`stream` supported), passing the inbound `Authorization` key through.
4. Writes back a naive observation, records the **routed model + cost** as external context.
5. Persists an audit record (`audit.jsonl`) reconstructing *answer -> memory used -> routed model*,
   and commits an AMFS decision-trace outcome.

## Run

```bash
export OPENROUTER_API_KEY=sk-or-...        # never commit this
python proxy.py                            # serves on 127.0.0.1:8088
# in another shell:
export OPENROUTER_API_KEY=sk-or-...
python benchmark.py
```

Requires only packages already present in this repo's env: `fastapi`, `uvicorn`, `requests`,
plus the `amfs` SDK. No `openai`/`httpx` needed.

## Files

- `proxy.py` — the memory-augmenting proxy.
- `benchmark.py` — forces a model switch across two sessions and grades whether memory followed;
  prints the audit trail.
- `audit.jsonl` / `.data/` / `benchmark_results.json` — generated locally, gitignored.
