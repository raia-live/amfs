---
title: Replay Webhook
layout: default
parent: Guides
nav_order: 11
description: "Let SenseLab prove a repair against your own agent before it ships: run replay requests on a memory branch and commit graded outcomes."
---

# Replay Webhook
{: .no_toc }

Answer SenseLab's replay requests from your agent — the customer's end of the repair loop.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What a replay is

When SenseLab Pro finds a behaviour worth fixing — a judge that keeps failing
the same kind of session, an incident, a discredited memory — it drafts a
*repair*: corrective memory entries on a branch of your agent's memory,
`repair/<fix-id>`. Before that branch is merged into `main`, the loop wants
proof that the fix actually changes what the agent does.

Tier 1 of that proof replays the failing sessions through SenseLab's own
harness. Tier 2 asks **your** agent: one webhook per test case, each saying
*run this task with your memory on this branch and tell us how it went*. The
grader then reads the trace your agent committed, on that branch, and passes
or fails the case with the same judge that found the problem. A fix under an
`auto_after_replay` policy ships to `main` when the cases pass; under `review`
it waits for a person with the replay's verdict beside it.

This page is what your side has to do. The SDKs do almost all of it.

---

## The contract

Every case is one HTTP request to the URL you registered for the agent:

```
POST <replay_webhook_url>
Content-Type: application/json
X-AMFS-Event: replay_requested
X-AMFS-Delivery: <uuid, unique per request>
X-AMFS-Signature: sha256=<hex HMAC-SHA256(secret, raw body)>

{
  "event": "replay_requested",
  "delivery_id": "…",
  "sent_at": "2026-09-20T01:00:00+00:00",
  "fix_id": "0d1f3a6c-…",
  "agent_id": "support-agent",
  "branch": "repair/0d1f3a6c",
  "case_id": "9b8a7c6d-…",
  "case_set_id": "…",
  "task_input": "customer says the invoice email never arrived",
  "expected": {"action": "resolve:resend_email"},
  "deadline_at": "2026-09-20T07:00:00+00:00",
  "instructions": "…"
}
```

Your receiver must:

1. **Verify `X-AMFS-Signature`** over the raw body with the shared secret
   (when one is registered; an unsigned webhook is sent without the header).
2. **Run the agent on `task_input` with its memory on `branch`.** The branch
   holds the fix. A run on `main` never read it and proves nothing.
3. **Commit the outcome with `attributes.case_id` and
   `attributes.memory_branch`** set to the values you were sent, plus
   `task_input` and the agent's `response_text` so the judge has something to
   grade. Both attributes are required: the grader reads the newest trace
   tagged with the case *on that branch*.

Answer any `2xx` to accept. A `4xx` other than `429` tells the sender the
endpoint refused for good; anything else is retried with backoff, a bounded
number of times. The sender waits **ten seconds** for the response, so a
receiver that runs the agent inline will time out on any real task —
acknowledge first, run after. A `ping` event with the same headers is sent
from the settings page so a person can check the endpoint.

---

## Python

```python
import os
from amfs import AgentMemory
from amfs.replay import ReplayReceiver

def run(task_input: str, memory: AgentMemory) -> str:
    # `memory` is already on the request's branch: every read here sees the fix.
    return my_agent.answer(task_input, memory=memory)

receiver = ReplayReceiver(secret=os.environ["AMFS_REPLAY_SECRET"], run=run)
```

`receiver.handle(headers, raw_body)` returns `(status, json_dict)`, so it
mounts in any framework:

```python
# Flask
@app.post("/amfs/replay")
def replay():
    status, body = receiver.handle(dict(request.headers), request.get_data())
    return jsonify(body), status

# FastAPI
@app.post("/amfs/replay")
async def replay(request: Request):
    status, body = receiver.handle(request.headers, await request.body())
    return JSONResponse(body, status_code=status)
```

What the receiver does for you:

| Concern | Behaviour |
|:--------|:----------|
| Signature | Verified in constant time; a mismatch is `401` (the sender stops retrying). No secret → unsigned requests accepted. |
| Acknowledgement | `202` at once; the run happens on a worker thread. `background=False` runs inline and answers `200` with the outcome. |
| Duplicates | Delivery is at-least-once. A `delivery_id` seen before is answered `200` with the first answer and is **not** run again. |
| Deadline | A request past `deadline_at` is `410` — too late to be graded. |
| Memory | `AgentMemory(agent_id=<request's>, branch=<request's>)` from your process configuration; pass `memory_factory=` to build it yourself (a specific adapter, a different agent id). |
| Commit | `commit_outcome(f"replay:<fix>:<case>", outcome, task_input=…, response_text=…, attributes={"case_id", "memory_branch", "fix_id", "replay_delivery_id"})`. |
| A runner that raises | Committed as a `failure` with the error as the answer. A graded failure tells the loop more than a case that never came back. |

The runner may return a string (the answer, taken as a success), a
`(answer, outcome_type)` pair, a `ReplayResult`, or a dict with
`response_text`, `outcome_type`, `tool_calls` and `attributes`.

### Without a web framework

```bash
export AMFS_REPLAY_SECRET=whsec_…          # the secret registered in Settings
amfs replay serve --run my_agent.replay:run --port 8787
```

`amfs replay serve` starts a stdlib HTTP server around the runner you name
as `module:function`; `GET /amfs/replay` answers `200` for a health check.
`--inline` runs inside the request, `--insecure` accepts unsigned requests,
`--agent` sets the agent id used when a request names none.

To exercise the whole path without the SaaS side:

```bash
amfs replay simulate http://localhost:8787/amfs/replay --ping
amfs replay simulate http://localhost:8787/amfs/replay \
    --task "refund the duplicate charge" --branch repair/local-test
```

`simulate` signs a synthetic request with the same secret and prints the
receiver's answer; the committed trace carries `memory_branch =
repair/local-test`, which you can confirm with `amfs inspect`.

---

## TypeScript

The receiver is runtime-agnostic (Web Crypto for the signature), so it runs
on Node 18+, Bun, Deno and edge runtimes:

```ts
import { AgentMemory, HttpAdapter, ReplayReceiver, createFetchHandler, serveReplay } from "@senselab-ai/amfs";

const http = new HttpAdapter({ url: process.env.AMFS_HTTP_URL!, apiKey: process.env.AMFS_API_KEY });

const receiver = new ReplayReceiver({
  secret: process.env.AMFS_REPLAY_SECRET!,
  memoryFactory: (req) => new AgentMemory(req.agentId, { adapter: http, branch: req.branch }),
  run: async (taskInput, memory) => myAgent.answer(taskInput, memory),
});

// Next.js / Hono / Bun.serve / workers: a (Request) => Response handler
export const POST = createFetchHandler(receiver);

// Express
app.post("/amfs/replay", express.raw({ type: "*/*" }), async (req, res) => {
  const { status, body } = await receiver.handle(req.headers, req.body);
  res.status(status).json(body);
});

// Or a node:http server of its own
await serveReplay(receiver, { port: 8787 });
```

`memoryFactory` is required in TypeScript: the SDK's default adapter is
in-memory, and a replay needs the server's branch. The receiver otherwise
behaves as the Python one — `202` and a background run, duplicates
acknowledged, `410` past the deadline, a thrown runner committed as a
failure — and commits with the same attributes, `taskInput` and
`responseText`.

---

## Memory branches from the SDK

A replay is one case of a general thing: pointing an agent at a memory
branch other than `main`. Both SDKs do this at three levels:

| Level | Python | TypeScript |
|:------|:-------|:-----------|
| Construction | `AgentMemory(agent_id, branch="repair/x")` | `new AgentMemory(id, { branch: "repair/x" })` |
| Environment | `AMFS_BRANCH=repair/x` — a process is pointed at a branch with no code change; the MCP server honours it too | `AMFS_BRANCH` likewise |
| At runtime | `memory.checkout("repair/x")` / `memory.checkout(None)` back to `main` | `memory.checkout("repair/x")` / `memory.checkout(null)` |

Every read, search, retrieve, briefing and write goes to the memory's branch
unless the call names another. And every outcome committed while off `main`
carries **`attributes.memory_branch`** automatically, so the trace says which
memory it read — the same stamp the hosted gateway puts on a canary session.
This is what lets a canary and a replay be graded at all; a caller who sets
the attribute themselves wins.

---

## Registering the webhook

In SenseLab Pro, under the agent's settings: **Replay webhook** — the URL and
an optional secret. *Send ping* delivers a signed `ping` so you can confirm
the endpoint before a fix ever needs it. Fixes for that agent can then set a
repair policy of `auto_after_replay`, or you can request a replay by hand from
the fix's page.

The webhook secret is stored encrypted and opened only to sign a request;
rotate it by saving a new one, and update `AMFS_REPLAY_SECRET` on your side.
