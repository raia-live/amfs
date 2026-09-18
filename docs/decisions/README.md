# Decision API and reference runtime (preview)

The portable contract is `amfs_core.decisions`. Decision recommendations do not
execute actions. Customer training, outcome verification, deployment control,
risk calibration and account billing belong to the hosted service.

```python
from amfs import DecisionClient
from amfs_core.decisions import Candidate, DecisionRequest, Question

request = DecisionRequest(
    decision="ci-recovery", spec_version="v1", state={"job": "failed"},
    questions={"action": Question(
        instructions="Choose the next permitted recovery step.",
        candidates=[
            Candidate(id="inspect", description="Read diagnostic logs", kind="observation"),
            Candidate(id="review", description="Request operator review", kind="defer"),
        ],
    )},
)
with DecisionClient() as client:  # AMFS_HTTP_URL and AMFS_API_KEY
    result = client.decide(request, idempotency_key="ci-job-42-attempt-1")
    # Persist result.decision_id before executing any action.
    # result.automate is false without eligible calibrated routing evidence.
```

Reuse the same idempotency key after a timeout. Reusing it with different input
returns 409. Responses are acknowledged after durable capture. Outcome events
are separate records and distinguish executed, overridden and unexecuted
actions. A client's claim of external verification is not trusted verification.

Accounts can create multiple named models via `DecisionClient.create_model` and
list them via `list_models`. A model name resolves only inside the authenticated
account. New models are drafts; a hosted operator activates evaluated versions.

Choice questions have explicit candidate IDs and descriptions. Boolean candidates
are `false`, `true`; score questions require increasing explicit numeric levels.
Dependencies must form a DAG. `allowed_candidates` narrows the declared set;
it is not a tool permission grant. Full `valid_tuples` reject incompatible
selections. The preview does not optimize over alternate feasible tuples.

`decision_probability` is a model preference; `estimated_success` remains null
until a separately validated outcome estimator is available. Fallback answers
are not calibrated action permissions. Route mode is an explicit request that
the server may decline; observe mode never permits automation.

## Native runtime

Install `amfs-decision-runtime[server]` separately; ordinary SDK users need no
PyTorch dependency. The runtime encodes state once for a batch of candidate
queries, then uses cross-attention and a decision head. Dependencies with
different parent contexts require separate batches.

Artifacts use safetensors, configuration and tokenizer files. Pretrained
backbones are pinned to a full revision and saved for offline loading. The
no-backbone byte encoder exists only for offline engineering tests.

`amfs_decision_runtime.server:app` is a private inference service. Configure an
operator-owned artifact registry using `AMFS_DECISION_ARTIFACTS`, an immutable
artifact digest per runtime key, and a bounded `AMFS_DECISION_CACHE_MODELS`.
Protect the service with authenticated ingress; the reference FastAPI server
does not implement public account authentication itself. Requests cannot supply
filesystem paths. It never executes proposed tools or fetches caller-specified
weights. Full-model cache entries are isolated; LoRA multiplexing is future work.

No useful pretrained SenseLab decision checkpoint is released with this code.
Public weights require data rights, a model card, pinned runtime, independent
evaluation and release review. Synthetic smoke-test metrics are not evidence
of superiority over Jev or a production automation guarantee.

### Optional startup artifact preload

The reference server remains lazy by default. An operator may set
`AMFS_DECISION_WARMUP_KEYS` to a JSON array of existing opaque runtime keys, for
example `["account-a:recovery:v1","account-a:triage:v2"]`. The array comes only
from process configuration; scoring clients cannot set warmup paths or keys.
Every key must exist in `AMFS_DECISION_ARTIFACTS`, be unique and fit within
`AMFS_DECISION_CACHE_MODELS`. Invalid configuration fails before loading models.

Startup constructs a private pool, checks each full artifact digest and version,
and loads the requested models onto the selected device. Readiness is published
only after the entire requested set succeeds. A missing/corrupt artifact, version
mismatch or device allocation failure aborts startup and clears the partial pool;
the service does not advertise partial readiness. Set startup-probe timeouts for
the actual measured preload duration. No new models are downloaded by this path.

This preloads artifacts; it does not execute a representative inference request
or establish a latency SLO. Model count bounds are not GPU memory byte bounds:
operators must choose capacity and artifact sizes that fit the device, including
inference activation memory. OOM is a startup failure, not permission to silently
drop a warmup key. Subsequent traffic still uses the bounded LRU and may evict
preloaded models. Cold container startup, new artifact misses, tokenization,
queueing and inference can still contribute latency. Benchmark the configured
service under representative concurrency before making performance claims.
