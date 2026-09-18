# Local Mapika adapter serving artifact

`amfs_decision_runtime.decider_artifact` supplies an opt-in local loader. It does not change the existing server, artifact pool, model dispatch or public-base `DeciderChoiceScorer`. A research training directory is **not** accepted directly, and no customer checkpoint is promoted by this module.

An operator-curated directory contains only:

- `manifest.json` with schema `amfs.mapika-peft.v1`, canonical UUID `account_id`/`model_id`, immutable `version`, `spec_hash`, fixed `base_id`/`base_revision`, and a complete relative-file SHA-256 map;
- `base/` with the pinned native text-model config, local tokenizer files and safetensor weights (single file or explicitly indexed shards);
- `adapter/adapter_config.json` and `adapter/adapter_model.safetensors`.

The trusted registry must supply the exact SHA-256 of `manifest.json` plus expected ownership, version and spec bindings. The manifest hashes all serving files; the loader verifies those hashes before loading any weights. It rejects symlinks, traversal, undeclared files, duplicate JSON keys, remote-code mappings, unsupported base identity, non-safetensor model weights, unsupported adapter configuration and non-language-layer LoRA targets. File/metadata/count limits bound validation. Files must remain immutable during load; this local loader does not defend against an operator concurrently replacing already-verified files.

```python
from amfs_decision_runtime.decider_artifact import load_artifact

scorer = load_artifact(
    directory,
    expected_digest=trusted_manifest_sha256,
    account_id=trusted_account_uuid,
    model_id=trusted_model_uuid,
    version=trusted_version,
    spec_hash=trusted_spec_sha256,
    device="cuda",
)
result = scorer.score(state, questions)
```

The loader needs the optional `adapters` dependency group and a Transformers release supporting native Qwen3.5. All model/tokenizer/PEFT loading is local-only; remote model code is disabled. CUDA requires BF16 support without fallback; CPU uses FP32. Each loaded scorer owns a separate adapted base. It does not mutate a shared base's active adapter. The native scorer receives PEFT's adapted base model, whose layers retain their LoRA modules, rather than an incompatible wrapper backbone.

`artifact_identity` is a frozen value. Scores report the tenant model version and artifact/adapter/base/tokenizer/spec identity, rather than labeling bare-base inference as the trained model. Choice probabilities remain uncalibrated. The caller must still enforce authenticated tenant routing and validate the request's actual specification; these metadata bindings are not a replacement for those checks.

Mock tests verify local-only loader arguments, actual attachment use, output identity, changed output compared with a mock bare base, frozen parameters and rejection of tampered files, wrong ownership/version/spec, remote code, traversal, duplicate metadata and symlinks. They do not prove a full checkpoint fits serving GPU memory or that the selected research adapter improves customer outcomes. A curated export, managed lifecycle integration, actual-checkpoint runtime test and tenant concurrency tests remain separate work.
