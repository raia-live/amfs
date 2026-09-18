# Audited external decision baseline

This is a separate eager baseline, not a replacement for the existing pretrained scorer or a SenseLab-trained model. No competitive result or workflow calibration is established by adding an adapter.

## Immutable inputs

- Model: [Mapika/decider-2b revision1d96be0093133e194fe18105a521b3e69be931d2](https://huggingface.co/Mapika/decider-2b/tree/1d96be0093133e194fe18105a521b3e69be931d2).
- Official source inspected: [Mapika/decider revision1c15a48f199adc0eb1952b17441516b5358d79a0](https://github.com/Mapika/decider/tree/1c15a48f199adc0eb1952b17441516b5358d79a0).
- Actual pinned configuration is v8, state-first, temperature1.3, no option neutralization. The model-card training narrative describes an older release; configuration governs this adapter.

The model metadata and official pyproject declare Apache-2.0. Neither inspected tree contains a standalone LICENSE file. Preserve upstream provenance and license declarations when redistributing weights; this inspection does not establish rights to every upstream training example. The adapter is an independent implementation of the documented format, not vendored remote code. Upstream calibration/throughput claims are not ours.

## Inference contract

```python
from amfs_decision_runtime.decider import DeciderChoiceScorer
scorer = DeciderChoiceScorer.from_pretrained(
    revision="1d96be0093133e194fe18105a521b3e69be931d2",
    device="cuda", local_files_only=True,
)
result = scorer.score(observed_state_string, questions, parallel=False)
scorer.close()
```

The same observed state string and declared candidates used for other engines enter this baseline. Each question receives its own native `Context`/`Question`/lettered `Options`/`Answer: (` row. Option text is candidate ID plus the same canonical description/kind/cost criteria supplied to Jev. Candidate order is preserved. No gold labels, permission filtering, hidden world state, option permutation, or automatic abstention is introduced.

This initial adapter supports2..10 candidates per question and up to16 independent questions. It preserves the native255-label projection before selecting declared options and applying temperature1.3. It returns full-precision probabilities without the official HTTP helper's four-decimal rounding. These probabilities are not certificates of correctness or safety.

Context above4096 tokens and batches above32768 padded tokens fail explicitly. The official helper can silently truncate context; this adapter cannot. Each row is padded to64-token boundaries. It uses eager full-context inference, with no shared cache, CUDA graphs, schema-first layout or FP8. Therefore it cannot substantiate the author's optimized latency figures. `parallel=False` runs one question row at a time; it is the conservative development baseline.

## Runtime and validation

Pinned config uses native `Qwen3_5ForCausalLM` (`qwen3_5_text`) with hybrid linear/full attention; transformers5.17.0 supports it without `trust_remote_code`. Use the frozen Torch/CUDA/compiler image and BF16 on a supported GPU. Official helper dependencies include flash-linear-attention; Transformers also reports causal-conv1d and flash-linear-attention as optimized optional kernels. The reference PyTorch path works locally but is slower. Adding these kernels would be a separately pinned runtime recipe, not a silent dependency installation.

Tests check exact native prompt structure, label-table uniqueness, candidate identity, temperature projection, padding, serial/batched equality on a tiny random hybrid Qwen3.5, and fail-closed bounds. The actual pinned tokenizer produced255 distinct labels (A..J tokenIDs32..41), pad/EOS248044. No checkpoint inference was performed during the local audit. Actual GPU development preflight remains required, with technical failures retained. The earlier BF16 numerical-equivalence failures in another model are not erased by changing the baseline.
