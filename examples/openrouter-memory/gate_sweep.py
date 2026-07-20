"""Deterministic gate sweep for the AMFS confidence+semantic retrieval gate.

Characterizes the precision/recall tradeoff of the semantic-relevance floor
(SEM_FLOOR) with NO LLM calls: it is purely a property of the embeddings.

For each candidate threshold T:
  recall_gate_pass(T) = fraction of (fact, its own question) pairs where the
                        correct fact's semantic similarity to the question is >= T
                        (i.e. the gate would inject the right fact -> a recall hit).
  miss_abstain(T)     = fraction of miss questions (answer never stored) where NO
                        stored fact clears T (i.e. the gate injects nothing ->
                        the proxy abstains instead of feeding an irrelevant fact).

The two curves move in opposite directions; the crossing region is the operating
point. This is the honest figure behind any abstain-on-miss claim.
"""

from __future__ import annotations

import json

from pro_benchmark import DATASET, MISS_QUESTIONS, FastEmbedEmbedder, cosine

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def main() -> None:
    emb = FastEmbedEmbedder()
    fact_vecs = [emb.embed(f) for f, _, _ in DATASET]
    q_vecs = [emb.embed(q) for _, q, _ in DATASET]
    miss_vecs = [emb.embed(q) for q in MISS_QUESTIONS]

    # similarity of each question to its own fact (should be high)
    self_sims = [cosine(q_vecs[i], fact_vecs[i]) for i in range(len(DATASET))]
    # similarity of each miss question to its NEAREST stored fact (should be low)
    miss_max_sims = [max(cosine(mv, fv) for fv in fact_vecs) for mv in miss_vecs]

    rows = []
    for t in THRESHOLDS:
        recall_pass = sum(s >= t for s in self_sims) / len(self_sims)
        miss_abstain = sum(m < t for m in miss_max_sims) / len(miss_max_sims)
        rows.append({
            "threshold": t,
            "recall_gate_pass_pct": round(100 * recall_pass, 1),
            "miss_abstain_pct": round(100 * miss_abstain, 1),
        })

    out = {
        "self_similarity_min": round(min(self_sims), 3),
        "self_similarity_median": round(sorted(self_sims)[len(self_sims) // 2], 3),
        "miss_nearest_similarity_max": round(max(miss_max_sims), 3),
        "miss_nearest_similarity_median": round(sorted(miss_max_sims)[len(miss_max_sims) // 2], 3),
        "sweep": rows,
    }
    from pathlib import Path
    Path(__file__).parent.joinpath("gate_sweep.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
