---
title: ML Layer
layout: default
parent: Guides
nav_order: 5
description: "Use AMFS Pro's ML layer for learned ranking, confidence calibration, and training data export."
---

# ML Layer
{: .no_toc }

AMFS Pro's ML layer learns from your outcome data to make retrieval smarter, confidence scoring more accurate, and your agents' decision traces exportable as fine-tuning datasets.
{: .fs-6 .fw-300 }

{: .note }
The ML layer is a Pro-only feature. The OSS layer captures all the data — reads, writes, outcomes, causal chains — and the ML layer learns from it.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

- AMFS Pro MCP server running (see [MCP Setup](/amfs/guides/mcp/))
- Outcome data in your memory store — the ML layer trains on `commit_outcome` history
- At least 20 outcome-linked entries for learned ranking, 5 per outcome type for calibration

---

## Learned Retrieval Ranking

### The Problem

AMFS's multi-strategy retrieval uses fixed weights (semantic: 0.4, keyword: 0.2, temporal: 0.2, confidence: 0.2) merged via Reciprocal Rank Fusion. These work well as defaults, but they can't capture domain-specific patterns like "for this entity, recency matters more than confidence" or "entries from production agents are more reliable for deployment decisions."

### How It Works

The learned ranker trains a gradient-boosted model on your outcome history:

- **Positive labels**: entries that were read before clean deploys
- **Negative labels**: entries read before incidents, or entries never linked to any outcome

The model learns which `MemoryEntry` features predict usefulness and integrates as an additional strategy in the retrieval pipeline. When trained, it automatically gets 30% weight in RRF fusion.

### Via MCP

```
amfs_retrain()
```

Train from all available data. Returns metrics:

```json
{
  "num_samples": 156,
  "num_positive": 98,
  "num_negative": 58,
  "accuracy": 0.82,
  "feature_importances": {
    "confidence": 0.23,
    "outcome_count": 0.18,
    "log_age_hours": 0.15,
    "tier_production_validated": 0.12,
    "version": 0.09
  },
  "trained_at": "2026-04-01T14:30:00Z"
}
```

Train for a specific entity:

```
amfs_retrain(entity_path="checkout-service")
```

### Via Python SDK

```python
from amfs_ml import LearnedRanker

ranker = LearnedRanker(adapter, model_path=Path(".amfs/ml/ranker.pkl"))

# Train
metrics = ranker.train()
print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"Top feature: {max(metrics.feature_importances, key=metrics.feature_importances.get)}")

# Score entries
scored = ranker.score(entries)
for entry, probability in scored[:5]:
    print(f"{entry.entry_key}: {probability:.3f}")
```

### Graceful Degradation

With fewer than 20 training samples, the ranker falls back to confidence-based scoring. The `amfs_retrieve` tool works identically whether a model is trained or not — the learned strategy simply receives zero weight until training completes.

---

## Adaptive Confidence Calibration

### The Problem

AMFS uses fixed outcome multipliers:

| Outcome | Default Multiplier |
|:--------|:-------------------|
| Critical Failure | × 1.15 |
| Failure | × 1.10 |
| Minor Failure | × 1.08 |
| Success | × 0.97 |

These are reasonable defaults, but the actual signal strength of each outcome type varies by domain. A P1 incident in a payment service carries different weight than a P1 in a logging service.

### How It Works

The calibrator analyzes your outcome history to learn domain-specific multipliers:

1. Groups outcomes by type
2. For each type, measures how often causally-linked entries later appear in incidents vs clean deploys
3. Adjusts multipliers based on observed signal strength
4. Estimates optimal decay half-life from the age distribution of actively-used entries

### Via MCP

```
amfs_calibrate()
```

Returns calibrated multipliers and analysis:

```json
{
  "global_multipliers": {
    "entity_path": null,
    "multipliers": {
      "critical_failure": 1.1845,
      "failure": 1.123,
      "minor_failure": 1.1016,
      "success": 0.9797
    },
    "decay_half_life_days": 21.5,
    "num_outcomes_analyzed": 89
  },
  "entity_multipliers": [],
  "total_outcomes": 89,
  "total_entries": 234
}
```

With per-entity overrides:

```
amfs_calibrate(per_entity=true)
```

Returns global multipliers plus entity-specific overrides for any entity with enough data (5+ outcomes per type).

### Via Python SDK

```python
from amfs_ml import ConfidenceCalibrator

calibrator = ConfidenceCalibrator(adapter)

# Global calibration
report = calibrator.calibrate()
print(report.global_multipliers.multipliers)

# Per-entity calibration
report = calibrator.calibrate(per_entity=True)
for em in report.entity_multipliers:
    print(f"{em.entity_path}: {em.multipliers}")
    if em.decay_half_life_days:
        print(f"  Estimated decay: {em.decay_half_life_days} days")
```

---

## Training Data Export

### The Problem

AMFS captures structured decision traces: what the agent read, what it decided, and what happened next. These traces are the exact data structure needed for fine-tuning — `(context, action, reward)` tuples — but they're locked inside the memory store.

### How It Works

The exporter queries historical outcomes and their causally-linked entries, then formats them as training datasets in three formats:

**SFT (Supervised Fine-Tuning)** — Each successful decision trace becomes a training example. Context entries (what was read) pair with the decision entry (what was written). Only clean deploys produce SFT examples.

**DPO (Direct Preference Optimization)** — Pairs a successful decision trace (chosen) with a failed one (rejected) for the same entity. The outcome replaces human preference annotation.

**Reward Model** — Each entry is labeled with a score based on its outcome history: clean deploys score +1.0, P1 incidents score -1.0, with intermediate values for P2 and regressions.

### Via MCP

Export as SFT:

```
amfs_export_training_data(format="sft")
```

Export as DPO:

```
amfs_export_training_data(format="dpo")
```

Export as reward model data:

```
amfs_export_training_data(format="reward_model", entity_path="checkout-service")
```

Returns:

```json
{
  "format": "reward_model",
  "num_examples": 42,
  "examples": [
    {
      "entry": {"entity_path": "checkout-service", "key": "retry-pattern", "...": "..."},
      "label": 0.85,
      "outcome_type": "success",
      "outcome_count": 7
    }
  ],
  "exported_at": "2026-04-01T15:00:00Z"
}
```

### Via Python SDK

```python
from amfs_ml import TrainingDataExporter
from amfs_ml.export.exporter import ExportFormat

exporter = TrainingDataExporter(adapter)

# Export as structured result
result = exporter.export(format=ExportFormat.DPO, entity_path="checkout-service")
print(f"Generated {result.num_examples} DPO pairs")

# Export as JSONL (ready for fine-tuning pipelines)
jsonl = exporter.export_jsonl(format=ExportFormat.SFT, limit=1000)
with open("training_data.jsonl", "w") as f:
    f.write(jsonl)
```

### Integration with Fine-Tuning Pipelines

AMFS generates the data; you bring the training infrastructure. The exported formats are compatible with common fine-tuning workflows:

| Format | Compatible With |
|:-------|:----------------|
| SFT | OpenAI fine-tuning API, Hugging Face `SFTTrainer`, Axolotl |
| DPO | TRL `DPOTrainer`, OpenRLHF |
| Reward Model | TRL `RewardTrainer`, custom reward model training |

---

## Data Requirements

The ML layer needs outcome data to learn from. Here's the minimum for each feature:

| Feature | Minimum Data | Recommended |
|:--------|:-------------|:------------|
| Learned Ranking | 20 outcome-linked entries | 100+ entries with mixed outcomes |
| Confidence Calibration | 5 outcomes per type | 20+ per type for reliable calibration |
| Training Data Export (SFT) | 1 clean deploy with 2+ causal entries | Dozens of successful traces |
| Training Data Export (DPO) | 1 positive + 1 negative outcome per entity | Multiple of each per entity |
| Training Data Export (Reward) | 1 outcome-linked entry | Hundreds of entries for a useful dataset |

{: .tip }
The ML layer works best with Postgres, which persists outcome records. The filesystem adapter tracks outcome effects on entries but doesn't persist the outcome records themselves, limiting the data available for training.

---

## Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `AMFS_ML_MODEL_DIR` | `.amfs/ml` | Directory for persisted ML models (ranker pickle files) |

---

## How the Pieces Fit Together

```
Agents use AMFS normally:
  read → decide → write → commit_outcome
      │                        │
      │                        ▼
      │               Outcome data accumulates
      │                        │
      ▼                        ▼
  amfs_retrieve ◄── amfs_retrain (learns which entries are useful)
      │
      │            amfs_calibrate (learns optimal multipliers)
      │
      │            amfs_export_training_data (generates fine-tuning datasets)
      │                        │
      ▼                        ▼
  Better retrieval      Better agents (via your fine-tuning pipeline)
```

The feedback loop: agents produce outcome data by working normally. The ML layer consumes that data to improve retrieval and generate training datasets. Better retrieval leads to better decisions, which produce more outcome data.

---

## Next Steps

- [MCP Setup](/amfs/guides/mcp/) — Configure the Pro MCP server
- [Confidence & Outcomes](/amfs/concepts/confidence/) — Understand the outcome system that feeds the ML layer
- [Context Graphs](/amfs/concepts/context-graphs/) — How decision traces are captured
- [OSS vs Pro](/amfs/editions/) — Full feature comparison
