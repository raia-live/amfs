---
title: What "recursive self-improvement" means here
layout: default
nav_order: 7
description: "The operational definition behind the SenseLab tagline, what it does not mean, and the evidence that stands behind it today."
permalink: /recursive-self-improvement/
---

# What "recursive self-improvement" means here

The README calls SenseLab "the recursive self-improvement engine for AI
agents". This page says what that sentence means, what it does not mean, and
what evidence stands behind it today. It follows §2 of the SenseLab position
paper on the engine; the paper is on its way to publication and this page
will link to it when it is public.

## The definition

We use the term in a narrow, operational sense. A system is recursively
self-improving when:

1. its own outcomes change the knowledge it acts on;
2. the changes are themselves tested against outcomes before they take
   effect; and
3. the machinery that makes the changes is subject to the same evidence as
   the knowledge it changes.

The word *recursive* points at the last clause: the corrections apply to the
thing that produces corrections, and the record the loop keeps is also what
trains its next version.

## What it does not mean

The term is also used for systems that rewrite their own code or training
toward open-ended capability gain. Nothing here does that. The engine changes
the operating knowledge its agents act on, and the prompts and models that
produce corrections to it. Every change is tested against outcomes it did not
write, exposed to a share of live traffic before all of it, reversible, and
logged. The gate is the design: generating corrections is the easy part, and
the engine's value is in refusing most of them and keeping a record of what
it refused and why.

## The four properties, and where they are checked

| Property | What it means | Where to look |
|---|---|---|
| Gated | no correction reaches agents until a stand-in replay with a control arm, the customer's own replay where one exists, and a live canary graded by a second judge family have each passed it | the gate benchmark below; `benchmarks/gate/` in this repository runs the same probes against any loop |
| Reversible | a shipped correction is verified over a window and rolled back to the pre-fix versions when it regresses | the fix ledger on each account; the canary and verify sweeps |
| Auditable | every read, decision, outcome and fix event is on a sealed, hash-chained trace | `amfs_verify` / the traces API in this repository |
| Preregistered | the measured claims are made against hypotheses and endpoints written down before the data | `benchmarks/continual_learning/HYPOTHESIS.md` (written 2026-09-16, before any harness code or data) and the amendments the paper names |

## What the evidence shows today

Two things are measured; they are different claims and sit on different
rungs.

**Level 0 — outcomes decide what the agent acts on.** The continual-learning
benchmark was preregistered in this repository
(`benchmarks/continual_learning/HYPOTHESIS.md`) before any harness code or
data: the same agent on SenseLab, on two memory products, on a plain
pgvector table, on a do-it-yourself pgvector build and with no memory at
all, with the metrics and the correction fixed in advance. The results are
reported in the paper with cell counts and the preregistered correction; the
sentence they support is that agents stop repeating mistakes because every
outcome moves the evidence behind what the agent relied on. No comparison
in that run survives the Holm correction, and the paper says so in its
abstract. This page will link to the results when the paper is public.

**The gate refuses known-wrong corrections.** The gate benchmark
(`benchmarks/gate/`, results under `benchmarks/gate/results/`) injects
placebo, poisoned and reward-hacking corrections into a running loop and
counts what reaches agents. On the SenseLab development deployment on
2026-10-18, 17 known-wrong corrections were injected and 17 were rejected at
the first gate, 0 false accepts; with 17 probes the one-sided 95% upper
bound on the false-accept rate is 13.7%, a sample-size bound rather than a
measured rate. The identical-arm (A/A) check fired falsely in 0 of 40 trials
against a nominal 7.5%. The run's `NOTES.md` says what the number is and is
not, including that every probe was drafted from one shape of change.

**What is not yet shown.** That level 1 (the engine proposing, testing and
shipping corrections) adds to level 0 on a preregistered endpoint. That
re-run is preregistered and not yet done; until it is, "recursive" here
names the mechanism and the gates it runs through, not a measured gain
from the loop over the memory alone.

## Reading the tagline

"Recursive self-improvement engine" on the README means: a loop whose
corrections are gated by outcomes they did not produce, reversible, and
logged, where the loop's own machinery is under the same evidence. It does
not mean open-ended capability gain, and it is not a claim that the loop has
been shown to beat its own level 0 — that claim waits on the preregistered
re-run.
