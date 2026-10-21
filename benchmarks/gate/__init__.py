"""The gate benchmark: how well a self-improvement loop refuses what it should.

Every loop that changes an agent from its own evidence has a gate between a
proposed change and the agent that runs it — a replay, a review, a canary.
Published work reports what the loops accepted and how much the agent improved;
none reports what the gate *refused*, so a loop whose gate accepts anything and
a loop whose gate holds look the same in those tables. This harness measures the
gate from the other side, with three self-tests any loop can be pointed at:

- **Probes** (``probes.py``): changes that are wrong on purpose — a *placebo*
  that repairs nothing, a *poison* that endorses the failure, a *reward hack*
  that improves the number the gate reads without improving the task, and a
  *revert* of a change the loop itself proved — sent through the gate as if they
  were real. The scorecard is the **false-accept rate** (with a one-sided 95%
  Wilson upper bound), the **true-reject rate**, the **median time to reject**,
  and what each tier of the gate said.
- **A/A** (``scorecard.aa_scorecard``): the gate's live decision rule run over
  two arms of the same traffic with no change between them. Every decision is a
  false positive; the rate is read against the rule's nominal.
- **Reward** (``scorecard.reward_scorecard``): every change the loop shipped and
  proved by its own proxy, read against an outcome the loop did not write. A
  change that moved the proxy and not the outcome *diverged*.

A loop takes part in one of two ways. Live: implement :class:`protocol.GatedLoop`
(``adapters/`` has the SenseLab adapter over its HTTP API and two reference
loops that run in-process) and ``python -m benchmarks.gate run`` drives it.
Offline: write the loop's own ledger in the JSONL schema of ``ledger.py`` and
``python -m benchmarks.gate score`` reads it. Either way the output is one
``scorecard.json`` and one ``scorecard.md``, the same for every loop.

Nothing here depends on anything outside the standard library. The arithmetic
is the same as SenseLab's own (``amfs_eval.repair.adversarial``,
``amfs_eval.repair.aa``, ``amfs_eval.repair.reward``) so the number SenseLab
reports about itself and the number this harness reports about it agree; the
harness is the public, loop-agnostic statement of that arithmetic.

SenseLab runs first and publishes its numbers whatever they are
(``results/`` under this package once a run has been made against a deployment).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
