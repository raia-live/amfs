# Preregistration: distilling an agent's experience into a small model

Companion to `../HYPOTHESIS.md` (the continual-learning grid). Written before any student
was trained; the grid v4 teacher runs it draws on were complete for the two scenarios named
below when this was frozen (2026-09-19). Every number the paper quotes for a student comes
from a held-out seed no teacher cell used.

## Question

WikiSkill (Google, 2026) argues that an agent's experience should be kept in three layers —
raw traces, a curated wiki, a procedural skill file — and shows the skill file carrying a
frontier model's task performance forward. We ask the next question: **can the experience
that a frontier agent accumulates in a continual-learning memory layer be distilled into a
small model, and how close does that student come to the frontier agent it learned from?**

SenseLab already keeps the three layers as a matter of course — decision traces (raw), a
memory with outcome-reinforced confidence and an action record (wiki), and Managed Models
tuned on the traces (skill in weights). The experiment is the product's own pipeline run
end to end on a benchmark whose ground truth we control, with the frontier teacher measured
on the same held-out seeds.

## Arms and conditions

Teacher grid (from `HYPOTHESIS.md`): `none`, `senselab` and its ablations, on
gpt-5.5 and claude-opus-5, 60 episodes with a regime change at 20, 300 distractors,
fleet 1 and 6, six seeds.

New arm — **`skillfile`** (WikiSkill-lite): raw episode records; a maintainer folds them into
`patterns.md` every 5 episodes; a proposer drafts `SKILL.md`; the draft is accepted only if a
validation replay on regenerated instances (another seed, same regime) does at least as well
as the current file and strictly better when one exists. The file is prepended to the system
prompt; the arm has no memory tools. Its maintainer, proposer and validation cost is charged
to the arm. Every episode is also sealed as a SenseLab trace so it can be a teacher.

Students — SenseLab Managed Models (`senselab-s1`, Gemini 2.5 Flash-Lite supervised tuning)
trained on the traces of a teacher's cells, selected by the cells' memory scopes
(`entity_prefix`), one student per (scenario, teacher arm, teacher model). Served through the
inference gateway: retrieval from the evaluation cell's own store, the renderer's context /
avoid / priors blocks, one action per call. Evaluated under the same harness in
**decision mode**: no tool loop, a failed attempt returned as a failed tool result (rendered
as "Tried in this task", the shape the fail-then-succeed teacher traces trained on),
diagnostics unavailable, retry budget 3.

| id | condition | what it isolates |
|----|-----------|------------------|
| F1 | frontier, full loop, `senselab` | the bar: the teacher on held-out seeds |
| F0 | frontier, full loop, `none` | the frontier without memory |
| D1 | frontier `@decision`, `senselab` | identical served prompt, frontier answers: the prompt's worth |
| D0 | frontier `@decision`, `none` | same, empty store |
| S1 | student, `senselab` | **the claim**: distilled weights + live memory |
| S0 | student, `none` | the weights alone (empty store, nothing served) |
| S1p | student `~plain`, `senselab` | context without the outcome record (no avoid rows, no priors) |
| S1k | student `~k0`, `senselab` | the weights with a store present but nothing served (task text only) |
| S1n | student `+notes`, `senselab` | a cheap note-writer (gpt-5.4-mini) reflects after each episode so the store keeps learning |
| B1 | untuned base (`serve_base`), `senselab` | Gemini 2.5 Flash-Lite behind the same prompt: what tuning added |

Scenarios: `support` and `ci-fix` first (the two with the most teacher traces: 720 per
teacher model per arm). Seeds 101, 113, 127; fleet 1 and 6; otherwise the teacher grid's
constants. `runner.py` is unchanged apart from the decision-mode branch and the skill-file hook.

## Hypotheses

- **H1 (distillation works).** S1 recovers most of the frontier teacher's advantage:
  success-rate gap S1 − F0 ≥ 0.5 × (F1 − F0), at ≤ 1/20 of F1's list-price cost per success.
  Falsified if S1 ≤ F0 on success rate.
- **H2 (memory matters to the student).** S1 > S0 and S1 > S1k: the served context is doing
  work at inference, not only at training. Falsified if S1 ≈ S0 (|Δ| < 3 points, paired by
  seed×episode).
- **H3 (outcome record matters).** S1 > S1p: avoid rows and priors change the student's
  choices in the direction the record points, most visibly after the regime change
  (episodes 20–40). Directional; we report the delta with its CI either way.
- **H4 (the prompt is not the whole story).** D1 − S1 measures what the frontier's weights
  add to the same prompt; F1 − D1 measures what the tool loop and the agent's own reflection
  add. We expect both positive and F1 − D1 to be the larger, because the student never
  writes to the store unless S1n does it for it.
- **H5 (a cheap note-writer closes part of the gap).** S1n > S1 after the regime change,
  because the store keeps learning; and S1n's cost stays under D1's.
- **H6 (tuning, not the base).** S1 > B1 by ≥ 10 points; B1 tells whether the rendered
  prompt alone makes a small model competent.
- **H7 (skill file vs memory as the source of experience).** On the teacher grid, the
  `skillfile` arm sits between `none` and `senselab`; after the regime change its recovery
  is slower than `senselab`'s because a file changes only at a checkpoint that passes a
  gate. Students distilled from `skillfile` traces (`ci-skill-*`, when available) do not beat
  students distilled from `senselab` traces.

## Primary metrics

As in `HYPOTHESIS.md`: success rate, first-attempt success, escalation rate, tokens and
list-price USD per success, recovery after the regime change (episodes to regain the
pre-change rate), stale-entry acts, discredited-entry reuse. Added for students:
`served_by` (tuned / base / fallback), unparseable answers, non-terminal answers, and the
product's own eval report per version (tool/decision/exact agreement, parse rate, stability)
with the default gate's verdict beside the relaxed one the study promotes under.

Paired comparisons by (scenario, seed, fleet, episode); bootstrap CIs over seeds; the
per-episode curves in figures. Costs at list price (`config.STUDENT_PRICES`) with the
provider cost SenseLab pays reported separately.

## What would falsify us

- S1 ≤ F0: the student did not learn a useful policy from the traces.
- S1 ≈ S0 ≈ S1k: memory at inference adds nothing; the paper becomes "distillation works,
  the store does not".
- B1 ≈ S1: tuning added nothing over the rendered prompt.
- D1 ≈ F1 and S1 ≪ D1: the whole benefit is the prompt, and the small model cannot use it.

## Known limits

- One family (Flash-Lite) and one training recipe (the product's). Deliberate: the question
  is what the product delivers, not the best possible student.
- Students are trained on all six teacher seeds and evaluated on three new ones; the eval
  scenarios share generators with the training ones. Generalisation across scenarios is a
  separate question we do not claim.
- The decision-mode harness gives the student no diagnostics. Scenarios where the
  frontier's advantage comes from diagnostic calls will under-state the student.
- ~5% of grid v4 teacher traces carry no memory reference (episodes where the agent read
  nothing) and fall outside the scope predicate; runs from this point stamp the scope on
  every trace.

## Amendments

Dated, in order; each written before the run it concerns finished.

- **2026-09-19 (identifier leak).** The first student scored 0.81 post-change success with
  no memory (S0) where the frontier without memory scores 0.04. Cause: ticket and PR
  numbers were counted from the episode index (`SUP-{4400+ep}`), and with the regime change
  always at episode 20 they encoded the regime; the student learned "SUP-442x and up → the
  new fix" from its training traces. Fix: `Scenario.opaque_ids` draws identifiers from the
  instance rng; every distillation cell runs with it (`-oid` in the cell id). The runs before
  the fix are archived under `results/leaky-distill-*` and appear in no table. The v1
  students keep their (leaky-trace) weights and are evaluated on the fixed scenario, where
  the shortcut cannot fire; v2 students are trained on fresh teacher traces from the fixed
  scenario (`distill-train-*`, seeds 11–73) so the two generations are replicates of the
  same recipe with and without the artefact in the training data.
- **2026-09-20 07:30 (first-try as the memory metric).** After the A-grid results and before
  the B grid below: with a retry budget of three, a student that has both regimes' fix sets
  in its weights reaches the post-change fix on a second attempt with no memory, so S1 − S0
  on eventual success under-states memory at inference. H2 and H3 are reported on both
  success and first-attempt success; first-attempt is the primary reading for them. The
  paper says so and reports both.
- **2026-09-20 07:30 (H8, novel change).** Under change set A every post-change (class, fix)
  pair is in the students' training traces. Change set B (`Scenario.change_variant="B"`,
  `-vB` in the cell id) changes three support classes and two CI classes that A leaves alone,
  to fixes no A episode pairs them with. **H8:** under B, S0's post-change first-try success
  on the changed classes falls to the level of F0's (nothing in the weights knows the new
  fix), and S1 recovers above S0 by at least the S1 − S0 first-try gap measured under A. If
  S1 ≈ S0 under B, memory at inference did not carry a change the weights had not seen, and
  the paper says so. Falsified in the other direction if S0 under B stays near its A level,
  which would mean the new fixes are guessable from the action vocabulary alone; the design
  includes one textbook-answer case per scenario where that is expected and reported as
  such. Conditions under B: S1, S1p, S0 for every student; F1, F0; D1 for one student per
  scenario. Same seeds, fleets and budget as the A grid. Written when 77 F0 episodes and 17
  S0 episodes of the B grid existed and no S1 rows.
