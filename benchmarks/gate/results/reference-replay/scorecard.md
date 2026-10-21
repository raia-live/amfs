# Gate benchmark — reference:replay

**Verdict: ok**

## Probes

the gate rejected all 8 known-wrong changes; median time to reject 0.4 h

| | injected | decided | false accepts | FA rate | FA upper (95% one-sided) | true-reject rate | median time to reject | would have shipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 8 | 8 | 0 | 0.0% | 25.3% | 100.0% | 23 min | 0 |
| placebo | 2 | 2 | 0 | 0.0% | 57.5% | 100.0% | 23 min | 0 |
| poison | 2 | 2 | 0 | 0.0% | 57.5% | 100.0% | 23 min | 0 |
| reward_hack | 2 | 2 | 0 | 0.0% | 57.5% | 100.0% | 23 min | 0 |
| revert | 2 | 2 | 0 | 0.0% | 57.5% | 100.0% | 23 min | 0 |

| tier | saw | rejected | accepted | undecided |
|---|---:|---:|---:|---:|
| t1 | 8 | 8 | 0 | 0 |

## A/A

the rule fired falsely in 1 of 40 decidable trials (2.5%), within its nominal 5.0%

| trials | decidable | false positives | FP rate | interval | nominal | median graded at pass |
|---:|---:|---:|---:|---|---:|---:|
| 40 | 40 | 1 | 2.5% | 0.4%–12.9% | 5.0% | 160 |

## Reward

1 shipped change(s) readable downstream; 1 aligned, 0 flat, none diverged

| change | verdict | proxy prevention | downstream before → after | self-report gap before → after |
|---|---|---:|---|---|
| Pin the smoke-test step before any deploy | aligned | 86.4% | 66.7% → 89.7% | 10.0% → 1.9% |

## Run

- harness_version: 0.1.0
- started_at: 2026-10-16T15:00:04.186442+00:00
- finished_at: 2026-10-16T15:00:04.187557+00:00
- probes_requested: 8
- kinds: ['placebo', 'poison', 'reward_hack', 'revert']
- timeout_s: 86400.0
- aa_trials: 40
- aa_window_days: 14
- loop_spec: reference:replay
