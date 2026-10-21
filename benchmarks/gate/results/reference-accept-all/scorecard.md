# Gate benchmark — reference:accept-all

**Verdict: alert**

## Probes

the gate accepted 8 of 8 known-wrong change(s); 8 would have shipped under the policy

| | injected | decided | false accepts | FA rate | FA upper (95% one-sided) | true-reject rate | median time to reject | would have shipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 8 | 8 | 8 | 100.0% | 100.0% | 0.0% | — | 8 |
| placebo | 2 | 2 | 2 | 100.0% | 100.0% | 0.0% | — | 2 |
| poison | 2 | 2 | 2 | 100.0% | 100.0% | 0.0% | — | 2 |
| reward_hack | 2 | 2 | 2 | 100.0% | 100.0% | 0.0% | — | 2 |
| revert | 2 | 2 | 2 | 100.0% | 100.0% | 0.0% | — | 2 |

| tier | saw | rejected | accepted | undecided |
|---|---:|---:|---:|---:|
| review | 8 | 0 | 8 | 0 |

## A/A

the rule fired falsely in 38 of 40 decidable trials (95.0%; interval 83.5%–98.6%), above its nominal 5.0%

| trials | decidable | false positives | FP rate | interval | nominal | median graded at pass |
|---:|---:|---:|---:|---|---:|---:|
| 40 | 40 | 38 | 95.0% | 83.5%–98.6% | 5.0% | 120.0 |

## Reward

1 of 1 shipped change(s) readable downstream improved the proxy and not the outcome

| change | verdict | proxy prevention | downstream before → after | self-report gap before → after |
|---|---|---:|---|---|
| Pin the smoke-test step before any deploy | diverged (downstream_fell) | 86.4% | 75.0% → 53.4% | 0.0% → 39.8% |

## Run

- harness_version: 0.1.0
- started_at: 2026-10-16T15:00:04.249445+00:00
- finished_at: 2026-10-16T15:00:04.250379+00:00
- probes_requested: 8
- kinds: ['placebo', 'poison', 'reward_hack', 'revert']
- timeout_s: 86400.0
- aa_trials: 40
- aa_window_days: 14
- loop_spec: reference:accept-all
