# RSI paper — confirmatory readout (set C, seeds 37, 41, 59, 73, gpt-5.5 / claude-opus-5; SenseLab from `rsi-conf-f1`)

## 1. After the rule changes conditionally

First-attempt success on every post-change episode (30–99), pooled over the 16 cells (support / ci-fix × seeds 37 / 41 / 59 / 73 × gpt-5.5 / claude-opus-5). Moved half: changed-class instances whose rule changed; kept half: changed-class instances whose rule did not. Brackets: cell-level bootstrap 95 % interval. Infra errors are excluded from every rate.

| system | cells | moved half | kept half | unchanged classes | pre-change | episode success | escalated | infra err | $/ep |
|---|---|---|---|---|---|---|---|---|---|
| Frontier alone | 16 | 25 % (72/284) [13, 39] | 27 % (75/276) [16, 39] | 77 % (432/560) | 65 % (312/480) | 85 % (1362/1600) | 15 % (238/1600) | 0 | 0.014 |
| Context window | 16 | 40 % (115/284) [27, 54] | 92 % (253/276) [84, 97] | 97 % (542/560) | 86 % (415/480) | 92 % (1476/1600) | 8 % (124/1600) | 0 | 0.130 |
| pgvector | 16 | 32 % (91/284) [20, 45] | 66 % (183/276) [59, 74] | 96 % (539/560) | 82 % (392/480) | 92 % (1469/1600) | 8 % (131/1600) | 0 | 0.024 |
| pgvector + rerank | 16 | 36 % (101/284) [23, 48] | 66 % (183/276) [58, 75] | 91 % (511/560) | 77 % (371/480) | 90 % (1435/1600) | 10 % (165/1600) | 0 | 0.027 |
| pgvector + outcomes (DIY) | 16 | 32 % (90/284) [20, 44] | 58 % (161/276) [48, 69] | 84 % (470/560) | 76 % (364/480) | 88 % (1411/1600) | 12 % (189/1600) | 0 | 0.032 |
| SenseLab | 16 | 45 % (127/283) [39, 51] | 69 % (191/276) [62, 76] | 100 % (556/557) | 85 % (406/480) | 95 % (1516/1596) | 5 % (80/1596) | 4 | 0.033 |

## 1b. Paired cell deltas, moved half

SenseLab minus each system on the moved half (percentage points), paired on (family, seed, model). p: exact two-sided sign-flip test of the mean paired delta over the cells; Holm: adjusted across the systems in this table. 16 cells; the smallest attainable p is 0.0000.

| pair | support gpt-5.5 | support claude-opus-5 | ci-fix gpt-5.5 | ci-fix claude-opus-5 | mean | cells won | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| SenseLab − Frontier alone | +28 | +56 | -9 | -1 | +19 | 11/16 | 0.024 | 0.121 |
| SenseLab − Context window | +17 | +36 | -17 | -19 | +4 | 8/16 | 0.544 | 0.544 |
| SenseLab − pgvector | +23 | +16 | +22 | -11 | +13 | 11/16 | 0.040 | 0.159 |
| SenseLab − pgvector + rerank | +24 | +7 | +3 | +4 | +9 | 9/16 | 0.119 | 0.239 |
| SenseLab − pgvector + outcomes (DIY) | +23 | +34 | -6 | +1 | +13 | 10/16 | 0.040 | 0.159 |

### 1b-kept. Secondary endpoint, paired: kept half

Holm within this endpoint.

| pair | support gpt-5.5 | support claude-opus-5 | ci-fix gpt-5.5 | ci-fix claude-opus-5 | mean | cells won | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| SenseLab − Frontier alone | +1 | +23 | +53 | +75 | +38 | 14/16 | 0.002 | 0.007 |
| SenseLab − Context window | -36 | -12 | -26 | -21 | -24 | 1/16 | 0.000 | 0.001 |
| SenseLab − pgvector | +2 | +8 | -13 | +9 | +1 | 9/16 | 0.834 | 1.000 |
| SenseLab − pgvector + rerank | -2 | +1 | +1 | +3 | +1 | 9/16 | 0.882 | 1.000 |
| SenseLab − pgvector + outcomes (DIY) | -17 | +3 | +23 | +25 | +8 | 10/16 | 0.236 | 0.708 |

### 1b-unchanged. Secondary endpoint, paired: unchanged classes

Holm within this endpoint.

| pair | support gpt-5.5 | support claude-opus-5 | ci-fix gpt-5.5 | ci-fix claude-opus-5 | mean | cells won | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| SenseLab − Frontier alone | +37 | +35 | +19 | +0 | +23 | 12/16 | 0.000 | 0.002 |
| SenseLab − Context window | +4 | +9 | -1 | +0 | +3 | 4/16 | 0.125 | 0.125 |
| SenseLab − pgvector | +4 | +4 | +0 | +6 | +4 | 6/16 | 0.031 | 0.062 |
| SenseLab − pgvector + rerank | +21 | +14 | -1 | +0 | +9 | 8/16 | 0.012 | 0.035 |
| SenseLab − pgvector + outcomes (DIY) | +26 | +25 | +12 | +0 | +16 | 11/16 | 0.001 | 0.004 |

## 1c. By family

First-attempt success per family: moved half, kept half, unchanged classes.

| system | support: moved | support: kept | support: unchanged | ci-fix: moved | ci-fix: kept | ci-fix: unchanged |
|---|---|---|---|---|---|---|
| Frontier alone | 0 % (0/150) | 52 % (68/130) | 64 % (179/280) | 54 % (72/134) | 5 % (7/146) | 90 % (253/280) |
| Context window | 17 % (25/150) | 92 % (119/130) | 94 % (262/280) | 67 % (90/134) | 92 % (134/146) | 100 % (280/280) |
| pgvector | 22 % (33/150) | 62 % (80/130) | 96 % (269/280) | 43 % (58/134) | 71 % (103/146) | 96 % (270/280) |
| pgvector + rerank | 27 % (40/150) | 66 % (86/130) | 82 % (231/280) | 46 % (61/134) | 66 % (97/146) | 100 % (280/280) |
| pgvector + outcomes (DIY) | 14 % (21/150) | 74 % (96/130) | 74 % (208/280) | 51 % (69/134) | 45 % (65/146) | 94 % (262/280) |
| SenseLab | 42 % (62/149) | 69 % (90/130) | 100 % (277/277) | 49 % (65/134) | 69 % (101/146) | 100 % (279/280) |

## 1d. By model

First-attempt success per agent model: moved half, kept half, unchanged classes.

| system | gpt-5.5: moved | gpt-5.5: kept | gpt-5.5: unchanged | claude-opus-5: moved | claude-opus-5: kept | claude-opus-5: unchanged |
|---|---|---|---|---|---|---|
| Frontier alone | 25 % (36/142) | 28 % (39/138) | 72 % (201/280) | 25 % (36/142) | 26 % (36/138) | 82 % (231/280) |
| Context window | 36 % (51/142) | 88 % (122/138) | 98 % (275/280) | 45 % (64/142) | 95 % (131/138) | 95 % (267/280) |
| pgvector | 13 % (18/142) | 63 % (87/138) | 98 % (274/280) | 51 % (73/142) | 70 % (96/138) | 95 % (265/280) |
| pgvector + rerank | 22 % (31/142) | 57 % (79/138) | 90 % (251/280) | 49 % (70/142) | 75 % (104/138) | 93 % (260/280) |
| pgvector + outcomes (DIY) | 27 % (38/142) | 54 % (74/138) | 80 % (225/280) | 37 % (52/142) | 63 % (87/138) | 88 % (245/280) |
| SenseLab | 36 % (51/142) | 60 % (83/138) | 100 % (279/280) | 54 % (76/141) | 78 % (108/138) | 100 % (277/277) |

## 2. SenseLab by engine cut

Every SenseLab run on set C, oldest engine first. The engine changed between runs (HYPOTHESIS_RSI.md cut records); each row is the same protocol on a different cut.

_(no rows)_

## 3. Attribution

SenseLab (`rsi-conf-f1`) against its own ablations on the same protocol: without the canary (same run) and without the repair loop (outcome learning alone, `rsi-conf-f1`).

| system | cells | moved half | kept half | unchanged classes | pre-change | episode success | escalated | infra err | $/ep |
|---|---|---|---|---|---|---|---|---|---|
| SenseLab | 16 | 45 % (127/283) [39, 51] | 69 % (191/276) [62, 76] | 100 % (556/557) | 85 % (406/480) | 95 % (1516/1596) | 5 % (80/1596) | 4 | 0.033 |
| SenseLab − canary | 16 | 44 % (125/283) [36, 54] | 71 % (197/276) [66, 76] | 99 % (548/556) | 86 % (412/480) | 94 % (1506/1595) | 6 % (89/1595) | 5 | 0.034 |
| SenseLab − repair loop | 16 | 43 % (122/284) [33, 52] | 68 % (187/276) [59, 76] | 99 % (557/560) | 85 % (410/480) | 93 % (1494/1600) | 7 % (106/1600) | 0 | 0.033 |

## 4. Fix lifecycle

Every fix the loop proposed in the SenseLab cells and the no-canary cells, by run. Tier 1 is the stand-in replay on the fix branch; Tier 2 the customer replay of the fix's case set; a canary grades live exposure over the sessions after it starts. `open` Tier 2 replays were still awaited when the cell ended; no canary reached a decision inside a cell (HEADROOM K1–K4).

| arm | proposed | regime / judge | T1 pass / inc. / fail | T2 pass / fail / open | canaries run / undecided | shipped | reverted | drafted by the repair agent |
|---|---|---|---|---|---|---|---|---|
| SenseLab, confirmatory | 32 | 23 / 9 | 16 / 13 / 3 | 6 / 22 / 2 | 5 / 3 | 2 | 0 | 94 % (30/32) |
| SenseLab − canary, confirmatory | 44 | 32 / 12 | 25 / 14 / 4 | 10 / 32 / 0 | 0 / 0 | 10 | 0 | 89 % (39/44) |

## 5. Cost and latency

Per episode, pooled over the four cells, by where the call is charged. Agent $: the agent's own calls. Arm $: reflection / consolidation the arm ran per episode. Judges $ and replays $: the loop's judge verdicts and the Tier 2 replays the harness re-ran on the fix branch, exact from the arm reports. Drafts: fixes the repair agent drafted per 100 episodes, priced at the account ledger's remainder per draft (repair agent, Tier 1 stand-in replay, lever, discovery — not persisted per fix, so an estimate: $0.134 per draft from $94.17 spent, $46.79 of it judges, over 354 drafts on the account at 2026-09-29T17:31:12Z). Total sums the five. Memory s: time in memory calls per episode against the production API.

| system | tokens/ep | agent $/ep | arm $/ep | judges $/ep | replays $/ep | drafts/100 ep | drafts $/ep (est.) | total $/ep | memory s/ep | wall s/ep |
|---|---|---|---|---|---|---|---|---|---|---|
| Frontier alone | 2,389 | 0.014 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.014 | 0.0 | 6 |
| Context window | 29,871 | 0.130 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.130 | 0.0 | 6 |
| pgvector | 5,005 | 0.024 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.024 | 0.0 | 10 |
| pgvector + rerank | 5,164 | 0.025 | 0.002 | 0.000 | 0.000 | 0.0 | 0.000 | 0.027 | 0.0 | 13 |
| pgvector + outcomes (DIY) | 5,784 | 0.027 | 0.005 | 0.000 | 0.000 | 0.0 | 0.000 | 0.032 | 0.0 | 11 |
| SenseLab | 7,856 | 0.033 | 0.000 | 0.003 | 0.003 | 1.9 | 0.003 | 0.042 | 18.4 | 31 |
| SenseLab − canary | 8,001 | 0.034 | 0.000 | 0.004 | 0.005 | 2.4 | 0.003 | 0.045 | 18.0 | 31 |
| SenseLab − repair loop | 7,983 | 0.033 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | 0.033 | 16.8 | 30 |

Account-ledger check (runs whose runner snapshotted the ledger; a delta is shared with any other run on the account in the same window):

- Frontier alone: `rsi-conf-f1` ledger delta $29.65 from 2026-09-30T01:42:49Z to 2026-09-30T08:01:03Z; estimate for the run's cells $0.00
- Context window: `rsi-conf-f1` ledger delta $29.65 from 2026-09-30T01:42:49Z to 2026-09-30T08:01:03Z; estimate for the run's cells $0.00
- pgvector: `rsi-conf-f1` ledger delta $29.65 from 2026-09-30T01:42:49Z to 2026-09-30T08:01:03Z; estimate for the run's cells $0.00
- pgvector + rerank: `rsi-conf-f1` ledger delta $29.65 from 2026-09-30T01:42:49Z to 2026-09-30T08:01:03Z; estimate for the run's cells $0.00
- pgvector + outcomes (DIY): `rsi-conf-f1` ledger delta $29.65 from 2026-09-30T01:42:49Z to 2026-09-30T08:01:03Z; estimate for the run's cells $0.00
- SenseLab: `rsi-conf-f1` ledger delta $29.65 from 2026-09-30T01:42:49Z to 2026-09-30T08:01:03Z; estimate for the run's cells $9.27
- SenseLab − canary: `rsi-conf-f1` ledger delta $29.65 from 2026-09-30T01:42:49Z to 2026-09-30T08:01:03Z; estimate for the run's cells $10.93
- SenseLab − repair loop: `rsi-conf-f1` ledger delta $29.65 from 2026-09-30T01:42:49Z to 2026-09-30T08:01:03Z; estimate for the run's cells $0.00

