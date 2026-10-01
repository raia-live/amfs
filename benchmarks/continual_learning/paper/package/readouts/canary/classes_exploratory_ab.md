## rsi-x-canary-s11: 4 canaries  classes={'decided_treated': 2, 'unexercised': 0, 'decided_untreated': 0, 'other': 2}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix s11 | dc722d08 | flaky integration timeout cassette | knowledge | running |  | other | 11/16/63 | 0/12/56 | 11/16 (66) | 1.00 (7) | 1.00 (30) | 95 |
| ci-fix s11 | 130b2a03 | pip audit pinned dependency | knowledge | passed |  | decided_treated | 58/58/102 | 0/25/107 | 57/58 (106) | 0.97 (32) | 0.84 (49) | 174 |
| support s11 | a09e0892 | web team data export error | knowledge | running |  | other | 10/21/116 | 0/17/124 | 9/21 (120) | 0.57 (7) | 0.47 (62) | 89 |
| support s11 | a3da998e | ios team payment declined | procedure | passed |  | decided_treated | 18/51/51 | 0/53/53 | 17/51 (69) | 0.11 (9) | 0.54 (41) | 99 |
p13prime: {"decided": 2, "share": 0.5, "met": true, "falsified": false, "zero_count_bound": 1, "control_recurring": 1}
p13doubleprime: {"median_read": 14.5, "reads": [10, 11, 18, 58], "floors": [8], "any_reached_floor": true}
p15prime: {"passed": 2, "harmful": ["a3da998e-c748-4c5b-a1f1-641e900e0258"], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 4, "share_exact_of_graded_pooled": 0.6438356164383562, "per_canary": {"dc722d08": 0.6875, "130b2a03": 0.9827586206896551, "a09e0892": 0.42857142857142855, "a3da998e": 0.3333333333333333}, "exact_vs_engine_read": {"dc722d08": [11, 11], "130b2a03": [57, 58], "a09e0892": [9, 10], "a3da998e": [17, 18]}, "met": true, "falsified": false}

## rsi-x-canary-s23: 4 canaries  classes={'decided_treated': 2, 'unexercised': 0, 'decided_untreated': 1, 'other': 1}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix s23 | e3a81d36 | pip audit requests reports export pinned related | knowledge | running |  | other | 6/7/14 | 0/4/16 | 6/7 (17) | 1.00 (4) | 1.00 (10) | 23 |
| ci-fix s23 | 0a2f5553 | pip audit pinned related exception | knowledge | passed |  | decided_treated | 25/25/38 | 0/18/42 | 25/25 (41) | 0.92 (12) | 0.88 (26) | 69 |
| ci-fix s23 | dd421419 | pip audit requests export compatibility | knowledge | passed |  | decided_treated | 9/30/69 | 0/39/65 | 8/30 (69) | 1.00 (1) | 0.97 (39) | 138 |
| support s23 | 7fd04380 | large data export timeout | knowledge | failed |  | decided_untreated | 7/8/12 | 0/20/101 | 4/8 (12) | 0.00 (2) | 0.67 (54) | 33 |
p13prime: {"decided": 3, "share": 0.75, "met": true, "falsified": false, "zero_count_bound": 2, "control_recurring": 1}
p13doubleprime: {"median_read": 8.0, "reads": [6, 7, 9, 25], "floors": [8], "any_reached_floor": true}
p15prime: {"passed": 2, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 4, "share_exact_of_graded_pooled": 0.6142857142857143, "per_canary": {"e3a81d36": 0.8571428571428571, "0a2f5553": 1.0, "dd421419": 0.26666666666666666, "7fd04380": 0.5}, "exact_vs_engine_read": {"e3a81d36": [6, 6], "0a2f5553": [25, 25], "dd421419": [8, 9], "7fd04380": [4, 7]}, "met": true, "falsified": false}

