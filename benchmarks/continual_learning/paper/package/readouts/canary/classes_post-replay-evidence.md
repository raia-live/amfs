## rsi-conf-e2b-r2-s37: 1 canaries  classes={'decided_treated': 0, 'unexercised': 0, 'decided_untreated': 1, 'other': 0}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| support s37 | 8a908156 | android free push notifications stopped | knowledge | failed |  | decided_untreated | 5/10/10 | 0/10/10 | 5/10 (10) | 0.00 (2) | 1.00 (1) | 20 |
p13prime: {"decided": 1, "share": 1.0, "met": true, "falsified": false, "zero_count_bound": 1, "control_recurring": 0}
p13doubleprime: {"median_read": 5, "reads": [5], "floors": [8], "any_reached_floor": false}
p15prime: {"passed": 0, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 1, "share_exact_of_graded_pooled": 0.5, "per_canary": {"8a908156": 0.5}, "exact_vs_engine_read": {"8a908156": [5, 5]}, "met": true, "falsified": false}

## rsi-conf-e2b-r2-s41: 2 canaries  classes={'decided_treated': 0, 'unexercised': 0, 'decided_untreated': 0, 'other': 2}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix s41 | c6263b74 | pip audit direct pin | knowledge | running |  | other | 12/13/36 | 0/4/29 | 13/13 (39) | 0.00 (3) | 0.70 (10) | 60 |
| support s41 | 15fdbbcb | large export failure | knowledge | running |  | other | 18/20/66 | 0/8/79 | 15/20 (81) | 0.33 (6) | 0.40 (25) | 84 |
p13prime: {"decided": 0, "share": 0.0, "met": false, "falsified": false, "zero_count_bound": 0, "control_recurring": 0}
p13doubleprime: {"median_read": 15.0, "reads": [12, 18], "floors": [8], "any_reached_floor": true}
p15prime: {"passed": 0, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 2, "share_exact_of_graded_pooled": 0.8484848484848485, "per_canary": {"c6263b74": 1.0, "15fdbbcb": 0.75}, "exact_vs_engine_read": {"c6263b74": [13, 12], "15fdbbcb": [15, 18]}, "met": true, "falsified": false}

## rsi-conf-e2b-r2-s59: 1 canaries  classes={'decided_treated': 0, 'unexercised': 0, 'decided_untreated': 0, 'other': 1}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix s59 | f2779fdc | billing webhook timeout code only | knowledge | running |  | other | 14/14/62 | 0/10/51 | 14/14 (72) | 1.00 (6) | 0.88 (17) | 94 |
p13prime: {"decided": 0, "share": 0.0, "met": false, "falsified": false, "zero_count_bound": 0, "control_recurring": 0}
p13doubleprime: {"median_read": 14, "reads": [14], "floors": [8], "any_reached_floor": true}
p15prime: {"passed": 0, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 1, "share_exact_of_graded_pooled": 1.0, "per_canary": {"f2779fdc": 1.0}, "exact_vs_engine_read": {"f2779fdc": [14, 14]}, "met": true, "falsified": false}

## rsi-conf-e2b-r2-s73: 2 canaries  classes={'decided_treated': 0, 'unexercised': 0, 'decided_untreated': 0, 'other': 2}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix s73 | 983594a6 | flaky integration ci | knowledge | running |  | other | 8/17/58 | 0/14/72 | 8/17 (61) | 1.00 (6) | 0.77 (22) | 88 |
| support s73 | ff763eab | data export error | knowledge | running |  | other | 2/2/6 | 0/8/78 | 2/2 (7) | — (0) | 0.59 (27) | 10 |
p13prime: {"decided": 0, "share": 0.0, "met": false, "falsified": false, "zero_count_bound": 0, "control_recurring": 0}
p13doubleprime: {"median_read": 5.0, "reads": [2, 8], "floors": [8], "any_reached_floor": true}
p15prime: {"passed": 0, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 2, "share_exact_of_graded_pooled": 0.5263157894736842, "per_canary": {"983594a6": 0.47058823529411764, "ff763eab": 1.0}, "exact_vs_engine_read": {"983594a6": [8, 8], "ff763eab": [2, 2]}, "met": true, "falsified": false}

