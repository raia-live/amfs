## rsi-conf-e2b-s37: 1 canaries  classes={'decided_treated': 0, 'unexercised': 0, 'decided_untreated': 0, 'other': 1}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix s37 | f5241329 | pip audit vulnerable pin | knowledge | running |  | other | 1/1/6 | 0/1/4 | 1/1 (8) | 1.00 (1) | 1.00 (2) | 15 |
p13prime: {"decided": 0, "share": 0.0, "met": false, "falsified": false, "zero_count_bound": 0, "control_recurring": 0}
p13doubleprime: {"median_read": 1, "reads": [1], "floors": [8], "any_reached_floor": false}
p15prime: {"passed": 0, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 1, "share_exact_of_graded_pooled": 1.0, "per_canary": {"f5241329": 1.0}, "exact_vs_engine_read": {"f5241329": [1, 1]}, "met": true, "falsified": false}

## rsi-conf-e2b-s41: 1 canaries  classes={'decided_treated': 0, 'unexercised': 0, 'decided_untreated': 0, 'other': 1}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| support s41 | 0b676549 | android push notifications stopped | knowledge | running |  | other | 1/9/38 | 0/3/32 | 1/9 (38) | — (0) | 0.62 (8) | 10 |
p13prime: {"decided": 0, "share": 0.0, "met": false, "falsified": false, "zero_count_bound": 0, "control_recurring": 0}
p13doubleprime: {"median_read": 1, "reads": [1], "floors": [8], "any_reached_floor": false}
p15prime: {"passed": 0, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 1, "share_exact_of_graded_pooled": 0.1111111111111111, "per_canary": {"0b676549": 0.1111111111111111}, "exact_vs_engine_read": {"0b676549": [1, 1]}, "met": false, "falsified": true}

## rsi-conf-e2b-s59: 2 canaries  classes={'decided_treated': 0, 'unexercised': 0, 'decided_untreated': 0, 'other': 2}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix s59 | 3be3ede8 | ci failure unknown | knowledge | running |  | other | 17/23/95 | 0/15/94 | 17/23 (110) | — (0) | 0.70 (27) | 29 |
| support s59 | b2b656c5 | csv export timeout | knowledge | running |  | other | 0/63/63 | 0/77/77 | 0/63 (74) | — (0) | 0.65 (17) | 11 |
p13prime: {"decided": 0, "share": 0.0, "met": false, "falsified": false, "zero_count_bound": 0, "control_recurring": 0}
p13doubleprime: {"median_read": 8.5, "reads": [0, 17], "floors": [8], "any_reached_floor": true}
p15prime: {"passed": 0, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 2, "share_exact_of_graded_pooled": 0.19767441860465115, "per_canary": {"3be3ede8": 0.7391304347826086, "b2b656c5": 0.0}, "exact_vs_engine_read": {"3be3ede8": [17, 17], "b2b656c5": [0, 0]}, "met": false, "falsified": true}

## rsi-conf-e2b-s73: 2 canaries  classes={'decided_treated': 0, 'unexercised': 0, 'decided_untreated': 1, 'other': 1}
| cell | fix | situation | lever | status | reason | class | can read/graded/exposed | ctrl read/graded/exposed | P16 exact/graded (sessions) | P15′ treated kept FAS | ctrl kept FAS | reuse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| support s73 | 30ff29d9 | android notifications stopped free no prior steps | knowledge | running |  | other | 1/4/36 | 0/2/24 | 1/4 (40) | — (0) | 0.25 (8) | 15 |
| support s73 | 2ae2f0e0 | android notifications not received | procedure | failed |  | decided_untreated | 0/8/8 | 0/22/22 | 0/8 (14) | — (0) | 0.88 (8) | 5 |
p13prime: {"decided": 1, "share": 0.5, "met": true, "falsified": false, "zero_count_bound": 1, "control_recurring": 0}
p13doubleprime: {"median_read": 0.5, "reads": [0, 1], "floors": [8], "any_reached_floor": false}
p15prime: {"passed": 0, "harmful": [], "underpowered": []}
p16: {"canaries_with_graded_canary_sessions": 2, "share_exact_of_graded_pooled": 0.08333333333333333, "per_canary": {"30ff29d9": 0.25, "2ae2f0e0": 0.0}, "exact_vs_engine_read": {"30ff29d9": [1, 1], "2ae2f0e0": [0, 0]}, "met": false, "falsified": true}

