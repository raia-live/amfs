## rsi-x-canary-s11: 4 canaries
| cell | fix | from | status | reason | canary graded/exposed | control graded/exposed | kept cases | kept FAS canary | kept FAS control | ctrl−can |
|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix senselab-rsi s11 | dc722d08 | test_passed | running |  | 16/63 | 12/56 | 2 | 0.89 (28) | 1.00 (30) | 0.11 |
| ci-fix senselab-rsi s11 | 130b2a03 | test_passed | passed |  | 58/102 | 25/107 | 1 | 0.90 (51) | 0.84 (49) | -0.07 |
| support senselab-rsi s11 | a09e0892 | test_passed | running |  | 21/116 | 17/124 | 1 | 0.46 (48) | 0.47 (62) | 0.01 |
| support senselab-rsi s11 | a3da998e | test_passed | passed |  | 51/51 | 53/53 | 0 | 0.45 (31) | 0.54 (41) | 0.08 |
p12: {"reached_floor": 4, "share": 1.0, "met": true, "falsified": false}
p13: {"decided": 2, "share": 0.5, "met": true, "falsified": false}
p14: {"population": 1, "passes": 1, "violations": [], "met": true, "undefined": false, "margin": 0.1}
p13 by start: {"test_passed": {"canaries": 4, "decided": 2, "passed": 2, "failed": 0}}
