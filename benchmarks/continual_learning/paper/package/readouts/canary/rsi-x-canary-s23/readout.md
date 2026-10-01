## rsi-x-canary-s23: 4 canaries
| cell | fix | from | status | reason | canary graded/exposed | control graded/exposed | kept cases | kept FAS canary | kept FAS control | ctrl−can |
|---|---|---|---|---|---|---|---|---|---|---|
| ci-fix senselab-rsi s23 | e3a81d36 | test_passed | running |  | 7/14 | 4/16 | 2 | 1.00 (10) | 1.00 (10) | 0.00 |
| ci-fix senselab-rsi s23 | 0a2f5553 | test_passed | passed |  | 25/38 | 18/42 | 2 | 0.89 (18) | 0.88 (26) | -0.00 |
| ci-fix senselab-rsi s23 | dd421419 | test_passed | passed |  | 30/69 | 39/65 | 1 | 0.96 (27) | 0.97 (39) | 0.01 |
| support senselab-rsi s23 | 7fd04380 | test_inconclusive | failed |  | 8/12 | 20/101 | 2 | 0.25 (4) | 0.67 (54) | 0.42 |
p12: {"reached_floor": 3, "share": 0.75, "met": true, "falsified": false}
p13: {"decided": 3, "share": 0.75, "met": true, "falsified": false}
p14: {"population": 3, "passes": 2, "violations": [], "met": true, "undefined": false, "margin": 0.1}
p13 by start: {"test_passed": {"canaries": 3, "decided": 2, "passed": 2, "failed": 0}, "test_inconclusive": {"canaries": 1, "decided": 1, "passed": 0, "failed": 1}}
