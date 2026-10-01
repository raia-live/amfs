# The team grows: what a joiner inherits (`rsi-teamgrows-dev`)

Veterans = replicas of one hosted identity; each joiner = a new identity. First exposure = the joiner's first episode on a task class after joining. *Poisoned* = failed first attempt with an untested peer note in context and no validated hit; *guided* = succeeded with a validated hit. *Platform* hits = keys no agent in the cell wrote (lesson-contrast entries, fix memories, seeds). A hit with no evidence status at all (arms that track no outcomes) counts as untested. *Disagreement* = the joiner's first answer differs from the veterans' modal answer on that class over their last 10 pre-join episodes; read it next to the veterans' late success — disagreeing with a team that had settled on the wrong answer is not a defect. CIs are bootstrap over cells.

## pioneer: agent 1 works alone until 60; agents 2..6 join; regime change at 20

Cells: 12 (support, ci-fix; seeds 11, 23); complete: 10/12 reached ep 65+.

### A. Joiner first exposures after ep 60

| arm                       | cells | veterans ep50-59 1st | veterans changed post 1st |      joiner 1st-exp all |         1st-exp changed |       1st-exp unchanged | stale-pick (changed 1st-exp) | escalated |       joiner all eps 1st | joiner stale (changed, all eps) |
|---------------------------|-------|----------------------|---------------------------|-------------------------|-------------------------|-------------------------|------------------------------|-----------|--------------------------|---------------------------------|
| SenseLab, outcome loop on |     4 |                 0.97 |                      0.69 | 1.00 [1.00,1.00] (n=90) | 1.00 [1.00,1.00] (n=36) | 1.00 [1.00,1.00] (n=54) |                         0.00 |      0.00 | 1.00 [1.00,1.00] (n=136) |                            0.00 |
| pgvector+rerank           |     4 |                 0.75 |                      0.28 | 0.68 [0.52,0.83] (n=90) | 0.31 [0.15,0.50] (n=36) | 0.93 [0.78,1.00] (n=54) |                         0.17 |      0.20 | 0.62 [0.46,0.78] (n=136) |                            0.20 |
| senselab-rsi              |     4 |                 1.00 |                      0.76 | 1.00 [1.00,1.00] (n=90) | 1.00 [1.00,1.00] (n=36) | 1.00 [1.00,1.00] (n=54) |                         0.00 |      0.00 | 1.00 [1.00,1.00] (n=136) |                            0.00 |

### B. Inheritance quality: what sat in the joiner's context on its first exposures

| arm                       | 1st-exposures | any validated hit | any untested peer note | any discredited (avoid) | any platform-authored | hits/ep validated / untested-peer / platform / avoid / stale |     veterans' own ep50-59 (same) |     poisoned | poisoned (changed) |       guided |
|---------------------------|---------------|-------------------|------------------------|-------------------------|-----------------------|--------------------------------------------------------------|----------------------------------|--------------|--------------------|--------------|
| SenseLab, outcome loop on |            90 |              1.00 |                   0.64 |                    1.00 |                  0.00 |                             2.61 / 1.13 / 0.00 / 3.00 / 0.03 | 2.83 / 0.00 / 0.00 / 3.15 / 0.10 |  0/90 (0.00) |        0/36 (0.00) | 90/90 (1.00) |
| pgvector+rerank           |            90 |              0.00 |                   1.00 |                    0.00 |                  0.12 |                             0.00 / 7.26 / 0.13 / 0.00 / 0.96 | 0.00 / 0.00 / 0.35 / 0.00 / 0.70 | 29/90 (0.32) |       25/36 (0.69) |  0/90 (0.00) |
| senselab-rsi              |            90 |              0.99 |                   0.57 |                    1.00 |                  0.00 |                             2.48 / 1.11 / 0.00 / 3.00 / 0.01 | 2.73 / 0.00 / 0.00 / 3.00 / 0.03 |  0/90 (0.00) |        0/36 (0.00) | 89/90 (0.99) |

### C. Onboarding velocity

| arm                       | joiners | reached parity | median episodes to parity |              never | median tokens to parity | cold ep 1 | cold ep 2 | cold ep 3 | cold ep 4 | cold ep 5 |
|---------------------------|---------|----------------|---------------------------|--------------------|-------------------------|-----------|-----------|-----------|-----------|-----------|
| SenseLab, outcome loop on |      20 |          20/20 |                       5.0 |  0% (n/a within 7) |                   22315 |      1.00 |      1.00 |      1.00 |      1.00 |      1.00 |
| pgvector+rerank           |      20 |          16/20 |                       5.0 | 20% (n/a within 7) |                   26577 |      0.70 |      0.60 |      0.40 |      0.70 |      0.60 |
| senselab-rsi              |      20 |          20/20 |                       5.0 |  0% (n/a within 7) |                   20992 |      1.00 |      1.00 |      1.00 |      1.00 |      1.00 |

### D/E. Disagreement with the veterans and cost

| arm                       | joiner answers differing from veterans' modal ep50-59 answer | joiner success (all eps) | tokens / joiner ep | tokens / joiner-solved task |
|---------------------------|--------------------------------------------------------------|--------------------------|--------------------|-----------------------------|
| SenseLab, outcome loop on |                                     0.02 [0.00,0.05] (n=112) | 1.00 [1.00,1.00] (n=136) |               4380 |                        4381 |
| pgvector+rerank           |                                     0.06 [0.02,0.10] (n=112) | 0.75 [0.71,0.80] (n=136) |               5615 |                        7556 |
| senselab-rsi              |                                     0.00 [0.00,0.00] (n=112) | 1.00 [1.00,1.00] (n=136) |               4331 |                        4331 |

### F. RSI arm: what the veterans' loop had produced by the join, and the joiners' own identities

| cell                                                                             | fixes (all) | proposed / shipped before join | first fix lag (eps after change) |           levers |             sources |                                                                                                      joiner identities |
|----------------------------------------------------------------------------------|-------------|--------------------------------|----------------------------------|------------------|---------------------|------------------------------------------------------------------------------------------------------------------------|
| support|senselab-rsi|gpt-5.5|s11|e100-d300-n0.0-f0-a6-pioneer60-c20-x0.5-oid-rep |           1 |                          1 / 0 |                                0 | {"knowledge": 1} | {"repair_agent": 1} | 2: 7 verdicts, 0 fixes; 3: 7 verdicts, 0 fixes; 4: 7 verdicts, 0 fixes; 5: 7 verdicts, 0 fixes; 6: 6 verdicts, 0 fixes |
| support|senselab-rsi|gpt-5.5|s23|e100-d300-n0.0-f0-a6-pioneer60-c20-x0.5-oid-rep |           0 |                          0 / 0 |                                - |               {} |                  {} | 2: 7 verdicts, 0 fixes; 3: 7 verdicts, 0 fixes; 4: 7 verdicts, 0 fixes; 5: 7 verdicts, 0 fixes; 6: 6 verdicts, 0 fixes |
| ci-fix|senselab-rsi|gpt-5.5|s11|e100-d300-n0.0-f0-a6-pioneer60-c20-x0.5-oid-rep  |           1 |                          1 / 0 |                              -13 | {"knowledge": 1} | {"repair_agent": 1} | 2: 7 verdicts, 0 fixes; 3: 7 verdicts, 0 fixes; 4: 7 verdicts, 0 fixes; 5: 7 verdicts, 0 fixes; 6: 6 verdicts, 0 fixes |
| ci-fix|senselab-rsi|gpt-5.5|s23|e100-d300-n0.0-f0-a6-pioneer60-c20-x0.5-oid-rep  |           0 |                          0 / 0 |                                - |               {} |                  {} | 2: 7 verdicts, 0 fixes; 3: 7 verdicts, 0 fixes; 4: 7 verdicts, 0 fixes; 5: 7 verdicts, 0 fixes; 6: 6 verdicts, 0 fixes |

## newcomer: agents 1..5 work until 60; agent 6 joins; regime change at 20

Cells: 12 (support, ci-fix; seeds 11, 23); complete: 10/12 reached ep 65+.

### A. Joiner first exposures after ep 60

| arm                       | cells | veterans ep50-59 1st | veterans changed post 1st |      joiner 1st-exp all |         1st-exp changed |       1st-exp unchanged | stale-pick (changed 1st-exp) | escalated |       joiner all eps 1st | joiner stale (changed, all eps) |
|---------------------------|-------|----------------------|---------------------------|-------------------------|-------------------------|-------------------------|------------------------------|-----------|--------------------------|---------------------------------|
| SenseLab, outcome loop on |     4 |                 1.00 |                      0.79 | 1.00 [1.00,1.00] (n=32) | 1.00 [1.00,1.00] (n=10) | 1.00 [1.00,1.00] (n=22) |                         0.00 |      0.00 | 1.00 [1.00,1.00] (n=160) |                            0.00 |
| pgvector+rerank           |     4 |                 0.65 |                      0.28 | 0.75 [0.62,0.88] (n=32) | 0.40 [0.33,0.50] (n=10) | 0.91 [0.80,1.00] (n=22) |                         0.00 |      0.16 | 0.62 [0.45,0.80] (n=160) |                            0.10 |
| senselab-rsi              |     4 |                 1.00 |                      0.70 | 1.00 [1.00,1.00] (n=32) | 1.00 [1.00,1.00] (n=10) | 1.00 [1.00,1.00] (n=22) |                         0.00 |      0.00 | 1.00 [1.00,1.00] (n=160) |                            0.00 |

### B. Inheritance quality: what sat in the joiner's context on its first exposures

| arm                       | 1st-exposures | any validated hit | any untested peer note | any discredited (avoid) | any platform-authored | hits/ep validated / untested-peer / platform / avoid / stale |     veterans' own ep50-59 (same) |    poisoned | poisoned (changed) |       guided |
|---------------------------|---------------|-------------------|------------------------|-------------------------|-----------------------|--------------------------------------------------------------|----------------------------------|-------------|--------------------|--------------|
| SenseLab, outcome loop on |            32 |              1.00 |                   0.72 |                    1.00 |                  0.00 |                             2.25 / 1.31 / 0.00 / 3.00 / 0.12 | 2.65 / 0.72 / 0.00 / 3.00 / 0.28 | 0/32 (0.00) |        0/10 (0.00) | 32/32 (1.00) |
| pgvector+rerank           |            32 |              0.00 |                   1.00 |                    0.00 |                  0.09 |                             0.00 / 5.88 / 0.09 / 0.00 / 0.28 | 0.00 / 5.20 / 0.20 / 0.00 / 0.65 | 8/32 (0.25) |        6/10 (0.60) |  0/32 (0.00) |
| senselab-rsi              |            32 |              1.00 |                   0.66 |                    1.00 |                  0.00 |                             2.91 / 1.06 / 0.00 / 3.00 / 0.12 | 2.85 / 0.80 / 0.00 / 3.00 / 0.25 | 0/32 (0.00) |        0/10 (0.00) | 32/32 (1.00) |

### C. Onboarding velocity

| arm                       | joiners | reached parity | median episodes to parity |              never | median tokens to parity | cold ep 1 | cold ep 2 | cold ep 3 | cold ep 4 | cold ep 5 |
|---------------------------|---------|----------------|---------------------------|--------------------|-------------------------|-----------|-----------|-----------|-----------|-----------|
| SenseLab, outcome loop on |       4 |            4/4 |                       5.0 | 0% (n/a within 40) |                   19328 |      1.00 |      1.00 |      1.00 |      1.00 |      1.00 |
| pgvector+rerank           |       4 |            4/4 |                      11.5 | 0% (n/a within 40) |                   55176 |      1.00 |      0.75 |      0.75 |      1.00 |      0.25 |
| senselab-rsi              |       4 |            4/4 |                       5.0 | 0% (n/a within 40) |                   22468 |      1.00 |      1.00 |      1.00 |      1.00 |      1.00 |

### D/E. Disagreement with the veterans and cost

| arm                       | joiner answers differing from veterans' modal ep50-59 answer | joiner success (all eps) | tokens / joiner ep | tokens / joiner-solved task |
|---------------------------|--------------------------------------------------------------|--------------------------|--------------------|-----------------------------|
| SenseLab, outcome loop on |                                     0.00 [0.00,0.00] (n=127) | 1.00 [1.00,1.00] (n=160) |               4085 |                        4085 |
| pgvector+rerank           |                                     0.05 [0.00,0.09] (n=127) | 0.76 [0.71,0.82] (n=160) |               5152 |                        6836 |
| senselab-rsi              |                                     0.00 [0.00,0.00] (n=127) | 1.00 [1.00,1.00] (n=160) |               4230 |                        4230 |

### F. RSI arm: what the veterans' loop had produced by the join, and the joiners' own identities

| cell                                                                              | fixes (all) | proposed / shipped before join | first fix lag (eps after change) |                           levers |             sources |       joiner identities |
|-----------------------------------------------------------------------------------|-------------|--------------------------------|----------------------------------|----------------------------------|---------------------|-------------------------|
| support|senselab-rsi|gpt-5.5|s11|e100-d300-n0.0-f0-a6-newcomer60-c20-x0.5-oid-rep |           0 |                          0 / 0 |                                - |                               {} |                  {} | 6: 40 verdicts, 0 fixes |
| support|senselab-rsi|gpt-5.5|s23|e100-d300-n0.0-f0-a6-newcomer60-c20-x0.5-oid-rep |           2 |                          2 / 0 |                                3 |                 {"knowledge": 2} | {"repair_agent": 2} | 6: 40 verdicts, 0 fixes |
| ci-fix|senselab-rsi|gpt-5.5|s11|e100-d300-n0.0-f0-a6-newcomer60-c20-x0.5-oid-rep  |           2 |                          2 / 0 |                              -13 | {"knowledge": 1, "procedure": 1} | {"repair_agent": 2} | 6: 40 verdicts, 0 fixes |
| ci-fix|senselab-rsi|gpt-5.5|s23|e100-d300-n0.0-f0-a6-newcomer60-c20-x0.5-oid-rep  |           2 |                          2 / 0 |                                2 |                 {"knowledge": 2} | {"repair_agent": 2} | 6: 40 verdicts, 0 fixes |
