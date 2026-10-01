# Reproducibility package for `rsi.pdf`

Everything the paper cites that is not already in the repository's tracked analysis code
(`benchmarks/continual_learning/analysis/`, frozen at tag `analysis-freeze-rsi`; the one
post-freeze wiring correction is commit `e04f78cf`) or in the preregistration
(`benchmarks/continual_learning/HYPOTHESIS_RSI.md`, amendments (a)–(aa) with their read records).

## scripts/ — supplementary reads, none of which redefines a metric

- `stage_conf.py` — the fleet-1 staging step: concatenates the four per-seed confirmatory runs
  (`rsi-conf-f1-s{37,41,59,73}`) file-by-file into `results/rsi-conf-f1`, because the frozen
  reader resolves one run per system. Data staging only; the development read is byte-identical
  before and after.
- `f1_cut_split.py` (output: `readouts/f1_cut_split.txt`) — the fleet-1 episodes split at the
  `post-canary-zero` cut (03:40:19Z), with the moved-half direction on the pre-cut episodes alone.
- `canary_classes.py` — the three-class canary read of amendment (z″) (decided on a treated arm /
  closed unexercised / other), P13″, P15′ and P16, from the sealed traces and `amfs_reuse_events`.
  Needs `PGPASS_DSN` (read-only access to the benchmark account's rows) and `CL_CUT`
  (`2026-09-30T08:21:05Z` for the `post-canary-visible` run, `2026-09-30T12:52:04Z` for the
  `post-replay-evidence` rerun). Reads by the diagnostic agent `diag-branch-treatment` are excluded.
- `fleet10_curve.py` — the long-horizon learning curve of the fleet-10 cells: first-attempt
  success per 10-episode block, pooled over the cells of several runs, cell-level bootstrap,
  using the frozen reader's population rule (`analysis.rsi.population`) with `SEEDS` and
  `CELL_EPISODES` set as the reader's CLI sets them for the confirmatory seeds; `CL_SEEDS=11,23
  CL_CELL_EPISODES=400` for the exploratory cells of amendment (ab). Defaults unchanged.

Run from the repository root with `PYTHONPATH=. .venv/bin/python <script> <runs...>`.

## readouts/

- `rsi-report-confirmatory/` — `REPORT.md` and `report.json` from
  `analysis.rsi --seeds 37 41 59 73 --models gpt-5.5 claude-opus-5 --runs rsi-conf-f1
  --report-dir results/rsi-report-confirmatory --paper` (the fleet-1 confirmatory read; every
  `CONF_TAB_*` table in the paper is filled from it).
- `canary/<run>/canary_readout.json` — `analysis.rsi_canary_readout --run <run>` for the fleet-10
  cells on `post-canary-visible` (`rsi-conf-e2b-s*`, finding 13, reported as measured) and on
  `post-replay-evidence` (`rsi-conf-e2b-r2-s*`, the confirmatory fleet-10 read).
- `canary/classes_*.md` — the three-class read on each of the two fleet-10 runs, and
  `classes_exploratory_ab.md` on the exploratory run (`CL_CUT=2026-09-30T12:52:04Z`).
- `canary/rsi-x-canary-s{11,23}/` — the exploratory 400-episode run of amendment (ab)
  (`--episodes 400 --change-focus 1.0`, development seeds): `canary_readout.json` and
  `readout.md` from the frozen `rsi_canary_readout`, `canary_classes.json` from the harness.
  Exploratory; enters no confirmatory table.
- `fleet10_curve_*.txt` — the long-horizon curve on each of the two fleet-10 runs, and
  `fleet10_curve_exploratory_ab.txt` on the exploratory run (focus 1.0; unchanged classes are
  not measured after episode 50 there, so that block is empty).
- `f1_cut_split.txt` — see above.

## The companion paper, `rsi_position.pdf`

`rsi_position.tex` is a standalone paper on what recursive self-improvement is, the problem it
solves and how it differs from giving an agent a memory. Every number in it is quoted from
`rsi.pdf` and from `readouts/rsi-report-confirmatory/report.json`; it has no filler markers.
Its two figures (`figures/pos_headline.png`, `figures/pos_curve.png`) are drawn from
`report.json` (headline block, with its cell-level intervals) and from
`readouts/fleet10_curve_post-replay-evidence.txt`. Build: `tectonic -X compile rsi_position.tex`.

## coding/ — P11(a′)

`rsi-coding-dev/` (37 drafts of `rsi-c6` and `rsi-e2b-c6`) and `rsi-coding-confirmatory/`
(39 drafts of `rsi-conf-f1-s*`): `sheet.csv` (the drafts as shown to the coder),
`INSTRUCTIONS.md`, `key.json` (fix id, arm, cell, status per item — withheld from the coder),
`answers_author.csv` (the author's coding: conditional Y/N, the quoted condition, and whether it
names the true set-C variable) and `score.json` (`analysis.rsi_coding_sheet score`). The true
set-C rules are in `scenarios/realworld.py` (`SUPPORT_CHANGES_C`, `CI_CHANGES_C`).

## Not in this package

Run records (`results/rsi-*`: episodes, arm reports, budget snapshots, run logs) are large and
carry the account's opaque scope ids; they are available on request. Nothing here contains a
credential or an endpoint.
