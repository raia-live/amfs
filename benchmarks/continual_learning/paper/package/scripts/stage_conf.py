"""Stage the per-seed confirmatory run directories as one run directory for the frozen reader.

analysis/rsi.py resolves ONE run per system (`resolve()` returns the first listed run holding
finished cells of the arm), while confirm_rsi.sh launched fleet-1 as four per-seed runs
(rsi-conf-f1-s37 … s73). Concatenating the four directories' jsonl files into results/rsi-conf-f1
is a data-layout step; no analysis code is touched. manifest.json records the sources.

  python /tmp/stage_conf.py f1        -> results/rsi-conf-f1   (from rsi-conf-f1-s{37,41,59,73})
  python /tmp/stage_conf.py e2b       -> results/rsi-conf-e2b  (from rsi-conf-e2b-s{37,41,59,73})
"""
import json
import shutil
import sys
from pathlib import Path

RESULTS = Path.home() / "Desktop/projects/amfs/benchmarks/continual_learning/results"
SEEDS = (37, 41, 59, 73)
JSONL = ("episodes.jsonl", "arm_reports.jsonl", "budget_snapshots.jsonl", "confidence_traces.jsonl")


def stage(proto: str) -> None:
    out = RESULTS / f"rsi-conf-{proto}"
    srcs = [RESULTS / f"rsi-conf-{proto}-s{s}" for s in SEEDS]
    missing = [s for s in srcs if not (s / "episodes.jsonl").exists()]
    if missing:
        sys.exit(f"missing: {[m.name for m in missing]}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    counts = {}
    for name in JSONL:
        n = 0
        with (out / name).open("w") as w:
            for s in srcs:
                p = s / name
                if p.exists():
                    for line in p.open():
                        if line.strip():
                            w.write(line if line.endswith("\n") else line + "\n")
                            n += 1
        counts[name] = n
    cells = []
    manifests = []
    for s in srcs:
        if (s / "cells.json").exists():
            cells += json.loads((s / "cells.json").read_text())
        if (s / "manifest.json").exists():
            manifests.append(json.loads((s / "manifest.json").read_text()))
    (out / "cells.json").write_text(json.dumps(cells, indent=1) + "\n")
    base = dict(manifests[0]) if manifests else {}
    base["run"] = out.name
    base["cells"] = [c for m in manifests for c in (m.get("cells") or [])]
    base["staged_from"] = [s.name for s in srcs]
    base["started"] = min((m.get("started") or "" for m in manifests), default=None)
    (out / "manifest.json").write_text(json.dumps(base, indent=1) + "\n")
    print(out.name, {**counts, "cells": len(cells), "from": [s.name for s in srcs]})


if __name__ == "__main__":
    stage(sys.argv[1] if len(sys.argv) > 1 else "f1")
