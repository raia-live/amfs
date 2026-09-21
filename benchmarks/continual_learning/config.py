"""Shared configuration for the continual-learning benchmark.

Everything that a reader might want to change lives here or in the ``.env`` file next
to this module. No credentials are ever hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
# The benchmark's .env is authoritative: it overrides any key already exported in the shell,
# so a stale ANTHROPIC_API_KEY/OPENAI_API_KEY in the environment cannot silently win.
# ``CL_ENV_FILE`` names another file — the way to point a run at a local SenseLab server
# (AMFS_HTTP_URL / AMFS_API_KEY) without editing the production .env.
load_dotenv(os.environ.get("CL_ENV_FILE") or HERE / ".env", override=True)


def env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"{name} is required (set it in benchmarks/continual_learning/.env)")
    return value


# ---------------------------------------------------------------------------
# Models and prices (USD per 1M tokens, standard tier, verified 2026-09-16)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Price:
    input: float
    cached_input: float
    output: float


PRICES: dict[str, Price] = {
    # OpenAI — developers.openai.com/api/docs/models
    "gpt-5.5": Price(5.00, 0.50, 30.00),
    "gpt-5.4": Price(2.50, 0.25, 15.00),
    "gpt-5.4-mini": Price(0.75, 0.075, 4.50),
    # Anthropic — platform.claude.com/docs/en/about-claude/pricing
    "claude-fable-5-1": Price(10.00, 0.25, 50.00),
    "claude-opus-5": Price(5.00, 0.50, 25.00),
    "claude-sonnet-5": Price(2.00, 0.20, 10.00),
    "claude-haiku-4-5-20251001": Price(1.00, 0.10, 5.00),
}


def cost_usd(model: str, prompt: int, completion: int, cached: int = 0) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    uncached = max(prompt - cached, 0)
    return (uncached * p.input + cached * p.cached_input + completion * p.output) / 1e6


AGENT_MODELS: list[str] = [
    m.strip() for m in (env("CL_AGENT_MODELS") or "gpt-5.5,claude-opus-5").split(",") if m.strip()
]
JUDGE_MODEL: str = env("CL_JUDGE_MODEL") or "gpt-5.4"
# Reasoning effort for the agents. Recorded in the manifest; identical across arms.
OPENAI_REASONING_EFFORT: str = env("CL_OPENAI_REASONING_EFFORT") or "low"
ANTHROPIC_EFFORT: str = env("CL_ANTHROPIC_EFFORT") or "low"

# ---------------------------------------------------------------------------
# Study constants (mirror HYPOTHESIS.md)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Study:
    # Scale (amended 2026-09-16, before the main run, on the owner's instruction to prioritise
    # credibility over cost): core scenarios run 40 episodes, in both a single-agent and a
    # 6-agent fleet configuration, over 6 seeds, against a store pre-loaded with 300 realistic
    # distractor entries. Probes run 20 episodes, single agent, 5 seeds, clean store.
    # Grid v2 (2026-09-17): core scenarios run 60 episodes with a regime change at episode 20,
    # and half of the post-change tasks drawn from the classes whose truth changed. 40 post-
    # change episodes is what it takes to see a stale lesson fall and a replacement rise
    # within one cell; grid v1's 40/20 split left ~15 changed-class tasks per cell.
    episodes: int = 20                 # probes
    core_episodes: int = 60            # core scenarios
    core_change_at: int = 20
    core_change_focus: float = 0.5
    long_horizon_episodes: int = 60
    seeds: tuple[int, ...] = (11, 23, 37, 41, 59, 73)          # core
    probe_seeds: tuple[int, ...] = (11, 23, 37, 41, 59)        # probes
    sweep_seeds: tuple[int, ...] = (11, 23, 37)
    fleet_sizes: tuple[int, ...] = (1, 6)                       # core scenarios run in both
    core_distractors: int = 300
    retry_budget: int = 3
    burn_in: int = 5
    convergence_streak: int = 5
    reliability_target: float = 0.95
    reliability_window: int = 10
    min_confidence_gate: float = 0.5
    top_k: int = 5
    max_tokens_per_turn: int = 700
    diy_consolidate_every: int = 5
    budget_usd: float = 0.0  # 0 = no ceiling (owner decision 2026-09-16; was 600)
    # Zep dropped from the study 2026-09-18 (owner decision): the account exhausted its
    # episode credits after the main run (403 "over the episode credit usage limit"), so no
    # regime-change cells could be produced. The arm implementation stays for reproduction.
    arms: tuple[str, ...] = (
        "none", "pgvector", "pgvector-diy", "pgvector-diy+outcomes", "mem0",
        "senselab", "senselab-episode", "senselab-nofeedback",
    )
    # Real-world families (what teams deploy today) — full seed count
    core_scenarios: tuple[str, ...] = (
        "support", "ci-fix", "analytics", "concierge", "retention", "order-ops", "diagnose",
    )
    # Mechanism probes (isolate one failure mode each) — reduced seed count
    probe_scenarios: tuple[str, ...] = (
        "runbook", "drift-fact", "drift-tool", "triage", "handoff", "unknowns", "fleet", "sizing",
    )
    scenarios: tuple[str, ...] = core_scenarios + probe_scenarios
    generalization_scenarios: tuple[str, ...] = (
        "support", "ci-fix", "analytics", "concierge", "retention", "order-ops", "diagnose", "sizing",
    )
    long_horizon_scenarios: tuple[str, ...] = ("support", "diagnose")
    sweep_arms: tuple[str, ...] = ("senselab", "pgvector-diy+outcomes", "mem0")
    # Recursive-learning proof (2026-09-20). Two protocols share the ``senselab-repair`` arm:
    #   repair-vs-demote  ``ci-fix`` x {senselab, senselab-repair, raw-traces}: does the shipped
    #                     repair loop beat demotion-only on success, unnecessary edits and cost,
    #                     on a fixed model? The narrow intervention claim.
    #   composition       ``fleet-disjoint`` x composition_arms: can partial discoveries held by
    #                     different agents be combined into procedures none of them had, and does
    #                     that beat the same model reading every raw trace?
    # Both need a dev Pro deployment (``AMFS_PRO_URL``) with the repair agent enabled; they are
    # not in the default grid. Run them with ``--arms`` / ``--scenarios`` explicitly.
    repair_arms: tuple[str, ...] = ("senselab", "senselab-repair", "raw-traces")
    composition_arms: tuple[str, ...] = ("raw-traces", "pgvector-diy", "senselab", "senselab-repair",
                                         "senselab-compose")
    composition_scenarios: tuple[str, ...] = ("fleet-disjoint",)
    composition_seeds: tuple[int, ...] = (11, 23, 37, 41, 59, 73)
    distractor_levels: tuple[int, ...] = (0, 200, 2000)
    label_noise_levels: tuple[float, ...] = (0.0, 0.10)
    feedback_delay_levels: tuple[int, ...] = (0, 3)


STUDY = Study()

# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------

AMFS_HTTP_URL = env("AMFS_HTTP_URL") or "https://amfs-login.sense-lab.ai"
# The Pro evaluation API (``/api/v1/eval``) the ``senselab-repair`` arm drives. Defaults to
# the memory host; the arm refuses to run against anything that does not look like a dev
# deployment (see ``pro_client.assert_dev``). ``AMFS_EVAL_API_KEY`` falls back to AMFS_API_KEY.
AMFS_PRO_URL = env("AMFS_PRO_URL") or AMFS_HTTP_URL
# ``raw-traces`` control arm: how much prior-episode transcript (in approximate tokens) the
# agent gets in context. Oldest episodes are dropped first when the cap is hit.
RAW_TRACES_TOKEN_CAP = int(env("CL_RAW_TRACES_TOKENS") or "20000")
PG_DSN = env("CL_PG_DSN") or "postgresql://amfs:amfs@localhost:5433/clbench"
EMBED_MODEL = env("CL_EMBED_MODEL") or "BAAI/bge-small-en-v1.5"

# Root scope under which every benchmark entry is written, so it can be found and
# cleaned up. Each (arm, scenario, seed) gets its own child scope.
SCOPE_ROOT = env("CL_SCOPE_ROOT") or "clbench"
