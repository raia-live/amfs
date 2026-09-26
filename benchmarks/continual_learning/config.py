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


# SenseLab Managed Models (the "students") and their Gemini base models. Students are priced at
# the customer list price for served inference (the overage tier in amfs-internal
# packages/managed-models/catalog.py, 2026-09-19): what a SenseLab user pays per token, which is
# the number comparable with the frontier list prices above. The provider cost SenseLab itself
# pays is lower (s1 0.25 / 1.50; s1-pro 0.30 / 2.50) and is reported separately by the analysis
# as ``STUDENT_PROVIDER_PRICES``. Base models (``serve_base`` twins) are priced at Google's list
# price for the same model, since that is what a team calling Gemini directly would pay.
STUDENT_PRICES: dict[str, Price] = {
    "senselab-s1": Price(1.00, 1.00, 5.00),
    "senselab-s1-pro": Price(1.25, 1.25, 8.00),
    "gemini-2.5-flash-lite": Price(0.10, 0.025, 0.40),
    "gemini-2.5-flash": Price(0.30, 0.03, 2.50),
}
STUDENT_PROVIDER_PRICES: dict[str, Price] = {
    "senselab-s1": Price(0.25, 0.25, 1.50),
    "senselab-s1-pro": Price(0.30, 0.30, 2.50),
}
# alias -> {public_model_id, family, price, serve_base, ...}; written by student/train.py.
STUDENT_REGISTRY = RESULTS_DIR / "students" / "registry.json"


def cost_usd(model: str, prompt: int, completion: int, cached: int = 0) -> float:
    p = PRICES.get(model) or STUDENT_PRICES.get(model)
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
    # Grid v4 (2026-09-19). Core scenarios run 60 episodes with a regime change at episode 20
    # and half of the post-change tasks drawn from the classes whose truth changed (the grid v2
    # split: 40 post-change episodes is what it takes to see a stale lesson fall and a
    # replacement rise within one cell). Every core scenario runs in three fleet configurations
    # — single agent, a 6-agent fleet, and a 10-agent fleet (see ``fleet_large*``) — with BOTH
    # agent models on the full grid, over 6 seeds, against a store pre-loaded with 300 realistic
    # distractor entries. Three new real-world families join the core tier (policy-qa, accounts,
    # dispatch), and the knowledge-transfer protocols (pioneer / newcomer, including a
    # cross-model newcomer via ``--newcomer-model``) report onboarding metrics: episodes and
    # tokens to parity with the veterans and the joining agent's cold-start curve
    # (analysis/transfer.py). Probes run 20 episodes, single agent, 5 seeds, clean store.
    episodes: int = 20                 # probes
    core_episodes: int = 60            # core scenarios
    core_change_at: int = 20
    core_change_focus: float = 0.5
    long_horizon_episodes: int = 60
    seeds: tuple[int, ...] = (11, 23, 37, 41, 59, 73)          # core
    probe_seeds: tuple[int, ...] = (11, 23, 37, 41, 59)        # probes
    sweep_seeds: tuple[int, ...] = (11, 23, 37)
    fleet_sizes: tuple[int, ...] = (1, 6)                       # core scenarios run in both
    # Large fleet. Ten agents over the standard 60 episodes would give each agent only 6
    # episodes — too few for a per-agent learning curve or an episodes-to-parity reading — so
    # the large-fleet cells run 100 episodes with the regime change at 30 (10 per agent, 7 of
    # them after the change). These are not loop defaults: the runner is invoked explicitly
    # with ``--fleet 10 --episodes 100 --change-at 30`` for those cells.
    fleet_large: int = 10
    fleet_large_episodes: int = 100
    fleet_large_change_at: int = 30
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
        "senselab", "senselab-episode", "senselab-nofeedback", "senselab-nopriors", "senselab-induce",
    )
    # Real-world families (what teams deploy today) — full seed count. policy-qa, accounts and
    # dispatch are the grid v4 additions (scenario modules land separately).
    core_scenarios: tuple[str, ...] = (
        "support", "ci-fix", "analytics", "concierge", "retention", "order-ops", "diagnose",
        "policy-qa", "accounts", "dispatch",
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
