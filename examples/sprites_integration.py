"""Fly.io Sprites × SenseLab — memory that survives a disposable microVM.

This simulates the Sprites lifecycle *locally* so you can see the magic moment
without provisioning a real Sprite: two independent sessions (each standing in
for a freshly spun-up microVM) share one durable ``entity_path``. The second
session boots already knowing what the first learned.

Run it:

    uv run python examples/sprites_integration.py

Against hosted SenseLab, set these first and the same code points at the SaaS:

    export AMFS_HTTP_URL=https://amfs-login.sense-lab.ai
    export AMFS_API_KEY=amfs_sk_...

On a real Sprite you would call ``provision_memory()`` with no arguments — the
base image exports AMFS_ENTITY_PATH / AMFS_HTTP_URL / AMFS_API_KEY for you.
"""

from __future__ import annotations

import os
import tempfile

from amfs_sprites import derive_entity_path, provision_memory


def sprite_one(entity_path: str) -> None:
    """First microVM: nothing to hydrate, so it learns and records."""
    print("\n=== Sprite #1 (cold) ===")
    with provision_memory(entity_path=entity_path, agent_id="checkout-agent") as s:
        print(s.hydrate_prompt())

        # The agent does work and forms durable knowledge.
        s.write(
            "decision-idempotency",
            "All charge calls send an idempotency key derived from the cart id, "
            "so client retries never double-bill. Learned after a duplicate-charge incident.",
            confidence=0.9,
        )
        s.write(
            "risk-provider-timeout",
            "The payments provider p99 spikes to ~8s during flash sales; keep the "
            "read timeout >= 10s or checkouts fail under load.",
            confidence=0.7,
            memory_type="belief",
        )

        # Snapshot the decision trace + a summary for the next Sprite.
        s.commit_outcome(
            "checkout-hardening",
            "success",
            task_input="make checkout resilient to retries and provider latency",
            summary="Hardened checkout: idempotency keys on charges; raised provider read timeout.",
        )
    print("Sprite #1 done — microVM discarded, memory persisted.")


def sprite_two(entity_path: str) -> None:
    """Second microVM: a brand-new VM that boots already briefed."""
    print("\n=== Sprite #2 (fresh VM, hydrated) ===")
    with provision_memory(entity_path=entity_path, agent_id="checkout-agent") as s:
        prompt = s.hydrate_prompt()
        print(prompt)

        assert "idempotency" in prompt, "expected prior memory to hydrate"
        print("\n>>> The fresh Sprite recalled decisions it never made itself. <<<")


def main() -> None:
    # Local demo store. Delete these two lines (and set AMFS_HTTP_URL /
    # AMFS_API_KEY) to run against hosted SenseLab instead.
    os.environ.setdefault("AMFS_DATA_DIR", tempfile.mkdtemp(prefix="sprite-demo-"))

    entity_path = derive_entity_path("acme", "checkout")
    print(f"Home entity for this workload: {entity_path}")

    sprite_one(entity_path)
    sprite_two(entity_path)


if __name__ == "__main__":
    main()
