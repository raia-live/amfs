"""AMFS integration for Fly.io Sprites.

Sprites are disposable, hardware-isolated microVMs. Their filesystem does not
survive across the fleet, so an agent that only remembers what's on local disk
starts every Sprite from zero. This package binds a Sprite to a durable
``entity_path`` in hosted SenseLab and hydrates the agent's context on boot, so
a freshly spun-up Sprite recalls everything prior sessions learned.

Typical use inside a Sprite:

    from amfs_sprites import provision_memory

    session = provision_memory(entity_path="sprites/acme/checkout")
    system_prompt = session.hydrate_prompt()   # inject into your agent
    # ... run the agent ...
    session.commit_outcome("checkout-refactor", "success",
                           task_input="reduce checkout latency")
"""

from amfs_sprites.integration import (
    SpriteSession,
    commit_sprite_outcome,
    derive_entity_path,
    hydrate_prompt,
    provision_memory,
)

__all__ = [
    "SpriteSession",
    "commit_sprite_outcome",
    "derive_entity_path",
    "hydrate_prompt",
    "provision_memory",
]
