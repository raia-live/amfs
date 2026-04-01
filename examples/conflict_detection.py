"""Conflict Detection — detect when another agent modified an entry since your last read.

Demonstrates AMFS conflict policies: LAST_WRITE_WINS (default) and RAISE,
plus custom merge callbacks for resolving conflicts.

    uv run python examples/conflict_detection.py
"""

from __future__ import annotations

from typing import Any

from amfs import AgentMemory, ConflictPolicy
from amfs_core.exceptions import StaleWriteError
from amfs_filesystem.adapter import FilesystemAdapter

import tempfile
from pathlib import Path


def demo_last_write_wins(root: Path) -> None:
    """Default behavior: last write wins, no error."""
    print("--- LAST_WRITE_WINS (default) ---")
    adapter = FilesystemAdapter(root=root / "lww", namespace="demo")

    agent_a = AgentMemory(agent_id="agent-a", adapter=adapter)
    agent_b = AgentMemory(agent_id="agent-b", adapter=adapter)

    agent_a.write("svc", "config", {"timeout": 30})

    agent_b.read("svc", "config")
    agent_a.write("svc", "config", {"timeout": 60})
    agent_b.write("svc", "config", {"timeout": 45})

    final = agent_a.read("svc", "config")
    print(f"  Agent A wrote 30, then 60. Agent B wrote 45.")
    print(f"  Final value: {final.value}")
    print()

    agent_a.close()
    agent_b.close()


def demo_raise_on_conflict(root: Path) -> None:
    """RAISE policy: throws StaleWriteError when another agent modified the entry."""
    print("--- RAISE on conflict ---")
    adapter = FilesystemAdapter(root=root / "raise", namespace="demo")

    agent_a = AgentMemory(agent_id="agent-a", adapter=adapter)
    agent_b = AgentMemory(
        agent_id="agent-b", adapter=adapter,
        conflict_policy=ConflictPolicy.RAISE,
    )

    agent_a.write("svc", "config", {"timeout": 30})

    agent_b.read("svc", "config")
    agent_a.write("svc", "config", {"timeout": 60})

    try:
        agent_b.write("svc", "config", {"timeout": 45})
    except StaleWriteError as e:
        print(f"  Caught StaleWriteError: {e}")
        print(f"  Agent B read v{e.read_version}, but v{e.current_version} exists now")
        print(f"  Modified by: {e.current_agent}")
    print()

    agent_a.close()
    agent_b.close()


def demo_merge_callback(root: Path) -> None:
    """Custom merge: callback resolves conflicts by merging values."""
    print("--- Custom merge callback ---")
    adapter = FilesystemAdapter(root=root / "merge", namespace="demo")

    def merge_configs(
        our_read: Any, current: Any, new_value: Any,
    ) -> dict:
        """Merge strategy: take the max timeout from both sides."""
        merged = {**current.value, **new_value}
        merged["timeout"] = max(current.value.get("timeout", 0), new_value.get("timeout", 0))
        merged["_merged"] = True
        return merged

    agent_a = AgentMemory(agent_id="agent-a", adapter=adapter)
    agent_b = AgentMemory(
        agent_id="agent-b", adapter=adapter,
        on_conflict=merge_configs,
    )

    agent_a.write("svc", "config", {"timeout": 30, "retries": 3})

    agent_b.read("svc", "config")
    agent_a.write("svc", "config", {"timeout": 60, "retries": 3})

    result = agent_b.write("svc", "config", {"timeout": 45, "retries": 5})
    print(f"  Agent A: timeout=60, retries=3")
    print(f"  Agent B wanted: timeout=45, retries=5")
    print(f"  Merged result: {result.value}")
    print()

    agent_a.close()
    agent_b.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        demo_last_write_wins(root)
        demo_raise_on_conflict(root)
        demo_merge_callback(root)


if __name__ == "__main__":
    main()
