"""AMFS Quickstart — get running in 30 seconds.

    uv run python examples/quickstart.py
"""

from amfs import AgentMemory, OutcomeType


def main() -> None:
    with AgentMemory(agent_id="quickstart-agent") as mem:
        # ---- Write some memories ----
        mem.write(
            "checkout-service",
            "retry-pattern",
            {"max_retries": 3, "backoff": "exponential", "ai_contribution": 0.8},
            confidence=0.9,
            pattern_refs=["retry-logic"],
        )
        mem.write(
            "checkout-service",
            "timeout-config",
            {"connect_ms": 3000, "read_ms": 10000},
            confidence=1.0,
        )
        mem.write(
            "payment-service",
            "circuit-breaker",
            {"threshold": 5, "reset_interval_s": 30},
            confidence=0.7,
            pattern_refs=["circuit-breaker"],
        )

        # ---- Read them back ----
        entry = mem.read("checkout-service", "retry-pattern")
        print(f"Read: {entry.entity_path}/{entry.key} = {entry.value}")
        print(f"  Written by: {entry.provenance.agent_id}")
        print(f"  Confidence: {entry.confidence}")
        print()

        # ---- Search across all entities ----
        print("High-confidence entries:")
        for e in mem.search(min_confidence=0.8, sort_by="confidence"):
            print(f"  {e.entry_key} (confidence={e.confidence})")
        print()

        # ---- Check what we read (auto-tracked) ----
        print(f"Session read log: {mem.read_log}")

        # ---- Record an outcome — auto-links to what we read ----
        mem.read("checkout-service", "retry-pattern")
        mem.read("checkout-service", "timeout-config")
        updated = mem.commit_outcome("INC-2047", OutcomeType.CRITICAL_FAILURE)
        print(f"\nOutcome back-propagated to {len(updated)} entries:")
        for e in updated:
            print(f"  {e.entry_key} → confidence={e.confidence:.4f}")

        # ---- Stats ----
        s = mem.stats()
        print(f"\nMemory stats: {s.total_entries} entries across {s.total_entities} entities")
        print(f"  Confidence range: {s.confidence_min:.2f} – {s.confidence_max:.2f}")
        print(f"  Outcome-linked: {s.outcome_linked_count}")


if __name__ == "__main__":
    main()
