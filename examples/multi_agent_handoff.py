"""Multi-Agent Handoff — Review agent writes context, Release agent reads it.

Demonstrates the core AMFS use case: cross-agent memory sharing with
automatic causal tracking and outcome back-propagation.

    uv run python examples/multi_agent_handoff.py
"""

from amfs import AgentMemory, OutcomeType


def review_agent_session() -> None:
    """Simulates the Review agent analysing a PR for checkout-service."""
    print("=" * 60)
    print("REVIEW AGENT SESSION")
    print("=" * 60)

    with AgentMemory(agent_id="review-agent") as mem:
        mem.write(
            "checkout-service",
            "risk_profile",
            {
                "retry_logic_risk": 0.84,
                "incident_correlation_count": 3,
                "ai_contribution_avg": 0.71,
                "flagged_patterns": ["retry-without-backoff", "silent-failure"],
            },
            confidence=0.9,
            pattern_refs=["retry-logic", "error-handling"],
        )

        mem.write(
            "checkout-service",
            "session_risk_signals",
            {
                "pr_number": 1842,
                "files_changed": 12,
                "risk_score": 0.78,
                "recommendation": "requires-senior-review",
            },
            confidence=0.85,
        )

        mem.write(
            "checkout-service",
            "team_context",
            {
                "team": "platform-eng",
                "recent_incident_count": 2,
                "deploy_frequency": "3x/week",
            },
            confidence=1.0,
        )

        print("Review agent wrote 3 memory entries for checkout-service")
        s = mem.stats()
        print(f"  Total entries: {s.total_entries}")
        print()


def release_agent_session() -> None:
    """Simulates the Release agent reading Review context before deploy."""
    print("=" * 60)
    print("RELEASE AGENT SESSION")
    print("=" * 60)

    with AgentMemory(agent_id="release-agent") as mem:
        risk = mem.read("checkout-service", "risk_profile")
        signals = mem.read("checkout-service", "session_risk_signals")
        team = mem.read("checkout-service", "team_context")

        print("Release agent reads Review agent's context (no cold start):")
        print(f"  Risk score: {signals.value['risk_score']}")
        print(f"  Retry risk: {risk.value['retry_logic_risk']}")
        print(f"  Team incidents: {team.value['recent_incident_count']}")
        print(f"  Written by: {risk.provenance.agent_id}")
        print()

        print(f"Auto-tracked read log: {mem.read_log}")
        print()

        # --- Simulate: deployment causes an incident ---
        print("Deployment goes out... P1 incident occurs!")
        updated = mem.commit_outcome("INC-2047", OutcomeType.CRITICAL_FAILURE)
        print(f"Outcome auto-linked to {len(updated)} entries the agent read:")
        for e in updated:
            print(f"  {e.entry_key} → confidence {e.confidence:.4f}")
        print()

        # --- Search for high-risk entries ---
        high_risk = mem.search(min_confidence=0.9, sort_by="confidence")
        print(f"High-confidence entries after incident ({len(high_risk)}):")
        for e in high_risk:
            print(f"  {e.entry_key} = {e.confidence:.4f}")


def main() -> None:
    review_agent_session()
    release_agent_session()

    print()
    print("=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("The Release agent started with full context from the Review")
    print("agent — no manual handoff code. When the incident happened,")
    print("commit_outcome() automatically knew which entries were causal")
    print("because the SDK tracked every read().")


if __name__ == "__main__":
    main()
