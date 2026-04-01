"""CrewAI Integration — give CrewAI agents shared memory via AMFS.

Shows how to wrap AgentMemory as CrewAI tools so agents can read/write
shared entity memory during crew execution.

    pip install crewai
    uv run python examples/crewai_integration.py

NOTE: Requires crewai to be installed. This example shows the pattern —
adapt the agent/task definitions to your use case.
"""

from __future__ import annotations


def main() -> None:
    try:
        from crewai import Agent, Crew, Task
    except ImportError:
        print("This example requires crewai: pip install crewai")
        print()
        print("Showing the integration pattern instead:")
        print()
        show_pattern()
        return

    from amfs import AgentMemory
    from amfs_crewai import AMFSTool

    mem = AgentMemory(agent_id="crewai-agent")
    tools = AMFSTool(mem).tools()

    review_agent = Agent(
        role="Code Review Agent",
        goal="Analyse code changes and write risk signals to shared memory",
        backstory="You are a senior engineer reviewing PRs for risk patterns.",
        tools=tools,
        verbose=True,
    )

    release_agent = Agent(
        role="Release Agent",
        goal="Read risk context from shared memory and decide deploy safety",
        backstory="You make release decisions based on accumulated context.",
        tools=tools,
        verbose=True,
    )

    review_task = Task(
        description=(
            "Review checkout-service PR #1842. Write risk signals to AMFS memory "
            "under entity 'checkout-service' with keys like 'risk_profile'."
        ),
        expected_output="Risk profile written to shared memory.",
        agent=review_agent,
    )

    release_task = Task(
        description=(
            "Read the risk profile for checkout-service from AMFS memory. "
            "Decide whether it is safe to deploy based on the risk signals."
        ),
        expected_output="Deploy decision with reasoning.",
        agent=release_agent,
    )

    crew = Crew(
        agents=[review_agent, release_agent],
        tasks=[review_task, release_task],
        verbose=True,
    )

    result = crew.kickoff()
    print(result)
    mem.close()


def show_pattern() -> None:
    """Show the integration pattern without requiring crewai."""
    print("from amfs import AgentMemory")
    print("from amfs_crewai import AMFSTool")
    print()
    print('mem = AgentMemory(agent_id="crewai-agent")')
    print("tools = AMFSTool(mem).tools()")
    print("# tools = [AMFSReadTool, AMFSWriteTool, AMFSListTool]")
    print()
    print("# Pass tools to your CrewAI agents:")
    print("agent = Agent(role='...', tools=tools)")
    print()
    print("# Agents can now read/write shared AMFS memory during execution.")
    print("# Memory persists across crew runs and across different agents.")


if __name__ == "__main__":
    main()
