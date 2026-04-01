"""LangGraph Integration — use AMFS as a checkpoint store for LangGraph.

Shows how AMFSCheckpointer stores graph state as AMFS memory entries,
giving you versioned, provenance-tracked checkpoints with CoW history.

    pip install langgraph
    uv run python examples/langgraph_integration.py

NOTE: Requires langgraph to be installed. This example shows the pattern —
adapt to your graph definition.
"""

from __future__ import annotations


def main() -> None:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        print("This example requires langgraph: pip install langgraph")
        print()
        print("Showing the integration pattern instead:")
        print()
        show_pattern()
        return

    show_pattern()


def show_pattern() -> None:
    """Show the integration pattern."""
    from amfs import AgentMemory
    from amfs_langgraph import AMFSCheckpointer

    mem = AgentMemory(agent_id="graph-agent")
    checkpointer = AMFSCheckpointer(mem)

    # Simulate what LangGraph does internally
    config = {"configurable": {"thread_id": "thread-001"}}

    # Save a checkpoint
    checkpoint = {
        "messages": [
            {"role": "user", "content": "Review checkout-service"},
            {"role": "assistant", "content": "Analysing risk patterns..."},
        ],
        "state": {"risk_score": 0.78, "step": 2},
    }
    checkpointer.put(config, checkpoint)
    print("Saved checkpoint for thread-001")

    # Load it back
    loaded = checkpointer.get(config)
    print(f"Loaded checkpoint: {loaded['state']}")
    print(f"Messages: {len(loaded['messages'])}")

    # Save another version
    checkpoint["state"]["step"] = 3
    checkpoint["state"]["risk_score"] = 0.85
    checkpointer.put(config, checkpoint)
    print(f"Updated checkpoint (step 3)")

    # List all versions (CoW history)
    history = checkpointer.list(config)
    print(f"Checkpoint history: {len(history)} versions")

    print()
    print("# In your LangGraph code:")
    print("# graph = StateGraph(MyState)")
    print("# graph.add_node(...)")
    print("# app = graph.compile(checkpointer=AMFSCheckpointer(mem))")
    print("# app.invoke(input, config={'configurable': {'thread_id': 'my-thread'}})")
    print()
    print("# Benefits over default checkpointers:")
    print("#   - CoW versioning: every checkpoint creates a new version")
    print("#   - Provenance: which agent, which session, when")
    print("#   - Shared: multiple agents can read checkpoint state")
    print("#   - Snapshot: export/restore checkpoints with amfs CLI")

    mem.close()


if __name__ == "__main__":
    main()
