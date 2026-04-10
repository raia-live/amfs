"""Multi-Framework Demo — CrewAI, LangChain, and AutoGen agents sharing AMFS memory.

Demonstrates all three frameworks collaborating through a shared AMFS memory store.
Each framework section gracefully degrades if the framework or API key is missing.

    # Full demo (requires frameworks + OPENAI_API_KEY):
    pip install crewai langchain-core langchain-openai pyautogen
    OPENAI_API_KEY=sk-... uv run python examples/multi_framework_demo.py

    # SDK-only demo (no extra deps needed):
    uv run python examples/multi_framework_demo.py
"""

from __future__ import annotations

import os
import json

from amfs import AgentMemory, OutcomeType


DIVIDER = "=" * 70


def run_crewai_phase(adapter) -> None:
    """Phase 1: CrewAI agents analyze a PR and write risk signals to AMFS."""
    print(f"\n{DIVIDER}")
    print("PHASE 1: CrewAI — Code Review Analysis")
    print(DIVIDER)

    mem = AgentMemory(agent_id="crewai-reviewer", adapter=adapter)

    try:
        from crewai import Agent, Crew, Task
        from amfs_crewai import AMFSTool

        tools = AMFSTool(mem).tools()

        reviewer = Agent(
            role="Security Reviewer",
            goal="Analyze checkout-service PR #2001 for security risks and write findings to AMFS",
            backstory="You are a senior security engineer. Write your findings to shared memory using the amfs_write tool.",
            tools=tools,
            verbose=True,
        )

        review_task = Task(
            description=(
                "Review checkout-service PR #2001. The PR modifies retry logic and adds "
                "a new payment gateway integration. Write a risk_profile to entity "
                "'checkout-service' with key 'risk_profile' containing a JSON object with "
                "fields: risk_score (0-1), flagged_patterns (list), and recommendation."
            ),
            expected_output="Risk profile written to AMFS memory.",
            agent=reviewer,
        )

        crew = Crew(agents=[reviewer], tasks=[review_task], verbose=True)
        crew.kickoff()
        print("CrewAI phase completed with real agents.")

    except ImportError:
        print("[CrewAI not installed — using direct SDK writes]")
        mem.write(
            "checkout-service",
            "risk_profile",
            {
                "risk_score": 0.78,
                "flagged_patterns": ["retry-without-backoff", "unvalidated-input"],
                "recommendation": "requires-senior-review",
                "pr_number": 2001,
            },
            confidence=0.9,
            pattern_refs=["retry-logic", "input-validation"],
        )
        mem.write(
            "checkout-service",
            "code_metrics",
            {
                "files_changed": 8,
                "lines_added": 142,
                "lines_removed": 37,
                "test_coverage_delta": -2.1,
            },
            confidence=1.0,
        )
        print("Wrote risk_profile and code_metrics to checkout-service.")

    except Exception as e:
        print(f"[CrewAI failed: {e} — falling back to SDK writes]")
        mem.write(
            "checkout-service",
            "risk_profile",
            {"risk_score": 0.78, "flagged_patterns": ["retry-without-backoff"], "recommendation": "requires-review"},
            confidence=0.9,
        )

    finally:
        entries = mem.list("checkout-service")
        print(f"  Entries written: {len(entries)}")
        for e in entries:
            print(f"    {e.entity_path}/{e.key} v{e.version} (confidence={e.confidence})")
        mem.close()


def run_langchain_phase(adapter) -> None:
    """Phase 2: LangChain reads risk signals and has a Q&A conversation."""
    print(f"\n{DIVIDER}")
    print("PHASE 2: LangChain — Risk Assessment Conversation")
    print(DIVIDER)

    mem = AgentMemory(agent_id="langchain-analyst", adapter=adapter)

    risk = mem.read("checkout-service", "risk_profile")
    if risk:
        print(f"  Read risk_profile from CrewAI agent:")
        print(f"    Written by: {risk.provenance.agent_id}")
        print(f"    Risk score: {risk.value.get('risk_score', 'N/A')}")
        print(f"    Flags: {risk.value.get('flagged_patterns', [])}")
    else:
        print("  No risk_profile found — CrewAI phase may not have run.")

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from amfs_langchain import AMFSChatMemory

        if not os.environ.get("OPENAI_API_KEY"):
            raise ImportError("OPENAI_API_KEY not set")

        chat_memory = AMFSChatMemory(mem, session_key="risk-review-2001")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        risk_context = json.dumps(risk.value, indent=2) if risk else "No risk data available"

        questions = [
            f"Given this risk profile for checkout-service PR #2001:\n{risk_context}\n\nWhat are the top 2 concerns?",
            "Should we require a canary deployment for this PR? Why or why not?",
        ]

        for q in questions:
            print(f"\n  Q: {q[:80]}...")
            response = llm.invoke([
                SystemMessage(content="You are a deployment risk analyst. Be concise (2-3 sentences max)."),
                HumanMessage(content=q),
            ])
            answer = response.content
            print(f"  A: {answer}")
            chat_memory.save_context({"input": q}, {"output": answer})

        history = chat_memory.load_memory_variables()
        print(f"\n  Chat history persisted: {len(history['history'])} turns")

        mem.write(
            "checkout-service",
            "analyst_recommendation",
            {"deploy_strategy": "canary", "approval_status": "conditional", "analyst": "langchain-analyst"},
            confidence=0.85,
        )
        print("  Wrote analyst_recommendation to AMFS.")

    except ImportError:
        print("[LangChain/OpenAI not available — using direct SDK writes]")
        from amfs_langchain import AMFSChatMemory

        chat_memory = AMFSChatMemory(mem, session_key="risk-review-2001")
        chat_memory.save_context(
            {"input": "What are the main risks in PR #2001?"},
            {"output": "The retry-without-backoff pattern could cause cascading failures under load."},
        )
        chat_memory.save_context(
            {"input": "Should we canary this?"},
            {"output": "Yes, a canary deployment is recommended given the risk score of 0.78."},
        )
        mem.write(
            "checkout-service",
            "analyst_recommendation",
            {"deploy_strategy": "canary", "approval_status": "conditional"},
            confidence=0.85,
        )
        print("  Wrote mock conversation and recommendation.")

    except Exception as e:
        print(f"[LangChain error: {e}]")

    finally:
        mem.close()


def run_autogen_phase(adapter) -> None:
    """Phase 3: AutoGen agents read all context and make a deploy decision."""
    print(f"\n{DIVIDER}")
    print("PHASE 3: AutoGen — Deploy Decision")
    print(DIVIDER)

    mem = AgentMemory(agent_id="autogen-deployer", adapter=adapter)

    from amfs_autogen import AMFSMemoryStore
    store = AMFSMemoryStore(mem, entity_path="checkout-service")

    risk = store.get("risk_profile")
    recommendation = store.get("analyst_recommendation")

    print("  Context loaded from AMFS:")
    print(f"    risk_profile: {risk}")
    print(f"    analyst_recommendation: {recommendation}")

    try:
        import autogen

        if not os.environ.get("OPENAI_API_KEY"):
            raise ImportError("OPENAI_API_KEY not set")

        config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

        deployer = autogen.AssistantAgent(
            name="deployer",
            system_message=(
                "You are a deployment manager. Based on the risk data provided, "
                "make a final deploy/no-deploy decision. Be concise (2-3 sentences)."
            ),
            llm_config={"config_list": config_list},
        )

        user_proxy = autogen.UserProxyAgent(
            name="system",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )

        context = json.dumps({"risk_profile": risk, "recommendation": recommendation}, indent=2, default=str)
        user_proxy.initiate_chat(
            deployer,
            message=f"Here is the accumulated context for checkout-service PR #2001:\n{context}\n\nMake your deploy decision.",
        )

        decision = "proceed-with-canary"
        print(f"\n  AutoGen decision: {decision}")

    except ImportError:
        print("[AutoGen/OpenAI not available — using direct SDK decision]")
        risk_score = risk.get("risk_score", 0.5) if isinstance(risk, dict) else 0.5
        decision = "proceed-with-canary" if risk_score < 0.85 else "block"
        print(f"  Automated decision (risk={risk_score}): {decision}")

    except Exception as e:
        print(f"[AutoGen error: {e}]")
        decision = "manual-review-required"

    store.add("deploy_decision", {
        "decision": decision,
        "pr": 2001,
        "decider": "autogen-deployer",
    })
    print(f"  Wrote deploy_decision to AMFS: {decision}")
    mem.close()


def show_summary(adapter) -> None:
    """Final summary: show all entries, stats, and outcome back-propagation."""
    print(f"\n{DIVIDER}")
    print("SUMMARY — Cross-Framework Memory State")
    print(DIVIDER)

    mem = AgentMemory(agent_id="summary-observer", adapter=adapter)

    stats = mem.stats()
    print(f"\n  Total entries: {stats.total_entries}")
    print(f"  Total entities: {stats.total_entities}")
    print(f"  Total agents: {stats.total_agents}")
    print(f"  Confidence range: {stats.confidence_min:.2f} – {stats.confidence_max:.2f}")

    print("\n  All entries:")
    entries = mem.list()
    for e in entries:
        print(f"    [{e.provenance.agent_id}] {e.entity_path}/{e.key} "
              f"v{e.version} confidence={e.confidence:.2f}")

    risk = mem.read("checkout-service", "risk_profile")
    decision = mem.read("checkout-service", "deploy_decision")
    if risk and decision:
        print("\n  Simulating deployment outcome (P1 incident)...")
        updated = mem.commit_outcome("INC-3001", OutcomeType.CRITICAL_FAILURE)
        print(f"  Outcome back-propagated to {len(updated)} entries:")
        for u in updated:
            print(f"    {u.entry_key} → confidence={u.confidence:.4f}")

    mem.close()


def main() -> None:
    print(DIVIDER)
    print("AMFS Multi-Framework Integration Demo")
    print("CrewAI → LangChain → AutoGen pipeline with shared memory")
    print(DIVIDER)

    from amfs_filesystem.adapter import FilesystemAdapter
    import tempfile
    from pathlib import Path

    data_dir = Path(tempfile.mkdtemp(prefix="amfs-demo-"))
    print(f"  AMFS data directory: {data_dir}")
    adapter = FilesystemAdapter(root=data_dir, namespace="demo")

    run_crewai_phase(adapter)
    run_langchain_phase(adapter)
    run_autogen_phase(adapter)
    show_summary(adapter)

    adapter.close()
    print(f"\n{DIVIDER}")
    print("Demo complete. Memory data at: {data_dir}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
