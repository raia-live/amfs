"""Live Multi-Agent Test — CrewAI and AutoGen agents reading and writing AMFS concurrently.

Spins up real LLM-powered agents from both frameworks, all sharing the same
AMFS memory store. Watch them discover each other's work in real time.

Backend resolution:
  1. AMFS_HTTP_URL + AMFS_API_KEY env vars → HTTP adapter (SaaS dashboard)
  2. ~/.config/amfs/credentials.json       → HTTP adapter (SaaS dashboard)
  3. Fallback                              → local temp filesystem

    OPENAI_API_KEY=sk-... uv run python examples/live_multi_agent_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from amfs import AgentMemory
from amfs_core.abc import AdapterABC
from amfs_autogen import AMFSMemoryStore

DIV = "=" * 72
THIN = "-" * 72


def _resolve_adapter() -> tuple[AdapterABC, str]:
    """Resolve the AMFS adapter — prefer SaaS HTTP, fall back to local filesystem."""
    http_url = os.environ.get("AMFS_HTTP_URL")
    api_key = os.environ.get("AMFS_API_KEY", "")

    if not http_url:
        creds_path = Path.home() / ".config" / "amfs" / "credentials.json"
        if creds_path.exists():
            creds = json.loads(creds_path.read_text())
            http_url = creds.get("url", "")
            api_key = creds.get("api_key", "")

    if http_url:
        try:
            from amfs_adapter_http import HttpAdapter
            adapter = HttpAdapter(base_url=http_url, api_key=api_key)
            return adapter, f"HTTP → {http_url}"
        except ImportError:
            print("  WARNING: amfs-adapter-http not installed, falling back to filesystem")

    from amfs_filesystem.adapter import FilesystemAdapter
    data_dir = Path(tempfile.mkdtemp(prefix="amfs-live-test-"))
    adapter = FilesystemAdapter(root=data_dir, namespace="live-test")
    return adapter, f"Filesystem → {data_dir}"


def log(agent: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] [{agent}] {msg}", flush=True)


def snapshot_memory(adapter: AdapterABC, label: str) -> None:
    """Print recent entries written by this test run."""
    mem = AgentMemory(agent_id="_observer", adapter=adapter)
    entries = mem.list("live-test/checkout-service")
    print(f"\n{THIN}")
    print(f"  MEMORY SNAPSHOT — {label}")
    print(f"  {len(entries)} entries in live-test/checkout-service")
    print(THIN)
    for e in entries:
        val_preview = str(e.value)[:90] + ("..." if len(str(e.value)) > 90 else "")
        print(f"  [{e.provenance.agent_id}] {e.entity_path}/{e.key} "
              f"v{e.version} conf={e.confidence:.2f}")
        print(f"    → {val_preview}")
    print(THIN, flush=True)
    mem.close()


# ---------------------------------------------------------------------------
# CrewAI agents
# ---------------------------------------------------------------------------

ENTITY = "live-test/checkout-service"


def run_crewai_crew(adapter: AdapterABC) -> None:
    """Run a CrewAI crew with 2 agents that write findings to AMFS."""
    from crewai import Agent, Crew, Task
    from amfs_crewai import AMFSTool

    print(f"\n{DIV}")
    print("  STARTING: CrewAI Crew (2 agents: Security Reviewer + Performance Analyst)")
    print(DIV, flush=True)

    mem = AgentMemory(agent_id="crewai-security-reviewer", adapter=adapter)
    tools = AMFSTool(mem).tools()

    security_reviewer = Agent(
        role="Security Reviewer",
        goal=(
            "Analyze the checkout-service for security risks. "
            "Write your findings to AMFS shared memory using the amfs_write tool. "
            f"Use entity_path '{ENTITY}' and key 'security-risks'."
        ),
        backstory=(
            "You are a senior application security engineer. You specialize in "
            "finding injection attacks, auth bypasses, and data leaks in payment systems."
        ),
        tools=tools,
        verbose=True,
        llm="gpt-4o-mini",
    )

    perf_analyst = Agent(
        role="Performance Analyst",
        goal=(
            "Analyze checkout-service for performance bottlenecks. "
            "First use amfs_read to check if the security reviewer has already "
            f"written findings (entity_path='{ENTITY}', key='security-risks'). "
            "Then write your own performance analysis using amfs_write with "
            f"entity_path '{ENTITY}' and key 'performance-analysis'."
        ),
        backstory=(
            "You are a performance engineer who profiles latency, throughput, and "
            "resource usage in high-traffic e-commerce services."
        ),
        tools=tools,
        verbose=True,
        llm="gpt-4o-mini",
    )

    security_task = Task(
        description=(
            "Review the checkout-service codebase for security vulnerabilities. "
            "The service handles payment processing with Stripe, user authentication "
            "via JWT tokens, and stores order data in PostgreSQL. "
            "Focus on: SQL injection, JWT validation, PII handling, and retry logic. "
            f"Write a JSON risk assessment to AMFS with entity_path='{ENTITY}' "
            "and key='security-risks'. The JSON should have fields: risk_score (0-1), "
            "critical_findings (list of strings), and recommendation (string)."
        ),
        expected_output="Security risk assessment written to AMFS shared memory.",
        agent=security_reviewer,
    )

    perf_task = Task(
        description=(
            "Analyze checkout-service for performance issues. The service handles "
            "~500 req/s during peak with P99 latency of 800ms (target: 200ms). "
            "First, read the security reviewer's findings from AMFS using amfs_read "
            f"with entity_path='{ENTITY}' and key='security-risks' — their "
            "findings may reveal performance-impacting issues. "
            "Then write your performance analysis to AMFS using amfs_write with "
            f"entity_path='{ENTITY}' and key='performance-analysis'. "
            "Include fields: latency_bottlenecks (list), optimization_priority (string), "
            "and estimated_improvement (string)."
        ),
        expected_output="Performance analysis written to AMFS, informed by security findings.",
        agent=perf_analyst,
    )

    crew = Crew(
        agents=[security_reviewer, perf_analyst],
        tasks=[security_task, perf_task],
        verbose=True,
    )

    log("crewai", "Kicking off crew...")
    result = crew.kickoff()
    log("crewai", f"Crew finished. Result: {str(result)[:120]}...")
    mem.close()


# ---------------------------------------------------------------------------
# AutoGen agents (v0.10+ API — autogen_agentchat)
# ---------------------------------------------------------------------------

async def run_autogen_agents(adapter: AdapterABC) -> None:
    """Run AutoGen agents that read CrewAI's findings and build on them."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    print(f"\n{DIV}")
    print("  STARTING: AutoGen Agent Group (3 agents: Architect + Ops + Summarizer)")
    print(DIV, flush=True)

    mem_arch = AgentMemory(agent_id="autogen-architect", adapter=adapter)
    store_arch = AMFSMemoryStore(mem_arch, entity_path=ENTITY)

    security = store_arch.get("security-risks")
    perf = store_arch.get("performance-analysis")

    log("autogen-architect", f"Read from AMFS — security-risks: {'FOUND' if security else 'NOT YET'}")
    log("autogen-architect", f"Read from AMFS — performance-analysis: {'FOUND' if perf else 'NOT YET'}")

    context_parts = []
    if security:
        context_parts.append(f"Security findings: {json.dumps(security, default=str)}")
    if perf:
        context_parts.append(f"Performance findings: {json.dumps(perf, default=str)}")

    context_str = "\n".join(context_parts) if context_parts else "No prior findings available yet."

    client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    architect = AssistantAgent(
        name="architect",
        system_message=(
            "You are a software architect making deployment decisions for checkout-service. "
            "You have access to findings from other agents. "
            "Make a concrete deploy/no-deploy decision with specific conditions. "
            "Be concise — 3-4 sentences max."
        ),
        model_client=client,
    )

    ops_engineer = AssistantAgent(
        name="ops_engineer",
        system_message=(
            "You are a DevOps/SRE engineer. Given the architect's decision, "
            "define the rollout plan: canary percentage, monitoring metrics, "
            "and rollback triggers. Be concise — 3-4 sentences max. "
            "End your message with TERMINATE when done."
        ),
        model_client=client,
    )

    termination = MaxMessageTermination(max_messages=4)
    team = RoundRobinGroupChat(
        participants=[architect, ops_engineer],
        termination_condition=termination,
    )

    log("autogen", "Starting architect + ops_engineer team chat...")

    prompt = (
        f"Here are the accumulated findings for checkout-service:\n\n"
        f"{context_str}\n\n"
        f"Based on these findings, should we deploy checkout-service to production? "
        f"What are the conditions? Define the rollout plan."
    )

    result = await team.run(task=prompt)

    architect_msg = ""
    ops_msg = ""
    for msg in result.messages:
        log("autogen", f"[{msg.source}] {str(msg.content)[:120]}...")
        if msg.source == "architect":
            architect_msg = str(msg.content)
        elif msg.source == "ops_engineer":
            ops_msg = str(msg.content)

    store_arch.add("deploy-decision", {
        "decision": "conditional-deploy",
        "architect_reasoning": architect_msg[:500],
        "decided_by": "autogen-architect",
    })
    log("autogen-architect", "Wrote deploy-decision to AMFS ✓")

    mem_ops = AgentMemory(agent_id="autogen-ops-engineer", adapter=adapter)
    store_ops = AMFSMemoryStore(mem_ops, entity_path=ENTITY)
    store_ops.add("rollout-config", {
        "rollout_plan": ops_msg[:500],
        "approved_by": "autogen-ops-engineer",
    })
    log("autogen-ops-engineer", "Wrote rollout-config to AMFS ✓")

    mem_sum = AgentMemory(agent_id="autogen-summarizer", adapter=adapter)
    store_sum = AMFSMemoryStore(mem_sum, entity_path=ENTITY)

    all_data = store_sum.get_all()
    log("autogen-summarizer", f"Read ALL entries from AMFS: {list(all_data.keys())}")

    summarizer = AssistantAgent(
        name="summarizer",
        system_message=(
            "You are a technical writer. Given a set of engineering findings, "
            "write a 2-3 sentence executive summary suitable for a Slack message. "
            "End with TERMINATE."
        ),
        model_client=client,
    )

    summary_team = RoundRobinGroupChat(
        participants=[summarizer],
        termination_condition=MaxMessageTermination(max_messages=2),
    )

    summary_prompt = (
        f"Summarize these engineering findings for checkout-service into "
        f"a brief executive summary:\n\n{json.dumps(all_data, default=str, indent=2)}"
    )

    summary_result = await summary_team.run(task=summary_prompt)
    summary_text = ""
    for msg in summary_result.messages:
        if msg.source == "summarizer":
            summary_text = str(msg.content)
            log("autogen-summarizer", f"Summary: {summary_text[:120]}...")

    store_sum.add("executive-summary", {
        "summary": summary_text.replace("TERMINATE", "").strip(),
        "summarized_by": "autogen-summarizer",
        "sources": list(all_data.keys()),
    })
    log("autogen-summarizer", "Wrote executive-summary to AMFS ✓")

    await client.close()
    mem_arch.close()
    mem_ops.close()
    mem_sum.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY to run this test.")
        sys.exit(1)

    print(DIV)
    print("  AMFS Live Multi-Agent Test")
    print("  CrewAI + AutoGen agents sharing memory in real time")
    print(DIV, flush=True)

    adapter, backend_desc = _resolve_adapter()
    print(f"  Backend: {backend_desc}")
    print(f"  Entity:  {ENTITY}")

    snapshot_memory(adapter, "Initial state")

    # Phase 1: CrewAI agents analyze the service
    print("\n\n  >>> PHASE 1: CrewAI crew analyzing checkout-service...")
    run_crewai_crew(adapter)
    snapshot_memory(adapter, "After CrewAI crew (2 agents wrote)")

    # Phase 2: AutoGen agents read CrewAI's work and make decisions
    print("\n\n  >>> PHASE 2: AutoGen agents reading CrewAI findings + making decisions...")
    asyncio.run(run_autogen_agents(adapter))
    snapshot_memory(adapter, "After AutoGen agents (3 more agents wrote)")

    # Phase 3: Outcome back-propagation
    print(f"\n{DIV}")
    print("  PHASE 3: Simulating deployment outcome — back-propagating to all entries")
    print(DIV, flush=True)

    mem = AgentMemory(agent_id="release-bot", adapter=adapter)
    mem.read(ENTITY, "security-risks")
    mem.read(ENTITY, "performance-analysis")
    mem.read(ENTITY, "deploy-decision")
    mem.read(ENTITY, "rollout-config")
    mem.read(ENTITY, "executive-summary")

    from amfs import OutcomeType
    updated = mem.commit_outcome("DEPLOY-2001", OutcomeType.SUCCESS)
    print(f"\n  Outcome 'SUCCESS' back-propagated to {len(updated)} entries:")
    for u in updated:
        print(f"    {u.entry_key} → confidence={u.confidence:.4f}")
    mem.close()

    snapshot_memory(adapter, "Final state (after outcome back-propagation)")

    # Provenance
    print(f"\n{DIV}")
    print("  PROVENANCE — Who wrote what?")
    print(DIV)
    observer = AgentMemory(agent_id="_observer", adapter=adapter)
    entries = observer.list(ENTITY)
    agents_seen = set()
    for e in entries:
        agents_seen.add(e.provenance.agent_id)
        print(f"  {e.entity_path}/{e.key}")
        print(f"    written_by: {e.provenance.agent_id}")
        print(f"    session:    {e.provenance.session_id}")
        print(f"    at:         {e.provenance.written_at.strftime('%H:%M:%S')}")
        print()

    print(f"  Total unique agent identities: {len(agents_seen)}")
    print(f"  Agents: {', '.join(sorted(agents_seen))}")
    observer.close()
    if hasattr(adapter, "close"):
        adapter.close()

    print(f"\n{DIV}")
    print("  Test complete. Check the dashboard for new agents!")
    print(DIV)


if __name__ == "__main__":
    main()
