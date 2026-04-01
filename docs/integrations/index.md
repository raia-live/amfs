---
title: Integrations
layout: default
nav_order: 6
description: "Use AMFS with CrewAI, LangGraph, LangChain, and AutoGen."
permalink: /integrations/
---

# Framework Integrations
{: .no_toc }

AMFS integrates with popular multi-agent frameworks so you can add persistent memory to your existing workflows.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## CrewAI

Add AMFS tools to your CrewAI agents:

```bash
pip install amfs-crewai
```

```python
from amfs import AgentMemory
from amfs_crewai import AMFSTool

mem = AgentMemory(agent_id="crewai-agent")
tools = AMFSTool(mem).tools()
# Returns [AMFSReadTool, AMFSWriteTool, AMFSListTool]

# Pass to your CrewAI agent
from crewai import Agent

agent = Agent(
    role="Researcher",
    goal="Analyze the codebase",
    tools=tools,
)
```

The tools give your CrewAI agents the ability to read, write, and list memory entries as part of their task execution.

---

## LangGraph

Use AMFS as a checkpointer for LangGraph:

```bash
pip install amfs-langgraph
```

```python
from amfs import AgentMemory
from amfs_langgraph import AMFSCheckpointer

mem = AgentMemory(agent_id="graph-agent")
checkpointer = AMFSCheckpointer(mem)

# Pass to your LangGraph graph builder
```

The checkpointer persists graph state to AMFS, enabling cross-session graph execution.

---

## LangChain

Use AMFS as chat memory for LangChain:

```bash
pip install amfs-langchain
```

```python
from amfs import AgentMemory
from amfs_langchain import AMFSChatMemory

mem = AgentMemory(agent_id="chat-agent")
chat_memory = AMFSChatMemory(mem, session_key="conv-123")

# Save conversation context
chat_memory.save_context(
    {"input": "What's the retry policy?"},
    {"output": "We use exponential backoff with max 3 retries."},
)

# Load conversation history
history = chat_memory.load_memory_variables({})
```

---

## AutoGen

Use AMFS as a memory store for AutoGen agents:

```bash
pip install amfs-autogen
```

```python
from amfs import AgentMemory
from amfs_autogen import AMFSMemoryStore

mem = AgentMemory(agent_id="autogen-agent")
store = AMFSMemoryStore(mem)

# Store and retrieve data
store.add("user_prefs", {"theme": "dark"})
prefs = store.get("user_prefs")
```

---

## Building Your Own Integration

All integrations are thin wrappers around the `AgentMemory` API. To build your own:

1. Accept an `AgentMemory` instance in your constructor
2. Map your framework's memory/tool API to `mem.read()`, `mem.write()`, `mem.list()`, etc.
3. Use `entity_path` to scope data (e.g., per-conversation, per-task, per-agent)

```python
class MyFrameworkMemory:
    def __init__(self, mem: AgentMemory, scope: str = "default"):
        self._mem = mem
        self._scope = scope

    def save(self, key: str, data: dict):
        self._mem.write(self._scope, key, data)

    def load(self, key: str):
        entry = self._mem.read(self._scope, key)
        return entry.value if entry else None
```
