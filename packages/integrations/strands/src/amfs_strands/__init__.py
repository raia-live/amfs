"""AMFS integration for Strands Agents.

Provides:
- ``AMFSPlugin`` — a Strands Plugin with memory tools and causal tracing hooks
- ``amfs_memory`` — a standalone @tool for action-based memory (mem0 drop-in)
"""

from amfs_strands.plugin import AMFSPlugin
from amfs_strands.tools import amfs_memory

__all__ = ["AMFSPlugin", "amfs_memory"]
