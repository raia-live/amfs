"""TracePatternExtractor — mines successful DecisionTraces for reusable patterns.

Extracts recurring read patterns, decision contexts, and agent sequences
from successful traces and emits them as first-class AMFS memories with
type ``experience`` so they decay slower than normal entries.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from amfs_core.models import (
    DecisionTrace,
    Digest,
    DigestType,
    MemoryEntry,
    MemoryType,
    Provenance,
)

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter

logger = logging.getLogger(__name__)

_MIN_TRACES_FOR_PATTERN = 3
_MAX_PATTERNS_PER_ENTITY = 10
_SYSTEM_AGENT_ID = "cortex/trace-miner"


class TracePatternExtractor:
    """Extract reusable patterns from successful decision traces.

    Groups traces by entity_path, finds recurring read-key co-occurrences,
    recurring external-context labels, and agent collaboration sequences,
    then emits pattern memories at confidence scaled by supporting trace count.
    """

    def __init__(
        self,
        adapter: PostgresAdapter,
        namespace: str = "default",
        min_traces: int = _MIN_TRACES_FOR_PATTERN,
    ) -> None:
        self._adapter = adapter
        self._namespace = namespace
        self._min_traces = min_traces

    def extract_patterns(
        self,
        entity_path: str,
        *,
        branch: str = "main",
    ) -> list[MemoryEntry]:
        """Mine successful traces for an entity and return pattern entries.

        Does NOT write to the adapter — the caller (DigestCompiler or
        CortexWorker) decides whether to persist.
        """
        traces = self._adapter.list_traces(
            entity_path=entity_path,
            outcome_type="success",
            limit=200,
        )
        if len(traces) < self._min_traces:
            logger.debug(
                "Not enough successful traces for %s (%d < %d)",
                entity_path, len(traces), self._min_traces,
            )
            return []

        patterns: list[MemoryEntry] = []
        patterns.extend(self._extract_read_cooccurrence(entity_path, traces, branch))
        patterns.extend(self._extract_context_patterns(entity_path, traces, branch))
        patterns.extend(self._extract_agent_sequences(entity_path, traces, branch))

        return patterns[:_MAX_PATTERNS_PER_ENTITY]

    def compile_trace_digest(
        self,
        entity_path: str,
        *,
        branch: str = "main",
    ) -> Digest | None:
        """Produce a TRACE_PATTERN digest summarising what successful agents do."""
        traces = self._adapter.list_traces(
            entity_path=entity_path,
            outcome_type="success",
            limit=200,
        )
        if len(traces) < self._min_traces:
            return None

        agent_ids: set[str] = set()
        total_reads = 0
        context_labels: Counter[str] = Counter()

        for t in traces:
            agent_ids.add(t.agent_id)
            total_reads += len(t.causal_entries)
            for ctx in t.external_contexts:
                label = ctx.label if hasattr(ctx, "label") else ctx.get("label", "") if isinstance(ctx, dict) else ""
                if label:
                    context_labels[label] += 1

        read_groups = self._group_reads_by_key(traces)
        top_keys = sorted(read_groups.items(), key=lambda kv: kv[1], reverse=True)[:10]

        narrative_parts = [
            f"Across {len(traces)} successful traces on {entity_path}, "
            f"{len(agent_ids)} agents read {total_reads} entries total.",
        ]
        if top_keys:
            keys_str = ", ".join(k for k, _ in top_keys[:5])
            narrative_parts.append(f"Most frequently read keys: {keys_str}.")
        if context_labels:
            ctx_str = ", ".join(f"{l} ({c}x)" for l, c in context_labels.most_common(3))
            narrative_parts.append(f"Recurring decision contexts: {ctx_str}.")

        return Digest(
            digest_type=DigestType.TRACE_PATTERN,
            scope=entity_path,
            summary={
                "narrative": " ".join(narrative_parts),
                "trace_count": len(traces),
                "agents": sorted(agent_ids),
                "top_read_keys": [{"key": k, "count": c} for k, c in top_keys],
                "recurring_contexts": dict(context_labels.most_common(10)),
                "total_reads": total_reads,
            },
            entry_count=len(traces),
            source_agents=sorted(agent_ids),
            compiled_at=datetime.now(timezone.utc),
            namespace=self._namespace,
            branch=branch,
        )

    # ------------------------------------------------------------------
    # Pattern extraction strategies
    # ------------------------------------------------------------------

    def _extract_read_cooccurrence(
        self,
        entity_path: str,
        traces: list[DecisionTrace],
        branch: str,
    ) -> list[MemoryEntry]:
        """Find keys that are frequently read together before success."""
        key_sets: list[frozenset[str]] = []
        for t in traces:
            keys = frozenset(
                ce.key for ce in t.causal_entries if ce.entity_path == entity_path
            )
            if len(keys) >= 2:
                key_sets.append(keys)

        if len(key_sets) < self._min_traces:
            return []

        pair_counts: Counter[frozenset[str]] = Counter()
        for ks in key_sets:
            keys_list = sorted(ks)
            for i in range(len(keys_list)):
                for j in range(i + 1, len(keys_list)):
                    pair_counts[frozenset({keys_list[i], keys_list[j]})] += 1

        patterns: list[MemoryEntry] = []
        for pair, count in pair_counts.most_common(5):
            if count < self._min_traces:
                break
            keys = sorted(pair)
            pattern_id = self._hash_pattern("read-cooccurrence", entity_path, keys)
            confidence = self._confidence_from_count(count, len(traces))
            patterns.append(MemoryEntry(
                entity_path=entity_path,
                key=f"pattern-read-cooccurrence-{pattern_id[:8]}",
                value={
                    "type": "read_cooccurrence",
                    "keys": keys,
                    "supporting_traces": count,
                    "description": (
                        f"Keys {keys[0]} and {keys[1]} are typically read together "
                        f"before successful outcomes ({count}/{len(traces)} traces)."
                    ),
                },
                confidence=confidence,
                version=1,
                memory_type=MemoryType.EXPERIENCE,
                provenance=Provenance(
                    agent_id=_SYSTEM_AGENT_ID,
                    session_id=f"trace-mining-{entity_path}",
                    written_at=datetime.now(timezone.utc),
                    pattern_refs=[t.id for t in traces[:count]],
                ),
            ))

        return patterns

    def _extract_context_patterns(
        self,
        entity_path: str,
        traces: list[DecisionTrace],
        branch: str,
    ) -> list[MemoryEntry]:
        """Find external_context labels that recur across successful traces."""
        label_counter: Counter[str] = Counter()
        label_sources: dict[str, set[str]] = defaultdict(set)

        for t in traces:
            for ctx in t.external_contexts:
                label = ctx.label if hasattr(ctx, "label") else ctx.get("label", "") if isinstance(ctx, dict) else ""
                if label:
                    label_counter[label] += 1
                    source = ctx.source if hasattr(ctx, "source") else ctx.get("source", "unknown") if isinstance(ctx, dict) else "unknown"
                    label_sources[label].add(source or "unknown")

        patterns: list[MemoryEntry] = []
        for label, count in label_counter.most_common(3):
            if count < self._min_traces:
                break
            pattern_id = self._hash_pattern("context", entity_path, [label])
            confidence = self._confidence_from_count(count, len(traces))
            sources = sorted(label_sources[label])
            patterns.append(MemoryEntry(
                entity_path=entity_path,
                key=f"pattern-context-{pattern_id[:8]}",
                value={
                    "type": "recurring_context",
                    "label": label,
                    "sources": sources,
                    "supporting_traces": count,
                    "description": (
                        f"Decision context '{label}' from {', '.join(sources)} appears in "
                        f"{count}/{len(traces)} successful traces — likely a prerequisite step."
                    ),
                },
                confidence=confidence,
                version=1,
                memory_type=MemoryType.EXPERIENCE,
                provenance=Provenance(
                    agent_id=_SYSTEM_AGENT_ID,
                    session_id=f"trace-mining-{entity_path}",
                    written_at=datetime.now(timezone.utc),
                ),
            ))

        return patterns

    def _extract_agent_sequences(
        self,
        entity_path: str,
        traces: list[DecisionTrace],
        branch: str,
    ) -> list[MemoryEntry]:
        """Find agent collaboration sequences that lead to success."""
        sequence_counter: Counter[tuple[str, ...]] = Counter()

        for t in traces:
            agents_in_order: list[str] = []
            seen: set[str] = set()
            for ce in t.causal_entries:
                if ce.entity_path == entity_path:
                    writer = ce.written_by or "unknown"
                    if writer not in seen:
                        agents_in_order.append(writer)
                        seen.add(writer)
            if len(agents_in_order) >= 2:
                sequence_counter[tuple(agents_in_order)] += 1

        patterns: list[MemoryEntry] = []
        for seq, count in sequence_counter.most_common(3):
            if count < self._min_traces:
                break
            pattern_id = self._hash_pattern("agent-seq", entity_path, list(seq))
            confidence = self._confidence_from_count(count, len(traces))
            patterns.append(MemoryEntry(
                entity_path=entity_path,
                key=f"pattern-agent-sequence-{pattern_id[:8]}",
                value={
                    "type": "agent_sequence",
                    "sequence": list(seq),
                    "supporting_traces": count,
                    "description": (
                        f"Agent sequence {' → '.join(seq)} on {entity_path} correlates with "
                        f"successful outcomes ({count}/{len(traces)} traces)."
                    ),
                },
                confidence=confidence,
                version=1,
                memory_type=MemoryType.EXPERIENCE,
                provenance=Provenance(
                    agent_id=_SYSTEM_AGENT_ID,
                    session_id=f"trace-mining-{entity_path}",
                    written_at=datetime.now(timezone.utc),
                ),
            ))

        return patterns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_reads_by_key(traces: list[DecisionTrace]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in traces:
            for ce in t.causal_entries:
                k = f"{ce.entity_path}/{ce.key}"
                counts[k] = counts.get(k, 0) + 1
        return counts

    @staticmethod
    def _hash_pattern(kind: str, entity_path: str, components: list[str]) -> str:
        raw = f"{kind}:{entity_path}:{':'.join(components)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _confidence_from_count(supporting: int, total: int) -> float:
        """Scale confidence by number of supporting traces.

        1 trace = 0.4 (speculative), 3 = 0.65, 5+ = 0.8, 10+ = 0.85.
        Never exceeds 0.85 since patterns are inferred, not verified.
        """
        ratio = supporting / max(total, 1)
        base = 0.4 + (ratio * 0.35)
        count_bonus = min(supporting / 15, 0.1)
        return round(min(base + count_bonus, 0.85), 3)
