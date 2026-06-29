"""AgentClusterStrategy — community detection over agent similarity graphs."""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import networkx as nx

from amfs_core.models import AgentCluster, Digest, DigestType, GraphEdge

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Helper functions ──────────────────────────────────────────────────


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two numeric vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _normalize_dist(counts: dict) -> list[float]:
    """Normalize a {label: count} dict into a probability vector (sorted by key)."""
    if not counts:
        return []
    keys = sorted(counts.keys())
    total = sum(counts.values())
    if total == 0:
        return [0.0] * len(keys)
    return [counts.get(k, 0) / total for k in keys]


# ── Strategy ──────────────────────────────────────────────────────────


class AgentClusterStrategy:
    """Detects communities of related agents using Louvain community detection.

    Feature vectors per agent:
      - entity_set: set of entity_paths the agent has written to
      - type_vec: normalized [fact, belief, experience] distribution
      - collab_set: set of agents this agent has learned_from (graph edges)

    Pairwise similarity:
      0.5 * jaccard(entity_sets) + 0.2 * cosine(type_vecs) + 0.3 * collab_overlap
    """

    def __init__(
        self, resolution: float = 1.0, sim_threshold: float = 0.15, scope: str = "cluster:all",
    ) -> None:
        self._resolution = resolution
        self._sim_threshold = sim_threshold
        self._scope = scope

    def compute_clusters(
        self, agents: list[dict], graph_edges: list[GraphEdge],
    ) -> Digest:
        """Compute agent clusters via Louvain community detection.

        Args:
            agents: enriched agent dicts (from list_agents_enriched), each containing
                    at minimum ``agent_id``, ``entity_paths``, ``memory_type_counts``,
                    and optionally ``platform``, ``entry_count``.
            graph_edges: learned_from edges from the knowledge graph.

        Returns:
            A Digest with DigestType.AGENT_CLUSTERS.
        """
        if len(agents) < 3 or not graph_edges:
            return self._empty_digest(agents)

        agent_ids = [a["agent_id"] for a in agents]
        agent_map = {a["agent_id"]: a for a in agents}

        entity_sets: dict[str, set] = {}
        type_vecs: dict[str, list[float]] = {}
        collab_sets: dict[str, set] = {}

        all_type_keys = {"fact", "belief", "experience"}

        for a in agents:
            aid = a["agent_id"]
            entity_sets[aid] = set(a.get("entity_paths") or [])

            raw_counts = a.get("type_dist") or a.get("memory_type_counts") or {}
            for k in all_type_keys:
                raw_counts.setdefault(k, 0)
            type_vecs[aid] = _normalize_dist(raw_counts)

            collab_sets[aid] = set()

        for edge in graph_edges:
            if edge.relation == "learned_from":
                src = edge.source_entity
                tgt = edge.target_entity
                if src in collab_sets:
                    collab_sets[src].add(tgt)
                if tgt in collab_sets:
                    collab_sets[tgt].add(src)

        G = nx.Graph()
        G.add_nodes_from(agent_ids)

        for i, aid_a in enumerate(agent_ids):
            for j in range(i + 1, len(agent_ids)):
                aid_b = agent_ids[j]

                ent_sim = jaccard(entity_sets[aid_a], entity_sets[aid_b])
                type_sim = cosine(type_vecs[aid_a], type_vecs[aid_b])
                collab_overlap = 1.0 if (
                    aid_b in collab_sets[aid_a] or aid_a in collab_sets[aid_b]
                ) else 0.0

                sim = 0.5 * ent_sim + 0.2 * type_sim + 0.3 * collab_overlap

                if sim > self._sim_threshold:
                    G.add_edge(aid_a, aid_b, weight=sim)

        if G.number_of_edges() == 0:
            return self._empty_digest(agents)

        communities = nx.community.louvain_communities(
            G, weight="weight", resolution=self._resolution, seed=42,
        )

        clusters: list[AgentCluster] = []
        clustered_agents: set[str] = set()

        for community in communities:
            members = sorted(community)
            if len(members) < 2:
                continue

            clustered_agents.update(members)

            cluster_id = hashlib.md5(
                "|".join(members).encode()
            ).hexdigest()[:12]

            entity_counter: Counter = Counter()
            platform_counter: Counter = Counter()
            total_entries = 0

            for aid in members:
                a = agent_map.get(aid, {})
                for ep in (a.get("entity_paths") or []):
                    entity_counter[ep] += 1
                platform = (
                    a.get("platform")
                    or (a.get("session_metadata") or {}).get("platform")
                )
                if platform:
                    platform_counter[platform] += 1
                total_entries += a.get("entries_written", 0) or a.get("entry_count", 0)

            dominant_entities = [ep for ep, _ in entity_counter.most_common(3)]

            dominant_platform = (
                platform_counter.most_common(1)[0][0] if platform_counter else None
            )

            suggested_name = self._suggest_name(dominant_entities, members)

            subgraph = G.subgraph(members)
            edge_weights = [
                subgraph[u][v].get("weight", 0) for u, v in subgraph.edges()
            ]
            max_possible = len(members) * (len(members) - 1) / 2
            cohesion_score = (
                round(sum(edge_weights) / max_possible, 3) if max_possible > 0 else 0.0
            )

            rationale_parts = []
            if dominant_entities:
                rationale_parts.append(
                    f"share entities: {', '.join(dominant_entities[:3])}"
                )
            if dominant_platform:
                rationale_parts.append(f"common platform: {dominant_platform}")
            collab_pairs = sum(
                1 for u, v in subgraph.edges()
                if u in collab_sets.get(v, set()) or v in collab_sets.get(u, set())
            )
            if collab_pairs:
                rationale_parts.append(f"{collab_pairs} collaboration edge(s)")

            clusters.append(AgentCluster(
                cluster_id=cluster_id,
                suggested_name=suggested_name,
                agents=members,
                dominant_entities=dominant_entities,
                dominant_platform=dominant_platform,
                cohesion_score=cohesion_score,
                rationale="; ".join(rationale_parts) if rationale_parts else "similarity-based",
                total_entries=total_entries,
            ))

        unclustered = sorted(set(agent_ids) - clustered_agents)

        return Digest(
            digest_type=DigestType.AGENT_CLUSTERS,
            scope=self._scope,
            summary={
                "clusters": [c.model_dump() for c in clusters],
                "cluster_count": len(clusters),
                "clustered_agent_count": len(clustered_agents),
                "unclustered_agents": unclustered,
                "total_agents": len(agents),
                "resolution": self._resolution,
                "sim_threshold": self._sim_threshold,
            },
            entry_count=len(agents),
            source_agents=agent_ids,
            compiled_at=datetime.now(timezone.utc),
        )

    def _empty_digest(self, agents: list[dict]) -> Digest:
        """Return a digest indicating clustering wasn't possible."""
        agent_ids = [a["agent_id"] for a in agents]
        return Digest(
            digest_type=DigestType.AGENT_CLUSTERS,
            scope=self._scope,
            summary={
                "clusters": [],
                "cluster_count": 0,
                "clustered_agent_count": 0,
                "unclustered_agents": sorted(agent_ids),
                "total_agents": len(agents),
                "reason": "insufficient data (< 3 agents or no edges)",
            },
            entry_count=len(agents),
            source_agents=agent_ids,
            compiled_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _suggest_name(dominant_entities: list[str], members: list[str]) -> str:
        """Generate a cluster name from dominant entity paths."""
        if dominant_entities:
            primary = dominant_entities[0]
            parts = primary.split("/")
            if len(parts) >= 2:
                return f"{parts[-1]}-agents"
            return f"{primary}-agents"
        return f"cluster-{len(members)}-agents"
