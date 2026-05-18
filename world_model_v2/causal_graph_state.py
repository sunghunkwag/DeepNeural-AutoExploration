"""Directed causal graph state with latent/confounder flags."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    relation: str = "causes"
    weight: float = 1.0
    latent_confounder: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CausalGraphState:
    edges: List[CausalEdge] = field(default_factory=list)
    latent_nodes: List[str] = field(default_factory=list)

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        relation: str = "causes",
        weight: float = 1.0,
        latent_confounder: bool = False,
    ) -> "CausalGraphState":
        edge = CausalEdge(source, target, relation, float(weight), bool(latent_confounder))
        return CausalGraphState(list(self.edges) + [edge], list(self.latent_nodes))

    def with_latent_node(self, node_id: str) -> "CausalGraphState":
        nodes = list(dict.fromkeys(list(self.latent_nodes) + [node_id]))
        return CausalGraphState(list(self.edges), nodes)

    def parents(self, node_id: str) -> List[str]:
        return [edge.source for edge in self.edges if edge.target == node_id]

    def children(self, node_id: str) -> List[str]:
        return [edge.target for edge in self.edges if edge.source == node_id]

    def has_latent_confounder(self, source: str, target: str) -> bool:
        return any(
            edge.source == source and edge.target == target and edge.latent_confounder
            for edge in self.edges
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "edges": [edge.to_dict() for edge in self.edges],
            "latent_nodes": list(self.latent_nodes),
        }
