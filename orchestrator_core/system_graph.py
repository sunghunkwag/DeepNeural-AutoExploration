"""Typed integration graph for orchestrator artifacts and routing."""

from __future__ import annotations

from typing import Any

from .schemas import SystemEdge, SystemNode, json_safe


class SystemGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, SystemNode] = {}
        self._edges: dict[str, SystemEdge] = {}

    @property
    def nodes(self) -> dict[str, SystemNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> dict[str, SystemEdge]:
        return dict(self._edges)

    def add_node(self, node: SystemNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        self._nodes[node.node_id] = node

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def add_edge(self, edge: SystemEdge) -> None:
        if edge.edge_id in self._edges:
            raise ValueError(f"duplicate edge_id: {edge.edge_id}")
        if edge.source_node_id not in self._nodes:
            raise ValueError(f"edge source does not exist: {edge.source_node_id}")
        if edge.target_node_id not in self._nodes:
            raise ValueError(f"edge target does not exist: {edge.target_node_id}")
        self._edges[edge.edge_id] = edge

    def downstream_nodes(self, node_id: str) -> tuple[SystemNode, ...]:
        if node_id not in self._nodes:
            raise KeyError(f"unknown node_id: {node_id}")
        targets = [edge.target_node_id for edge in self._sorted_edges() if edge.source_node_id == node_id]
        return tuple(self._nodes[target] for target in targets)

    def upstream_nodes(self, node_id: str) -> tuple[SystemNode, ...]:
        if node_id not in self._nodes:
            raise KeyError(f"unknown node_id: {node_id}")
        sources = [edge.source_node_id for edge in self._sorted_edges() if edge.target_node_id == node_id]
        return tuple(self._nodes[source] for source in sources)

    def artifact_lineage(self, artifact_id: str) -> dict[str, Any]:
        node_id = artifact_id if artifact_id in self._nodes else f"artifact:{artifact_id}"
        if node_id not in self._nodes:
            raise KeyError(f"unknown artifact_id: {artifact_id}")
        incoming = [edge for edge in self._sorted_edges() if edge.target_node_id == node_id]
        outgoing = [edge for edge in self._sorted_edges() if edge.source_node_id == node_id]
        upstream = [self._nodes[edge.source_node_id] for edge in incoming]
        downstream = [self._nodes[edge.target_node_id] for edge in outgoing]
        return {
            "artifact_id": artifact_id,
            "artifact_node": self._nodes[node_id].to_dict(),
            "upstream_nodes": [node.to_dict() for node in upstream],
            "downstream_nodes": [node.to_dict() for node in downstream],
            "incoming_edges": [edge.to_dict() for edge in incoming],
            "outgoing_edges": [edge.to_dict() for edge in outgoing],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in sorted(self._nodes.values(), key=lambda item: item.node_id)],
            "edges": [edge.to_dict() for edge in self._sorted_edges()],
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
        }

    def _sorted_edges(self) -> list[SystemEdge]:
        return sorted(self._edges.values(), key=lambda item: item.edge_id)

    def __len__(self) -> int:
        return len(self._nodes)

    def json_safe(self) -> dict[str, Any]:
        return json_safe(self.to_dict())
