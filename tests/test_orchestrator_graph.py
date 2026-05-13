import json

import pytest

from orchestrator_core import SystemEdge, SystemGraph, SystemNode


def _node(node_id, node_type="orchestrator"):
    return SystemNode(node_id, node_type, None, None, (), {})


def test_graph_adds_nodes_edges_and_exports_json():
    graph = SystemGraph()
    graph.add_node(_node("producer", "candidate_generator"))
    graph.add_node(_node("artifact:a0", "artifact"))
    graph.add_node(_node("memory", "memory_store"))
    graph.add_edge(SystemEdge("e0", "producer", "artifact:a0", "produces", "operator_program", {}))
    graph.add_edge(SystemEdge("e1", "artifact:a0", "memory", "stores", "operator_program", {}))
    payload = graph.to_dict()
    assert payload["node_count"] == 3
    assert payload["edge_count"] == 2
    json.dumps(payload, sort_keys=True)


def test_graph_rejects_duplicate_nodes_and_missing_endpoints():
    graph = SystemGraph()
    graph.add_node(_node("n0"))
    with pytest.raises(ValueError):
        graph.add_node(_node("n0"))
    with pytest.raises(ValueError):
        graph.add_edge(SystemEdge("e0", "n0", "missing", "routes_to", None, {}))


def test_graph_upstream_downstream_and_artifact_lineage():
    graph = SystemGraph()
    graph.add_node(_node("producer", "candidate_generator"))
    graph.add_node(_node("artifact:a0", "artifact"))
    graph.add_node(_node("version:v0", "problem_space_version"))
    graph.add_edge(SystemEdge("e0", "producer", "artifact:a0", "produces", "operator_program", {}))
    graph.add_edge(SystemEdge("e1", "artifact:a0", "version:v0", "accepts", "operator_program", {}))
    assert [node.node_id for node in graph.upstream_nodes("artifact:a0")] == ["producer"]
    assert [node.node_id for node in graph.downstream_nodes("artifact:a0")] == ["version:v0"]
    lineage = graph.artifact_lineage("a0")
    assert lineage["artifact_id"] == "a0"
    assert len(lineage["incoming_edges"]) == 1
    assert len(lineage["outgoing_edges"]) == 1


def test_graph_rejects_invalid_schema_values():
    with pytest.raises(ValueError):
        _node("bad", "invalid")
    with pytest.raises(ValueError):
        SystemEdge("e0", "a", "b", "invalid", None, {})
