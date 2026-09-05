"""CPU-only contracts for the shared causal topometric graph.

These tests deliberately use hand-authored snapshots.  They do not load ROS,
Gazebo, bags, checkpoints, or evaluator ground truth: the shared graph must be
testable from the causal evidence exchanged by robots alone.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mtare_topo.topology.shared_causal_graph import SharedCausalTopometricGraph


def _node(
    node_id: int,
    xyz: tuple[float, float, float],
    *,
    role: str = "junction",
    heading: float = 0.0,
    state: str = "observed",
    embedding: float = 1.0,
    node_kind: str = "structural",
    observation_count: int = 1,
) -> dict:
    return {
        "id": node_id,
        "node_kind": node_kind,
        "xyz_m": list(xyz),
        "role": role,
        "role_probabilities_mean": [0.1, 0.8, 0.1],
        "exit_headings_world_deg": [heading],
        "exit_stubs": [{"heading_world_deg": heading, "state": state, "confidence": 0.8}],
        "confidence_mean": 0.8,
        "z_role_mean": (np.ones(128) * embedding).tolist(),
        "observation_count": observation_count,
    }


def _snapshot(nodes: list[dict], *, edges: list[dict] | None = None, current_node: int | None = None) -> dict:
    return {
        "schema_version": "causal_topometric_graph_v2",
        "nodes": nodes,
        "edges": [] if edges is None else edges,
        "current_node": current_node,
    }


def _edge(edge_id: int, first: int, second: int, length: float, count: int) -> dict:
    return {
        "id": edge_id,
        "from": first,
        "to": second,
        "kind": "verified_traversed",
        "traversals": [{"length_m": length}],
        "verified_traversal_count": count,
        "minimum_traversed_length_m": length,
    }


def test_latest_snapshot_rebuild_does_not_repeat_evidence() -> None:
    graph = SharedCausalTopometricGraph()
    first = _snapshot([_node(0, (0.0, 0.0, 0.0), observation_count=2)])
    graph.ingest(robot_id="r1", revision=1, stamp_sec=1.0, snapshot=first)
    graph.ingest(robot_id="r2", revision=1, stamp_sec=1.0, snapshot=first)
    assert graph.nodes[0]["observation_count"] == 4

    # A newer snapshot replaces r1's previous snapshot; it is not appended to
    # the shared evidence a second time.
    graph.ingest(robot_id="r1", revision=2, stamp_sec=2.0, snapshot=first)
    assert graph.nodes[0]["observation_count"] == 4
    assert len(graph.nodes[0]["source_nodes"]) == 2


def test_same_structure_fuses_but_role_distance_and_embedding_do_not() -> None:
    graph = SharedCausalTopometricGraph()
    graph.ingest(robot_id="r1", revision=1, stamp_sec=1.0, snapshot=_snapshot([
        _node(0, (0.0, 0.0, 0.0)),
        _node(1, (20.0, 0.0, 0.0)),
        _node(2, (0.5, 0.0, 0.0), role="terminal"),
        _node(3, (1.0, 0.0, 0.0), role="interior", embedding=-1.0),
    ]))
    graph.ingest(robot_id="r2", revision=1, stamp_sec=1.0, snapshot=_snapshot([
        _node(0, (0.2, 0.1, 0.0)),  # same role/heading/embedding: fuse
        _node(1, (20.2, 0.0, 0.0)),  # remote from node 0: fuse with r1 node 1
        _node(2, (0.3, 0.0, 0.0), role="interior", heading=90.0),  # role/heading differ from nearby structures
        _node(3, (0.4, 0.0, 0.0), role="interior", embedding=1.0),  # cosine is -1: no fuse
    ]))
    assert len(graph.nodes) == 6
    assert graph.local_to_shared["r2"][0] == graph.local_to_shared["r1"][0]
    assert graph.local_to_shared["r2"][1] == graph.local_to_shared["r1"][1]
    assert graph.local_to_shared["r2"][2] != graph.local_to_shared["r1"][0]
    assert graph.local_to_shared["r2"][3] != graph.local_to_shared["r1"][0]


def test_verified_edge_evidence_is_merged_and_minimum_is_retained() -> None:
    graph = SharedCausalTopometricGraph()
    graph.ingest(robot_id="r1", revision=1, stamp_sec=1.0, snapshot=_snapshot(
        [_node(0, (0.0, 0.0, 0.0)), _node(1, (10.0, 0.0, 0.0))],
        edges=[_edge(0, 0, 1, 10.0, 2)], current_node=0,
    ))
    graph.ingest(robot_id="r2", revision=1, stamp_sec=1.0, snapshot=_snapshot(
        [_node(0, (0.2, 0.0, 0.0)), _node(1, (10.2, 0.0, 0.0))],
        edges=[_edge(0, 0, 1, 8.0, 3)], current_node=0,
    ))
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge["minimum_traversed_length_m"] == 8.0
    assert edge["verified_traversal_count"] == 5
    assert {(item["robot_id"], item["local_edge_id"]) for item in edge["source_edges"]} == {("r1", 0), ("r2", 0)}


def test_stub_lifecycle_and_frontier_tasks() -> None:
    graph = SharedCausalTopometricGraph()
    graph.ingest(robot_id="r1", revision=1, stamp_sec=1.0, snapshot=_snapshot([
        _node(0, (0.0, 0.0, 0.0), heading=0.0, state="observed"),
    ], current_node=0))
    assert len(graph.frontier_tasks()) == 1
    graph.ingest(robot_id="r2", revision=1, stamp_sec=1.0, snapshot=_snapshot([
        _node(0, (0.1, 0.0, 0.0), heading=0.0, state="traversed"),
    ], current_node=0))
    assert graph.nodes[0]["exit_stubs"][0]["state"] == "traversed"
    assert graph.frontier_tasks() == ()


def test_graph_costs_are_per_robot_and_frontier_specific() -> None:
    graph = SharedCausalTopometricGraph()
    nodes = [
        _node(0, (0.0, 0.0, 0.0), state="traversed", heading=0.0),
        _node(1, (10.0, 0.0, 0.0), state="observed", heading=90.0),
        _node(2, (20.0, 0.0, 0.0), state="observed", heading=180.0),
    ]
    edges = [_edge(0, 0, 1, 10.0, 1), _edge(1, 1, 2, 5.0, 1)]
    graph.ingest(robot_id="r1", revision=1, stamp_sec=1.0, snapshot=_snapshot(copy.deepcopy(nodes), edges=edges, current_node=0))
    graph.ingest(robot_id="r2", revision=1, stamp_sec=1.0, snapshot=_snapshot(copy.deepcopy(nodes), edges=edges, current_node=2))
    costs = graph.graph_costs()
    task_ids = {task.frontier_id for task in graph.frontier_tasks()}
    assert len(task_ids) == 2
    assert costs[("r1", "n1:s0")] == 10.0
    assert costs[("r1", "n2:s0")] == 15.0
    assert costs[("r2", "n2:s0")] == 0.0
    assert costs[("r2", "n1:s0")] == 5.0


def test_revision_timestamp_and_wire_bytes_are_monotonic_deterministic() -> None:
    graph = SharedCausalTopometricGraph()
    snapshot = _snapshot([_node(0, (0.0, 0.0, 0.0))])
    result = graph.ingest(robot_id="r1", revision=1, stamp_sec=1.0, snapshot=snapshot)
    reordered = {"current_node": None, "edges": [], "nodes": snapshot["nodes"], "schema_version": snapshot["schema_version"]}
    other = SharedCausalTopometricGraph()
    result_other = other.ingest(robot_id="r1", revision=1, stamp_sec=1.0, snapshot=reordered)
    assert result["received_bytes"] == result_other["received_bytes"]
    assert graph.snapshot()["received_bytes_total"] == result["received_bytes"]
    with pytest.raises(ValueError, match="revision"):
        graph.ingest(robot_id="r1", revision=1, stamp_sec=2.0, snapshot=snapshot)
    with pytest.raises(ValueError, match="timestamp"):
        graph.ingest(robot_id="r1", revision=2, stamp_sec=1.0, snapshot=snapshot)
    # The graph must remain independent of any evaluator-only annotation.
    assert "evaluator_gt_edge_ids" not in graph.snapshot()
