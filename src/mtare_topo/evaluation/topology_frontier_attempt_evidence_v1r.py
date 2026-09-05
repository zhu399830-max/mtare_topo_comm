"""Fail-closed V1R validation for frontier-attempt execution evidence."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from mtare_topo.evaluation.topology_frontier_attempt_evidence import (
    audit_frontier_attempt_evidence,
)


def _frontier_key(target: Mapping[str, Any]) -> tuple[int, int] | None:
    frontier = target.get("frontier")
    if not isinstance(frontier, Mapping):
        return None
    return int(frontier["node_id"]), int(frontier["stub_index"])


def _verified_edges(
    graph_snapshot: Mapping[str, Any], node_ids: set[int]
) -> dict[frozenset[int], Mapping[str, Any]]:
    raw_edges = graph_snapshot.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError("frontier attempt graph requires an edge list")
    edges: dict[frozenset[int], Mapping[str, Any]] = {}
    edge_ids: set[int] = set()
    for edge in raw_edges:
        edge_id = int(edge["id"])
        first, second = int(edge["from"]), int(edge["to"])
        pair = frozenset((first, second))
        if (
            edge_id in edge_ids
            or len(pair) != 2
            or not pair <= node_ids
            or pair in edges
            or edge.get("kind") != "verified_traversed"
            or not isinstance(edge.get("traversals"), list)
            or not edge["traversals"]
        ):
            raise ValueError("frontier attempt graph has an invalid verified edge")
        edge_ids.add(edge_id)
        edges[pair] = edge
    return edges


def _validate_target(
    target: Mapping[str, Any],
    *,
    current_node: int,
    nodes: Mapping[int, Mapping[str, Any]],
    edges: Mapping[frozenset[int], Mapping[str, Any]],
) -> None:
    mode = str(target.get("mode"))
    frontier = _frontier_key(target)
    if frontier is not None:
        node_id, stub_index = frontier
        if node_id not in nodes or not 0 <= stub_index < len(nodes[node_id]["exit_stubs"]):
            raise ValueError("target frontier stub identity does not exist in final append-only graph")
    if mode == "frontier_exit":
        if (
            frontier is None
            or frontier[0] != current_node
            or int(target.get("frontier_node_id", -1)) != current_node
            or int(target.get("next_hop_node_id", -1)) != current_node
            or [int(value) for value in target.get("graph_path_node_ids", [])]
            != [current_node]
        ):
            raise ValueError("frontier_exit target identity is inconsistent with current node")
    elif mode == "graph_backtrack":
        path = [int(value) for value in target.get("graph_path_node_ids", [])]
        if (
            frontier is None
            or len(path) < 2
            or path[0] != current_node
            or path[1] != int(target.get("next_hop_node_id", -1))
            or path[-1] != int(target.get("frontier_node_id", -1))
            or path[-1] != frontier[0]
            or any(node_id not in nodes for node_id in path)
            or any(frozenset(pair) not in edges for pair in zip(path, path[1:]))
        ):
            raise ValueError("graph_backtrack target path is not trace-verified or identity-consistent")
    elif frontier is not None:
        raise ValueError("only frontier target modes may carry frontier identity")


def audit_frontier_attempt_evidence_v1r(
    decision_rows: Sequence[Mapping[str, Any]],
    graph_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every trace identity before classifying execution outcomes."""

    if graph_snapshot.get("schema_version") != "causal_topometric_graph_v2":
        raise ValueError("frontier attempt audit requires the frozen V2 graph schema")
    raw_nodes = graph_snapshot.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("frontier attempt graph requires nonempty nodes")
    nodes = {int(node["id"]): node for node in raw_nodes}
    if len(nodes) != len(raw_nodes):
        raise ValueError("frontier attempt graph node IDs must be unique")
    for node in nodes.values():
        stubs = node.get("exit_stubs")
        if not isinstance(stubs, list):
            raise ValueError("frontier attempt node requires append-only exit stubs")
        for stub in stubs:
            heading = float(stub.get("heading_world_deg", math.nan))
            if not math.isfinite(heading) or stub.get("state") not in {"observed", "traversed"}:
                raise ValueError("frontier attempt node has an invalid exit stub")
    edges = _verified_edges(graph_snapshot, set(nodes))

    previous_arc: float | None = None
    for expected_index, row in enumerate(decision_rows):
        if int(row.get("frame_index", -1)) != expected_index:
            raise ValueError("decision frames must be contiguous and zero-based")
        arc = float(row.get("route_arc_m", math.nan))
        if not math.isfinite(arc) or (previous_arc is not None and arc < previous_arc):
            raise ValueError("route arc must be finite and nondecreasing")
        previous_arc = arc
        update = row.get("graph_update")
        target = row.get("target")
        if not isinstance(update, Mapping) or not isinstance(target, Mapping):
            raise ValueError("decision graph update and target must be mappings")
        update_node = int(update.get("node_id", -1))
        update_arc = float(update.get("route_arc_m", math.nan))
        if (
            int(update.get("frame_index", -1)) != expected_index
            or not math.isclose(update_arc, arc, rel_tol=0.0, abs_tol=1e-12)
            or update_node not in nodes
        ):
            raise ValueError("graph update identity does not match its decision row")
        _validate_target(
            target,
            current_node=update_node,
            nodes=nodes,
            edges=edges,
        )

    verified_transition_count = 0
    transition_evidence: dict[int, dict[str, Any]] = {}
    for frame_index in range(1, len(decision_rows)):
        previous_node = int(decision_rows[frame_index - 1]["graph_update"]["node_id"])
        current_node = int(decision_rows[frame_index]["graph_update"]["node_id"])
        if previous_node == current_node:
            continue
        edge = edges.get(frozenset((previous_node, current_node)))
        if edge is None:
            raise ValueError("decision node transition lacks a trace-verified graph edge")
        arc = float(decision_rows[frame_index]["route_arc_m"])
        matches = [
            traversal
            for traversal in edge["traversals"]
            if int(traversal.get("from", -1)) == previous_node
            and int(traversal.get("to", -1)) == current_node
            and int(traversal.get("end_frame", -1)) == frame_index
            and math.isclose(
                float(traversal.get("end_route_arc_m", math.nan)),
                arc,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if len(matches) != 1:
            raise ValueError("decision node transition lacks one exact verified traversal")
        verified_transition_count += 1
        transition_evidence[frame_index] = {
            "verified_edge_id": int(edge["id"]),
            "verified_traversal_start_frame": int(matches[0]["start_frame"]),
            "verified_traversal_end_frame": int(matches[0]["end_frame"]),
        }

    audit = audit_frontier_attempt_evidence(decision_rows, graph_snapshot)
    events = []
    for event in audit["events"]:
        value = dict(event)
        if value["source_node_id"] != value["destination_node_id"]:
            value.update(transition_evidence[value["frame_index"]])
        elif value["graph_update_reason"] != "loop_merge":
            raise ValueError("same-node attempt evidence requires a loop merge")
        events.append(value)
    outcome_counts = Counter(event["outcome"] for event in events)
    if dict(sorted(outcome_counts.items())) != audit["outcome_counts"]:
        raise ValueError("frontier attempt outcome count is inconsistent with events")
    result = dict(audit)
    result.update({
        "schema_version": "topology_frontier_attempt_evidence_v1r",
        "verified_graph_transition_count": verified_transition_count,
        "verified_attempt_departure_count": sum(
            event["source_node_id"] != event["destination_node_id"]
            for event in events
        ),
        "append_only_stub_identity_required": True,
        "events": events,
    })
    return result


__all__ = ["audit_frontier_attempt_evidence_v1r"]
