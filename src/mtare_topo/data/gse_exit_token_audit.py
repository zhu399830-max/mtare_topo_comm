"""Deterministic incident-only exit-token audit for one GSE development world."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.data.gse_sensor_export import world_unique_frame_poses
from mtare_topo.teacher.gse_exit_teacher import (
    ExitRouteGeometryContext,
    audit_exit_route_targets,
    incident_exit_candidates,
)
from mtare_topo.teacher.gse_mesh_geometry_teacher import (
    BatchRaycastFunction,
    MeshGeometryTeacherConfig,
)


NODE_EVENTS = frozenset({"junction", "terminal"})


def selected_node_id_from_observation(
    observation: Mapping[str, Any],
    *,
    graph_node_ids: set[str],
) -> str | None:
    """Decode only objective node-event identities; edge events remain route-local."""

    event = str(observation["event"])
    identity = observation.get("identity")
    if event not in NODE_EVENTS:
        return None
    prefix = f"{observation['parent_id']}:node:"
    if not isinstance(identity, str) or not identity.startswith(prefix):
        raise ValueError("node event lacks a parent-scoped node identity")
    node_id = identity[len(prefix) :]
    if not node_id or node_id not in graph_node_ids:
        raise ValueError("node-event identity does not resolve in the current graph")
    return node_id


def world_exit_token_audit(
    *,
    parent_id: str,
    split: str,
    observations: Sequence[Mapping[str, Any]],
    traversal_manifest: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    geometry_parameters: Mapping[str, Any],
    cast_distances: BatchRaycastFunction,
    mesh_config: MeshGeometryTeacherConfig | None = None,
    representative_offset_m: float = 2.5,
    los_margin_m: float = 0.25,
) -> dict[str, Any]:
    """Audit every candidate and visible exit token without collapsing physical edges."""

    if split not in {"train", "validation"} or parent_id.endswith("_C10"):
        raise ValueError("exit-token audit permits only train/validation parents")
    node_ids = {str(node["id"]) for node in graph["nodes"]}
    edge_by_id = {str(edge["id"]): edge for edge in graph["edges"]}
    if len(node_ids) != len(graph["nodes"]) or len(edge_by_id) != len(graph["edges"]):
        raise ValueError("graph node/edge identities must be unique")
    poses = world_unique_frame_poses(
        parent_id=parent_id,
        traversal_manifest=traversal_manifest,
        graph=graph,
        spline_document=spline_document,
        geometry_parameters=geometry_parameters,
    )
    pose_by_global_index = {
        int(global_index): index
        for index, global_index in enumerate(poses.global_frame_indices)
    }
    context = ExitRouteGeometryContext(
        parent_id=parent_id,
        graph=graph,
        spline_document=spline_document,
        geometry_parameters=geometry_parameters,
    )

    ordered_observations = sorted(
        observations,
        key=lambda row: (int(row["global_sequence_index"]), str(row["observation_id"])),
    )
    route_targets = []
    current_sensors: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    observation_candidate_counts: Counter[str] = Counter()
    event_by_observation: dict[str, str] = {}
    event_observation_counts: Counter[str] = Counter()
    same_tunnel_distinct_edge_observations = 0
    seen_observation_ids: set[str] = set()

    for observation in ordered_observations:
        if str(observation["parent_id"]) != parent_id or str(observation["split"]) != split:
            raise ValueError("observation crosses the requested parent or split")
        observation_id = str(observation["observation_id"])
        if observation_id in seen_observation_ids:
            raise ValueError("duplicate observation identity")
        seen_observation_ids.add(observation_id)
        references = tuple(int(value) for value in observation["global_frame_references"])
        if len(references) != 5 or any(b != a + 1 for a, b in zip(references, references[1:])):
            raise RuntimeError("observation does not preserve a contiguous five-frame history")
        pose_index = pose_by_global_index.get(references[-1])
        if pose_index is None:
            raise RuntimeError("observation anchor does not resolve to a unique frame pose")
        if (
            poses.traversal_ids[pose_index] != str(observation["traversal_id"])
            or int(poses.local_frame_indices[pose_index]) != int(observation["frame_index"])
            or not np.isclose(
                float(poses.arc_m[pose_index]),
                float(observation["traversal_arc_m"]),
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise RuntimeError("teacher observation and reconstructed sensor pose disagree")
        current_edge_id = str(observation["edge_id"])
        if current_edge_id not in edge_by_id:
            raise ValueError("observation edge is absent from current graph")
        selected_node_id = selected_node_id_from_observation(
            observation,
            graph_node_ids=node_ids,
        )
        candidates = incident_exit_candidates(
            parent_id=parent_id,
            graph=dict(graph),
            current_edge_id=current_edge_id,
            selected_node_id=selected_node_id,
        )
        if selected_node_id is None:
            valid_scope = all(candidate.edge_id == current_edge_id for candidate in candidates)
        else:
            valid_scope = all(
                candidate.from_node_id == selected_node_id
                and selected_node_id in {str(value) for value in edge_by_id[candidate.edge_id]["node_ids"]}
                for candidate in candidates
            )
        if not valid_scope:
            raise RuntimeError("nonincident physical edge entered the exit candidate set")
        tunnel_to_edges: dict[str, set[str]] = {}
        for candidate in candidates:
            for tunnel_id in candidate.source_tunnel_ids:
                tunnel_to_edges.setdefault(tunnel_id, set()).add(candidate.edge_id)
        if any(len(edge_ids) > 1 for edge_ids in tunnel_to_edges.values()):
            same_tunnel_distinct_edge_observations += 1
        event = str(observation["event"])
        event_by_observation[observation_id] = event
        event_observation_counts[event] += 1
        observation_candidate_counts[observation_id] = len(candidates)
        for candidate_index, candidate in enumerate(candidates):
            route_targets.append(
                context.route_target(
                    candidate=candidate,
                    current_axis_xyz_m=poses.axis_xyz_m[pose_index],
                    robot_yaw_deg=float(poses.yaw_deg[pose_index]),
                    selected_node_id=selected_node_id,
                    representative_offset_m=representative_offset_m,
                )
            )
            current_sensors.append(poses.sensor_xyz_m[pose_index])
            provenance.append(
                {
                    "observation_id": observation_id,
                    "parent_id": parent_id,
                    "split": split,
                    "event": event,
                    "structural_identity": observation.get("identity"),
                    "traversal_id": str(observation["traversal_id"]),
                    "frame_index": int(observation["frame_index"]),
                    "global_frame_index": references[-1],
                    "edge_id": current_edge_id,
                    "selected_node_id": selected_node_id,
                    "candidate_index": candidate_index,
                    "candidate_count": len(candidates),
                }
            )

    audited = audit_exit_route_targets(
        route_targets,
        current_sensor_xyz_m=np.asarray(current_sensors, dtype=np.float64).reshape(-1, 3),
        cast_distances=cast_distances,
        mesh_config=mesh_config,
        los_margin_m=los_margin_m,
    )
    rows: list[dict[str, Any]] = []
    visible_by_observation: Counter[str] = Counter()
    event_candidate_counts: Counter[str] = Counter()
    event_visible_counts: Counter[str] = Counter()
    width_valid_counts: Counter[str] = Counter()
    for source, target, token in zip(provenance, route_targets, audited, strict=True):
        if target.candidate.identity != token.identity:
            raise RuntimeError("audited token order or identity drift")
        event = source["event"]
        event_candidate_counts[event] += 1
        width_valid_counts[event] += int(token.opening_width_valid)
        if token.visible:
            visible_by_observation[source["observation_id"]] += 1
            event_visible_counts[event] += 1
        rows.append(
            {
                **source,
                "candidate": target.candidate.to_dict(),
                "representative_axis_xyz_m": list(target.representative_axis_xyz_m),
                "representative_sensor_xyz_m": list(target.representative_sensor_xyz_m),
                "away_tangent_xyz": list(target.away_tangent_xyz),
                "token": token.to_dict(),
            }
        )
    zero_visible_node_events = sum(
        int(visible_by_observation[observation_id] == 0)
        for observation_id in observation_candidate_counts
        if event_by_observation[observation_id] in NODE_EVENTS
    )
    summary = {
        "parent_id": parent_id,
        "split": split,
        "observation_count": len(ordered_observations),
        "candidate_count": len(rows),
        "visible_token_count": sum(int(row["token"]["visible"]) for row in rows),
        "width_valid_count": sum(int(row["token"]["opening_width_valid"]) for row in rows),
        "node_event_observation_count": sum(event_observation_counts[event] for event in NODE_EVENTS),
        "zero_visible_node_event_count": zero_visible_node_events,
        "same_tunnel_distinct_edge_observation_count": same_tunnel_distinct_edge_observations,
        "maximum_candidates_per_observation": max(observation_candidate_counts.values(), default=0),
        "event_observation_counts": dict(sorted(event_observation_counts.items())),
        "event_candidate_counts": dict(sorted(event_candidate_counts.items())),
        "event_visible_counts": dict(sorted(event_visible_counts.items())),
        "event_width_valid_counts": dict(sorted(width_valid_counts.items())),
        "nonincident_candidate_count": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    if len(rows) != len(route_targets) or len(rows) != sum(observation_candidate_counts.values()):
        raise RuntimeError("exit-token candidate coverage mismatch")
    return {"rows": rows, "summary": summary}


__all__ = [
    "NODE_EVENTS",
    "selected_node_id_from_observation",
    "world_exit_token_audit",
]
