"""Read-only world proof for the causal geometry change-point Teacher."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping

import numpy as np

from mtare_topo.data.gse_sequence_inventory import (
    deduplicated_frame_arcs_and_sequence_indices,
    enumerate_directed_traversals,
)
from mtare_topo.data.gse_teacher_inventory import oriented_edge_polyline
from mtare_topo.teacher.gse_causal_change_point_teacher import (
    CausalChangePointTeacherConfig,
    DirectionalGeometryProfile,
    assign_change_point_identity,
    causal_episode_matches_point,
    pair_bidirectional_change_points,
    past_only_causal_change_episodes,
    persistent_two_sided_change_episodes,
)
from mtare_topo.teacher.gse_geometry_teacher import (
    GSETeacherConfig,
    PolylineGeometrySampler,
)
from mtare_topo.teacher.gse_mesh_geometry_teacher import (
    BatchRaycastFunction,
    MeshGeometryTeacherConfig,
    measure_mesh_geometry_batch,
    sensor_origin_from_axis,
)


def world_causal_change_point_proof(
    *,
    parent_id: str,
    split: str,
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    geometry_parameters: Mapping[str, Any],
    cast_distances: BatchRaycastFunction,
    spacing_m: float = 1.0,
    history_frames: int = 5,
) -> dict[str, Any]:
    """Recompute full mesh profiles and prove offline/online identity agreement."""

    if split != "train" or not parent_id.endswith(tuple(f"_C{i:02d}" for i in range(1, 9))):
        raise ValueError("change-point development proof permits only C01-C08 train parents")
    teacher_cfg = GSETeacherConfig()
    mesh_cfg = MeshGeometryTeacherConfig()
    cfg = CausalChangePointTeacherConfig(
        spacing_m=spacing_m,
        comparison_span_m=mesh_cfg.transition_comparison_span_m,
        width_change_m=mesh_cfg.transition_width_change_m,
        height_change_m=mesh_cfg.transition_height_change_m,
        endpoint_merge_radius_m=teacher_cfg.node_event_radius_m,
    )
    node_by_id = {str(node["id"]): node for node in graph["nodes"]}
    nodes = {node_id: np.asarray(node["xyz"], dtype=np.float64) for node_id, node in node_by_id.items()}
    node_degree = {node_id: int(node["degree"]) for node_id, node in node_by_id.items()}
    splines = {
        str(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in spline_document["tunnels"]
    }
    radii = {
        str(item["tunnel_id"]): float(item["radius_m"])
        for item in geometry_parameters["tunnels"]
    }
    fta_distance_m = float(geometry_parameters["fta_distance_m"])
    if not math.isfinite(fta_distance_m):
        raise ValueError("fta_distance_m must be finite")
    inventory = enumerate_directed_traversals(
        parent_id=parent_id,
        split=split,
        graph=graph,
        spline_document=spline_document,
        spacing_m=spacing_m,
        history_frames=history_frames,
    )
    inventory_by_id = {record.traversal_id: record for record in inventory}
    totals: Counter[str] = Counter()
    points: list[dict[str, Any]] = []
    directional_episodes: list[dict[str, Any]] = []
    causal_episodes_out: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    identity_edges: dict[str, set[str]] = defaultdict(set)
    identity_edge_point_counts: Counter[tuple[str, str]] = Counter()
    identity_kinds: dict[str, str] = {}
    assigned_observation: dict[tuple[str, int], str] = {}

    for edge in sorted(graph["edges"], key=lambda item: str(item["id"])):
        edge_id = str(edge["id"])
        profiles: list[DirectionalGeometryProfile] = []
        edge_endpoints: tuple[str, str] | None = None
        edge_length_m: float | None = None
        anchor_by_direction: list[np.ndarray] = []
        for direction_index, reverse in enumerate((False, True)):
            oriented, tunnel_id, from_node_id, to_node_id = oriented_edge_polyline(
                edge=edge,
                nodes=nodes,
                splines=splines,
                reverse=reverse,
            )
            traversal_id = f"{parent_id}:{edge_id}:d{direction_index}"
            record = inventory_by_id[traversal_id]
            sampler = PolylineGeometrySampler(oriented)
            if not math.isclose(sampler.length_m, record.length_m, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError("proof and traversal inventory lengths disagree")
            frame_arcs, anchors, _ = deduplicated_frame_arcs_and_sequence_indices(
                sampler.length_m,
                spacing_m=spacing_m,
                history_frames=history_frames,
            )
            anchor_by_direction.append(anchors)
            if len(frame_arcs) == 0:
                totals["zero_frame_traversals"] += 1
                continue
            spline_targets = [
                sampler.target(float(arc), tunnel_radius_m=radii[tunnel_id], span_m=teacher_cfg.local_geometry_span_m)
                for arc in frame_arcs
            ]
            axes = np.asarray([sampler.interpolate(float(arc)) for arc in frame_arcs])
            origins = np.asarray(
                [
                    sensor_origin_from_axis(
                        axis,
                        fta_distance_m=fta_distance_m,
                        sensor_height_above_floor_m=mesh_cfg.sensor_height_above_floor_m,
                    )
                    for axis in axes
                ]
            )
            tangents = np.asarray([target.axis for target in spline_targets])
            measurements, complete = measure_mesh_geometry_batch(
                origins_xyz_m=origins,
                tangents_xyz=tangents,
                cast_distances=cast_distances,
                config=mesh_cfg,
            )
            widths = tuple(
                float(item.width_m) if item is not None else float("nan") for item in measurements
            )
            heights = tuple(
                float(item.height_m) if item is not None else float("nan") for item in measurements
            )
            profile = DirectionalGeometryProfile(
                traversal_id=traversal_id,
                edge_id=edge_id,
                direction_index=direction_index,
                edge_length_m=sampler.length_m,
                traversal_arc_m=tuple(float(value) for value in frame_arcs),
                width_m=widths,
                height_m=heights,
                valid_mask=tuple(bool(value) for value in complete),
            )
            profiles.append(profile)
            totals["unique_frame_count"] += len(frame_arcs)
            totals["complete_mesh_frame_count"] += int(np.sum(complete))
            totals["incomplete_mesh_frame_count"] += int(len(complete) - np.sum(complete))
            if direction_index == 0:
                edge_endpoints = (from_node_id, to_node_id)
                edge_length_m = sampler.length_m
        if len(profiles) != 2:
            totals["unqualified_short_edges"] += 1
            continue
        offline = [persistent_two_sided_change_episodes(profile, config=cfg) for profile in profiles]
        causal = [past_only_causal_change_episodes(profile, config=cfg) for profile in profiles]
        totals["persistent_directional_episode_count"] += len(offline[0]) + len(offline[1])
        totals["causal_directional_episode_count"] += len(causal[0]) + len(causal[1])
        for episode in offline[0] + offline[1]:
            directional_episodes.append({"parent_id": parent_id, **episode.to_dict()})
        for episode in causal[0] + causal[1]:
            causal_episodes_out.append({"parent_id": parent_id, **episode.to_dict()})
        pairing = pair_bidirectional_change_points(offline[0], offline[1])
        totals["bidirectional_candidate_count"] += len(pairing.accepted)
        totals["unilateral_episode_count"] += len(pairing.unilateral)
        totals["ambiguous_episode_count"] += len(pairing.ambiguous)
        assert edge_endpoints is not None and edge_length_m is not None
        for point_index, point in enumerate(pairing.accepted):
            assignment = assign_change_point_identity(
                parent_id=parent_id,
                point=point,
                edge_length_m=edge_length_m,
                from_node_id=edge_endpoints[0],
                to_node_id=edge_endpoints[1],
                node_degree_by_id=node_degree,
                point_index=point_index,
                config=cfg,
            )
            matches = [
                tuple(item for item in causal[direction] if causal_episode_matches_point(item, point))
                for direction in (0, 1)
            ]
            unique_causal = len(matches[0]) == 1 and len(matches[1]) == 1
            if unique_causal:
                totals["causal_bidirectional_candidate_count"] += 1
            elif not matches[0] or not matches[1]:
                totals["causal_missing_candidate_count"] += 1
            else:
                totals["causal_ambiguous_candidate_count"] += 1
            emitted = assignment.identity is not None and unique_causal
            if assignment.identity_kind == "suppressed_by_terminal_or_junction":
                totals["suppressed_by_terminal_or_junction_count"] += 1
            elif assignment.identity_kind == "ambiguous_degree_two_endpoints":
                totals["ambiguous_degree_two_endpoint_count"] += 1
            elif emitted:
                totals[f"emitted_{assignment.identity_kind}_count"] += 1
                identity_edges[str(assignment.identity)].add(edge_id)
                identity_edge_point_counts[(str(assignment.identity), edge_id)] += 1
                previous_kind = identity_kinds.setdefault(str(assignment.identity), assignment.identity_kind)
                if previous_kind != assignment.identity_kind:
                    raise RuntimeError("one change-point identity has inconsistent kinds")
                for direction, episode_group in enumerate(matches):
                    episode = episode_group[0]
                    anchors = anchor_by_direction[direction]
                    selected = np.flatnonzero(
                        (anchors >= episode.first_detection_arc_m - 1e-9)
                        & (anchors <= episode.emission_arc_m + 1e-9)
                    )
                    if len(selected) == 0:
                        raise RuntimeError("causal episode has no five-frame observation anchor")
                    for sequence_index in selected:
                        key = (profiles[direction].traversal_id, int(sequence_index))
                        old_identity = assigned_observation.setdefault(key, str(assignment.identity))
                        if old_identity != assignment.identity:
                            raise RuntimeError("one causal observation maps to multiple change identities")
                        label_rows.append(
                            {
                                "parent_id": parent_id,
                                "traversal_id": profiles[direction].traversal_id,
                                "edge_id": edge_id,
                                "sequence_index": int(sequence_index),
                                "traversal_arc_m": float(anchors[sequence_index]),
                                "canonical_edge_arc_m": profiles[direction].canonical_arc(
                                    float(anchors[sequence_index])
                                ),
                                "identity": assignment.identity,
                                "identity_kind": assignment.identity_kind,
                            }
                        )
            points.append(
                {
                    "parent_id": parent_id,
                    **point.to_dict(),
                    "identity_assignment": assignment.to_dict(),
                    "direction0_causal_match_count": len(matches[0]),
                    "direction1_causal_match_count": len(matches[1]),
                    "causal_qualified": unique_causal,
                    "emitted": emitted,
                    "causal_matches": [
                        [episode.to_dict() for episode in group] for group in matches
                    ],
                }
            )

    emitted_identities = sorted(identity_edges)
    totals["emitted_identity_count"] = len(emitted_identities)
    totals["emitted_observation_count"] = len(label_rows)
    totals["edge_count"] = len(graph["edges"])
    totals["directed_traversal_count"] = len(inventory)
    totals["sequence_count"] = sum(record.sequence_count for record in inventory)
    degree_two_identities = [identity for identity in emitted_identities if identity_kinds[identity] == "degree_two_endpoint"]
    interior_identities = [identity for identity in emitted_identities if identity_kinds[identity] == "interior_edge"]
    checks = {
        "exact_two_traversals_per_edge": len(inventory) == 2 * len(graph["edges"]),
        "all_emitted_points_have_unique_causal_support_both_directions": all(
            (not row["emitted"])
            or (row["direction0_causal_match_count"] == 1 and row["direction1_causal_match_count"] == 1)
            for row in points
        ),
        "no_emitted_identity_conflicts": len(assigned_observation) == len(label_rows),
        "no_degree_two_identity_uses_multiple_points_on_one_edge": all(
            count == 1 for count in identity_edge_point_counts.values()
        ),
        "all_causal_delays_are_nonnegative": all(
            episode["causal_delay_m"] >= -1e-9 for episode in causal_episodes_out
        ),
        "no_strict_test_or_mtare_read": True,
    }
    return {
        "schema_version": "gse_causal_change_point_world_proof_v1",
        "parent_id": parent_id,
        "split": split,
        "family": parent_id.split("_", 1)[0],
        "config": cfg.to_dict(),
        "totals": dict(totals),
        "checks": checks,
        "points": points,
        "directional_episodes": directional_episodes,
        "causal_episodes": causal_episodes_out,
        "labels": label_rows,
        "identity_summary": {
            "emitted_identities": emitted_identities,
            "degree_two_endpoint_identities": degree_two_identities,
            "interior_edge_identities": interior_identities,
            "identity_incident_edges": {
                identity: sorted(edges) for identity, edges in sorted(identity_edges.items())
            },
        },
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }


__all__ = ["world_causal_change_point_proof"]
