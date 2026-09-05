#!/usr/bin/env python3
"""Read-only attribution of P1 sensor origins outside realized primitive unions."""

from __future__ import annotations

import json
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_sensor_export import world_unique_frame_poses
from mtare_topo.governance import load_json
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField
from run_primitive_slot_capacity_audit_v1 import MESH_ROOT, _load_traversals, _select_windows


def main():
    worlds = load_json(PROJECT_ROOT / "configs/v3/gate3/data_cards/geometry_variant_inventory_v1.json")["worlds"]["train"]
    traversals = _load_traversals(); failures = []; checked = 0
    for world in worlds:
        primary = MESH_ROOT / world / "primary"
        graph = load_json(primary / "graph.json"); splines = load_json(primary / "splines.json"); geometry = load_json(primary / "geometry_parameters.json")
        construction = build_primitive_construction_graph(graph, splines, geometry, endpoint_attachment_mode="free_space_overlap", node_degree_source="edge_incidence")
        poses = world_unique_frame_poses(parent_id=world, traversal_manifest=traversals[world], graph=graph, spline_document=splines, geometry_parameters=geometry)
        windows = _select_windows(poses, graph); indices = sorted(set(int(index) for values in windows.values() for index in values))
        origins = poses.sensor_xyz_m[indices]
        for realization in GeometryRealization:
            field = SweptSuperellipseProvenanceField(realize_construction(world, construction, realization), spacing_m=.025)
            distances = field.operand_signed_distances(origins); minimum = np.min(distances, axis=1); checked += len(indices)
            for local, residual in enumerate(minimum):
                if residual > 0:
                    index = indices[local]; failures.append({
                        "world": world, "realization": realization.value, "pose_index": index,
                        "traversal_id": poses.traversal_ids[index], "local_frame_index": int(poses.local_frame_indices[index]),
                        "sensor_xyz_m": poses.sensor_xyz_m[index].tolist(), "axis_xyz_m": poses.axis_xyz_m[index].tolist(),
                        "minimum_operand_signed_distance_m": float(residual),
                        "nearest_primitive_index": int(np.argmin(distances[local])),
                        "roles": sorted(role for role, values in windows.items() if index in values),
                    })
        print(json.dumps({"world": world, "failures_so_far": len(failures)}), flush=True)
    result = {
        "status": "PASS_ALL_ORIGINS_INSIDE" if not failures else "ATTRIBUTED_ORIGINS_OUTSIDE_REALIZED_UNION",
        "worlds": len(worlds), "checked_task_origins": checked, "outside_count": len(failures),
        "outside_worlds": sorted(set(row["world"] for row in failures)),
        "outside_by_realization": {value.value: sum(row["realization"] == value.value for row in failures) for value in GeometryRealization},
        "maximum_positive_residual_m": max((row["minimum_operand_signed_distance_m"] for row in failures), default=0.),
        "failures": failures,
    }
    print("FINAL " + json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
