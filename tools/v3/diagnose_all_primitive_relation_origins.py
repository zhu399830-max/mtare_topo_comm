#!/usr/bin/env python3
"""Read-only full P1 origin-versus-finite-union attribution."""

from __future__ import annotations

import json
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_sensor_export import world_finite_union_qualified_frame_poses
from mtare_topo.governance import load_json
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField
from run_primitive_slot_capacity_audit_v1 import MESH_ROOT, _load_traversals


def main():
    worlds = load_json(PROJECT_ROOT / "configs/v3/gate3/data_cards/geometry_variant_inventory_v1.json")["worlds"]["train"]
    traversals = _load_traversals(); rows = []; examples = []; totals = {value.value: 0 for value in GeometryRealization}; residuals_by_realization = {value.value: [] for value in GeometryRealization}; total_poses = 0
    correction_count = 0; maximum_shift_m = 0.0; minimum_adjacent_arc_spacing_m = float("inf")
    for world in worlds:
        primary = MESH_ROOT / world / "primary"; graph = load_json(primary / "graph.json"); splines = load_json(primary / "splines.json"); geometry = load_json(primary / "geometry_parameters.json")
        construction = build_primitive_construction_graph(graph, splines, geometry, endpoint_attachment_mode="free_space_overlap", node_degree_source="edge_incidence")
        fields = {
            realization: SweptSuperellipseProvenanceField(
                realize_construction(world, construction, realization), spacing_m=.025
            )
            for realization in GeometryRealization
        }
        poses, corrections = world_finite_union_qualified_frame_poses(
            parent_id=world, traversal_manifest=traversals[world], graph=graph,
            spline_document=splines, geometry_parameters=geometry,
            signed_distance_fields=[fields[value].signed_distance for value in GeometryRealization],
        )
        total_poses += len(poses.global_frame_indices)
        correction_count += len(corrections)
        if corrections:
            maximum_shift_m = max(maximum_shift_m, max(item.inward_shift_m for item in corrections))
        for traversal_id in sorted(set(poses.traversal_ids)):
            indices = np.asarray([index for index, value in enumerate(poses.traversal_ids) if value == traversal_id])
            if len(indices) > 1:
                minimum_adjacent_arc_spacing_m = min(minimum_adjacent_arc_spacing_m, float(np.min(np.diff(poses.arc_m[indices]))))
        for realization in GeometryRealization:
            field = fields[realization]
            minimum_parts = []; nearest_parts = []
            for start in range(0, len(poses.sensor_xyz_m), 2048):
                values = field.operand_signed_distances_sparse(poses.sensor_xyz_m[start:start+2048])
                minimum_parts.append(np.min(values, axis=1)); nearest_parts.append(np.argmin(values, axis=1))
            minimum = np.concatenate(minimum_parts); nearest = np.concatenate(nearest_parts); outside = np.flatnonzero(minimum > 0)
            endpoint_nearest = 0
            for index in outside:
                operand = field.operands[int(nearest[index])]; _, sample = operand.tree.query(poses.sensor_xyz_m[index], k=1, workers=1)
                endpoint_nearest += int(sample in (0, len(operand.points)-1))
                if minimum[index] > 1e-9 and len(examples) < 100:
                    examples.append({"world": world, "realization": realization.value, "pose_index": int(index), "traversal_id": poses.traversal_ids[index], "local_frame_index": int(poses.local_frame_indices[index]), "residual_m": float(minimum[index]), "nearest_primitive": field.primitive_ids[int(nearest[index])], "nearest_sample_is_endpoint": bool(sample in (0, len(operand.points)-1))})
            totals[realization.value] += len(outside)
            residuals_by_realization[realization.value].extend(float(minimum[index]) for index in outside)
            rows.append({"world": world, "realization": realization.value, "frames": len(minimum), "outside": len(outside), "endpoint_nearest": endpoint_nearest, "maximum_residual_m": float(np.max(minimum[outside])) if len(outside) else 0.0})
        print(json.dumps({"world": world, "outside_so_far": sum(totals.values())}), flush=True)
    thresholds = (0.0, 1e-9, .001, .01, .05, .1, .25)
    meaningful = sum(residual > 1e-9 for values in residuals_by_realization.values() for residual in values)
    result = {"status": "PASS_ALL_P1_ORIGINS_INSIDE_1E9_TOLERANCE" if meaningful == 0 else "ATTRIBUTED_FULL_P1_ORIGIN_CONTRACT_FAILURE", "parent_worlds": len(worlds), "unique_source_frames": total_poses, "planned_variant_frames": 3*total_poses, "corrected_source_frames": correction_count, "maximum_inward_shift_m": maximum_shift_m, "minimum_adjacent_arc_spacing_m": minimum_adjacent_arc_spacing_m, "outside_by_realization": totals, "outside_total": sum(totals.values()), "outside_by_threshold_m": {str(value): sum(residual > value for values in residuals_by_realization.values() for residual in values) for value in thresholds}, "outside_by_realization_and_threshold_m": {name: {str(value): sum(residual > value for residual in values) for value in thresholds} for name, values in residuals_by_realization.items()}, "outside_worlds": sorted(set(row["world"] for row in rows if row["maximum_residual_m"] > 1e-9)), "endpoint_nearest_total": sum(row["endpoint_nearest"] for row in rows), "maximum_residual_m": max(row["maximum_residual_m"] for row in rows), "meaningful_rows": [row for row in rows if row["maximum_residual_m"] > 1e-9], "examples": examples}
    print("FINAL " + json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
