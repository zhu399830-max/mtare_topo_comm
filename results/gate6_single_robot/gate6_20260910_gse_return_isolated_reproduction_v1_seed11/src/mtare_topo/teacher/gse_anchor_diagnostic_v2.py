"""Source-anchor incidence versus the same causal observed grid, diagnostic only."""
from dataclasses import asdict

import numpy as np

from mtare_topo.data.gse_surface_material_v1 import build_observed_material
from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from mtare_topo.representation.gse_surface_ray_evidence_v1 import ObservedRayGrid
from mtare_topo.teacher.gse_construction_paths_v2 import construction_incident_paths
from mtare_topo.teacher.gse_junction_support_v1 import junction_support
from mtare_topo.teacher.gse_roi_crossings_v1 import roi_crossings


def local_path(points, origin, yaw):
    p=_current_sensor_transform(points,origin,yaw)
    if np.linalg.norm(p[0])>=10:
        raise ValueError("reference anchor outside open10m observation sphere")
    intersections=roi_crossings(p,center_m=np.zeros(3))
    if intersections.ambiguous_segment_indices:
        return None
    if len(intersections.positions_m):
        i=int(intersections.segment_indices[0])
        p=np.vstack([p[:i+1],intersections.positions_m[0]])
    return p


def diagnose_observation(bundle, *, range_error_bound_m):
    # Same signature as the frozen section runner; this value is NOT applied
    # to invent more FREE grid cells. Existing grid ULP bound remains authority.
    if range_error_bound_m!=0.:
        raise ValueError("grid diagnostic does not accept extra range-error tuning")
    student=bundle["student"];sensor=bundle["sensor_teacher_only"]
    arrays,metrics=build_observed_material(student["ranges_m"],student["valid_mask"],
        student["relative_translation_current_sensor_m"],student["relative_yaw_current_sensor_deg"])
    # Material exports every ndarray field, but scalar provenance belongs in
    # metrics. Reconstructing from source rays below avoids unbound grid flags.
    from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
    grid=build_surface_ray_grid(arrays["ray_origins_current_sensor_m"],arrays["points_current_sensor_m"],
        arrays["first_return_valid"],arrays["history_slot"])
    rows=[];groups=construction_incident_paths(bundle["construction_teacher_only"])
    origin=sensor["sensor_xyz_m"][-1];yaw=float(sensor["yaw_deg"][-1])
    for group in groups:
        anchor=_current_sensor_transform(np.asarray(group["anchor_world_m"]),origin,yaw)
        if np.linalg.norm(anchor)>=10:
            continue
        paths=tuple(local_path(p["points_world_m"],origin,yaw) for p in group["paths"])
        row=dict(node_id_teacher_only=group["node_id_teacher_only"],anchor_current_sensor_m=anchor.tolist(),
            source_degree=len(paths),anchor_axis_offsets_m=[p["anchor_axis_offset_m"] for p in group["paths"]])
        if any(p is None for p in paths):
            row.update(status="UNKNOWN",reason="ROI_CLIP_GEOMETRY_AMBIGUITY",semantic_label=None)
        else:
            support=junction_support(grid,anchor_m=anchor,incident_paths_m=paths,
                endpoint_keys=tuple(p["endpoint_key"] for p in group["paths"]))
            row.update(asdict(support));row["reference_paths_current_sensor_m"]=[p.tolist() for p in paths]
        rows.append(row)
    return dict(source=bundle["source"],anchors=rows,local_anchor_count=len(rows),
        reference_junctions=sum(r["source_degree"]>=3 for r in rows),
        supported_reference_junctions=sum(r["status"]=="REFERENCE_JUNCTION_SUPPORTED" for r in rows),
        observed_grid_sha256=grid.content_sha256,labels_generated=0,teacher_complete=False,scientific_gate_pass=False,
        limitation="Reference-conditioned axis-cell evidence only;not aperture boundary completeness,semantic labels,or physical safety.")
