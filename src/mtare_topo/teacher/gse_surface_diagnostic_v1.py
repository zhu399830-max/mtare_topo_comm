"""All-source ROI/mesh/ray diagnostic; never exports semantic training labels.

Uses exact original mesh builder (.05m/64) and finite first returns. Primitive
and composition IDs are teacher provenance only. No renderer/model is called.
Any unresolved mesh section aborts instead of hiding a candidate. Source
polylines, meshes, ROI boundary planes and ray witnesses are distinct objects.
"""
from collections import Counter

import numpy as np

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_mesh_sections_v1 import section_ray_witnesses
from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments
from mtare_topo.teacher.gse_roi_crossings_v1 import roi_crossings


def diagnose_observation(bundle, *, range_error_bound_m):
    """One fixed fiveframe observation; sources already validated by reader.

    The caller binds range error explicitly; zero denotes no extra measurement
    allowance beyond the existing first-return storage ULP, not physical truth.
    All candidates compete globally for ray indices. Positive section evidence
    alone does not qualify an opening/anchor, visibility completeness or width.
    """
    sensor = bundle["sensor_teacher_only"]; student = bundle["student"]
    construction, primitives = load_p1a_realized_construction(bundle["construction_teacher_only"])
    origins = np.repeat(sensor["sensor_xyz_m"],16*720,axis=0)
    local = lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions = np.concatenate([world_directions(local,float(yaw)) for yaw in sensor["yaw_deg"]])
    directions /= np.linalg.norm(directions,axis=1,keepdims=True)
    frames = np.repeat(np.asarray(bundle["source"]["frame_rows"],dtype=np.int64),16*720)
    rays = CausalRaySegments(origins,directions,student["ranges_m"].reshape(-1),
        student["valid_mask"].reshape(-1).astype(bool),frames,int(frames[-1]),range_error_bound_m)
    rays.validate()
    center = sensor["sensor_xyz_m"][-1]
    rows = []; meshes = 0
    for primitive in primitives:
        intersections = roi_crossings(primitive.centerline_xyz_m,center_m=center)
        if intersections.ambiguous_segment_indices:
            raise ValueError("ROI crossing ambiguity in source:"+primitive.primitive_id)
        if not len(intersections.positions_m):
            continue
        mesh = mesh_swept_superellipse(primitive,axial_spacing_m=.05,angular_segments=64)
        meshes += 1
        for segment,fraction,position,normal,arc in zip(intersections.segment_indices,intersections.segment_fractions,
                intersections.positions_m,intersections.outward_tangents,intersections.source_arc_m):
            section,witnesses = section_ray_witnesses(mesh.vertices_xyz_m,mesh.triangle_vertex_indices,
                center_m=position,normal=normal,rays=rays,exclusive=False)
            for loop,edges,hits in zip(section.loops_m,section.source_edges,witnesses):
                rows.append(dict(primitive_id_teacher_only=primitive.primitive_id,
                    segment_index=int(segment),segment_fraction=float(fraction),source_arc_m=float(arc),
                    boundary_axis_position_world_m=position.tolist(),outward_direction_world=normal.tolist(),
                    mesh_loop_world_m=loop.tolist(),mesh_edge_indices=[list(e) for e in edges],
                    section_local_witness_indices=list(hits)))
        del mesh
    counts = Counter(ray for row in rows for ray in row["section_local_witness_indices"])
    for row in rows:
        hits = [ray for ray in row["section_local_witness_indices"] if counts[ray]==1]
        row.update(exclusive_witness_indices=hits,section_status="SECTION_POSITIVE" if hits else "UNKNOWN",
            semantic_opening_label=None,anchor_label=None,physical_root_reachable=None)
    return dict(source=bundle["source"],candidate_sections=rows,source_primitives=len(primitives),
        constructed_meshes=meshes,construction_anchor_inventory=len(construction.compositions),
        section_positive_count=sum(r["section_status"]=="SECTION_POSITIVE" for r in rows),
        labels_generated=0,teacher_complete=False,scientific_gate_pass=False,
        limitation="Source section evidence is not semantic opening/anchor or physical reachability; no complete-background labels.")
