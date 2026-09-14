"""Teacher-only source-interior alternative to crossing a virtual end cap.

Locality is supplied by simultaneous source containment, not remote return
counts. Ambiguous overlap between reference nodes stays unknown. This remains
reference-conditioned evidence, not a physical connectivity certificate.
"""
import numpy as np
from .source_witness_binding import named_sources


def interior_branch_evidence(axes, *, source_ids, origin_operand_distance,
                             directions, valid, return_sources):
    distances = np.asarray(origin_operand_distance, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    valid = np.asarray(valid)
    # Existing sparse field uses +inf for operands outside their bounds.
    if (distances.shape != (5,len(source_ids)) or np.isnan(distances).any() or np.isneginf(distances).any()
            or directions.shape != (57600,3) or not np.isfinite(directions).all()
            or valid.shape != (57600,) or valid.dtype != bool
            or len(return_sources) != 57600 or len(set(source_ids)) != len(source_ids)):
        raise ValueError('bound five-frame geometry and source arrays required')
    lookup = {s:i for i,s in enumerate(source_ids)}
    by_node = {}
    result = {}
    for axis in axes:
        key = axis['interface_id_teacher_only']
        direction = np.asarray(axis['inward_direction'],dtype=np.float64)
        if (key in result or axis['source_key_teacher_only'] not in lookup
                or direction.shape != (3,) or not np.isfinite(direction).all()
                or abs(np.linalg.norm(direction)-1.) > 1e-12):
            raise ValueError('unique bound interface and unit direction required')
        by_node.setdefault(axis['node_id_teacher_only'],[]).append(axis)
        result[key] = []
    ambiguous = []
    for frame in range(5):
        candidates = {}
        for node, members in by_node.items():
            # Strict interior excludes a numerical source boundary. Extra
            # hidden branches do not erase three available local directions.
            inside = [a for a in members if distances[frame,lookup[a['source_key_teacher_only']]] < 0.]
            if len({tuple(a['inward_direction']) for a in inside}) >= 3:
                candidates[node] = inside
        if len(candidates) != 1:
            if candidates: ambiguous.append(frame)
            continue
        members = next(iter(candidates.values()))
        start, end = frame*11520, (frame+1)*11520
        for axis in members:
            source = axis['source_key_teacher_only']
            outward = directions[start:end] @ np.asarray(axis['inward_direction']) > 0.
            indices = np.flatnonzero(outward & valid[start:end]) + start
            result[axis['interface_id_teacher_only']].extend(
                int(i) for i in indices if return_sources[i] == [source])
    return dict(interface_ray_indices=result, ambiguous_frame_slots=ambiguous,
        semantic_label=None, physical_reachable=None)


def bound_interior_evidence(bundle, raw_interfaces, *, field_spacing_m=0.01):
    from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
    from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
    from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
    from .gse_construction_paths_v3 import construction_incident_paths
    from .gse_directed_interface_binding_v1 import bind_axes
    from .swept_superellipse_field import SweptSuperellipseProvenanceField
    s, student, doc, book = (bundle[k] for k in ('sensor_teacher_only','student','construction_teacher_only','codebook_teacher_only'))
    verify_alignment(s,student,doc,book,bundle['source'])
    _, primitives = load_p1a_realized_construction(doc)
    # Keep the legacy default for sealed V2/V6 callers. New source-bound
    # producers must pass the original export's field discretization.
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=field_spacing_m)
    distances = field.operand_signed_distances_sparse(s['sensor_xyz_m'])
    local = lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions = np.concatenate([world_directions(local,float(y)) for y in s['yaw_deg']])
    ids = book['primitive_ids']
    owners = named_sources(bundle)
    axes = bind_axes(construction_incident_paths(doc),raw_interfaces['interfaces_teacher_only'])
    return interior_branch_evidence(axes,source_ids=ids,origin_operand_distance=distances,
        directions=directions,valid=student['valid_mask'].reshape(-1).astype(bool),return_sources=owners)
