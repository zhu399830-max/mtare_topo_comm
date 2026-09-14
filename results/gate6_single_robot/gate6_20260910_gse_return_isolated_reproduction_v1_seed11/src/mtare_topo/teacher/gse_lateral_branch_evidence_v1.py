"""Source-bound observed overlap entries; no ID-only branch confirmation."""
import numpy as np
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from .gse_directed_interface_binding_v1 import interpret_bound_result, bind_axes
from .gse_construction_paths_v3 import construction_incident_paths
from .gse_caster_cap_replay_v1 import pack_caster_inputs
from .swept_superellipse_field import SweptSuperellipseProvenanceField
from .csg_mesh_provenance import mesh_swept_superellipse, CSGMeshProvenanceRaycaster
from .gse_observed_operand_entries_v1 import observed_operand_entries
from .source_witness_binding import named_sources


def bound_lateral_evidence(bundle, raw, *, axial_spacing_m, angular_segments, field_spacing_m):
    interpret_bound_result(bundle, raw)  # validate original scan, pose, source and raw XYZ
    sensor = bundle['sensor_teacher_only']; student = bundle['student']
    _, primitives = load_p1a_realized_construction(bundle['construction_teacher_only'])
    ids = [p.primitive_id for p in primitives]; lookup = {s: i for i, s in enumerate(ids)}
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=field_spacing_m)
    caster = CSGMeshProvenanceRaycaster([mesh_swept_superellipse(p,
        axial_spacing_m=axial_spacing_m, angular_segments=angular_segments) for p in primitives])
    local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
    directions = np.concatenate([world_directions(local, float(y)) for y in sensor['yaw_deg']])
    _, packed = pack_caster_inputs(np.empty((0, 3)), np.repeat(sensor['sensor_xyz_m'], 11520, axis=0), directions)
    book = bundle['codebook_teacher_only']
    owners = named_sources(bundle)
    entries = observed_operand_entries(caster, packed, first_return=student['ranges_m'].reshape(-1),
        valid=student['valid_mask'].reshape(-1).astype(bool), return_sources=owners,
        center_m=sensor['sensor_xyz_m'][-1])['entries']
    axes = bind_axes(construction_incident_paths(bundle['construction_teacher_only']), raw['interfaces_teacher_only'])
    incoming = {a['interface_id_teacher_only']: set() for a in axes}
    departure = {k: set() for k in incoming}
    origins = field.operand_signed_distances_sparse(sensor['sensor_xyz_m'])
    candidates = {}
    # Bound transient evaluation to one entry point, not all points x sources.
    for entry in entries:
        ray = entry['ray_index']; frame = ray // 11520
        inside = np.flatnonzero(origins[frame] < 0)
        if len(inside) != 1:
            continue
        source = ids[inside[0]]; target = entry['source_id_teacher_only']
        if source == target:
            continue
        at_entry = field.operand_signed_distances_sparse(np.asarray([entry['intersection_world_m']]))[0]
        if at_entry[lookup[source]] >= 0:
            continue
        left = [a for a in axes if a['source_key_teacher_only'] == source]
        right = [a for a in axes if a['source_key_teacher_only'] == target]
        pairs = [(a, b) for a in left for b in right if a['node_id_teacher_only'] == b['node_id_teacher_only']]
        if len(pairs) != 1:
            continue
        a, b = pairs[0]; d = packed[ray, 3:].astype(np.float64)
        guard = 64*np.finfo(np.float64).eps*np.linalg.norm(d)
        if d @ a['inward_direction'] >= -guard or d @ b['inward_direction'] <= guard:
            continue
        candidates.setdefault(ray, []).append((a, b))
    ambiguous = []
    for ray, pairs in candidates.items():
        if len({a['node_id_teacher_only'] for a, _ in pairs}) != 1:
            ambiguous.append(ray); continue
        for a, b in pairs:
            incoming[b['interface_id_teacher_only']].add(ray)
            departure[a['interface_id_teacher_only']].add(ray)
    return dict(entry_ray_indices={k: sorted(v) for k, v in incoming.items()},
        departure_ray_indices={k: sorted(v) for k, v in departure.items()},
        ambiguous_ray_indices=ambiguous, semantic_label=None)
