"""Source-bound numerical cap audit; does not produce or modify labels."""
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.gse_directed_interface_binding_v1 import interpret_bound_result, bind_axes
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_caster_cap_replay_v1 import pack_caster_inputs
from .gse_cap_branch_precision import cap_branch_precision, summarize_cap_directions
from .gse_ray_triangle_audit import ray_triangle_audit


def audit_bound_cap_precision(bundle, raw, *, axial_spacing_m, angular_segments, direction_kind='entering'):
    if direction_kind not in ('entering','leaving'):
        raise ValueError('entering or leaving direction required')
    if not np.isfinite(axial_spacing_m) or axial_spacing_m <= 0:
        raise ValueError('explicit positive source axial spacing required')
    if type(angular_segments) is not int or angular_segments < 3:
        raise ValueError('explicit source angular resolution required')
    directed = interpret_bound_result(bundle, raw)
    doc = bundle['construction_teacher_only']
    axes = {a['interface_id_teacher_only']:a for a in bind_axes(
        construction_incident_paths(doc), raw['interfaces_teacher_only'])}
    _, primitives = load_p1a_realized_construction(doc)
    byid = {p.primitive_id:p for p in primitives}
    sensor = bundle['sensor_teacher_only']
    local = lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions = np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
    normalized = directions / np.linalg.norm(directions,axis=1)[:,None]
    origins = np.repeat(sensor['sensor_xyz_m'],len(local),axis=0)
    ranges = bundle['student']['ranges_m'].reshape(-1)
    reports = {}; nodes = {}
    for interface in raw['interfaces_teacher_only']:
        key = interface['interface_id_teacher_only']
        sid, side = interface['endpoint_key_teacher_only']
        wanted = set(directed['interfaces'][key][direction_kind+'_ray_indices'])
        # Only archived entering candidates, never new intersections or labels.
        hits = [h for h in raw['raw_interface_intersections'] if h['interface_id_teacher_only']==key
                and h['ray_index'] in wanted and h['inside_roi'] and 0<=h['t']<float(ranges[h['ray_index']])]
        if {h['ray_index'] for h in hits} != wanted:
            raise ValueError('missing bound entrance hit')
        ri = np.asarray([h['ray_index'] for h in hits],dtype=np.int64)
        if hits:
            mesh = mesh_swept_superellipse(byid[sid],axial_spacing_m=axial_spacing_m,angular_segments=angular_segments)
            fi = np.asarray([h['triangle_index'] for h in hits],dtype=np.int64)
            cap_start = len(mesh.triangle_vertex_indices)-2*angular_segments
            if np.any(fi<cap_start) or np.any(fi>=len(mesh.triangle_vertex_indices)) or np.any((fi-cap_start)%2!=side):
                raise ValueError('archived hit not on declared source endpoint')
            vertices,packed = pack_caster_inputs(mesh.vertices_xyz_m,origins,directions)
            triangle_check = ray_triangle_audit(vertices,mesh.triangle_vertex_indices,packed,ri,fi,[h['t'] for h in hits])
            tri = mesh.vertices_xyz_m[mesh.triangle_vertex_indices[fi]]
        else:
            tri = np.empty((0,3,3),dtype=np.float64); triangle_check = None
        precision = cap_branch_precision(tri,origins[ri],normalized[ri],ri,axes[key]['inward_direction'],direction_kind=direction_kind)
        # The triangle report exposes numerical residuals; do not turn this
        # precision-only report into a whole observation or safety certificate.
        reports[key] = dict(**precision,triangle_check=triangle_check)
        nodes.setdefault(interface['node_id_teacher_only'],[]).append(precision)
    return dict(source=dict(bundle['source']),interfaces=reports,
        nodes={n:summarize_cap_directions(rows) for n,rows in nodes.items()} if direction_kind=='entering' else {},
        geometry_settings=dict(axial_spacing_m=axial_spacing_m,angular_segments=angular_segments),
        label_or_physical_connectivity_certified=False)
