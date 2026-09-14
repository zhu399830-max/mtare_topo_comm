"""Diagnose the unchanged failing T fixture using analytic cylinder geometry.

The -Y branch is a radius-two finite cylinder, y in [-15,1]. Solve its
lateral entry analytically against the *existing* synthetic scan rays.
This is not a general teacher and never generates structural labels.
"""
import json
import numpy as np
from check_gse_joint_producer_backend import main
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.teacher.gse_caster_cap_replay_v1 import pack_caster_inputs
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_directed_interface_binding_v1 import interpret_bound_result


def run():
    bundle = main(return_bundle=True, terminal_branch=True, terminal_length_m=8.,
        sensor_positions=[[-x, 0., .04] for x in (2.5, 3., 3.5, 4., 4.5)])
    raw = diagnose_observation(bundle)
    directed = interpret_bound_result(bundle, raw)
    sensor = bundle['sensor_teacher_only']; student = bundle['student']
    document = bundle['construction_teacher_only']
    branch = next(p for p in document['realized_primitives'] if p['primitive_id'] == '2')
    assert branch['centerline_xyz_m'] == [[0., 1., 0.], [0., -15., 0.]]
    assert branch['endpoint_half_axes_m'] == [[2., 2.], [2., 2.]]
    assert branch['endpoint_shape_exponent'] == [2., 2.]
    local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
    directions = np.concatenate([world_directions(local, float(y)) for y in sensor['yaw_deg']])
    _, packed = pack_caster_inputs(np.empty((0, 3)),
        np.repeat(sensor['sensor_xyz_m'], 11520, axis=0), directions)
    origin, direction = packed[:, :3].astype(np.float64), packed[:, 3:].astype(np.float64)
    book = bundle['codebook_teacher_only']
    owners = [[book['primitive_ids'][i] for i in book['source_sets'][int(code)]]
              for code in sensor['primitive_membership_code'].reshape(-1)]
    selected = np.array([owner == ['2'] for owner in owners]) & student['valid_mask'].reshape(-1).astype(bool)
    ox, oz = origin[:, 0], origin[:, 2]
    dx, dz = direction[:, 0], direction[:, 2]
    a = dx*dx + dz*dz
    b = 2*(ox*dx + oz*dz)
    c = ox*ox + oz*oz - 4.
    disc = b*b - 4*a*c
    possible = selected & (a > 0) & (disc > 0) & (c > 0)
    rays = np.flatnonzero(possible)
    entry = (-b[rays] - np.sqrt(disc[rays])) / (2*a[rays])
    xyz = origin[rays] + entry[:, None] * direction[rays]
    ranges = student['ranges_m'].reshape(-1)
    limit = ranges[rays].astype(np.float64) - np.abs(np.spacing(ranges[rays]).astype(np.float64))
    finite = (entry > 0) & (entry < limit) & (xyz[:, 1] > -15) & (xyz[:, 1] < 1)
    finite &= np.linalg.norm(xyz - sensor['sensor_xyz_m'][-1], axis=1) < 10.
    rays, entry, xyz = rays[finite], entry[finite], xyz[finite]
    # Existing terminal branch is the radius-two X cylinder, x in [-8,1].
    # Entry into the -Y source can occur while still in terminal-source air.
    overlap = ((xyz[:, 0] > -8) & (xyz[:, 0] < 1)
               & (xyz[:, 1]**2 + xyz[:, 2]**2 < 4.))
    interface = next(i['interface_id_teacher_only'] for i in raw['interfaces_teacher_only']
        if i['node_id_teacher_only'] == 'junction0' and i['endpoint_key_teacher_only'] == ['2', 0])
    entering = directed['interfaces'][interface]['entering_ray_indices']
    assert len(rays) > 0 and np.count_nonzero(overlap) > 0
    assert not entering
    from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
    from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse, CSGMeshProvenanceRaycaster
    from mtare_topo.teacher.gse_observed_operand_entries_v1 import observed_operand_entries
    _, primitives = load_p1a_realized_construction(document)
    caster = CSGMeshProvenanceRaycaster([mesh_swept_superellipse(p,
        axial_spacing_m=.05, angular_segments=64) for p in primitives])
    kwargs = dict(first_return=ranges, valid=student['valid_mask'].reshape(-1).astype(bool),
        return_sources=owners, center_m=sensor['sensor_xyz_m'][-1])
    mesh_result = observed_operand_entries(caster, packed, **kwargs)
    branch_entries = [e for e in mesh_result['entries'] if e['source_id_teacher_only'] == '2']
    mesh_rays = {e['ray_index'] for e in branch_entries}
    assert mesh_rays and mesh_rays <= set(np.flatnonzero(selected))
    assert mesh_rays & set(rays.tolist())
    assert all(-15 < e['intersection_world_m'][1] < 1 for e in branch_entries)
    assert mesh_result['membership'] is None
    blocked = observed_operand_entries(caster, packed,
        **dict(kwargs, first_return=np.zeros_like(ranges)))
    assert not blocked['entries'], 'surfaces behind a first return must not be witnesses'
    ambiguous = observed_operand_entries(caster, packed,
        **dict(kwargs, return_sources=[s + ['ambiguous_other'] for s in owners]))
    assert not ambiguous['entries'], 'multi-source returns cannot become unique branch witnesses'
    print(json.dumps(dict(status='SYNTHETIC_LATERAL_ENTRY_EXPLAINS_CAP_PROXY_MISS',
        frames=5, original_ray_slots=57600,
        exclusive_valid_branch_return_rays=int(np.count_nonzero(selected)),
        analytic_lateral_entries_before_first_return_inside_ROI=len(rays),
        lateral_entries_inside_terminal_operand=int(np.count_nonzero(overlap)),
        old_virtual_cap_entering_witnesses=len(entering),
        complete_mesh_entry_rays=len(mesh_rays),
        mesh_and_analytic_common_rays=len(mesh_rays & set(rays.tolist())),
        occluded_entries=len(blocked['entries']), ambiguous_entries=len(ambiguous['entries']),
        supported_frame_slots=sorted(set((rays[overlap] // 11520).tolist())),
        entry_y_range_m=[float(xyz[:, 1].min()), float(xyz[:, 1].max())],
        new_labels=0, real_dataset_reads=0, optimizer_steps=0)), flush=True)


if __name__ == '__main__':
    run()
