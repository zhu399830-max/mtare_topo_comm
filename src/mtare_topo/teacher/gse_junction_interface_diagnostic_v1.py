"""Bind ordered source-interface evidence to a verified five-frame bundle."""
import numpy as np

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from .csg_mesh_provenance import mesh_swept_superellipse
from .gse_cap_return_evidence_v1 import source_endpoint_cap_faces
from .gse_construction_paths_v3 import construction_incident_paths
from .gse_interface_caster_v1 import SourceInterfaces, replay_interfaces


def diagnose_observation(bundle):
    sensor, student = bundle['sensor_teacher_only'], bundle['student']
    document, book = bundle['construction_teacher_only'], bundle['codebook_teacher_only']
    source = bundle['source']
    verify_alignment(sensor, student, document, book, source)
    frames = np.asarray(source['frame_rows'])
    if (frames.shape != (5,) or frames.dtype.kind not in 'iu'
            or np.any(frames < 0) or not np.all(np.diff(frames) > 0)):
        raise ValueError('five increasing original frame rows required')
    ranges = np.asarray(student['ranges_m'])
    if ranges.shape != (5, 16, 720) or ranges.dtype != np.float32:
        raise ValueError('original five-frame range layout required')
    groups = construction_incident_paths(document)
    origin = np.asarray(sensor['sensor_xyz_m'][-1])
    nearby = [g for g in groups if len(g['paths']) >= 3
              and np.linalg.norm(np.asarray(g['anchor_world_m'])-origin) < 10.]
    _, primitives = load_p1a_realized_construction(document)
    lookup = {p.primitive_id: p for p in primitives}
    if len(lookup) != len(primitives):
        raise ValueError('unique source primitive identities required')
    interfaces = []
    by_source = {}
    for group in sorted(nearby, key=lambda g: g['node_id_teacher_only']):
        for path in sorted(group['paths'], key=lambda p: p['endpoint_key']):
            identity, side = path['endpoint_key']
            index = len(interfaces)
            interfaces.append(dict(interface_id_teacher_only=index,
                endpoint_key_teacher_only=[identity, side],
                node_id_teacher_only=group['node_id_teacher_only'],
                anchor_world_m=list(group['anchor_world_m'])))
            by_source.setdefault(identity, []).append((side, index))
    sources = []
    for identity, endpoints in sorted(by_source.items()):
        mesh = mesh_swept_superellipse(lookup[identity], axial_spacing_m=.05, angular_segments=64)
        face_map = {}
        for side, index in endpoints:
            for face in source_endpoint_cap_faces(mesh, angular_segments=64, endpoint_index=side):
                if int(face) in face_map:
                    raise ValueError('source endpoint assigned to multiple references')
                face_map[int(face)] = index
        sources.append(SourceInterfaces(identity, mesh, face_map))
    result = dict(rays=[], raw_interface_intersections=[], semantic_label=None,
                  training_eligible=False)
    if sources:
        local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
        directions = np.concatenate([world_directions(local, float(y)) for y in sensor['yaw_deg']])
        result = replay_interfaces(sources,
            origins=np.repeat(sensor['sensor_xyz_m'], 11520, axis=0), directions=directions,
            first_return=ranges.reshape(-1), valid=student['valid_mask'].reshape(-1).astype(bool),
            source_frame_indices=np.repeat(np.arange(5), 11520), roi_center=origin)
        for record in result['raw_interface_intersections']:
            slot = record['source_frame_index']
            record['source_frame_row'] = int(frames[slot])
            code = int(sensor['primitive_membership_code'].reshape(-1)[record['ray_index']])
            # Preserve return provenance, including multi-source membership;
            # crossing an interface is not a return from that source.
            record['return_source_indices_teacher_only'] = book['source_sets'][code]
    return dict(source=source, interfaces_teacher_only=interfaces,
        nearby_junction_references=len(nearby), labels_generated=0,
        teacher_complete=False, scientific_gate_pass=False, **result)
