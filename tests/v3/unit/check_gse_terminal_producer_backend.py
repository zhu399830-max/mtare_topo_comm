"""Executable in the locked Open3D sidecar; no dataset or output files.

Synthetic integration check, not a population experiment or human annotation.
"""
import json
from copy import deepcopy
import numpy as np
import open3d as o3d

from mtare_topo.teacher.primitive_construction_supervisor import (
    EndpointComposition, PrimitiveConstructionGraph, PrimitiveEndpoint, SweptPrimitive,
)
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_caster_cap_replay_v1 import pack_caster_inputs
from mtare_topo.teacher.gse_reference_terminal_targets_v1 import produce_terminal_reference_targets
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry


def bundle(axis=None):
    axis = np.array([[-3.,0.,0.], [3.,0.,0.]]) if axis is None else np.asarray(axis, dtype=float)
    endpoints = tuple(PrimitiveEndpoint('p', i, str(i), tuple(p), tuple(p))
                      for i,p in enumerate((axis[0], axis[-1])))
    primitive = SweptPrimitive('p', 'edge', 'tunnel', axis, 2., (1.,0.), endpoints)
    graph = PrimitiveConstructionGraph('cano_world', (primitive,), tuple(
        EndpointComposition(e.node_id, (e,), e.composition_anchor_xyz_m) for e in endpoints),
        endpoint_attachment_mode='free_space_overlap', node_degree_source='edge_incidence')
    doc = dict(schema_version='primitive_relation_realized_construction_v1',
        parent_id='synthetic', geometry_realization='ellipse', base_construction=graph.as_dict(),
        realized_primitives=[dict(primitive_id='p', centerline_xyz_m=axis.tolist(),
            endpoint_half_axes_m=[[2.,2.],[2.,2.]], endpoint_shape_exponent=[2.,2.])])
    mesh = mesh_swept_superellipse(SweptSuperellipsePrimitive('p', axis, ((2.,2.),(2.,2.)), (2.,2.)),
        axial_spacing_m=.05, angular_segments=64)
    xyz = np.array([[-.4,.1,0.],[-.3,.1,0.],[-.2,.1,0.],[-.1,.1,0.],[0.,.1,0.]], dtype=np.float64)
    yaw = np.array([-2.,-1.,0.,1.,2.], dtype=np.float64)
    local = lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions = np.concatenate([world_directions(local,float(y)) for y in yaw])
    vertices, rays = pack_caster_inputs(mesh.vertices_xyz_m, np.repeat(xyz,11520,axis=0), directions)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices),
        o3d.core.Tensor(mesh.triangle_vertex_indices.astype(np.uint32))))
    ranges = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy().reshape(5,16,720)
    assert ranges.dtype == np.float32 and np.isfinite(ranges).all()
    motion = causal_relative_odometry(xyz,yaw)
    return dict(source=dict(task='synthetic__ellipse', frame_rows=[0,1,2,3,4]),
        construction_teacher_only=doc,
        codebook_teacher_only=dict(parent_id='synthetic', geometry_realization='ellipse',
            primitive_ids=['p'], source_sets=[[],[0]]),
        sensor_teacher_only=dict(sensor_xyz_m=xyz, yaw_deg=yaw,
            primitive_membership_code=np.ones((5,16,720),dtype=np.uint16)),
        student=dict(ranges_m=ranges, valid_mask=np.ones((5,16,720),dtype=np.uint8),
            relative_translation_current_sensor_m=motion.translation_current_sensor_m.astype(np.float32),
            relative_yaw_current_sensor_deg=motion.yaw_current_sensor_deg.astype(np.float32)))


def main():
    b = bundle()
    result = produce_terminal_reference_targets(b)
    assert len(result['record']['anchors']) == 2
    assert all(p['witnesses'] for p in result['teacher_provenance'])
    assert not result['record']['score_region']['anchors_complete']
    assert not result['full_training_gate_eligible']
    missing = deepcopy(b)
    missing['student']['valid_mask'][:] = 0
    missing['sensor_teacher_only']['primitive_membership_code'][:] = 0
    empty = produce_terminal_reference_targets(missing)
    assert empty['record']['anchors'] == [] and len(empty['unknown_candidates']) == 2
    print(json.dumps(dict(status='SYNTHETIC_BACKEND_INTEGRATION_PASS',
        scenes=1, frames=5, rays=57600, partial_terminal_targets=2,
        masked_observation_unknown_terminals=2, real_dataset_reads=0,
        full_training_gate_eligible=False)))


if __name__ == '__main__':
    main()
