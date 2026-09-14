"""Actual synthetic CSG -> five scans -> joint target integration; no data IO."""
import json
import numpy as np
from mtare_topo.teacher.primitive_construction_supervisor import (
    EndpointComposition, PrimitiveConstructionGraph, PrimitiveEndpoint, SweptPrimitive)
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive, SweptSuperellipseProvenanceField
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse, CSGMeshProvenanceRaycaster
from mtare_topo.data.primitive_relation_dataset import PrimitiveMembershipCodebook
from mtare_topo.data.primitive_relation_sensor_export import render_primitive_sensor_frame
from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v1 import produce_joint_reference_targets
from mtare_topo.teacher.gse_directed_interface_binding_v1 import interpret_bound_result
from mtare_topo.teacher.gse_interior_branch_evidence_v1 import bound_interior_evidence


def main(*, revised=False, scene='single', return_bundle=False, terminal_branch=False,
         terminal_length_m=3., sensor_positions=None, rigid_yaw_deg=0., translation_xyz_m=(0.,0.,0.)):
    axes = [np.array([[-1.,0.,0.],[15.,0.,0.]]),
            np.array([[1.,0.,0.],[-15.,0.,0.]]),
            np.array([[0.,1.,0.],[0.,-15.,0.]])]
    if terminal_branch:
        axes[1][-1] = [-terminal_length_m, 0., 0.]
    offsets=[np.zeros(3)]
    if scene == 'layered': offsets.append(np.array([0.,0.,6.]))
    if scene == 'overlapping': offsets.append(np.array([.5,0.,0.]))
    axes=[a+offset for offset in offsets for a in axes]
    angle=np.deg2rad(rigid_yaw_deg);c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    translation=np.asarray(translation_xyz_m,dtype=np.float64)
    if translation.shape!=(3,) or not np.isfinite(translation).all():raise ValueError('finite translation required')
    axes=[a@rotation.T+translation for a in axes]
    offsets=[a@rotation.T+translation for a in offsets]
    primitives, base, far, rows = [], [], [], []
    near=[[] for _ in offsets]
    for i, axis in enumerate(axes):
        key = str(i)
        node='junction'+str(i//3); center=tuple(offsets[i//3])
        a = PrimitiveEndpoint(key,0,node,tuple(axis[0]),center)
        b = PrimitiveEndpoint(key,1,'end'+key,tuple(axis[1]),tuple(axis[1]))
        near[i//3].append(a); far.append(EndpointComposition(b.node_id,(b,),b.composition_anchor_xyz_m))
        base.append(SweptPrimitive(key,'edge'+key,'tunnel'+key,axis,2.,(1.,0.),(a,b)))
        primitives.append(SweptSuperellipsePrimitive(key,axis,((2.,2.),(2.,2.)),(2.,2.)))
        rows.append(dict(primitive_id=key,centerline_xyz_m=axis.tolist(),
            endpoint_half_axes_m=[[2.,2.],[2.,2.]],endpoint_shape_exponent=[2.,2.]))
    graph = PrimitiveConstructionGraph('cano_world',tuple(base),
        tuple(EndpointComposition('junction'+str(i),tuple(group),tuple(offsets[i])) for i,group in enumerate(near))+tuple(far),
        endpoint_attachment_mode='free_space_overlap',node_degree_source='edge_incidence')
    doc = dict(schema_version='primitive_relation_realized_construction_v1',parent_id='synthetic',
        geometry_realization='ellipse',base_construction=graph.as_dict(),realized_primitives=rows)
    field = SweptSuperellipseProvenanceField(primitives)
    meshes = [mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives]
    caster = CSGMeshProvenanceRaycaster(meshes)
    book = PrimitiveMembershipCodebook([str(i) for i in range(len(axes))])
    xyz = np.array([[0.,y,.04] for y in [-3.,-1.,0.,.5,1.5]],dtype=np.float64)
    if sensor_positions is not None:
        xyz = np.asarray(sensor_positions, dtype=np.float64)
        assert xyz.shape == (5,3)
    xyz=xyz@rotation.T+translation
    yaw = np.full(5,rigid_yaw_deg,dtype=np.float64)
    scans = [render_primitive_sensor_frame(raycaster=caster,field=field,codebook=book,
        sensor_xyz_m=p,yaw_deg=rigid_yaw_deg) for p in xyz]
    motion = causal_relative_odometry(xyz,yaw)
    bundle = dict(source=dict(task='synthetic__ellipse',source_sequence_id=0,frame_rows=[0,1,2,3,4]),
        construction_teacher_only=doc,
        codebook_teacher_only=dict(parent_id='synthetic',geometry_realization='ellipse',
            primitive_ids=[str(i) for i in range(len(axes))],source_sets=[list(s) for s in book.source_sets]),
        sensor_teacher_only=dict(sensor_xyz_m=xyz,yaw_deg=yaw,
            primitive_membership_code=np.stack([s.primitive_membership_code for s in scans])),
        student=dict(ranges_m=np.stack([s.range_m for s in scans]),
            valid_mask=np.stack([s.valid_mask for s in scans]),
            relative_translation_current_sensor_m=motion.translation_current_sensor_m.astype(np.float32),
            relative_yaw_current_sensor_deg=motion.yaw_current_sensor_deg.astype(np.float32)))
    if return_bundle:
        return bundle
    raw = diagnose_observation(bundle)
    directed = interpret_bound_result(bundle,raw)
    interior = bound_interior_evidence(bundle,raw)
    producer = produce_joint_reference_targets
    if revised:
        from mtare_topo.teacher.gse_joint_reference_targets_v2 import produce_joint_reference_targets as producer
    output = producer(bundle,raw)
    if revised:
        from mtare_topo.data.gse_structure_review_v1 import canonical_sha
        assert output['target_record_sha256']==canonical_sha(output['record'])
        assert output['source_binding']['source']['frame_rows']==output['record']['source_frame_indices']
    record = output['record']
    print(json.dumps(dict(status='SYNTHETIC_JOINT_BACKEND_RESULT',scene=scene,
        anchors=len(record['anchors']),openings=len(record['openings']),
        positive_memberships=sum(v is True for r in record['membership'] for v in r),
        unknown_memberships=sum(v is None for r in record['membership'] for v in r),
        entering_rays_per_interface={str(k):len(v['entering_ray_indices']) for k,v in directed['interfaces'].items()},
        interior_rays_per_interface={str(k):len(v) for k,v in interior['interface_ray_indices'].items()},
        interior_ambiguous_frames=interior['ambiguous_frame_slots'],
        unknown_anchor_reasons=output['unknown_candidates']['anchors'],
        relation_reasons=[dict(opening_index=r['opening_index'],status=r['status'],
            blocked_rays=len(r.get('blocked_ray_indices',[]))) for r in output['teacher_provenance']['relations']],
        full_training_gate_eligible=output['full_training_gate_eligible'])))
    if scene == 'overlapping':
        assert len(record['anchors']) == 0, 'overlapping references must not be resolved by identity'
        assert interior['ambiguous_frame_slots'], 'actual overlapping geometry must trigger competition'
    else:
        assert len(record['anchors']) == 1, 'synthetic T should supply one partial junction reference'
        assert len(record['openings']) == 3, 'synthetic T should supply three window opening references'
        assert output['teacher_provenance']['anchors'][0]['node_id_teacher_only'] == 'junction0'
        if revised:
            assert record['membership'] == [[True],[True],[True]], 'three supported partial reference correspondences required'
    assert not output['full_training_gate_eligible']


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--revised',action='store_true')
    parser.add_argument('--scene',choices=['single','layered','overlapping'],default='single')
    args=parser.parse_args()
    main(revised=args.revised,scene=args.scene)
