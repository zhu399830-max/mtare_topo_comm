import json
from copy import deepcopy
import numpy as np
import pytest
from mtare_topo.teacher.primitive_construction_supervisor import PrimitiveEndpoint,SweptPrimitive,PrimitiveConstructionGraph,EndpointComposition
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.data.development_branch_targets import archived_junction_targets


def fixture():
    primitives=[];ends=[];compositions=[];realized=[]
    for i,p in enumerate(([2.,0.,0.],[-2.,0.,0.],[0.,2.,0.])):
        name=str(i);pair=(PrimitiveEndpoint(name,0,'n',(0.,0.,0.),(0.,0.,0.)),PrimitiveEndpoint(name,1,'end'+name,tuple(p),tuple(p)))
        primitives.append(SweptPrimitive(name,'edge'+name,'tunnel'+name,np.array([[0.,0.,0.],p]),2.,(1.,0.),pair))
        ends.append(pair[0]);compositions.append(EndpointComposition(pair[1].node_id,(pair[1],),pair[1].composition_anchor_xyz_m))
        realized.append(dict(primitive_id=name,centerline_xyz_m=[[0.,0.,0.],p],endpoint_half_axes_m=[[2,2],[2,2]],endpoint_shape_exponent=[2,2]))
    compositions.append(EndpointComposition('n',tuple(ends),(0.,0.,0.)))
    graph=PrimitiveConstructionGraph('cano_world',tuple(primitives),tuple(compositions),endpoint_attachment_mode='free_space_overlap',node_degree_source='edge_incidence')
    doc=json.loads(json.dumps(dict(schema_version='primitive_relation_realized_construction_v1',base_construction=graph.as_dict(),realized_primitives=realized)))
    source=dict(task='synthetic',source_sequence_id=1,frame_rows=[0,1,2,3,4]);binding=dict(source=source,construction_sha256=canonical_sha(doc))
    record=dict(coordinate_frame='current_sensor_m',source_frame_indices=source['frame_rows'],anchors=[dict(position_m=[0.,0.,0.])])
    provenance=dict(terminal_anchor_start=1,terminals=[],anchors=[dict(node_id_teacher_only='n',interface_ids=[0,1,2],witness_ray_indices=[[1],[2],[3]])])
    produced=dict(record=record,teacher_provenance=provenance,source_binding=binding,target_record_sha256=canonical_sha(record))
    raw=dict(source=source,interfaces_teacher_only=[dict(interface_id_teacher_only=i,node_id_teacher_only='n',endpoint_key_teacher_only=[str(i),0]) for i in range(3)])
    return produced,raw,doc,binding


def read(p,r,d,b):return archived_junction_targets(p,r,d,current_yaw_deg=90,expected_binding=b,expected_record_sha256=p['target_record_sha256'])


def test_reference_complete_opposite_directions_not_collapsed():
    p,r,d,b=fixture();out=read(p,r,d,b)
    assert out['target'].branches_complete==(True,) and not out['target'].anchors_complete
    assert out['observed_branch_count']==3 and not out['full_detection_eligible']
    np.testing.assert_allclose(out['target'].directions[0][0].numpy(),[0,-1,0],atol=1e-7)
    np.testing.assert_allclose(out['target'].directions[0][1].numpy(),[0,1,0],atol=1e-7)


def test_unwitnessed_reference_remains_unknown():
    p,r,d,b=fixture();p['teacher_provenance']['anchors'][0]['witness_ray_indices'][2]=[]
    before=deepcopy(p);out=read(p,r,d,b)
    assert out['target'].branches_complete==(False,) and out['observed_branch_count']==2
    assert out['missing_interface_ids_teacher_only']==((2,),) and p==before


def test_incomplete_reference_inventory_rejected():
    p,r,d,b=fixture();r['interfaces_teacher_only'].pop();p['teacher_provenance']['anchors'][0]['interface_ids'].pop();p['teacher_provenance']['anchors'][0]['witness_ray_indices'].pop()
    with pytest.raises(ValueError,match='inventory'):read(p,r,d,b)


def test_target_content_drift_rejected():
    p,r,d,b=fixture();p['record']['anchors'][0]['position_m'][0]=1
    with pytest.raises(ValueError,match='binding'):read(p,r,d,b)
