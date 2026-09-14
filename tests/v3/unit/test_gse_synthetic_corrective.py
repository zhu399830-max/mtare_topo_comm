from copy import deepcopy
import json
import numpy as np
import pytest
from mtare_topo.data.gse_synthetic_corrective import derive_corrective,restore_declared_case
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry
from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit


def fixture():
    case=next(x for x in matrix() if x['case_id']=='hidden_branch__ellipse__view3')
    doc=construction_document(case);xyz=np.asarray(case['poses_world_m']);yaw=np.asarray(case['yaw_deg'])
    motion=causal_relative_odometry(xyz,yaw)
    student=dict(ranges_m=np.full((5,16,720),2.,dtype=np.float32),valid_mask=np.ones((5,16,720),dtype=np.uint8),
        relative_translation_current_sensor_m=motion.translation_current_sensor_m.astype(np.float32),
        relative_yaw_current_sensor_deg=motion.yaw_current_sensor_deg.astype(np.float32))
    student['valid_mask'][0,14,1]=0;student['ranges_m'][0,14,1]=50.
    codes=np.ones((5,16,720),dtype=np.uint16);codes[0,14,1]=0
    bundle=dict(source=dict(task=doc['parent_id']+'__'+doc['geometry_realization'],case_id=case['case_id']),
        student=student,sensor_teacher_only=dict(sensor_xyz_m=xyz,yaw_deg=yaw,primitive_membership_code=codes),
        construction_teacher_only=doc,codebook_teacher_only=dict(parent_id=doc['parent_id'],geometry_realization=doc['geometry_realization'],
            primitive_ids=[p['primitive_id'] for p in doc['realized_primitives']],source_sets=[[],[0]]))
    return case,bundle


def test_only_invalid_mutates_and_original_remains():
    case,b=fixture();before=deepcopy(b);calls=[]
    def repair(i):calls.append(i);return PrimitiveRayHit(6.,(0.,0.,0.),('approach','continuation'),False)
    out=derive_corrective(case,b,repair=repair)
    assert calls==[(0,14,1)]
    for k,v in b['student'].items():np.testing.assert_array_equal(v,before['student'][k])
    assert out['student']['valid_mask'].all()
    assert out['codebook_teacher_only']['source_sets']==[[],[0],[0,1]]
    assert out['derivation_provenance']['unchanged_ray_positions']==57599
    with pytest.raises(ValueError):derive_corrective(case,out,repair=repair)


def test_sorted_json_restores_original_constructor_order_without_changing_content():
    case,b=fixture();loaded=json.loads(json.dumps(case,sort_keys=True))
    assert list(loaded['program']['anchors'])!=list(case['program']['anchors'])
    restored=restore_declared_case(loaded)
    assert list(restored['program']['anchors'])==list(case['program']['anchors'])
    out=derive_corrective(loaded,b,repair=lambda i:PrimitiveRayHit(6.,(0.,0.,0.),('approach',),True))
    assert out['student']['valid_mask'].all()
    loaded['program']['anchors']['J'][0]+=1
    with pytest.raises(ValueError):restore_declared_case(loaded)


@pytest.mark.parametrize('problem',['count','pose','source','missing'])
def test_drift_and_unqualified_returns_reject(problem):
    case,b=fixture()
    if problem=='count':b['student']['valid_mask'][0,0,0]=0;b['sensor_teacher_only']['primitive_membership_code'][0,0,0]=0
    if problem=='pose':b['sensor_teacher_only']['sensor_xyz_m'][0,0]+=1
    hit=None if problem=='missing' else PrimitiveRayHit(6.,(0.,0.,0.),('wrong' if problem=='source' else 'approach',),True)
    with pytest.raises(ValueError):derive_corrective(case,b,repair=lambda i:hit)
