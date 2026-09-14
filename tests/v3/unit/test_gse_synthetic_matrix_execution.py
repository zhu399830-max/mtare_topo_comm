import numpy as np
from mtare_topo.data import gse_synthetic_matrix_execution as module
from mtare_topo.data.gse_synthetic_matrix import matrix


def test_scene_reuse_and_semantic_failures_preserved(tmp_path,monkeypatch):
    cases=matrix()[:2];builds=[];checks=[]
    monkeypatch.setattr(module,'matrix',lambda:cases)
    class Scene:
        def __init__(self,case):builds.append(case['case_id'])
        def render(self,case):
            return dict(source={},construction_teacher_only={},codebook_teacher_only={},student={'ranges_m':np.ones((5,1),np.float32)},
                sensor_teacher_only=dict(sensor_xyz_m=np.zeros((5,3)),yaw_deg=np.zeros(5),primitive_membership_code=np.ones((5,1),np.uint16)))
    monkeypatch.setattr(module,'SyntheticSensorScene',Scene)
    monkeypatch.setattr(module,'evaluate',lambda b,c:({},dict(record={}),dict(case_id=c['case_id'],independent_oracle_covered=True,
        status='FIXTURE_GEOMETRY_PASS' if c==cases[0] else 'FIXTURE_GEOMETRY_FAIL')))
    (tmp_path/'logs').mkdir();(tmp_path/'artifacts').mkdir()
    result=module.execute_matrix(tmp_path,progress=lambda x:None,resource_check=lambda:checks.append(1))
    assert len(builds)==1 and len(checks)==2
    assert result['geometry_pass']==result['geometry_fail']==1
    assert result['primary_observations']==2 and not result['full_matrix_qualification']
    assert len(list((tmp_path/'artifacts').glob('*.npz')))==2
