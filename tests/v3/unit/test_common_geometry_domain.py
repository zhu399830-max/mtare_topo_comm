from copy import deepcopy
import numpy as np
from mtare_topo.integration.common_geometry_domain import common_domain_record


def record(axis):
    return dict(sensor_to_local_odometry=np.eye(4).tolist(),structures=[{'unused':True}],
                primitives=[dict(index=7,axis_controls_world_m=axis,source_rays=[])])


def test_bend_kept_and_original_not_mutated():
    r=record([[-20,0,0],[0,0,0],[0,20,0]]);old=deepcopy(r)
    x=common_domain_record(r)
    assert r==old and x['structures']==[] and not x['physical_openings_confirmed']
    np.testing.assert_allclose(x['primitives'][0]['axis_controls_world_m'],[[-10,0,0],[0,0,0],[0,10,0]])
    assert x['primitives'][0]['parent_primitive_index']==7


def test_disconnected_clips_do_not_gain_bridge():
    x=common_domain_record(record([[-2,0,0],[0,20,0],[2,0,0]]))
    assert len(x['primitives'])==2
    for p in x['primitives']:assert np.linalg.norm(p['axis_controls_world_m'],axis=1).max()<=10+1e-10


def test_wholly_outside_rejected_explicitly():
    x=common_domain_record(record([[11,0,0],[12,0,0],[13,0,0]]))
    assert x['primitives']==[] and x['domain_rejections']==[[7,'outside_common_domain']]
