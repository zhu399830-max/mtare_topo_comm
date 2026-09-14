import numpy as np
import pytest
from mtare_topo.integration.learned_geometry_trace import learned_geometry_record
from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay


def args():
    return dict(axis_controls_sensor_m=[[[-4,0,0],[0,0,0],[4,0,0]]],
        existence_probabilities=[.8],existence_threshold=.5,checkpoint_sha256='a'*64,
        source_frame_keys=['a','b','c','d','e'],sensor_to_world=np.eye(4),
        coordinate_frame='map',timestamp=0,current_sector_columns=[0])


def test_prediction_enters_same_backend_without_fitted_rays_or_edges():
    r=learned_geometry_record(**args());p=r['primitives'][0]
    assert p['source_rays']==[] and p['residual'] is None and r['structures']==[]
    backend=GeometryNavigationReplay(anchor_spacing_m=4,lookahead_m=4)
    d=backend.update(r)
    assert len(d['proposals'])==2 and backend.edges==[]
    for q in d['proposals']:
        assert q['geometry_source_kind']=='model_prediction'
        assert '/ray:' not in q['source_refs'][0]
        assert q['current_direction_supported']==(q['axis_target_xyz_m'][0]>0)
        assert q['physical_opening'] is False


def test_world_transform_and_unknown_are_preserved():
    a=args();a['current_sector_columns']=None
    a['sensor_to_world'][:3,3]=[3,4,5]
    r=learned_geometry_record(**a)
    assert r['primitives'][0]['axis_controls_world_m'][1]==[3,4,5]
    d=GeometryNavigationReplay(anchor_spacing_m=4,lookahead_m=4).update(r)
    assert all(q['current_direction_supported'] is None for q in d['proposals'])


def test_low_probability_and_degenerate_are_reported_not_repaired():
    a=args();a['existence_probabilities']=[.4]
    r=learned_geometry_record(**a);assert not r['primitives'] and len(r['rejected_axes'])==1
    a=args();a['axis_controls_sensor_m']=[[[0,0,0]]*3]
    r=learned_geometry_record(**a);assert r['rejected_axes']==[[0,'degenerate_predicted_axis']]


def test_fake_ray_evidence_is_rejected():
    r=learned_geometry_record(**args())
    r['primitives'][0]['source_rays']=[dict(frame_key='a',ray_index=0)]
    with pytest.raises(ValueError,match='masquerade'):
        GeometryNavigationReplay(anchor_spacing_m=4,lookahead_m=4).update(r)
