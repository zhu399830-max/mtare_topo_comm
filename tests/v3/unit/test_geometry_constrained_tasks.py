import numpy as np
import pytest
from mtare_topo.integration.geometry_constrained_tasks import constrained_tasks

def record(axis):
    return dict(timestamp=4.,source_frame_keys=['0','1','2','3','4'],sensor_to_local_odometry=np.eye(4).tolist(),
        primitives=[dict(index=2,axis_controls_world_m=axis,prediction_provenance=dict(
            input_frame_keys=['0','1','2','3','4'],observation_order=4.,slot=2,checkpoint_sha256='a'*64))])

def tasks(r):
    return constrained_tasks(r,[dict(heading_robot_deg=h,source_refs=['4/ray:0']) for h in [0,90]],
        lookahead_m=4,direction_tolerance_deg=15)

def test_model_changes_tasks_not_just_visualization():
    x,_=tasks(record([[-8,0,0],[0,0,0],[8,0,0]]))
    y,_=tasks(record([[0,-8,0],[0,0,0],[0,8,0]]))
    assert len(x)==len(y)==1
    assert x[0]['axis_target_xyz_m']==[4.,0.,0.]
    assert np.allclose(y[0]['axis_target_xyz_m'],[0,4,0])
    r=record([[-8,0,0],[0,0,0],[8,0,0]]);r['primitives']=[]
    assert tasks(r)[0]==[]  # no nonlearning fallback

def test_fold_is_not_executed_or_counted_twice_and_raw_unchanged():
    r=record([[8,0,0],[-8,0,0],[8,0,0]]);before=repr(r)
    p,_=tasks(r)
    assert len(p)==1 and p[0]['axis_target_xyz_m']==[4.,0.,0.]
    assert len(p[0]['learned_geometry_support'])==2
    assert repr(r)==before and p[0]['physical_opening'] is False
    assert p[0]['target_coordinate_source']=='current_observed_sector_and_fixed_lookahead'

def test_future_prediction_and_old_observation_rejected():
    r=record([[-8,0,0],[0,0,0],[8,0,0]])
    with pytest.raises(ValueError,match='current scan'):
        constrained_tasks(r,[dict(heading_robot_deg=0,source_refs=['3/ray:0'])],lookahead_m=4,direction_tolerance_deg=15)
    r['primitives'][0]['prediction_provenance']['observation_order']=5
    with pytest.raises(ValueError,match='bound'):tasks(r)

def test_shared_graph_consumes_constrained_tasks_not_folded_targets():
    from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay
    r=record([[8,0,0],[-8,0,0],[8,0,0]])
    r.update(schema_version='geometry_structure_trace_v1',coordinate_frame='map',structures=[],
        frontend='frozen_learned_geometry_constrained')
    r['primitives'][0].update(fit_status='model_prediction_not_surface_fit',source_rays=[])
    p,_=tasks(r)
    graph=GeometryNavigationReplay(anchor_spacing_m=4,lookahead_m=4)
    d=graph.update(r,task_proposals=p)
    assert d['selected']['axis_target_xyz_m']==[4.,0.,0.]
    assert len(d['proposals'])==1
    assert 'raw_axis_proposals_not_executable' in d
    assert graph.edges==[]  # neither learned nor observed candidates invent edges
