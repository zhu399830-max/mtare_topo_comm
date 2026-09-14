import pytest
from mtare_topo.integration.recorded_return_execution import RecordedReturnExecution

def runtime():return RecordedReturnExecution(dict(kind='recorded_trace_return',new_edge_inferred=False,
    samples=[dict(order=3,xyz_m=[2,2,0]),dict(order=2,xyz_m=[2,0,0]),dict(order=1,xyz_m=[0,0,0])]),
    created_sec=4,arrival_radius_m=.5,target_timeout_s=30)

def test_returns_to_bend_before_final_place():
    r=runtime();a=r.update([2,2,0],4)
    assert a['xyz_m']==[2,0,0]
    assert r.update([2,0,0],5)['xyz_m']==[0,0,0]
    assert r.update([0,0,0],6)['status']=='RETURN_OBSERVATION_REACHED'

def test_stale_feedback_or_timeout_does_not_advance_route():
    r=runtime();r.update([2,2,0],4)
    with pytest.raises(ValueError):r.update([0,0,0],3)
    assert r.update([2,2,0],35)['status']=='RETURN_TIMEOUT_HOLD'

def test_return_can_revisit_previously_seen_positions_without_marking_new_edges():
    r=runtime()
    for i,p in enumerate(([2,2,0],[2,0,0],[0,0,0])):
        assert not r.update(p,i+4)['new_edge_inferred']
