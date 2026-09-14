import pytest
from mtare_topo.integration.local_path_feedback import inspect_local_path


def check(points,**kwargs):
    args=dict(frame_id='vehicle',stamp_sec=11.,target_sent_sec=10.,now_sec=11.1,maximum_age_sec=.5)
    args.update(kwargs)
    return inspect_local_path(points,**args)


def test_original_failure_origin_is_not_success():
    assert check([[0,0,0]]).status=='NO_NONTRIVIAL_LOCAL_PATH'


def test_path_is_neither_goal_ack_nor_execution():
    value=check([[0,0,0],[1,0,0],[2,0,1]])
    assert value.status=='LOCAL_PATH_AVAILABLE_NOT_EXECUTED'
    assert value.path_length_m==pytest.approx(1+2**.5)
    assert not value.target_acknowledged and not value.arrival_verified and not value.edge_verified


def test_old_path_cannot_satisfy_new_target():
    assert check([[0,0,0],[1,0,0]],stamp_sec=9.).status=='STALE_OR_PREVIOUS_TARGET'


def test_clock_reset_rejected():
    assert check([[0,0,0],[1,0,0]],now_sec=8.).status=='CLOCK_INCONSISTENT'


def test_map_frame_not_vehicle_frame():
    with pytest.raises(ValueError):check([[0,0,0]],frame_id='map')
