import pytest
from mtare_topo.integration.cmu_ground_target import prepare_ground_target


def call(**kwargs):
    args=dict(target_xyz_m=(1.,2.,8.),coordinate_frame='map',stamp_sec=1.,
        connected_ground_verified=False,verification_ref=None)
    args.update(kwargs)
    return prepare_ground_target(**args)


def test_unknown_layer_not_published():
    result=call()
    assert not result.accepted and result.waypoint is None
    assert result.structural_target_xyz_m==(1.,2.,8.)


def test_consumer_axes_not_misrepresented():
    result=call(connected_ground_verified=True,verification_ref='terrain:42/candidate:3')
    assert result.accepted and result.consumed_axes==('x','y')
    assert result.waypoint['xyz_m']==[1.,2.,8.]


def test_evidence_required():
    with pytest.raises(ValueError):call(connected_ground_verified=True)


def test_local_odometry_not_silently_renamed_map():
    with pytest.raises(ValueError):call(coordinate_frame='first_sensor')
