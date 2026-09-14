import math
import numpy as np
import pytest
from mtare_topo.integration.aee_lidar_pose import commanded_lidar_pose


@pytest.mark.parametrize('yaw,rate',[(0,0),(1.7,.8),(-math.pi+.001,-.7),(math.pi-.001,.7)])
def test_inverse_matches_original_simulator_update_order(yaw,rate):
    terrain_roll=.12;terrain_pitch=-.08
    previous=yaw-.005*rate
    roll=terrain_roll*math.cos(previous)+terrain_pitch*math.sin(previous)
    pitch=-terrain_roll*math.sin(previous)+terrain_pitch*math.cos(previous)
    m=commanded_lidar_pose([2,3,4],[roll,pitch,yaw],rate)
    # Independent elementary roll then pitch construction, no vehicle yaw.
    rx=np.array([[1,0,0],[0,math.cos(terrain_roll),-math.sin(terrain_roll)],
                 [0,math.sin(terrain_roll),math.cos(terrain_roll)]])
    ry=np.array([[math.cos(terrain_pitch),0,math.sin(terrain_pitch)],
                 [0,1,0],[-math.sin(terrain_pitch),0,math.cos(terrain_pitch)]])
    np.testing.assert_allclose(m[:3,:3],ry@rx,atol=1e-12)
    np.testing.assert_allclose(m[:3,3],[2,3,4]+(ry@rx)[:,2]*.0377)


def test_flat_sensor_does_not_rotate_with_vehicle_heading():
    m=commanded_lidar_pose([0,0,.75],[0,0,1.2],0)
    np.testing.assert_allclose(m[:3,:3],np.eye(3))
    assert m[2,3]==pytest.approx(.7877)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):commanded_lidar_pose([0,0,0],[0,0,0],float('nan'))
