import numpy as np
import pytest
from mtare_topo.data.continuous_input_check_v1 import inspect_arrays


def fixture():
    return dict(range_m=np.ones((14,16,720),dtype=np.float32),
                valid_mask=np.ones((14,16,720),dtype=np.uint8),
                sensor_xyz_m=np.column_stack((np.arange(14.),np.zeros((14,2)))),
                yaw_deg=np.zeros(14),route_arc_m=np.arange(14.),
                local_frame_index=np.arange(14),traversal_index=np.zeros(14,dtype=int))


def test_finite_source_order_reports_not_safety():
    a=fixture();r=inspect_arrays(a)
    assert r['sensor_step_m']==[1.]*13 and r['valid_returns']==161280
    assert not r['physical_safety_verified'] and not r['continuous_global_route']


@pytest.mark.parametrize('field,value',[('traversal_index',1),('local_frame_index',50),
    ('route_arc_m',50.),('sensor_xyz_m',np.nan),('range_m',51.),('valid_mask',2)])
def test_corruption_rejected(field,value):
    a=fixture();a[field][-1]=value
    with pytest.raises(ValueError):inspect_arrays(a)
