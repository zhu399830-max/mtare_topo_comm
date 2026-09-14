import numpy as np
import pytest
from mtare_topo.data.gse_surface_review_export_v1 import observed_material_bundle


def arrays():
    p=np.zeros((57600,3),dtype=np.float32);v=np.zeros(57600,dtype=bool)
    v[[0,11520,57599]]=True;p[57599]=[20,0,0]
    return dict(points_current_sensor_m=p,first_return_valid=v,history_slot=np.repeat(np.arange(5),11520))


def test_preserve_valid_returns_and_history_without_teacher():
    a=arrays();a['teacher_node_id']='never copy'
    b=observed_material_bundle(a,observation_id='opaque',source_frame_indices=[1,2,3,4,5])
    assert len(b['points_xyz_m'])==3 and b['point_history_slots']==[0,1,4]
    assert b['points_xyz_m'][-1]==[20,0,0]
    assert 'teacher_node_id' not in b


def test_wrong_history_rejected():
    a=arrays();a['history_slot'][0]=4
    with pytest.raises(ValueError):observed_material_bundle(a,observation_id='opaque',source_frame_indices=[1,2,3,4,5])


def test_invalid_returns_are_not_imputed_points():
    a=arrays();a['points_current_sensor_m'][1]=np.nan
    assert len(observed_material_bundle(a,observation_id='opaque',source_frame_indices=[1,2,3,4,5])['points_xyz_m'])==3
