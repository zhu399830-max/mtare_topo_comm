from copy import deepcopy
import numpy as np
from mtare_topo.teacher.gse_reference_opening_targets_v1 import _produce_opening_reference_targets


def test_nonowning_contour_preserved_without_duplicate_target():
    row=dict(reference_position_m=[10.,0.,0.],reference_direction=[1.,0.,0.],
        exclusive_outward_crossing_ray_indices=[1],surface_return_ray_indices=[2],
        primitive_id_teacher_only='p',reference_arc_m=10.,axis_reference_owned=True)
    other=dict(row,axis_reference_owned=False)
    bundle=dict(source=dict(frame_rows=[0,1,2,3,4]),sensor_teacher_only=dict(sensor_xyz_m=np.zeros((5,3)),yaw_deg=np.zeros(5)))
    result=_produce_opening_reference_targets(bundle,lambda b:dict(proposals=[other,row]),require_axis_ownership=True)
    assert len(result['record']['openings'])==1
    assert result['unknown_candidates'][0]['source_proposal']==other
    assert result['teacher_provenance'][0]['proposal_index']==1
    assert not result['full_training_gate_eligible']


def test_no_unique_owner_cannot_supply_positive():
    bundle=dict(source=dict(frame_rows=[0,1,2,3,4]),sensor_teacher_only=dict(sensor_xyz_m=np.zeros((5,3)),yaw_deg=np.zeros(5)))
    row=dict(axis_reference_owned=False)
    result=_produce_opening_reference_targets(bundle,lambda b:dict(proposals=[row]),require_axis_ownership=True)
    assert not result['record']['openings']
    assert result['unknown_candidates'][0]['source_proposal']==row
