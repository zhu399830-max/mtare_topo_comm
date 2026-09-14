import copy
import numpy as np
from mtare_topo.teacher import gse_reference_anchor_targets_v1 as module
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets


def case(monkeypatch,extra_hidden=False,visible=3,**options):
    directions=[[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]]
    count=4 if extra_hidden else 3
    group=dict(node_id_teacher_only='n',anchor_world_m=[1.,0.,0.],paths=[
        dict(endpoint_key=(str(i),0),axis_start_index=0,points_world_m=[[1.,0.,0.],(np.array([1.,0.,0.])+d).tolist()]) for i,d in enumerate(directions[:count])])
    refs=[dict(interface_id_teacher_only=i,node_id_teacher_only='n',endpoint_key_teacher_only=[str(i),0]) for i in range(count)]
    monkeypatch.setattr(module,'construction_incident_paths',lambda doc:[copy.deepcopy(group)])
    monkeypatch.setattr(module,'interpret_bound_result',lambda b,r:dict(interfaces={i:dict(entering_ray_indices=[i] if i<visible else []) for i in range(count)}))
    b=dict(construction_teacher_only={},source=dict(frame_rows=[1,2,3,4,5]),
        sensor_teacher_only=dict(sensor_xyz_m=np.zeros((5,3)),yaw_deg=np.zeros(5)))
    return module._produce_junction_reference_targets(b,dict(interfaces_teacher_only=refs),**options)


def test_nonempty_loss_targets_without_teacher_identity(monkeypatch):
    output=case(monkeypatch)
    target=observed_targets([output['record']])
    assert target.anchor_valid.tolist()==[[True]]
    assert target.anchor_position_m.tolist()==[[[1.,0.,0.]]]
    assert not target.anchor_region_complete.any() and not target.opening_region_complete.any()
    assert output['record']['openings']==[] and not output['full_training_gate_eligible']
    assert 'node_id_teacher_only' not in str(output['record'])


def test_extra_hidden_branch_does_not_erase_visible_event(monkeypatch):
    a=case(monkeypatch);b=case(monkeypatch,extra_hidden=True)
    assert a['record']==b['record']


def test_two_visible_directions_stay_unknown(monkeypatch):
    output=case(monkeypatch,visible=2)
    assert output['record']['anchors']==[] and len(output['unknown_candidates'])==1
    assert not output['record']['score_region']['anchors_complete']


def test_explicit_precision_removes_only_cap_support(monkeypatch):
    from mtare_topo.evaluation import gse_bound_cap_precision as audit
    monkeypatch.setattr(audit,'audit_bound_cap_precision',lambda *a,**k:dict(
        interfaces={i:dict(stable_entering_ray_indices=[0] if i==0 else []) for i in range(3)}))
    legacy=case(monkeypatch)
    assert len(legacy['record']['anchors'])==1
    options=dict(cap_source_settings=dict(axial_spacing_m=.05,angular_segments=64))
    guarded=case(monkeypatch,**options)
    assert guarded['record']['anchors']==[]
    assert len(guarded['unknown_candidates'])==1
    interior=dict(interface_ray_indices={0:[],1:[10],2:[11]})
    rescued=case(monkeypatch,interior=interior,**options)
    assert len(rescued['record']['anchors'])==1
    assert rescued['teacher_provenance'][0]['entering_witness_ray_indices']==[[0],[],[]]
    lateral=dict(entry_ray_indices={0:[],1:[10],2:[11]},departure_ray_indices={0:[],1:[],2:[]})
    assert len(case(monkeypatch,additional=lateral,**options)['record']['anchors'])==1
    assert 'cap_precision_audit' not in legacy


def test_precision_cannot_invent_witness(monkeypatch):
    import pytest
    from mtare_topo.evaluation import gse_bound_cap_precision as audit
    monkeypatch.setattr(audit,'audit_bound_cap_precision',lambda *a,**k:dict(
        interfaces={i:dict(stable_entering_ray_indices=[999]) for i in range(3)}))
    with pytest.raises(ValueError,match='invent'):
        case(monkeypatch,cap_source_settings=dict(axial_spacing_m=.05,angular_segments=64))
