import copy
import numpy as np
import pytest
from mtare_topo.teacher.saved_branch_evidence import read_saved_branches
from mtare_topo.teacher.saved_branch_evidence import read_saved_junction_branches


def fixture():
    record=dict(coordinate_frame='current_sensor_m',anchors=[dict(position_m=[1,2,3])])
    provenance=dict(anchors=[dict(node_id_teacher_only='n',interface_ids=[0,1],
        witness_ray_indices=[[0],[]],interior_witness_ray_indices=[[0],[]])])
    axes=[dict(interface_id_teacher_only=i,node_id_teacher_only='n',source_key_teacher_only='same',
        inward_direction=[sign,0.,0.]) for i,sign in enumerate((1.,-1.))]
    return record,provenance,axes


def test_opposite_same_source_not_collapsed_unknown_not_filled():
    r,p,a=fixture();before=copy.deepcopy((r,p,a))
    out=read_saved_branches(r,p,a,current_yaw_deg=90,ray_count=2)
    assert len(out)==2 and out[0].supported and not out[1].supported
    np.testing.assert_allclose(out[0].reference_direction_current_sensor,[0,-1,0],atol=1e-15)
    np.testing.assert_allclose(out[1].reference_direction_current_sensor,[0,1,0],atol=1e-15)
    assert (r,p,a)==before
    assert all(x.aperture_position_m is None and x.width_m is None and not x.training_eligible for x in out)


def test_mixed_archive_reads_only_explicit_junction_prefix_without_mutation():
    r,p,a=fixture()
    expected=read_saved_branches(r,p,a,current_yaw_deg=0,ray_count=2)
    r['anchors'].append(dict(position_m=[9,0,0]))
    p.update(terminal_anchor_start=1,terminals=[dict(node_id_teacher_only='terminal')])
    before=copy.deepcopy((r,p,a))
    assert read_saved_junction_branches(r,p,a,current_yaw_deg=0,ray_count=2)==expected
    assert (r,p,a)==before


def test_terminal_only_is_not_an_empty_branch_target():
    r,p,a=fixture();p.update(anchors=[],terminal_anchor_start=0,terminals=[{}])
    assert read_saved_junction_branches(r,p,[],current_yaw_deg=0,ray_count=2)==()


@pytest.mark.parametrize('start',[None,True,-1,0,2])
def test_mixed_archive_rejects_missing_or_wrong_partition(start):
    r,p,a=fixture();p.update(terminal_anchor_start=start,terminals=[])
    with pytest.raises(ValueError,match='partition'):
        read_saved_junction_branches(r,p,a,current_yaw_deg=0,ray_count=2)


@pytest.mark.parametrize('damage',['foreign','duplicate','ray','position','kind'])
def test_invalid_binding(damage):
    r,p,a=fixture()
    if damage=='foreign':a[0]['node_id_teacher_only']='other'
    if damage=='duplicate':a.append(copy.deepcopy(a[0]))
    if damage=='ray':p['anchors'][0]['witness_ray_indices'][0]=[2]
    if damage=='position':r['anchors'][0]['position_m']=[float('nan'),0,0]
    if damage=='kind':p['anchors'][0]['interior_witness_ray_indices'][1]=[1]
    with pytest.raises(ValueError):read_saved_branches(r,p,a,current_yaw_deg=0,ray_count=2)
