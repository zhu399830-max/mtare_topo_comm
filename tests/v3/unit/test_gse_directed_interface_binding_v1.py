import copy
import pytest
from mtare_topo.teacher.gse_directed_interface_binding_v1 import bind_axes


def fixture():
    groups=[dict(node_id_teacher_only='n',paths=[dict(endpoint_key=('p',0),axis_start_index=1,
                    points_world_m=[[0.,0.,0.],[0.,2.,0.],[3.,2.,0.]])])]
    refs=[dict(interface_id_teacher_only=0,node_id_teacher_only='n',endpoint_key_teacher_only=['p',0])]
    return groups,refs


def test_connector_is_not_branch_direction():
    groups,refs=fixture()
    assert bind_axes(groups,refs)[0]['inward_direction']==[1.,0.,0.]


def test_opposite_incidence_same_source_kept():
    groups,refs=fixture()
    p=copy.deepcopy(groups[0]['paths'][0]);p['endpoint_key']=('p',1);p['points_world_m'][-1]=[-3.,2.,0.]
    groups[0]['paths'].append(p)
    refs.append(dict(interface_id_teacher_only=1,node_id_teacher_only='n',endpoint_key_teacher_only=['p',1]))
    result=bind_axes(groups,refs)
    assert [r['inward_direction'] for r in result]==[[1.,0.,0.],[-1.,0.,0.]]


@pytest.mark.parametrize('damage',['missing_axis','duplicate','foreign_node','degenerate'])
def test_invalid_binding_rejected(damage):
    groups,refs=fixture()
    if damage=='missing_axis':groups[0]['paths'][0]['points_world_m'].pop()
    elif damage=='duplicate':refs.append(copy.deepcopy(refs[0]))
    elif damage=='foreign_node':refs[0]['node_id_teacher_only']='other'
    else:groups[0]['paths'][0]['points_world_m'][-1]=[0.,2.,0.]
    with pytest.raises(ValueError):bind_axes(groups,refs)
