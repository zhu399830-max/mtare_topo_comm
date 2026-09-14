import numpy as np
from mtare_topo.teacher.gse_interior_branch_evidence_v1 import interior_branch_evidence


def case(second_node=False, stacked=False, hidden=False):
    axes = [dict(interface_id_teacher_only=i,node_id_teacher_only='a',
        source_key_teacher_only=str(i),inward_direction=d) for i,d in enumerate(
        [[1.,0.,0.],[-1.,0.,0.],[0.,-1.,0.]])]
    ids = ['0','1','2']
    distances = np.full((5,3),-1.)
    if second_node:
        axes += [dict(a,interface_id_teacher_only=i+3,node_id_teacher_only='b',
            source_key_teacher_only=str(i+3)) for i,a in enumerate(axes[:3])]
        ids += ['3','4','5']
        distances = np.concatenate([distances,np.full((5,3),1. if stacked else -1.)],axis=1)
    if hidden:
        axes.append(dict(interface_id_teacher_only=9,node_id_teacher_only='a',source_key_teacher_only='h',inward_direction=[0.,1.,0.]))
        ids.append('h'); distances=np.column_stack([distances,np.ones(5)])
    dirs = np.tile([1.,0.,0.],(57600,1)); owners=[[] for _ in range(57600)]
    for frame in range(5):
        for i,d in enumerate([[1.,0.,0.],[-1.,0.,0.],[0.,-1.,0.]]):
            ray=frame*11520+i;dirs[ray]=d;owners[ray]=[str(i)]
    return interior_branch_evidence(axes,source_ids=ids,origin_operand_distance=distances,
        directions=dirs,valid=np.ones(57600,dtype=bool),return_sources=owners)


def test_inside_three_branches_has_directional_source_evidence():
    assert [len(v) for v in case()['interface_ray_indices'].values()] == [5,5,5]


def test_nearby_reference_overlap_is_unknown():
    result=case(second_node=True)
    assert not any(result['interface_ray_indices'].values())
    assert result['ambiguous_frame_slots'] == list(range(5))


def test_separate_height_layer_does_not_compete():
    result=case(second_node=True,stacked=True)
    assert [len(v) for v in result['interface_ray_indices'].values()] == [5,5,5,0,0,0]


def test_hidden_extra_direction_does_not_erase_observed_three():
    result=case(hidden=True)['interface_ray_indices']
    assert [len(result[i]) for i in range(3)] == [5,5,5]
    assert result[9] == []
