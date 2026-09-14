import numpy as np
from mtare_topo.teacher.gse_joint_reference_targets_v2 import _add_interior_correspondence


def case(competing=False, shared=True):
    result=dict(record=dict(anchors=[{}],openings=[dict(position_m=[10.,0.,0.],direction=[1.,0.,0.])],membership=[[None]]),
        teacher_provenance=dict(anchors=[dict(node_id_teacher_only='a',interface_ids=[0],interior_witness_ray_indices=[[7]])],
            openings=[dict(primitive_id_teacher_only='p',crossing_ray_indices=[7 if shared else 8])],
            relations=[dict(status='UNKNOWN')]))
    bundle=dict(sensor_teacher_only=dict(sensor_xyz_m=np.zeros((5,3)),yaw_deg=np.zeros(5)))
    raw=dict(interfaces_teacher_only=[dict(interface_id_teacher_only=0,node_id_teacher_only='a',endpoint_key_teacher_only=['p',0])],raw_interface_intersections=[])
    if competing:
        raw['interfaces_teacher_only'].append(dict(interface_id_teacher_only=1,node_id_teacher_only='b',endpoint_key_teacher_only=['q',0]))
        raw['raw_interface_intersections'].append(dict(ray_index=7,inside_roi=True,intersection_world_m=[5.,0.,0.],interface_id_teacher_only=1))
    _add_interior_correspondence(result,bundle,raw)
    return result


def test_same_ray_can_supply_interior_reference_correspondence():
    assert case()['record']['membership']==[[True]]


def test_source_identity_without_shared_ray_is_unknown():
    assert case(shared=False)['record']['membership']==[[None]]


def test_intermediate_unlabelled_reference_blocks_correspondence():
    r=case(competing=True)
    assert r['record']['membership']==[[None]]
    assert r['teacher_provenance']['relations'][0]['blocked_ray_indices']==[7]
