import numpy as np
from mtare_topo.teacher.gse_terminal_relation_witness_v1 import terminal_relation_witness


def args():
    return dict(source_id='a', node_id='end', cap_rays=[0], opening_rays=[1],
                source_ids=['a', 'b'], origin_distances=np.tile([-1., 2.], (5, 1)),
                interface_hits=[], interfaces=[dict(interface_id_teacher_only='j', node_id_teacher_only='junction')],
                first_return=np.ones(57600)*5, frame_rows=[10, 11, 12, 13, 14])


def test_two_sided_is_necessary_not_sufficient():
    r = terminal_relation_witness(**args())
    assert r['common_frame_slots'] == [0] and r['membership'] is None


def test_different_frames_do_not_invent_common_origin():
    a = args(); a['opening_rays'] = [11520]
    assert terminal_relation_witness(**a)['common_frame_slots'] == []


def test_overlap_does_not_choose_a_convenient_source():
    a = args(); a['origin_distances'][0, 1] = -1
    assert terminal_relation_witness(**a)['common_frame_slots'] == []


def test_other_node_on_one_ray_vetoes_clean_alternative():
    a = args(); a['opening_rays'] = [1, 2]
    a['interface_hits'] = [dict(ray_index=2, interface_id_teacher_only='j', t=2., inside_roi=True)]
    assert terminal_relation_witness(**a)['status'] == 'UNKNOWN_INTERVENING_NODE'


def test_hidden_hit_after_return_does_not_create_observed_conflict():
    a = args(); a['interface_hits'] = [dict(ray_index=1, interface_id_teacher_only='j', t=6., inside_roi=True)]
    assert terminal_relation_witness(**a)['common_frame_slots'] == [0]
