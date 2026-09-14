import numpy as np
import pytest
from mtare_topo.teacher.gse_reference_continuations_v1 import ReferenceSource
from mtare_topo.teacher.gse_terminal_continuation_contract import terminal_continuation_component,continuation_origin_witness


def source(name,nodes,points):
    return ReferenceSource(name,nodes,(tuple(map(float,points[0])),tuple(map(float,points[-1]))))


@pytest.mark.parametrize('cut',[3.,10.,15.])
def test_partition_and_source_order_do_not_move_window_exit(cut):
    full=[[0.,0.,0.],[20.,0.,0.]]
    a=terminal_continuation_component([source('a',('t','end'),full)],{'a':full},terminal_node='t',center_m=[0,0,0])
    p=[[0.,0.,0.],[cut,0.,0.]];q=[[20.,0.,0.],[cut,0.,0.]]
    split=terminal_continuation_component([source('q',('end','cut'),q),source('p',('t','cut'),p)],{'p':p,'q':q},terminal_node='t',center_m=[0,0,0])
    assert split['opening_position_m']==a['opening_position_m']==[10.,0.,0.]
    assert split['membership'] is None and not split['observation_support_verified']
    assert split['source_arc_offsets'][1]==dict(source_id='q',entry_side=1,offset_m=cut,length_m=20.-cut)


def test_rigid_axis_permutation_and_translation_keep_height():
    p=np.array([[0.,0.,0.],[3.,0.,0.]])+[2.,4.,7.]
    q=np.array([[3.,0.,0.],[20.,0.,0.]])+[2.,4.,7.]
    p=p[:,[1,2,0]];q=q[:,[1,2,0]]
    r=terminal_continuation_component([source('a',('t','c'),p),source('b',('c','e'),q)],{'a':p,'b':q},terminal_node='t',center_m=[4,7,2])
    assert r['opening_position_m']==[4.,7.,12.]


def test_junction_is_not_an_artificial_cut():
    lines={'a':[[0,0,0],[5,0,0]],'b':[[5,0,0],[20,0,0]],'c':[[5,0,0],[5,20,0]]}
    sources=[source(k,n,lines[k]) for k,n in [('a',('t','j')),('b',('j','x')),('c',('j','y'))]]
    r=terminal_continuation_component(sources,lines,terminal_node='t',center_m=[0,0,0])
    assert r['source_ids']==['a'] and r['status']=='UNKNOWN'


def test_displaced_height_join_is_not_snapped():
    lines={'a':[[0,0,0],[5,0,0]],'b':[[5,0,1],[20,0,1]]}
    r=terminal_continuation_component([source('a',('t','c'),lines['a']),source('b',('c','e'),lines['b'])],lines,terminal_node='t',center_m=[0,0,0])
    assert r['status']=='UNKNOWN' and r['membership'] is None


def test_later_reentry_is_not_the_terminal_component():
    p=[[0,0,0],[12,0,0]];q=[[12,0,0],[12,5,0],[0,5,0],[-12,5,0]]
    r=terminal_continuation_component([source('a',('t','c'),p),source('b',('c','e'),q)],{'a':p,'b':q},terminal_node='t',center_m=[0,0,0])
    assert r['opening_position_m']==[10.,0.,0.]


def witness(distances,**overrides):
    data=dict(node_id='t',cap_rays=[0],opening_rays=[1],interface_hits=[],interfaces=[],first_return=np.ones((5,16,720)),frame_rows=list(range(5)))
    data.update(overrides)
    return continuation_origin_witness(continuation_source_ids=['a','b'],source_ids=['a','b','other'],origin_distances=distances,**data)


def test_internal_overlap_can_support_but_not_label():
    r=witness(np.tile([-1.,-1.,2.],(5,1)))
    assert r['status']=='TWO_SIDED_SOURCE_WITNESS_ONLY' and r['membership'] is None
    assert not r['complete_correspondence_verified']


@pytest.mark.parametrize('distances',[[1.,1.,-1.],[-1.,1.,-1.],[0.,1.,2.]])
def test_wrong_layer_unrelated_overlap_or_boundary_stays_unknown(distances):
    r=witness(np.tile(distances,(5,1)))
    assert r['status']=='UNKNOWN_NO_COMMON_SOURCE_ORIGIN' and r['membership'] is None


def test_observed_intervening_junction_veto_is_reused():
    r=witness(np.tile([-1.,1.,2.],(5,1)),interfaces=[dict(interface_id_teacher_only=0,node_id_teacher_only='j')],
        interface_hits=[dict(ray_index=0,interface_id_teacher_only=0,t=.5,inside_roi=True)])
    assert r['status']=='UNKNOWN_INTERVENING_NODE' and r['membership'] is None


def test_different_frames_are_not_two_sided_support():
    r=witness(np.tile([-1.,1.,2.],(5,1)),opening_rays=[11520])
    assert r['status']=='UNKNOWN_NO_COMMON_SOURCE_ORIGIN'


def test_future_ray_is_rejected():
    with pytest.raises(ValueError):witness(np.tile([-1.,1.,2.],(5,1)),opening_rays=[57600])
