import numpy as np
from mtare_topo.teacher.gse_section_reference_ownership_v1 import section_reference_owner


def square(cx=0.,size=1.):
    return np.array([[cx-size,-size,0],[cx+size,-size,0],[cx+size,size,0],[cx-size,size,0]])


def test_remote_loop_does_not_share_axis_reference():
    r=section_reference_owner([square(5),square()],reference_m=[0,0,0],normal=[0,0,1])
    assert r['owner_loop_index']==1


def test_nested_or_boundary_or_no_owner_stays_unknown():
    for loops,point in [([square(),square(size=2)],[0,0,0]),([square()],[1,0,0]),([square(5)],[0,0,0])]:
        assert section_reference_owner(loops,reference_m=point,normal=[0,0,1])['owner_loop_index'] is None


def test_rotation_and_permutation_preserve_correspondence():
    rot=np.array([[0,0,1],[1,0,0],[0,1,0.]])
    loops=[square(),square(5)]
    r=section_reference_owner([p@rot.T for p in loops],reference_m=[0,0,0],normal=rot@[0,0,1])
    assert r['owner_loop_index']==0
