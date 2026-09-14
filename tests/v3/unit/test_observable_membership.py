import numpy as np
from mtare_topo.representation.observable_membership import bind_observable_support

def test_no_implicit_negative_or_shared_label():
    boxes=[dict(min_m=[0,0,0],max_m=[2,2,2]),dict(min_m=[1,1,1],max_m=[3,3,3])]
    r=bind_observable_support(np.array([[0,0,0],[3,3,3],[1,1,1],[9,9,9],[np.nan,0,0]]),np.ones(5,dtype=bool),boxes)
    assert r['support'].tolist()==[[1,-1],[-1,1],[-1,-1],[-1,-1],[-1,-1]]
    assert r['overlap_unresolved'].tolist()==[False,False,True,False,False]
    assert not (r['support']==0).any()

def test_permutation_and_invalid():
    xyz=np.array([[0.,0,0],[2,2,2],[4,4,4]])
    boxes=[dict(min_m=[0,0,0],max_m=[2,2,2])];valid=np.array([True,False,True])
    r=bind_observable_support(xyz,valid,boxes)
    order=[2,0,1];s=bind_observable_support(xyz[order],valid[order],boxes)
    assert np.array_equal(s['support'],r['support'][order])
    assert r['support'].tolist()==[[1],[-1],[-1]]
