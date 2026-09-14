"""Analytic synthetic ray counterexample, not real-dataset qualification.

Two box-union interiors differ by a side branch. Sparse identical five-frame
measurements cannot certify its absence. This tests the actual ray-grid path.
"""
import numpy as np
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid,FREE

MAIN=np.array([[-8.,-1.,-1.],[8.,1.,1.]])
BRANCH=np.array([[2.,0.,-1.],[3.,6.,1.]])


def first_exit(origin,direction,boxes):
    intervals=[]
    for box in boxes:
        lo,hi=0.,float('inf')
        for j in range(3):
            if direction[j]==0:
                if not box[0,j]<=origin[j]<=box[1,j]:hi=-1.;break
            else:
                a,b=sorted((box[:,j]-origin[j])/direction[j]);lo=max(lo,a);hi=min(hi,b)
        if lo<=hi:intervals.append((lo,hi))
    end=0.
    for lo,hi in sorted(intervals):
        if lo>end:break
        end=max(end,hi)
    assert np.isfinite(end) and end>0
    return origin+end*direction


def scans(boxes):
    origins=[];ends=[];slots=[]
    directions=np.concatenate((np.eye(3),-np.eye(3)))
    for i in range(5):
        origin=np.array([-2.+i*.1,.125,.125])
        for direction in directions:
            origins.append(origin);ends.append(first_exit(origin,direction,boxes));slots.append(i)
    return np.array(origins),np.array(ends),np.array(slots)


def test_free_candidate_does_not_certify_no_side_branch():
    origins,a,slots=scans([MAIN]);_,b,_=scans([MAIN,BRANCH])
    assert np.array_equal(a,b)  # same first returns, different actual geometry
    ga=build_surface_ray_grid(origins,a,np.ones(len(a),bool),slots)
    gb=build_surface_ray_grid(origins,b,np.ones(len(b),bool),slots)
    assert np.array_equal(ga.state,gb.state)
    cell=tuple(np.floor((np.array([2.625,.125,.125])+10)/.25).astype(int))
    assert ga.state[cell]==FREE and gb.state[cell]==FREE
    # The candidate is on observed free space in both; branching differs.
    assert not ga.physical_connectivity and not gb.physical_connectivity


def test_additional_ray_can_distinguish_worlds_not_hidden_identity():
    origin=np.array([2.5,.125,.125]);direction=np.array([0.,1.,0.])
    a=first_exit(origin,direction,[MAIN]);b=first_exit(origin,direction,[MAIN,BRANCH])
    assert a[1]==1. and b[1]==6.
    # Moving to a new viewpoint changes the evidence, not the label ledger.


def test_invalid_discriminating_ray_does_not_add_evidence():
    origin=np.array([[2.5,.125,.125]])
    a=np.array([[2.5,1.,.125]]);b=np.array([[2.5,6.,.125]])
    ga=build_surface_ray_grid(origin,a,np.array([False]),np.array([0]))
    gb=build_surface_ray_grid(origin,b,np.array([False]),np.array([0]))
    assert np.array_equal(ga.state,gb.state) and not ga.state.any()
