from dataclasses import replace
import numpy as np
import pytest
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.representation.candidate_ray_evidence_v1 import candidate_ray_evidence


def grid(empty=False):
    o=np.array([[.125,.125,.125]],dtype=float)
    e=np.array([[1.125,.125,.125]],dtype=float)
    return build_surface_ray_grid(o,e,np.array([not empty]),np.array([2]))


def test_interior_height_unknown_return_and_no_confirmation():
    p=np.array([[.375,.125,.125],[.375,.125,1.125],[1.125,.125,.125]])
    saved=p.copy(); g=grid(); r=candidate_ray_evidence(g,p)
    assert [x.states for x in r]==[(1,),(0,),(2,)]
    assert r[0].free_frame_bits==(4,) and r[2].occupied_frame_bits==(4,)
    assert all(not x.structure_confirmed and not x.physical_connectivity for x in r)
    np.testing.assert_array_equal(p,saved)
    assert candidate_ray_evidence(g,p[::-1])==r[::-1]


def test_face_edge_corner_and_outer_boundary_alternatives():
    p=np.array([[.25,.125,.125],[.25,.25,.125],[.25,.25,.25],[10.,.125,.125],[11.,0.,0.]])
    r=candidate_ray_evidence(grid(),p)
    assert [len(x.cell_indices) for x in r]==[2,4,8,1,0]
    assert all(x.boundary_ambiguous for x in r)
    assert [x.touches_outside for x in r]==[False,False,False,True,True]
    assert r[-1].states==() and not r[-1].structure_confirmed


def test_empty_observations_and_queries():
    g=grid(True)
    assert candidate_ray_evidence(g,np.empty((0,3)))==()
    assert candidate_ray_evidence(g,np.array([[.125,.125,.125]]))[0].states==(0,)


def test_bad_inputs_and_grid_drift_rejected():
    g=grid()
    for p in [np.full((1,3),np.nan),np.zeros((33,3)),np.zeros((1,3),dtype=int)]:
        with pytest.raises(ValueError):candidate_ray_evidence(g,p)
    with pytest.raises(ValueError):
        candidate_ray_evidence(replace(g,content_sha256='0'*64),np.empty((0,3)))
