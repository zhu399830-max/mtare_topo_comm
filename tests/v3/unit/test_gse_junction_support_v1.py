import numpy as np
import pytest

from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.teacher.gse_junction_support_v1 import junction_support


def inputs():
    anchor=np.array([.1,.1,.1])
    ends=np.array([[1.9,.1,.1],[-1.9,.1,.1],[.1,1.9,.1]])
    grid=build_surface_ray_grid(np.repeat(anchor[None],3,axis=0),ends,np.ones(3,dtype=bool),np.arange(3))
    paths=tuple(np.stack([anchor,anchor+.7*(end-anchor)]) for end in ends)
    return grid,dict(anchor_m=anchor,incident_paths_m=paths,endpoint_keys=(("edge",0),("edge",1),("branch",0)))


def test_three_observed_arms_remain_reference_proxy():
    grid,args=inputs();r=junction_support(grid,**args)
    assert r.status=="REFERENCE_JUNCTION_SUPPORTED"
    assert r.semantic_label is None and not r.human_reviewed


def test_unobserved_branch_does_not_reduce_degree_and_label_corridor():
    grid,args=inputs();paths=list(args["incident_paths_m"])
    paths[-1]=np.array([[.1,.1,.1],[.1,.1,1.5]])
    args["incident_paths_m"]=tuple(paths)
    assert junction_support(grid,**args).status=="UNKNOWN"


def test_degree_two_subdivision_never_becomes_structure_node():
    grid,args=inputs();args["incident_paths_m"]=args["incident_paths_m"][:2];args["endpoint_keys"]=args["endpoint_keys"][:2]
    assert junction_support(grid,**args).status=="UNKNOWN"


def test_xy_coincident_stacked_anchor_cannot_be_snapped():
    grid,args=inputs();args["anchor_m"]=np.array([.1,.1,3.1])
    with pytest.raises(ValueError,match="exact reference anchor"):junction_support(grid,**args)


def test_duplicate_arm_not_new_observable_branch():
    grid,args=inputs();args["incident_paths_m"]=(args["incident_paths_m"][0],)*3
    assert junction_support(grid,**args).status=="UNKNOWN"


def test_duplicate_incidence_rejected():
    grid,args=inputs();args["endpoint_keys"]=(args["endpoint_keys"][0],)*3
    with pytest.raises(ValueError,match="duplicate incidence"):junction_support(grid,**args)
