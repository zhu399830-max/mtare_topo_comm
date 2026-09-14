import numpy as np
import pytest

from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.teacher.gse_junction_support_v1 import junction_support as old_support
from mtare_topo.teacher.gse_junction_support_v2 import junction_support


def inputs():
    anchor=np.array([.1,.1,.1]); axis=anchor+np.array([0.,.001,0.])
    ends=axis+np.array([[1.8,0,0],[-1.8,0,0],[0,1.8,0]])
    grid=build_surface_ray_grid(np.repeat(axis[None],3,axis=0),ends,np.ones(3,dtype=bool),np.arange(3))
    paths=tuple(np.stack([anchor,axis,axis+.7*(end-axis)]) for end in ends)
    return grid,dict(anchor_m=anchor,incident_paths_m=paths,endpoint_keys=(("p",0),("p",1),("b",0)))


def test_shared_connector_is_not_shared_axis():
    grid,args=inputs()
    old=old_support(grid,**args)
    new=junction_support(grid,**args,axis_start_indices=(1,1,1))
    assert old.reason=="COINCIDENT_INITIAL_ARMS_NOT_INDEPENDENT_OBSERVATIONS"
    assert new.status=="REFERENCE_JUNCTION_SUPPORTED"
    assert old.branch_support==new.branch_support
    assert new.semantic_label is None


def test_missing_observation_stays_unknown_after_direction_fix():
    grid,args=inputs(); paths=list(args["incident_paths_m"])
    paths[-1]=paths[-1].copy();paths[-1][-1]=[.1,.101,1.5]
    args["incident_paths_m"]=tuple(paths)
    r=junction_support(grid,**args,axis_start_indices=(1,1,1))
    assert r.reason=="INCIDENT_OBSERVATION_SUPPORT_INCOMPLETE"
    assert r.status=="UNKNOWN" and r.semantic_label is None


@pytest.mark.parametrize("indices", [(1,1), (2,1,1), (True,1,1), ("1",1,1)])
def test_bad_provenance_rejected(indices):
    grid,args=inputs()
    with pytest.raises(ValueError):junction_support(grid,**args,axis_start_indices=indices)


def test_real_duplicate_axes_remain_unknown():
    grid,args=inputs();args["incident_paths_m"]=(args["incident_paths_m"][0],)*3
    r=junction_support(grid,**args,axis_start_indices=(1,1,1))
    assert r.status=="UNKNOWN" and "SOURCE_AXIS" in r.reason
