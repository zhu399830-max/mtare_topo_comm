"""Counterexamples: full-axis observation is not a local semantic teacher."""
import numpy as np

from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.teacher.gse_junction_support_v2 import junction_support


def fixture():
    a=np.array([.1,.1,.1]);d=np.array([[1.,0,0],[-1.,0,0],[0,1.,0]])
    grid=build_surface_ray_grid(np.repeat(a[None],3,axis=0),a+1.8*d,
                                np.ones(3,dtype=bool),np.arange(3))
    return a,d,grid


def evaluate(a,grid,paths):
    return junction_support(grid,anchor_m=a,incident_paths_m=tuple(paths),
        endpoint_keys=(("a",0),("b",0),("c",0)),axis_start_indices=(0,0,0))


def test_identical_observation_and_local_prefix_different_remote_extent():
    a,d,grid=fixture()
    local=tuple(np.stack([a,a+direction]) for direction in d)
    extended=tuple(np.vstack([p,a+4*direction]) for p,direction in zip(local,d))
    short=evaluate(a,grid,local);long=evaluate(a,grid,extended)
    assert short.status=="REFERENCE_JUNCTION_SUPPORTED"
    assert long.status=="UNKNOWN"
    assert short.grid_content_sha256==long.grid_content_sha256
    # Neither outcome establishes a local node label. Do not silently truncate
    # actual reference paths to reproduce the convenient first outcome.
    assert short.semantic_label is long.semantic_label is None


def test_three_free_lines_do_not_certify_aperture_or_wall_geometry():
    a,d,grid=fixture()
    result=evaluate(a,grid,[np.stack([a,a+v]) for v in d])
    assert grid.first_return_count==3
    assert result.status=="REFERENCE_JUNCTION_SUPPORTED"
    # This can pass without any measured opening contour, terminal cap,
    # cross-section completeness, or physical ground/body contract.
    assert result.semantic_label is None and result.human_reviewed is False


def test_supported_subset_must_not_delete_hidden_incident_branch():
    a,d,grid=fixture();paths=[np.stack([a,a+v]) for v in d]
    paths[-1]=np.stack([a,a+np.array([0.,0.,1.])])
    result=evaluate(a,grid,paths)
    assert result.status=="UNKNOWN"
    assert len(result.branch_support)==3
    assert result.semantic_label is None
