import numpy as np
import pytest
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.representation.gse_observed_regions import build_observed_regions
from mtare_topo.representation.gse_ray_region_support import bind_ray_support, compare_support


def setup_rays(origins, ends, frames=None, valid=None):
    o=np.asarray(origins,dtype=np.float64); e=np.asarray(ends,dtype=np.float64)
    v=np.ones(len(o),dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    f=np.zeros(len(o),dtype=np.int64) if frames is None else np.asarray(frames,dtype=np.int64)
    g=build_surface_ray_grid(o,e,v,f); r=build_observed_regions(g)
    return g,r,o,e,v,f


def test_two_branch_locations_share_free_component_not_structure_identity():
    # Main corridor and branches at x=1.125 and x=3.125: two distinct places.
    args=setup_rays([[.125,.125,.125],[1.125,.125,.125],[3.125,.125,.125]],
                    [[5.125,.125,.125],[1.125,2.125,.125],[3.125,-2.125,.125]])
    s=bind_ray_support(*args)
    c=compare_support(s[1],s[2])
    assert c['common_components'] and c['shared_free_cells']==0
    assert c['structural_membership'] is None and not c['training_qualified']


def test_return_surface_remains_surface_with_same_ray_free_context():
    s=bind_ray_support(*setup_rays([[.125,.125,.125]],[[2.125,.125,.125]]))[0]
    assert s.free_cells and s.surface_cells and s.components
    assert not set(s.free_cells)&set(s.surface_cells)
    assert s.structural_membership is None


def test_outside_return_does_not_create_window_surface():
    s=bind_ray_support(*setup_rays([[.125,.125,.125]],[[12.125,.125,.125]]))[0]
    assert s.free_cells and not s.surface_cells


def test_invalid_ray_has_no_record_or_background_label():
    s=bind_ray_support(*setup_rays([[.125,.125,.125],[0,0,0]],
        [[2.125,.125,.125],[np.nan,np.nan,np.nan]],valid=[True,False]))
    assert len(s)==1 and s[0].ray_index==0


def test_registered_frame_or_ray_payload_drift_rejected():
    args=list(setup_rays([[.125,.125,.125]],[[2.125,.125,.125]]))
    args[-1]=np.ones(1,dtype=np.int64)
    with pytest.raises(ValueError,match='drift'): bind_ray_support(*args)


def test_stacked_support_does_not_merge_in_xy():
    s=bind_ray_support(*setup_rays([[.125,.125,.125],[.125,.125,2.125]],
                                   [[2.125,.125,.125],[2.125,.125,2.125]]))
    assert compare_support(*s)['common_components']==[]
    assert compare_support(*s)['structural_membership'] is None


def test_occupied_conflict_never_becomes_free_support():
    args=setup_rays([[.125,.125,.125],[.125,.125,.125]],
                    [[1.125,.125,.125],[3.125,.125,.125]],[0,4])
    s=bind_ray_support(*args)
    assert not set(s[0].surface_cells)&set(s[1].free_cells)
    assert s[1].frame_slot==4


def test_repeated_binding_deterministic():
    args=setup_rays([[.125,.125,.125]],[[2.125,.125,.125]])
    assert bind_ray_support(*args)==bind_ray_support(*args)
