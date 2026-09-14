import numpy as np
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.representation.gse_observed_regions import build_observed_regions,query_observed_pair


def make(origins,ends,frames=None,valid=None):
    n=len(origins)
    grid=build_surface_ray_grid(np.asarray(origins,dtype=np.float64).reshape(-1,3),
        np.asarray(ends,dtype=np.float64).reshape(-1,3),np.ones(n,dtype=bool) if valid is None else valid,
        np.zeros(n,dtype=np.int64) if frames is None else np.asarray(frames,dtype=np.int64))
    return grid,build_observed_regions(grid)


def query(g,r,a,b):return query_observed_pair(r,g,a,b)['status']


def test_two_rays_around_corner_connect_without_single_through_ray():
    g,r=make([[.125,.125,.125],[1.125,.125,.125]],[[2.125,.125,.125],[1.125,2.125,.125]],[0,4])
    assert query(g,r,[.375,.125,.125],[1.125,1.625,.125])=='SAME_OBSERVED_CELL_COMPONENT'
    assert 17 in r.frame_bits and not r.training_qualified


def test_unobserved_gap_is_not_filled():
    g,r=make([[.125,.125,.125],[2.125,.125,.125]],[[1.125,.125,.125],[3.125,.125,.125]])
    assert query(g,r,[.375,.125,.125],[2.375,.125,.125])=='UNKNOWN'


def test_occupied_return_blocks_even_if_other_ray_crosses():
    g,r=make([[.125,.125,.125],[.125,.125,.125]],[[1.125,.125,.125],[3.125,.125,.125]],[0,1])
    assert query(g,r,[.375,.125,.125],[2.375,.125,.125])=='UNKNOWN'


def test_xy_overlap_does_not_merge_different_heights():
    g,r=make([[.125,.125,.125],[1.125,-1.125,2.125]],[[2.125,.125,.125],[1.125,1.125,2.125]])
    assert query(g,r,[.375,.125,.125],[1.125,.375,2.125])=='UNKNOWN'


def test_same_observations_do_not_change_with_hidden_map():
    # Neither hidden-world version can enter this API; unknown gap remains so.
    g,a=make([[.125,.125,.125]],[[1.125,.125,.125]])
    b=build_observed_regions(g)
    assert np.array_equal(a.labels,b.labels)
    assert query(g,a,[.375,.125,.125],[2.375,.125,.125])=='UNKNOWN'


def test_ray_permutation_does_not_change_regions():
    o=[[.125,.125,.125],[1.125,.125,.125]];e=[[2.125,.125,.125],[1.125,2.125,.125]]
    g,r=make(o,e,[0,4]);h,s=make(o[::-1],e[::-1],[4,0])
    assert np.array_equal(r.labels,s.labels)


def test_invalid_ray_never_connects_gap():
    g,r=make([[.125,.125,.125]],[[3.125,.125,.125]],valid=np.zeros(1,dtype=bool))
    assert not r.sizes


def test_no_boundary_snapping_or_outside_sphere_path():
    g,r=make([[9.125,9.125,.125]],[[9.125,9.875,.125]])
    assert not r.sizes


def test_occupied_and_unknown_endpoints_not_background():
    g,r=make([[.125,.125,.125]],[[1.125,.125,.125]])
    assert query(g,r,[.375,.125,.125],[1.125,.125,.125])=='UNKNOWN'


def test_corner_only_contact_does_not_connect():
    g,r=make([[.125,.125,.125],[.375,.375,.125]],[[-.375,.125,.125],[.875,.375,.125]])
    assert r.labels[40,40,40]>0 and r.labels[41,41,40]>0
    assert query(g,r,[.125,.125,.125],[.375,.375,.125])=='UNKNOWN'
