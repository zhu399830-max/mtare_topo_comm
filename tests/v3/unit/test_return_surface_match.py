import numpy as np
import pytest
from mtare_topo.teacher.gse_return_surface_match import triangle_distances, SourceSurfaceMatcher


def test_face_edge_vertex_and_degenerate_distances():
    t = np.array([[[0,0,0],[1,0,0],[0,1,0]]],dtype=float)
    assert triangle_distances([.2,.2,2],t)[0]==2
    assert np.isclose(triangle_distances([1,1,0],t)[0],np.sqrt(.5))
    assert triangle_distances([-1,0,0],t)[0]==1
    assert triangle_distances([0,0,1],np.zeros((1,3,3)))[0]==1


def test_overlapping_surfaces_not_arbitrarily_collapsed():
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.01],[1,0,.01],[0,1,.01]])
    m = SourceSurfaceMatcher(vertices,[[0,1,2],[3,4,5]])
    r = m.match([.2,.2,.005],maximum_distance_m=.006)
    assert r['triangle_indices'].tolist()==[0,1]
    assert r['membership'] is None and not r['point_label_qualified']
    assert len(m.match([.2,.2,.005],maximum_distance_m=.001)['triangle_indices'])==0


def test_candidate_budget_fails_without_dropping_triangles():
    m = SourceSurfaceMatcher([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[0,1,2]],max_candidates_per_return=1)
    with pytest.raises(MemoryError): m.match([.2,.2,0],maximum_distance_m=.001)


def test_broad_phase_matches_exhaustive_distances():
    rng = np.random.default_rng(123)
    vertices = rng.normal(size=(300,3)); faces = np.arange(300).reshape(-1,3)
    m = SourceSurfaceMatcher(vertices,faces)
    for p in rng.normal(size=(20,3)):
        expected = np.flatnonzero(triangle_distances(p,vertices[faces])<=.02)
        assert np.array_equal(m.match(p,maximum_distance_m=.02)['triangle_indices'],expected)
        nearest=m.nearest_residual(p)
        assert np.isclose(nearest['minimum_distance_m'],triangle_distances(p,vertices[faces]).min(),rtol=0,atol=1e-14)
        assert nearest['point_label_qualified'] is False
