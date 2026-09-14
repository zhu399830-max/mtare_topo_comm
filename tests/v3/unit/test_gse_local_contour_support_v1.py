import numpy as np
from mtare_topo.teacher.gse_local_contour_support_v1 import contour_surface_support


def test_all_contour_cells_supported_with_no_remote_axis_input():
    loop=np.array([[.1,.1,.1],[.6,.1,.1],[.6,.6,.1],[.1,.6,.1]])
    pts=np.array([[x,y,.1] for x in (.1,.35,.6) for y in (.1,.35,.6)])
    r=contour_surface_support(loop,pts,np.ones(len(pts),dtype=bool))
    assert r['contour_fully_surface_supported']
    assert r['semantic_opening_label'] is None and r['anchor_label'] is None


def test_vertices_do_not_certify_missing_edge_interiors():
    loop=np.array([[.1,.1,.1],[1.1,.1,.1],[1.1,1.1,.1],[.1,1.1,.1]])
    r=contour_surface_support(loop,loop,np.ones(4,dtype=bool))
    assert not r['contour_fully_surface_supported'] and r['unobserved_contour_cells']


def test_xy_overlap_does_not_supply_other_layer():
    loop=np.array([[.1,.1,.1],[.6,.1,.1],[.6,.6,.1],[.1,.6,.1]])
    pts=loop+np.array([0,0,2.])
    r=contour_surface_support(loop,pts,np.ones(4,dtype=bool))
    assert not r['supported_cell_point_indices']


def test_invalid_returns_do_not_supply_contour():
    loop=np.array([[.1,.1,.1],[.2,.1,.1],[.2,.2,.1]])
    r=contour_surface_support(loop,loop,np.zeros(3,dtype=bool))
    assert not r['contour_fully_surface_supported']
