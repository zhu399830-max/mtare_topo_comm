from dataclasses import replace
import numpy as np
import pytest
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_surface_arc_index import index_surface_arcs


def fixture():
    p=SweptSuperellipsePrimitive('source',np.array([[0.,0,0],[2.,0,0]]),((1.,1.),(1.,1.)),(2.,2.))
    m=mesh_swept_superellipse(p,axial_spacing_m=.5,angular_segments=12)
    return p,m


def test_wall_triangles_and_end_caps_keep_exact_arc_bounds():
    p,m=fixture();r=index_surface_arcs(p,m,axial_spacing_m=.5,angular_segments=12)
    assert r.triangle_arc_bounds_m.shape==(120,2)
    assert np.array_equal(r.triangle_arc_bounds_m[:24],np.tile([0.,.5],(24,1)))
    assert np.array_equal(r.triangle_arc_bounds_m[-24::2],np.zeros((12,2)))
    assert np.array_equal(r.triangle_arc_bounds_m[-23::2],np.full((12,2),2.))
    assert not r.point_membership_qualified
    assert not r.triangle_arc_bounds_m.flags.writeable


def test_wrong_tessellation_or_order_not_reinterpreted():
    p,m=fixture()
    with pytest.raises(ValueError,match='drift'):
        index_surface_arcs(p,m,axial_spacing_m=.25,angular_segments=12)
    with pytest.raises(ValueError,match='drift'):
        index_surface_arcs(p,replace(m,triangle_vertex_indices=m.triangle_vertex_indices[::-1]),
                           axial_spacing_m=.5,angular_segments=12)


def test_different_source_rejected():
    p,m=fixture()
    with pytest.raises(ValueError,match='same source'):
        index_surface_arcs(p,replace(m,primitive_id='other'),axial_spacing_m=.5,angular_segments=12)
