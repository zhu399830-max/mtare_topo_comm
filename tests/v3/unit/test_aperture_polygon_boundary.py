import numpy as np
import pytest
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive
from mtare_topo.teacher.aperture_polygon_boundary import straight_primitive_prism,classify_polygon_union
from mtare_topo.teacher.aperture_spherical_boundary import octahedral_sphere
from mtare_topo.teacher.aperture_component_contract import partition_boundary


def prism(start,end,axes=(2.,2.),exponent=2.):
    return straight_primitive_prism(SweptSuperellipsePrimitive('anonymous',np.array([start,end],float),(axes,axes),(exponent,exponent)))


@pytest.mark.parametrize('axes,exponent',[((2.,2.),2.),((2.,1.5),2.),((2.,1.5),8.)])
def test_finite_offset_overlap_has_one_negative_y_component(axes,exponent):
    v,f,a=octahedral_sphere(4)
    ps=[prism([x,1,0],[x,-30,0],axes,exponent) for x in (0.,.5)]
    s=classify_polygon_union(v,f,ps,center_m=(0,0,.04))
    p=partition_boundary(a,s,np.full_like(s,-1))
    assert len(p.geometry_components)==1 and not p.unresolved_component_pairs
    assert not p.observed_components and not p.training_labels_qualified
    np.testing.assert_array_equal(s,classify_polygon_union(v,f,[ps[1],ps[0],ps[0]],center_m=(0,0,.04)))


@pytest.mark.parametrize('offset',[(0.,6.,0.),(0.,0.,6.)])
def test_parallel_and_stacked_do_not_merge(offset):
    v,f,a=octahedral_sphere(4);d=np.array(offset)
    ps=[prism([-30,0,0],[30,0,0]),prism(np.array([-30,0,0])+d,np.array([30,0,0])+d)]
    s=classify_polygon_union(v,f,ps,center_m=(0,0,.04))
    p=partition_boundary(a,s,np.full_like(s,-1))
    # Whole physical boundary includes unobserved other tunnel: FOUR, not
    # two visible targets. This intentionally exposes geometry/visibility split.
    assert len(p.geometry_components)==4 and not p.unresolved_component_pairs
    assert len(p.observed_components)==0


def test_finite_end_inside_window_not_an_aperture():
    v,f,a=octahedral_sphere(4);ps=[prism([-30,0,0],[6,0,0])]
    s=classify_polygon_union(v,f,ps,center_m=(0,0,.04))
    assert len(partition_boundary(a,s,np.full_like(s,-1)).geometry_components)==1


def test_curved_or_tapered_inputs_rejected():
    for points,axes in [([[0,0,0],[1,0,0],[1,1,0]],((2,2),(2,2))),([[0,0,0],[1,0,0]],((2,2),(1,1)))]:
        with pytest.raises(ValueError):straight_primitive_prism(SweptSuperellipsePrimitive('p',np.array(points,float),axes,(2.,2.)))


def test_classified_cells_respect_halfspaces_at_interior_samples():
    v,f,a=octahedral_sphere(3);p=prism([.5,1,0],[.5,-30,0],(2,1.5),8.)
    s=classify_polygon_union(v,f,[p])
    for w in ([.1,.1,.8],[.3,.4,.3]):
        x=np.einsum('ijk,j->ik',v[f],w);x=10*x/np.linalg.norm(x,axis=1)[:,None]
        inside=np.all(x@p.normals.T-p.offsets<=p.numerical_padding_m,axis=1)
        assert inside[s==1].all() and not inside[s==0].any()
