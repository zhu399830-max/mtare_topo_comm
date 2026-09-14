from fractions import Fraction as F
import numpy as np
import pytest
from mtare_topo.evaluation.gse_source_plane_intervals import source_plane_sign_bounds


def test_clear_entering_leaving_and_backward():
    tri=np.repeat(np.array([[[0.,0,1],[1.,0,1],[0.,1,1]]]),3,axis=0)
    origins=np.array([[.2,.2,2.],[.2,.2,0.],[.2,.2,2.]])
    rays=np.array([[0.,0,-1.],[0.,0,1.],[0.,0,1.]])
    result=source_plane_sign_bounds(tri,origins,rays)
    assert result['forward'].tolist()==[True,True,False]
    assert result['entering'].tolist()==[True,False,False]
    assert result['backward'].tolist()==[False,False,True]


def test_quantization_flip_and_exact_boundary_are_unknown():
    tri=np.array([[[70.000003,0,0],[70.123003,1,0],[70.000003,0,1]]])
    origin=np.array([[np.nextafter(70.000003+.123*.2,-np.inf),.2,.2]])
    assert source_plane_sign_bounds(tri,origin,np.array([[-1.,0,0]]))['unknown'][0]
    tri=np.array([[[0.,0,0],[1.,0,0],[0.,1,0]]])
    assert source_plane_sign_bounds(tri,np.array([[.25,.25,0.]]),np.array([[0.,0,1.]]))['unknown'][0]


def test_exact_rational_determinants_are_enclosed():
    rng=np.random.default_rng(51)
    t=rng.normal(size=(24,3,3))*70;o=rng.normal(size=(24,3))*70;d=rng.normal(size=(24,3))
    result=source_plane_sign_bounds(t,o,d)
    def sub(a,b):return [x-y for x,y in zip(a,b)]
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    for quantize in (False,True):
        tt,oo,dd=(x.astype(np.float32).astype(float) if quantize else x for x in (t,o,d))
        for i in range(24):
            vertices=[[F(float(x)) for x in row] for row in tt[i]]
            origin=[F(float(x)) for x in oo[i]];direction=[F(float(x)) for x in dd[i]]
            n=cross(sub(vertices[1],vertices[0]),sub(vertices[2],vertices[0]))
            for key,value in [('numerator_bounds',dot(sub(vertices[0],origin),n)),('denominator_bounds',dot(direction,n))]:
                lo,hi=result[key][i]
                assert F(float(lo))<=value<=F(float(hi))


def test_no_implicit_float32_source_or_invalid_ray():
    with pytest.raises(ValueError):
        source_plane_sign_bounds(np.zeros((1,3,3),np.float32),np.zeros((1,3)),np.ones((1,3)))
