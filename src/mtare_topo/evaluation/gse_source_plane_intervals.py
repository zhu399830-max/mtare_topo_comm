"""Outward-rounded plane-sign bounds over source/float32 coordinate hulls.

This bounds the numerical representation change, not sensor noise, physical
clearance, triangle inclusion, or topology. No distance tolerance is fitted.
"""
import numpy as np


def _hull(value):
    value=np.asarray(value)
    if value.dtype!=np.float64 or not np.isfinite(value).all():
        raise ValueError('finite original float64 geometry required')
    quantized=value.astype(np.float32).astype(np.float64)
    if not np.isfinite(quantized).all():raise ValueError('float32 quantization overflow')
    return np.minimum(value,quantized),np.maximum(value,quantized)


def _add(a,b):
    return np.nextafter(a[0]+b[0],-np.inf),np.nextafter(a[1]+b[1],np.inf)


def _sub(a,b):
    return np.nextafter(a[0]-b[1],-np.inf),np.nextafter(a[1]-b[0],np.inf)


def _mul(a,b):
    products=np.stack((a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]))
    return np.nextafter(products.min(axis=0),-np.inf),np.nextafter(products.max(axis=0),np.inf)


def _component(a,i):return a[0][...,i],a[1][...,i]


def _cross(a,b):
    parts=[_sub(_mul(_component(a,j),_component(b,k)),
                _mul(_component(a,k),_component(b,j))) for j,k in ((1,2),(2,0),(0,1))]
    return tuple(np.stack([p[side] for p in parts],axis=-1) for side in (0,1))


def _dot(a,b):
    products=_mul(a,b)
    return _add(_add(_component(products,0),_component(products,1)),_component(products,2))


def source_plane_sign_bounds(triangles,origins,directions):
    triangles=np.asarray(triangles);origins=np.asarray(origins);directions=np.asarray(directions)
    if (triangles.ndim!=3 or triangles.shape[1:]!=(3,3) or origins.shape!=(len(triangles),3)
            or directions.shape!=origins.shape or np.any(np.linalg.norm(directions,axis=1)==0)):
        raise ValueError('matching triangles, origins and nonzero packed-order directions required')
    vertex=[_hull(triangles[:,i]) for i in range(3)]
    origin=_hull(origins);direction=_hull(directions)
    normal=_cross(_sub(vertex[1],vertex[0]),_sub(vertex[2],vertex[0]))
    numerator=_dot(_sub(vertex[0],origin),normal)
    denominator=_dot(direction,normal)
    if not all(np.isfinite(x).all() for x in (*numerator,*denominator)):
        raise ValueError('interval arithmetic overflow; no direction certificate')
    np_,nn=numerator[0]>0,numerator[1]<0
    dp,dn=denominator[0]>0,denominator[1]<0
    forward=(np_&dp)|(nn&dn)
    backward=(np_&dn)|(nn&dp)
    return dict(numerator_bounds=np.stack(numerator,axis=-1),denominator_bounds=np.stack(denominator,axis=-1),
        forward=forward,backward=backward,unknown=~(forward|backward),
        entering=forward&dn,leaving=forward&dp,
        physical_or_membership_certificate=False)
