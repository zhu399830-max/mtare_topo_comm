"""Independent convex-prism ray intervals for declared horizontal fixtures.

No production mesh/teacher/SDF. Boundary-only contact is not certified as an
open connection. Results are geometric ray evidence, not robot reachability.
"""
import numpy as np


def prism_planes(points,half_axes,power):
    points=np.asarray(points,float);axes=np.asarray(half_axes,float)
    if points.shape!=(2,3) or not np.isfinite(points).all() or axes.shape!=(2,) or not np.isfinite(axes).all() or (axes<=0).any():
        raise ValueError('finite straight endpoints and positive axes required')
    delta=points[1]-points[0];length=np.linalg.norm(delta)
    if length==0 or delta[2]!=0 or not np.isfinite(power) or power<1:
        raise ValueError('nonzero horizontal straight convex prism required')
    tangent=delta/length;lateral=np.array([-tangent[1],tangent[0],0.]);up=np.array([0.,0.,1.])
    theta=np.arange(64)*2*np.pi/64
    polygon=np.stack((axes[0]*np.sign(np.cos(theta))*np.abs(np.cos(theta))**(2/power),
                      axes[1]*np.sign(np.sin(theta))*np.abs(np.sin(theta))**(2/power)),1)
    edges=np.roll(polygon,-1,axis=0)-polygon
    cross_normals=np.column_stack((edges[:,1],-edges[:,0]))
    normals=cross_normals[:,0,None]*lateral+cross_normals[:,1,None]*up
    bounds=(cross_normals*polygon).sum(1)+normals@points[0]
    return np.vstack((normals,tangent,-tangent)),np.r_[bounds,tangent@points[1],-tangent@points[0]]


def ray_intervals(origins,directions,planes):
    origins=np.asarray(origins,float);directions=np.asarray(directions,float)
    if origins.shape!=directions.shape or origins.ndim!=2 or origins.shape[1]!=3 or not np.isfinite(origins).all() or not np.isfinite(directions).all() or (np.linalg.norm(directions,axis=1)==0).any():
        raise ValueError('aligned finite nonzero rays required')
    n,b=planes;slack=b[None]-origins@n.T;den=directions@n.T
    ratio=np.zeros_like(den);np.divide(slack,den,out=ratio,where=den!=0)
    lower=np.where(den<0,ratio,-np.inf).max(1)
    upper=np.where(den>0,ratio,np.inf).min(1)
    valid=(lower<upper)&~((den==0)&(slack<=0)).any(1)
    return np.where(valid,lower,np.inf),np.where(valid,upper,-np.inf)


def origin_component_exit(intervals):
    """Union only positive-overlap intervals connected to strict origin interior."""
    intervals=np.asarray(intervals,float)
    if intervals.ndim!=3 or intervals.shape[2]!=2 or intervals.shape[1]<1 or np.isnan(intervals).any():
        raise ValueError('N rays by K operands by 2 interval endpoints required')
    lo,hi=intervals[:,:,0],intervals[:,:,1]
    inside=(lo<0)&(hi>0)
    supported=inside.any(1)
    exit=np.where(inside,hi,-np.inf).max(1)
    # At most K interval extensions; permutation independent fixed-point union.
    for _ in range(intervals.shape[1]):
        reaches=(lo<exit[:,None])&(hi>0)&(lo<hi)
        exit=np.maximum(exit,np.where(reaches,hi,-np.inf).max(1))
    return np.where(supported,exit,np.nan),supported


def declared_union_exit(case,origins,directions):
    intervals=np.stack([np.stack(ray_intervals(origins,directions,prism_planes(
        e['points'],case['half_axes_m'],case['shape_exponent'])),axis=-1)
        for e in case['program']['edges']],axis=1)
    end,supported=origin_component_exit(intervals)
    return end,supported,intervals
