"""Declared convex-tube occlusion control, NOT an archived-scan certification.

Strictly disjoint bounding boxes suffice (not necessary) for operand isolation.
Every ray starting inside the convex main tube must exit its boundary before
it can meet the other tube. No production target/mesh/SDF call is used.
"""
from copy import deepcopy
import numpy as np
from .gse_synthetic_field_scoring import expected_geometry


def convex_main_exit(case, origins, directions):
    """Exact halfspace exit of declared straight 64-gon prism; returns ray t."""
    separated_tube_expectation(case)
    origins=np.asarray(origins,dtype=float);directions=np.asarray(directions,dtype=float)
    if origins.shape!=directions.shape or origins.ndim!=2 or origins.shape[1]!=3:
        raise ValueError('aligned N-by-3 rays required')
    if not np.isfinite(origins).all() or not np.isfinite(directions).all() or (np.linalg.norm(directions,axis=1)==0).any():
        raise ValueError('finite nonzero rays required')
    theta=np.arange(64)*2*np.pi/64;ay,az=case['half_axes_m'];power=case['shape_exponent']
    polygon=np.stack((ay*np.sign(np.cos(theta))*np.abs(np.cos(theta))**(2/power),
                      az*np.sign(np.sin(theta))*np.abs(np.sin(theta))**(2/power)),1)
    edge=np.roll(polygon,-1,axis=0)-polygon
    normals=np.column_stack((np.zeros(64),edge[:,1],-edge[:,0]))
    bounds=(normals[:,1:]*polygon).sum(1)
    a,b=case['program']['edges'][0]['points']
    normals=np.vstack((normals,[1,0,0],[-1,0,0]));bounds=np.r_[bounds,b[0],-a[0]]
    slack=bounds[None]-origins@normals.T
    if (slack<=0).any():raise ValueError('ray origin not strictly inside prism')
    denominator=directions@normals.T
    times=np.full_like(denominator,np.inf)
    np.divide(slack,denominator,out=times,where=denominator>0)
    return times.min(1)


def separated_tube_expectation(case):
    if case['program']['type'] not in ('parallel','stacked'):
        raise ValueError('only declared parallel/stacked straight tubes supported')
    edges=case['program']['edges']
    if len(edges)!=2:raise ValueError('exactly two operands required')
    main,other=(np.asarray(e['points'],dtype=float) for e in edges)
    if main.shape!=(2,3) or other.shape!=(2,3) or not np.isfinite([main,other]).all():
        raise ValueError('finite two-point tubes required')
    if not np.array_equal(main[:,1:],np.zeros((2,2))) or main[1,0]<=main[0,0]:
        raise ValueError('main must be an increasing X-axis tube')
    delta=other-main
    if not np.array_equal(delta[0],delta[1]) or delta[0,0]!=0:
        raise ValueError('other tube must be a transverse translation')
    axes=np.asarray(case['half_axes_m'],dtype=float)
    if axes.shape!=(2,) or not np.isfinite(axes).all() or (axes<=0).any():
        raise ValueError('positive finite half axes required')
    power=float(case['shape_exponent'])
    if not np.isfinite(power) or power<1:raise ValueError('convex section required')
    gaps=np.abs(delta[0,1:])-2*axes
    if not (gaps>0).any():raise ValueError('bounding boxes touch/overlap: no isolation proof')
    # Reuse independently declared convex straight-tube expectation. It checks
    # every causal pose against the actual 64-gon and finite axial interval.
    direct=deepcopy(case);direct['program']['type']='straight';direct['program']['edges']=[edges[0]]
    result=expected_geometry(direct)
    return dict(**result,scope='declared_geometry_only',archived_scan_verified=False,
                isolated_axis_gaps_m=gaps.tolist(),other_operand_observable=False,
                teacher_qualified=False,training_eligible=False)
