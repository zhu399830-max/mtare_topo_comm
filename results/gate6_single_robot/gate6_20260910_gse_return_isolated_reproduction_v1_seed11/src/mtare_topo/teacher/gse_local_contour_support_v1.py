"""Local surface support of a reference aperture contour, independent of arms.

Fixed .25m observation cells, no reference-axis extension requirement. This
produces a geometric support mask, not a semantic or physical safety label.
All segments contribute; sparse vertices alone cannot certify a whole loop.
"""
import itertools
import numpy as np

STEP=.25


def _cells(point):
    scaled=np.asarray(point,dtype=np.longdouble)/STEP
    options=[]
    for x in scaled:
        # Exact boundary belongs to both incident cells. No tuned tolerance.
        i=int(np.floor(x));options.append((i-1,i) if x==i else (i,))
    return set(itertools.product(*options))


def contour_surface_support(loop_m, points_m, valid):
    loop=np.asarray(loop_m);points=np.asarray(points_m);valid=np.asarray(valid)
    if loop.dtype.kind!='f' or loop.ndim!=2 or loop.shape[1:]!=(3,) or len(loop)<3 or not np.isfinite(loop).all():
        raise ValueError('finite closed contour vertices required')
    if points.dtype.kind!='f' or points.ndim!=2 or points.shape[1:]!=(3,) or valid.shape!=(len(points),) or valid.dtype!=bool:
        raise ValueError('observed points and explicit validity required')
    if len(points)>57600 or not np.isfinite(points[valid]).all():raise ValueError('bounded causal point set required')
    if len(set(map(tuple,loop.tolist())))!=len(loop):raise ValueError('distinct unclosed vertex list required')
    # Avoid silently filling an observed boundary cell's neighbors: a return
    # exactly on a plane is ambiguous and supplies no definite surface cell.
    observed={}
    for i in np.flatnonzero(valid):
        cells=_cells(points[i])
        if len(cells)==1:observed.setdefault(next(iter(cells)),[]).append(int(i))
    required=set();segment_cells=[]
    for a,b in zip(loop,np.roll(loop,-1,axis=0)):
        a=a.astype(np.longdouble);b=b.astype(np.longdouble);d=b-a
        if not np.any(d):raise ValueError('zero contour segment')
        ts=[np.longdouble(0),np.longdouble(1)]
        for k in range(3):
            if d[k]==0:continue
            lo,hi=sorted((a[k],b[k]));nlo=int(np.floor(lo/STEP));nhi=int(np.ceil(hi/STEP))
            if nhi-nlo>4096:raise ValueError('contour segment capacity exceeded')
            t=(np.arange(nlo,nhi+1,dtype=np.longdouble)*STEP-a[k])/d[k]
            ts.extend(t[(t>0)&(t<1)])
        ts=np.unique(ts);cells=set()
        for t in (ts[:-1]+ts[1:])/2:cells.update(_cells(a+t*d))
        cells.update(_cells(a));cells.update(_cells(b));required.update(cells)
        segment_cells.append(sorted(cells))
    missing=sorted(required-set(observed))
    return dict(reference_contour_cells=sorted(required),unobserved_contour_cells=missing,
        supported_cell_point_indices=[dict(cell=c,point_indices=observed[c]) for c in sorted(required&set(observed))],
        segment_cells=segment_cells,contour_fully_surface_supported=bool(required) and not missing,
        semantic_opening_label=None,anchor_label=None,physical_reachable=None,
        limitation='Observed cell support is a discretized surface proxy, not aperture existence, complete background, or robot clearance. Combine with finite-ray evidence before proposing labels.')
