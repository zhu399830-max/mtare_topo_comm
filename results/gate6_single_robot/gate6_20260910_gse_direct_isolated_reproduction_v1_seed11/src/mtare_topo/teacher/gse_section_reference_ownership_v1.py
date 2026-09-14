"""Geometric ownership of an axis-plane intersection, never visibility truth."""
import numpy as np


def section_reference_owner(loops_m, *, reference_m, normal):
    center=np.asarray(reference_m,dtype=np.float64);n=np.asarray(normal,dtype=np.float64)
    if center.shape!=(3,) or n.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-12:
        raise ValueError('finite reference and unit normal required')
    axis=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u)
    inside=[];boundary=[];eps=128*np.finfo(float).eps
    for index,loop in enumerate(loops_m):
        loop=np.asarray(loop,dtype=np.float64)
        if loop.ndim!=2 or loop.shape[1:]!=(3,) or len(loop)<3 or not np.isfinite(loop).all():
            raise ValueError('finite section polygon required')
        delta=loop-center
        if np.any(np.abs(delta@n)>eps*np.maximum(1.,np.linalg.norm(delta,axis=1))):
            raise ValueError('reference and loop are not coplanar')
        polygon=np.stack((delta@u,delta@v),axis=1);within=False;touch=False
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            edge=b-a;rel=-a;length2=float(edge@edge)
            if length2==0:raise ValueError('zero length section edge')
            cross=edge[0]*rel[1]-edge[1]*rel[0];projection=float(rel@edge)
            guard=eps*max(1.,float(np.linalg.norm(rel))*np.sqrt(length2))
            touch|=abs(cross)<=guard and -guard<=projection<=length2+guard
            if edge[1]!=0:
                crossing_x=a[0]-a[1]*edge[0]/edge[1]
                within^=bool((a[1]>0)!=(b[1]>0) and 0<crossing_x)
        if touch:boundary.append(index)
        elif within:inside.append(index)
    owner=inside[0] if len(inside)==1 and not boundary else None
    return dict(owner_loop_index=owner,containing_loop_indices=inside,boundary_loop_indices=boundary,
                status='UNIQUE_AXIS_REFERENCE_LOOP' if owner is not None else 'UNKNOWN_AXIS_REFERENCE_OWNERSHIP',
                semantic_opening=False)
