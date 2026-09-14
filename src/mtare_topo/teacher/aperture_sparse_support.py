"""Sparse causal first-return witnesses, never whole-cell visibility labels."""
from dataclasses import dataclass
import numpy as np
from .gse_portal_ray_evidence import CausalRaySegments


@dataclass(frozen=True)
class SparseBoundarySupport:
    crossing_before_return: np.ndarray
    ray_component: np.ndarray
    component_ray_counts: np.ndarray
    ambiguous_crossings: int
    whole_cells_observed: bool=False
    complete_background: bool=False


def window_crossing_support(rays:CausalRaySegments,vertices,faces,geometry_state,partition,
                            *,center_m=(0.,0.,0.),radius_m=10.,chunk_size=256):
    rays.validate()
    center=np.asarray(center_m,dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(radius_m) or radius_m<=0:
        raise ValueError('finite window sphere required')
    if type(chunk_size) is not int or not 1<=chunk_size<=256:raise ValueError('bounded chunk required')
    v=np.asarray(vertices,dtype=float);f=np.asarray(faces,dtype=int);state=np.asarray(geometry_state)
    if state.shape!=(len(f),) or not np.isin(state,[-1,0,1]).all():raise ValueError('cell states required')
    owner=np.full(len(f),-1,dtype=int)
    for i,component in enumerate(partition.geometry_components):
        for cell in component:
            if state[cell]!=1 or owner[cell]!=-1:raise ValueError('partition differs from free geometry cells')
            owner[cell]=i
    if not np.array_equal(owner>=0,state==1):raise ValueError('incomplete geometry partition')
    offset=rays.origins_m-center;direction=rays.directions
    b=np.einsum('ij,ij->i',offset,direction);c=np.einsum('ij,ij->i',offset,offset)-radius_m**2
    eps=128*np.finfo(float).eps
    # Only origins strictly inside this local window provide the outgoing
    # crossing used here; outside/on-sphere starts remain unqualified.
    discriminant=b*b-c
    distance=-b+np.sqrt(np.maximum(discriminant,0))
    stored=rays.first_return_m
    limit=stored.astype(float)-np.abs(np.spacing(stored)).astype(float)-rays.range_error_bound_m
    guard=eps*np.maximum(1.,np.abs(distance)+np.linalg.norm(offset,axis=1)+radius_m)
    eligible=rays.valid&(c<-eps*radius_m**2)&(discriminant>0)&(distance>guard)&(distance+guard<limit)
    assigned=np.full(len(stored),-1,dtype=int);tri=v[f]
    edge_normals=[]
    for a,bidx,opposite in ((0,1,2),(1,2,0),(2,0,1)):
        n=np.cross(tri[:,a],tri[:,bidx]);norm=np.linalg.norm(n,axis=1)
        if np.any(norm==0):raise ValueError('degenerate spherical triangle')
        n/=norm[:,None];n[np.einsum('ij,ij->i',n,tri[:,opposite])<0]*=-1
        edge_normals.append(n)
    indices=np.flatnonzero(eligible)
    for start in range(0,len(indices),chunk_size):
        ids=indices[start:start+chunk_size]
        points=offset[ids]+distance[ids,None]*direction[ids]
        points/=np.linalg.norm(points,axis=1)[:,None]
        contains=np.ones((len(ids),len(f)),dtype=bool)
        for n in edge_normals:contains&=points@n.T>=-eps
        for row,index in enumerate(ids):
            cells=np.flatnonzero(contains[row]);groups=np.unique(owner[cells])
            # Every adjacent possibility must be known free and in the SAME
            # component. Unknown bands cannot be assigned by nearest center.
            if len(cells) and len(groups)==1 and groups[0]>=0:assigned[index]=groups[0]
    counts=np.bincount(assigned[assigned>=0],minlength=len(partition.geometry_components))
    return SparseBoundarySupport(eligible,assigned,counts,int(np.sum(eligible&(assigned<0))))
