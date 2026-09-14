"""Ambiguity-preserving diagnostic only; never a complete instance teacher."""
import numpy as np


def diagnostic_readout(partition,support,reference_components=()):
    n=len(partition.geometry_components)
    rays=np.asarray(support.ray_component)
    if rays.dtype.kind not in 'iu' or np.any(rays<-1) or np.any(rays>=n):raise ValueError('invalid support assignment')
    counts=np.bincount(rays[rays>=0],minlength=n)
    if not np.array_equal(counts,support.component_ray_counts):raise ValueError('support counters differ from ray evidence')
    if partition.conflicting_cells:raise ValueError('geometry/observation conflicts cannot yield candidates')
    adjacency=[set() for _ in range(n)]
    for a,b in partition.unresolved_component_pairs:
        if not 0<=a<b<n:raise ValueError('invalid unresolved component pair')
        adjacency[a].add(b);adjacency[b].add(a)
    for r in reference_components:
        if r is not None and (type(r) is not int or not 0<=r<n):raise ValueError('invalid reference bookkeeping')
    remaining=set(range(n));groups=[]
    while remaining:
        seed=min(remaining);remaining.remove(seed);stack=[seed];members=[]
        while stack:
            i=stack.pop();members.append(i)
            for j in sorted(adjacency[i]):
                if j in remaining:remaining.remove(j);stack.append(j)
        members=sorted(members);witnesses=int(counts[members].sum());uncertain=len(members)>1
        status=('SUPPORTED_UNRESOLVED_COMPONENT_SET' if uncertain else 'SUPPORTED_ISOLATED_GEOMETRY_COMPONENT') if witnesses else 'NO_POSITIVE_OBSERVATION_EVIDENCE'
        groups.append(dict(component_indices=members,status=status,support_ray_count=witnesses,
            reference_seed_indices=[i for i,r in enumerate(reference_components) if r in members],
            independent_instance_count=None))
    return dict(schema='aperture_component_diagnostic_v1',groups=groups,
        supported_isolated_components=sum(g['status']=='SUPPORTED_ISOLATED_GEOMETRY_COMPONENT' for g in groups),
        supported_unresolved_sets=sum(g['status']=='SUPPORTED_UNRESOLVED_COMPONENT_SET' for g in groups),
        unmapped_reference_seed_indices=[i for i,r in enumerate(reference_components) if r is None],
        unassigned_crossing_rays=int(support.ambiguous_crossings),
        complete_instance_count=None,training_labels_qualified=False)


def reference_seed_components(points_current_m,vertices,faces,partition,*,radius_m=10.):
    """Map only a supplied section's SEED position, never its entire footprint.

    This is reference bookkeeping, not observed evidence. All incident cell
    possibilities must agree; no nearest-center assignment is made.
    """
    points=np.asarray(points_current_m,dtype=float).reshape(-1,3)
    if not np.isfinite(points).all():raise ValueError('finite reference seeds required')
    tri=np.asarray(vertices)[np.asarray(faces)];owner=np.full(len(tri),-1,dtype=int)
    for i,cells in enumerate(partition.geometry_components):owner[list(cells)]=i
    normals=[]
    for a,b,c in ((0,1,2),(1,2,0),(2,0,1)):
        normal=np.cross(tri[:,a],tri[:,b]);normal/=np.linalg.norm(normal,axis=1)[:,None]
        normal[np.einsum('ij,ij->i',normal,tri[:,c])<0]*=-1;normals.append(normal)
    result=[];eps=128*np.finfo(float).eps
    for p in points:
        length=np.linalg.norm(p)
        if abs(length-radius_m)>eps*max(1.,radius_m):result.append(None);continue
        direction=p/length;inside=np.ones(len(tri),dtype=bool)
        for normal in normals:inside&=normal@direction>=-eps
        ids=np.unique(owner[inside]);result.append(int(ids[0]) if len(ids)==1 and ids[0]>=0 else None)
    return tuple(result)
