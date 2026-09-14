"""Exact triangle-plane section topology, without a convex-hull replacement.

Mesh edge identities join intersection segments; proximity never merges loops.
Coplanar/vertex contacts fail explicitly rather than moving the section plane.
This is source geometry only: loops are not visible or traversable openings.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshSection:
    loops_m: tuple[np.ndarray, ...]
    source_edges: tuple[tuple[tuple[int, int], ...], ...]
    intersected_triangles: tuple[int, ...]


def mesh_section(vertices_m, triangles, *, center_m, normal):
    vertices = np.asarray(vertices_m,dtype=np.float64)
    faces = np.asarray(triangles)
    center = np.asarray(center_m,dtype=np.float64); n = np.asarray(normal,dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("finite mesh vertices required")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or faces.dtype.kind not in "iu" or (faces<0).any() or (faces>=len(vertices)).any():
        raise ValueError("valid integer mesh triangles required")
    if center.shape != (3,) or n.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-12:
        raise ValueError("finite center and unit plane normal required")
    relative = vertices-center
    distance = relative@n
    tolerance = 128*np.finfo(np.float64).eps*np.maximum(1.,np.linalg.norm(relative,axis=1))
    adjacency, coordinates, used = {}, {}, []
    for index, face in enumerate(faces):
        if len(set(map(int,face))) != 3:
            raise ValueError("degenerate source face")
        d = distance[face]; tol = tolerance[face]
        if (d>tol).all() or (d < -tol).all():
            continue
        if (np.abs(d)<=tol).any():
            raise ValueError("section intersects source vertex/edge; unresolved geometry, no plane shift")
        keys = []
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            if (distance[a]>0) == (distance[b]>0):
                continue
            key = tuple(sorted((int(a),int(b))))
            left,right = key
            ratio = distance[left]/(distance[left]-distance[right])
            coordinates[key] = vertices[left]+ratio*(vertices[right]-vertices[left])
            keys.append(key)
        if len(keys)!=2:
            raise ValueError("ambiguous section face")
        a,b = keys
        if b in adjacency.get(a,[]):
            raise ValueError("duplicate section segment/nonmanifold source")
        adjacency.setdefault(a,[]).append(b); adjacency.setdefault(b,[]).append(a)
        used.append(index)
    if any(len(v)!=2 for v in adjacency.values()):
        raise ValueError("open or nonmanifold section; do not close gaps")
    unseen = set(adjacency); loops, origins = [], []
    while unseen:
        first = min(unseen); previous = None; current = first; keys = []
        while True:
            if current not in unseen:
                if current != first:
                    raise ValueError("section revisits another loop")
                break
            unseen.remove(current); keys.append(current)
            choices = sorted(x for x in adjacency[current] if x!=previous)
            previous,current = current,choices[0]
        if len(keys)<3:
            raise ValueError("degenerate section cycle")
        points = np.array([coordinates[key] for key in keys]); points.setflags(write=False)
        loops.append(points); origins.append(tuple(keys))
    return MeshSection(tuple(loops),tuple(origins),tuple(used))


def section_ray_witnesses(vertices_m, triangles, *, center_m, normal, rays, exclusive=True):
    """Return positive-only exclusive ray indices per exact mesh loop.

    Concave loops are not replaced with a convex hull. Rays crossing multiple
    loops (including nested loops/holes) are ambiguous and witness none.
    No return, boundary contact and occluded sections stay unknown. A witness
    does not establish that the source loop is a semantic or navigable opening.
    """
    if type(exclusive) is not bool:
        raise ValueError("explicit boolean competition mode required")
    rays.validate()
    section = mesh_section(vertices_m,triangles,center_m=center_m,normal=normal)
    center = np.asarray(center_m,dtype=float); n = np.asarray(normal,dtype=float)
    # Deterministic orthonormal basis, independent of world axes up assumption.
    axis = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n,axis); u /= np.linalg.norm(u); v = np.cross(n,u)
    numerator = (center-rays.origins_m)@n
    denominator = rays.directions@n
    eps = 128*np.finfo(float).eps
    scale = np.maximum(1.,np.linalg.norm(center-rays.origins_m,axis=1))
    distance = np.divide(numerator,denominator,out=np.zeros_like(numerator),where=np.abs(denominator)>eps)
    limit = rays.first_return_m.astype(float)-np.abs(np.spacing(rays.first_return_m)).astype(float)-rays.range_error_bound_m
    eligible = (rays.valid & (np.abs(denominator)>eps) & (np.abs(numerator)>eps*scale)
                & (distance>0) & (distance+eps*scale<limit))
    selected = np.flatnonzero(eligible)
    point = rays.origins_m[selected]+distance[selected,None]*rays.directions[selected]-center
    xy = np.stack([point@u,point@v],axis=1)
    hits = []; any_boundary = np.zeros(len(xy),dtype=bool)
    for loop in section.loops_m:
        delta = loop-center
        polygon = np.stack([delta@u,delta@v],axis=1)
        inside = np.zeros(len(xy),dtype=bool); boundary = inside.copy()
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            edge = b-a; rel = xy-a
            length2 = float(edge@edge)
            if length2 == 0:
                raise ValueError("zero length section edge")
            cross = edge[0]*rel[:,1]-edge[1]*rel[:,0]
            projection = rel@edge
            guard = eps*np.maximum(1.,np.linalg.norm(rel,axis=1)*np.sqrt(length2))
            boundary |= (np.abs(cross)<=guard)&(projection>=-guard)&(projection<=length2+guard)
            if edge[1] != 0:
                crossing_x = a[0]+(xy[:,1]-a[1])*edge[0]/edge[1]
                inside ^= ((a[1]>xy[:,1])!=(b[1]>xy[:,1])) & (xy[:,0]<crossing_x)
        any_boundary |= boundary
        hits.append(inside & ~boundary)
    if not hits:
        return section, ()
    matrix = np.asarray(hits)
    unique = ((matrix.sum(axis=0)==1) if exclusive else np.ones(len(xy),dtype=bool)) & ~any_boundary
    witnesses = tuple(tuple(map(int,selected[row & unique])) for row in matrix)
    return section,witnesses
