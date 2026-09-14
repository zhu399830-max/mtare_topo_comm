"""Conservative, opt-in rescue of missing mesh-union exits.

This is a numerical mesh interpretation, not a physical safety certificate.
No spatial grouping tolerance is changed. Noninteger, surface, precision-
disagreeing, or over-capacity interval evidence produces no rescued return.
"""
import numpy as np
from mtare_topo.evaluation.mesh_winding_diagnostic import oriented_winding


def interval_winding_exit(meshes, origin, direction, initial_inside, distances,
                          source_operands, *, maximum_m=50.0):
    origin=np.asarray(origin,dtype=float);direction=np.asarray(direction,dtype=float)
    direction=direction/np.linalg.norm(direction)
    ts=np.asarray(distances,dtype=float)
    if not len(ts) or not np.isfinite(ts).all() or np.any(ts<=0):return None
    unique=np.unique(ts)
    # Bounded diagnostic fallback: never truncate a ray's intersections.
    if len(unique)>128:return None
    bounds=np.r_[0.,unique,unique[-1]+max(1.,abs(unique[-1]))]
    mids=(bounds[:-1]+bounds[1:])/2
    parameters=np.r_[0.,mids]
    packed_origin=origin.astype(np.float32).astype(float)
    packed_direction=direction.astype(np.float32).astype(float)
    occupancies=[]
    for mesh in meshes:
        states=[]
        for packed in (False,True):
            vertices=mesh.vertices_xyz_m
            points=origin+parameters[:,None]*direction
            if packed:
                vertices=vertices.astype(np.float32).astype(float)
                points=packed_origin+parameters[:,None]*packed_direction
            try:w=oriented_winding(vertices[mesh.triangle_vertex_indices],points)
            except ValueError:return None
            rounded=np.rint(w)
            # Numerical integrality check, not a spatial/safety threshold.
            if np.any(np.abs(w-rounded)>1e-6) or np.any(rounded<0):return None
            states.append(rounded.astype(np.int64))
        if not np.array_equal(states[0],states[1]):return None
        occupancies.append(states[0]>0)
    inside=np.asarray(occupancies).T
    if not np.array_equal(inside[0],np.asarray(initial_inside,dtype=bool)):return None
    union=inside[1:].any(axis=1)
    if not union[0]:return None
    for i,t in enumerate(unique):
        if t>maximum_m:break
        if union[i] and not union[i+1]:
            leaving=tuple(int(x) for x in np.flatnonzero(inside[i+1]&~inside[i+2]))
            raw_sources=set(int(source_operands[j]) for j in np.flatnonzero(ts==t))
            if not leaving or not set(leaving).issubset(raw_sources):return None
            return float(t),leaving
    return None
