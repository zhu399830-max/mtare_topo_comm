"""Exact same-plane shared-edge attribution; does not certify a full ray."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from mtare_topo.evaluation.exact_triangle_ray import exact_triangle_ray


@dataclass(frozen=True)
class ExactHitAttribution:
    original_triangle: int
    valid_triangles: tuple[int, ...]
    exact_t: Fraction
    replaced: bool


def attribute_hit(mesh,triangle_id,origin,direction):
    indices=mesh.triangle_vertex_indices
    if not isinstance(triangle_id,(int,np.integer)) or triangle_id<0 or triangle_id>=len(indices):
        raise ValueError('valid triangle id required')
    triangle_id=int(triangle_id)
    original=exact_triangle_ray(mesh.vertices_xyz_m[indices[triangle_id]],origin,direction)
    if original['status']=='hit':return ExactHitAttribution(triangle_id,(triangle_id,),original['t'],False)
    if original['status']!='outside_triangle' or original['t']<=0:return None
    neighbors=np.flatnonzero(np.isin(indices,indices[triangle_id]).sum(axis=1)==2)
    valid=[]
    for other in neighbors:
        hit=exact_triangle_ray(mesh.vertices_xyz_m[indices[other]],origin,direction)
        if hit['status']=='hit' and hit['plane']==original['plane'] and hit['t']==original['t']:
            valid.append(int(other))
    if not valid:return None
    # Several shared-edge triangles may describe the same proven point;
    # preserve all instead of silently selecting a supposedly unique face.
    return ExactHitAttribution(triangle_id,tuple(valid),original['t'],True)
