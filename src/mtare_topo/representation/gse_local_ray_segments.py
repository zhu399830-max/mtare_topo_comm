"""Observed first-return segments inside the fixed local10m sphere.

Student-only geometry: no construction IDs, teacher openings or support masks.
A segment ending at the crop boundary is NOT an observed surface or opening.
Invalid returns give no free-space assertion. No traversability is inferred.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalRaySegments:
    ray_indices: np.ndarray
    start_xyz_m: np.ndarray
    end_xyz_m: np.ndarray
    original_return_xyz_m: np.ndarray
    end_is_observed_return: np.ndarray
    start_distance_from_origin_m: np.ndarray
    end_distance_from_origin_m: np.ndarray
    crop_endpoint_is_opening: bool = False
    physical_traversability_qualified: bool = False


def clip_observed_rays(origins_xyz_m, returns_xyz_m, valid):
    origins=np.asarray(origins_xyz_m,dtype=np.float64)
    returns=np.asarray(returns_xyz_m,dtype=np.float64)
    mask=np.asarray(valid)
    if origins.ndim!=2 or origins.shape[1:]!=(3,) or returns.shape!=origins.shape or mask.shape!=(len(origins),) or mask.dtype!=np.bool_:
        raise ValueError('Nx3 origins/returns and boolean validity required')
    if not np.isfinite(origins).all() or not np.isfinite(returns).all():raise ValueError('finite registered ray coordinates required')
    indices=np.flatnonzero(mask);o=origins[indices];r=returns[indices]
    length=np.linalg.norm(r-o,axis=1)
    if np.any(length<=0):raise ValueError('valid return cannot coincide with origin')
    direction=(r-o)/length[:,None]
    along=-np.einsum('ij,ij->i',o,direction)
    closest=o+along[:,None]*direction
    discriminant=100.-np.einsum('ij,ij->i',closest,closest)
    # Tangency has no positive-length observed interior. Do not pad it.
    offset=np.sqrt(np.maximum(discriminant,0.))
    lo=np.maximum(0.,along-offset);hi=np.minimum(length,along+offset)
    keep=(discriminant>0)&(hi>lo)
    ids=indices[keep];start=o[keep]+lo[keep,None]*direction[keep]
    end=o[keep]+hi[keep,None]*direction[keep]
    observed=(np.linalg.norm(r[keep],axis=1)<=10.)&(hi[keep]==length[keep])
    # Use actual stored return coordinates for a measured endpoint.
    end[observed]=r[keep][observed]
    arrays=[ids,start,end,r[keep].copy(),observed,lo[keep],hi[keep]]
    for a in arrays:a.setflags(write=False)
    return LocalRaySegments(*arrays)
