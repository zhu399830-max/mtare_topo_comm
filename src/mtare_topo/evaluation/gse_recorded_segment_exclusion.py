"""Conservative mesh-AABB exclusion of recorded finite ray segments.

Not a sensor-noise bound, general occlusion detector, or whole-world equivalence
certificate. Failure to separate means unproved, not necessarily intersecting.
"""
import numpy as np
from .gse_source_plane_intervals import _hull,_mul,_add


def recorded_segment_exclusion(vertices,origins,directions,ranges,valid):
    vertices=np.asarray(vertices);origins=np.asarray(origins);directions=np.asarray(directions)
    ranges=np.asarray(ranges);valid=np.asarray(valid)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not len(vertices)
        or origins.ndim!=2 or origins.shape[1:]!=(3,) or not len(origins)
        or directions.shape!=origins.shape or ranges.shape!=(len(origins),)
        or ranges.dtype!=np.float32 or not np.isfinite(ranges).all() or np.any(ranges<0)
        or valid.shape!=ranges.shape or valid.dtype!=np.bool_
        or np.any(np.linalg.norm(directions,axis=1)==0)):
        raise ValueError('finite source vertices and nonempty recorded ray arrays required')
    vl,vh=_hull(vertices);boxlo=np.nextafter(vl.min(axis=0),-np.inf);boxhi=np.nextafter(vh.max(axis=0),np.inf)
    origin=_hull(origins);direction=_hull(directions)
    t=ranges.astype(np.float64)[:,None]
    end=_add(origin,_mul((t,t),direction))
    lo=np.minimum(origin[0],end[0]);hi=np.maximum(origin[1],end[1])
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():raise ValueError('segment bound overflow')
    separated=np.any((hi<boxlo)|(lo>boxhi),axis=1)&valid
    gap=np.maximum(boxlo-hi,lo-boxhi).max(axis=1)
    return dict(separated=separated,unproved=~separated,
        mesh_bounds_m=[boxlo.tolist(),boxhi.tolist()],
        minimum_separating_axis_gap_m=float(gap[separated].min()) if separated.any() else None,
        all_recorded_segments_excluded=bool(separated.all()),
        full_sensor_equivalence_certified=False)
