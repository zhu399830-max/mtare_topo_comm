"""Cell-by-cell observation support for a supplied geometric reference path.

Unlike a min/max source-return interval, no unobserved gap is filled. This is
an observation proxy, NEVER a ground path, body clearance, reachability or
semantic-label producer. The reference path is teacher-only and must not enter
student candidate selection. Every touched endpoint/interior cell is retained.
"""
from dataclasses import dataclass

import numpy as np

from mtare_topo.representation.gse_surface_ray_evidence_v1 import (
    FREE, OCCUPIED, query_patch_gaps, _segment_cells, _point_cells, _numerical_bound,
)


@dataclass(frozen=True)
class ObservedAxisSupport:
    status: str
    cell_indices: tuple[int, ...]
    free_cells: int
    occupied_cells: int
    unknown_cells: int
    ambiguous_cells: tuple[int, ...]
    segment_cells: tuple[tuple[int, ...], ...]
    outside_observation_cube: bool
    grid_content_sha256: str
    physical_reachability: None = None
    semantic_anchor: None = None


def observed_axis_support(grid, points_m):
    """Validate grid identity and inspect every finite segment, no resampling.

    OBSERVED_AXIS means every touched discrete cell has observed free evidence
    and no numerical contact ambiguity. It is not a physical connectivity
    assertion. UNKNOWN includes out-of-domain and blocked-reference cases;
    occupied cells are reported but never label a whole opening unreachable.
    """
    # Reuse existing bit/state/hash validation without querying any patch gap.
    query_patch_gaps(grid,np.empty((0,3),dtype=np.float64),
        np.empty((0,8),dtype=np.int64),np.empty((0,8),dtype=bool))
    p=np.asarray(points_m)
    if p.dtype not in (np.float32,np.float64) or p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<2 or not np.isfinite(p).all():
        raise ValueError("at least two finite float3 reference positions required")
    if np.any(np.linalg.norm(np.diff(p.astype(float),axis=0),axis=1)==0):
        raise ValueError("zero length reference segment; no silent pruning")
    tolerance=max(grid.numerical_bound_m,_numerical_bound(p))
    outside=bool(np.any(p < -10.) or np.any(p >= 10.))
    cells=set();ambiguous=set();segments=[]
    for a,b in zip(p[:-1],p[1:]):
        touched,uncertain=_segment_cells(a,b,tolerance)
        for endpoint in (a,b):
            options=_point_cells(endpoint,tolerance)
            touched.update(options)
            if len(options)!=1:uncertain.update(options)
        cells.update(touched);ambiguous.update(uncertain);segments.append(tuple(sorted(touched)))
    ordered=tuple(sorted(cells));state=grid.state.reshape(-1)[list(ordered)]
    free=int(np.count_nonzero(state==FREE));occupied=int(np.count_nonzero(state==OCCUPIED))
    unknown=len(ordered)-free-occupied
    supported=bool(ordered) and not outside and not ambiguous and not occupied and not unknown
    return ObservedAxisSupport("OBSERVED_AXIS" if supported else "UNKNOWN",ordered,free,occupied,unknown,
        tuple(sorted(ambiguous)),tuple(segments),outside,grid.content_sha256)
