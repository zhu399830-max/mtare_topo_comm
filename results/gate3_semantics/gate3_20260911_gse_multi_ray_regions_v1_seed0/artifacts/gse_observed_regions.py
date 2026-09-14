"""Multi-ray observed-cell components; NOT structural labels or safe paths.

Reuses the existing .25m grid and its conservative return/unknown semantics.
Only full cells inside the10m sphere participate. Source IDs/GT paths are not
inputs. A common component is a resolution-limited evidence proxy: neither
whole-cell free volume nor physical passage connectivity is certified.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from .gse_surface_ray_evidence_v1 import (
    ObservedRayGrid, SHAPE, FREE, OCCUPIED, UNKNOWN, RESOLUTION_M,
    _digest, _point_cells,
)


@dataclass(frozen=True)
class ObservedRegions:
    labels: np.ndarray
    sizes: tuple
    frame_bits: tuple
    grid_sha256: str
    physical_connectivity: bool = False
    training_qualified: bool = False


def build_observed_regions(grid):
    if type(grid) is not ObservedRayGrid or grid.physical_connectivity is not False:
        raise ValueError('original typed observation grid required')
    for a in (grid.state,grid.free_frame_bits,grid.occupied_frame_bits):
        if a.shape!=SHAPE or a.dtype!=np.uint8:raise ValueError('fixed grid contract')
    expected=np.where(grid.occupied_frame_bits!=0,OCCUPIED,
                      np.where(grid.free_frame_bits!=0,FREE,UNKNOWN))
    if (not np.array_equal(expected,grid.state) or np.any(grid.free_frame_bits>31)
        or np.any(grid.occupied_frame_bits>31) or not np.isfinite(grid.numerical_bound_m)
        or grid.numerical_bound_m<0 or _digest(grid.state,grid.free_frame_bits,grid.occupied_frame_bits,
                 np.asarray([grid.numerical_bound_m],dtype='<f8'))!=grid.content_sha256):
        raise ValueError('inconsistent grid evidence')
    # Farthest corner test: never allow a connection to detour outside the ROI.
    centers=-10.+(np.arange(80)+.5)*RESOLUTION_M
    far=np.abs(centers)+RESOLUTION_M/2
    inside=far[:,None,None]**2+far[None,:,None]**2+far[None,None,:]**2<=100.
    free=(grid.state==FREE)&inside
    labels=np.zeros(SHAPE,dtype=np.int32);sizes=[];bits=[]
    offsets=((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))
    for seed in np.argwhere(free):
        seed=tuple(seed)
        if labels[seed]:continue
        number=len(sizes)+1;labels[seed]=number;queue=deque([seed]);size=0;frames=0
        while queue:
            p=queue.popleft();size+=1;frames|=int(grid.free_frame_bits[p])
            for d in offsets:
                q=tuple(p[i]+d[i] for i in range(3))
                if all(0<=x<80 for x in q) and free[q] and not labels[q]:
                    labels[q]=number;queue.append(q)
        sizes.append(size);bits.append(frames)
    labels.setflags(write=False)
    return ObservedRegions(labels,tuple(sizes),tuple(bits),grid.content_sha256)


def query_observed_pair(regions,grid,first_xyz,second_xyz):
    """Evaluation-only association query; does not seed/modify components.

    No nearest-free-cell snapping. Unknown/occupied/boundary ambiguous endpoints
    or disconnected cells are UNKNOWN, never nonmembership/background labels.
    """
    if regions.grid_sha256!=grid.content_sha256:raise ValueError('different source grid')
    ids=[]
    for xyz in (first_xyz,second_xyz):
        p=np.asarray(xyz,dtype=np.float64)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('finite3D endpoints')
        cells=_point_cells(p,grid.numerical_bound_m)
        values={int(regions.labels.flat[c]) for c in cells}
        if len(values)!=1 or not values or 0 in values:
            return {'status':'UNKNOWN','reason':'ENDPOINT_NOT_UNAMBIGUOUSLY_OBSERVED_FREE'}
        ids.append(next(iter(values)))
    if ids[0]!=ids[1]:return {'status':'UNKNOWN','reason':'NO_OBSERVED_CELL_CONNECTION'}
    return {'status':'SAME_OBSERVED_CELL_COMPONENT','component':ids[0],
            'physical_connectivity':False,'training_qualified':False}
