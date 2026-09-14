"""Reference-junction observability diagnostic, not a training label factory.

Construction endpoints are loss/evaluation provenance only. The geometry and
all branch paths must come from a source-bound producer. This rejects ordinary
degree2 subdivisions, unsupported arms and duplicate geometric arms. It does
not establish aperture boundary completeness, terminal caps or robot passage.
"""
from dataclasses import dataclass

import numpy as np

from mtare_topo.teacher.gse_observed_axis_support_v1 import observed_axis_support


@dataclass(frozen=True)
class JunctionSupport:
    status: str
    reason: str
    reference_anchor_m: tuple[float,float,float]
    branch_support: tuple
    grid_content_sha256: str
    semantic_label: None = None
    human_reviewed: bool = False


def junction_support(grid, *, anchor_m, incident_paths_m, endpoint_keys, axis_start_indices):
    """All supplied incident paths start at the same 3D reference anchor.

    axis_start_indices are source-adapter provenance, not estimated by length.
    Auxiliary connector cells remain in the unchanged observation checks.
    Only the physical source axis segment determines branch direction.

    At least three independently referenced and geometrically distinct arms
    must have full cell evidence. Missing arms remain UNKNOWN, not a lower
    degree label. Endpoints must stay individual even on one physical tunnel.
    A positive result is still a reference-conditioned observability proxy;
    it must not be fed to a model as its candidate set or auto-certified label.
    """
    anchor=np.asarray(anchor_m,dtype=np.float64)
    if anchor.shape!=(3,) or not np.isfinite(anchor).all():
        raise ValueError("finite 3D reference anchor required")
    if type(incident_paths_m) is not tuple or type(endpoint_keys) is not tuple or not incident_paths_m or len(incident_paths_m)!=len(endpoint_keys):
        raise ValueError("complete explicit incident path/endpoint inventory required")
    for key in endpoint_keys:
        if type(key) is not tuple or len(key)!=2 or type(key[0]) is not str or not key[0] or type(key[1]) is not int or key[1] not in (0,1):
            raise ValueError("teacher endpoint key must retain edge and side")
    if len(set(endpoint_keys))!=len(endpoint_keys):
        raise ValueError("duplicate incidence, no artificial degree inflation")
    if type(axis_start_indices) is not tuple or len(axis_start_indices)!=len(incident_paths_m):
        raise ValueError("source-bound axis segment indices required")
    supports=[]; directions=[]
    for points, axis_start in zip(incident_paths_m, axis_start_indices):
        p=np.asarray(points)
        if p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<2 or not np.array_equal(p[0],anchor):
            raise ValueError("path must begin at exact reference anchor; no nearest-node snapping")
        support=observed_axis_support(grid,p);supports.append(support)
        if type(axis_start) is not int or axis_start not in (0,1) or axis_start+1>=len(p):
            raise ValueError("invalid source axis segment index")
        direction=p[axis_start+1].astype(float)-p[axis_start].astype(float)
        directions.append(direction/np.linalg.norm(direction))
    reason="ALL_INCIDENT_AXIS_PROXIES_OBSERVED"
    status="REFERENCE_JUNCTION_SUPPORTED"
    if len(supports)<3:
        status,reason="UNKNOWN","NOT_A_JUNCTION_REFERENCE_NO_TERMINAL_OR_CORRIDOR_LABEL"
    elif len({tuple(d) for d in directions})!=len(directions):
        status,reason="UNKNOWN","COINCIDENT_SOURCE_AXIS_DIRECTIONS_NOT_INDEPENDENT_OBSERVATIONS"
    elif any(s.status!="OBSERVED_AXIS" for s in supports):
        status,reason="UNKNOWN","INCIDENT_OBSERVATION_SUPPORT_INCOMPLETE"
    return JunctionSupport(status,reason,tuple(map(float,anchor)),tuple(supports),grid.content_sha256)
