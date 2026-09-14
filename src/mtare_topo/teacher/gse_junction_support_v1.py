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


def junction_support(grid, *, anchor_m, incident_paths_m, endpoint_keys):
    """All supplied incident paths start at the same 3D reference anchor.

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
    supports=[]; directions=[]
    for points in incident_paths_m:
        p=np.asarray(points)
        if p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<2 or not np.array_equal(p[0],anchor):
            raise ValueError("path must begin at exact reference anchor; no nearest-node snapping")
        support=observed_axis_support(grid,p);supports.append(support)
        direction=p[1].astype(float)-anchor
        directions.append(direction/np.linalg.norm(direction))
    reason="ALL_INCIDENT_AXIS_PROXIES_OBSERVED"
    status="REFERENCE_JUNCTION_SUPPORTED"
    if len(supports)<3:
        status,reason="UNKNOWN","NOT_A_JUNCTION_REFERENCE_NO_TERMINAL_OR_CORRIDOR_LABEL"
    elif len({tuple(d) for d in directions})!=len(directions):
        status,reason="UNKNOWN","COINCIDENT_INITIAL_ARMS_NOT_INDEPENDENT_OBSERVATIONS"
    elif any(s.status!="OBSERVED_AXIS" for s in supports):
        status,reason="UNKNOWN","INCIDENT_OBSERVATION_SUPPORT_INCOMPLETE"
    return JunctionSupport(status,reason,tuple(map(float,anchor)),tuple(supports),grid.content_sha256)
