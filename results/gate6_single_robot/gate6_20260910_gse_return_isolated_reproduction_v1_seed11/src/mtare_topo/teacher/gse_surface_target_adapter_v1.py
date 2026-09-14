"""Loss-only interchange for observed structures; no annotation or source IO.

Unlike legacy review records, openings are independent of anchors. Membership
is tri-state per pair. This observed-only adapter never produces physical root
reach supervision. Evidence strings are producer assertions, not proof of human
review or geometric validity; real producers require a separately frozen scope.
"""
import math

import torch

from mtare_topo.representation.gse_surface_losses_v1 import SurfaceLossTargets


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError("unexpected or missing observed-target fields")


def _xyz(value):
    if type(value) is not list or len(value) != 3 or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in value
    ):
        raise ValueError("finite XYZ triple required")


def _evidence(value):
    if type(value) is not str or not value.strip():
        raise ValueError("explicit observation evidence required")


def observed_targets(records, *, device="cpu"):
    """Convert a batch of explicit labels to detached float32 masked targets.

    Null attributes stay unknown. Empty lists never imply complete background.
    No default scoring region/completeness is inferred from sensor range.
    Labels use current_sensor_m; construction IDs are deliberately absent.
    """
    if type(records) is not list or not records:
        raise ValueError("nonempty observation batch required")
    for record in records:
        _keys(record, "schema coordinate_frame source_frame_indices anchors openings membership score_region")
        if record["schema"] != "gse_surface_observed_targets_v1" or record["coordinate_frame"] != "current_sensor_m":
            raise ValueError("versioned current sensor labels required")
        frames = record["source_frame_indices"]
        if type(frames) is not list or len(frames) != 5 or any(type(x) is not int or x < 0 for x in frames) or sorted(set(frames)) != frames:
            raise ValueError("five distinct causal frame indices required")
        for kind, capacity in (("anchors", 32), ("openings", 64)):
            entries = record[kind]
            if type(entries) is not list:
                raise ValueError("explicit target lists required")
            if len(entries) > capacity:
                raise OverflowError("target capacity exceeded; never truncate")
            for item in entries:
                _keys(item, "position_m evidence" if kind == "anchors" else "position_m direction width_m height_m evidence")
                _xyz(item["position_m"]); _evidence(item["evidence"])
                if kind == "openings":
                    if item["direction"] is not None:
                        _xyz(item["direction"])
                        if abs(math.hypot(*item["direction"]) - 1) > 1e-6:
                            raise ValueError("known direction must be unit length")
                    for key in ("width_m", "height_m"):
                        v = item[key]
                        if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v <= 0):
                            raise ValueError("known dimensions must be positive")
            if len({tuple(x["position_m"]) for x in entries}) != len(entries):
                raise ValueError("duplicate centers are ambiguous; do not silently merge")
        pairs = record["membership"]
        if type(pairs) is not list or len(pairs) != len(record["openings"]):
            raise ValueError("one explicit membership row per opening")
        for row in pairs:
            if type(row) is not list or len(row) != len(record["anchors"]) or any(v is not None and type(v) is not bool for v in row):
                raise ValueError("membership is independently true/false/null per anchor")
        region = record["score_region"]
        _keys(region, "center_m radius_m anchors_complete openings_complete evidence")
        _xyz(region["center_m"])
        r = region["radius_m"]
        if type(r) not in (int, float) or not math.isfinite(r) or not 0 < r <= 10:
            raise ValueError("explicit scoring radius in (0,10]m required")
        if any(type(region[k]) is not bool for k in ("anchors_complete", "openings_complete")):
            raise ValueError("explicit completeness booleans required")
        _evidence(region["evidence"])

    b = len(records); t = max(len(x["anchors"]) for x in records); u = max(len(x["openings"]) for x in records)
    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, device=device)
    a = zeros(b,t,3); av = zeros(b,t,dtype=torch.bool)
    p = zeros(b,u,3); pv = zeros(b,u,dtype=torch.bool)
    d = zeros(b,u,3); dv = zeros(b,u,dtype=torch.bool)
    sizes = zeros(b,u,2); sv = zeros(b,u,2,dtype=torch.bool)
    m = zeros(b,u,t); mv = zeros(b,u,t,dtype=torch.bool)
    centers = zeros(b,3); radii = zeros(b)
    ac = zeros(b,dtype=torch.bool); oc = zeros(b,dtype=torch.bool)
    for i, record in enumerate(records):
        for j, item in enumerate(record["anchors"]):
            a[i,j] = torch.tensor(item["position_m"], device=device); av[i,j] = True
        for j, item in enumerate(record["openings"]):
            p[i,j] = torch.tensor(item["position_m"], device=device); pv[i,j] = True
            if item["direction"] is not None:
                d[i,j] = torch.tensor(item["direction"], device=device); dv[i,j] = True
            for k, key in enumerate(("width_m", "height_m")):
                if item[key] is not None:
                    sizes[i,j,k] = item[key]; sv[i,j,k] = True
            for k, value in enumerate(record["membership"][j]):
                if value is not None:
                    m[i,j,k] = float(value); mv[i,j,k] = True
        region = record["score_region"]
        centers[i] = torch.tensor(region["center_m"], device=device); radii[i] = region["radius_m"]
        ac[i] = region["anchors_complete"]; oc[i] = region["openings_complete"]
    return SurfaceLossTargets(a,av,p,pv,d,dv,sizes,sv,
        torch.full((b,u),2,dtype=torch.long,device=device), zeros(b,u,dtype=torch.bool),
        zeros(b,u,dtype=torch.bool),m,mv,centers,radii,ac,oc)
