"""No-training local connection interface probe, NOT a navigation graph.

Synthetic callers supply observed surface samples and explicit observed bridge
witnesses. These are not inferred from nearest neighbours or hidden topology.
This module tests consuming evidence, not obtaining it from real LiDAR.
No robot envelope, traversability certification, node identity or GT input.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Surface:
    key: str
    points: np.ndarray


@dataclass(frozen=True)
class BridgeWitness:
    left: str
    right: str
    # An observed surface strip; an empty strip carries NO connection evidence.
    points: np.ndarray


def _points(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("finite N x 3 observed coordinates required")
    return p


def _plane(points):
    p = _points(points)
    if len(p) < 3:
        return None
    center = p.mean(axis=0)
    _, s, vh = np.linalg.svd(p - center, full_matrices=False)
    if s[1] <= 1e-9 or s[-1] > 1e-8:
        return None
    return center, vh[-1]


def _touches(points, strip):
    # Fixture contract: exact shared observed seam samples, not radius merging.
    return bool(set(map(tuple, points)) & set(map(tuple, strip)))


def local_connections(surfaces, witnesses, *, representation="points"):
    """Return local evidence links and unknown pairs, never verified edges.

    Both routes require the SAME observed bridge and planar support. The point
    route fits combined samples; the primitive route composes fitted planes.
    Numerical tolerances here are exact-fixture tolerances, NOT robot limits.
    Absence/rejection stays unknown, never a claim of a physical obstruction.
    """
    if representation not in {"points", "primitives"}:
        raise ValueError("unknown representation")
    by_key = {s.key: _points(s.points) for s in surfaces}
    if len(by_key) != len(surfaces):
        raise ValueError("duplicate surface key")
    accepted, unknown = set(), set()
    for witness in witnesses:
        a, b = witness.left, witness.right
        if a == b or a not in by_key or b not in by_key:
            raise ValueError("invalid observed surface reference")
        pair = tuple(sorted((a, b)))
        strip = _points(witness.points)
        parts = [by_key[a], strip, by_key[b]]
        seam = len(strip) >= 3 and all(_touches(p, strip) for p in (parts[0], parts[2]))
        compatible = False
        if seam and representation == "points":
            compatible = _plane(np.concatenate(parts)) is not None
        elif seam:
            fits = [_plane(p) for p in parts]
            if all(f is not None for f in fits):
                center, normal = fits[0]
                compatible = all(abs(abs(normal @ n) - 1) < 1e-8
                                 and abs((c - center) @ normal) < 1e-8
                                 for c, n in fits[1:])
        (accepted if compatible else unknown).add(pair)
    unknown -= accepted
    adjacency = {key: set() for key in by_key}
    for a, b in accepted:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining, components = set(by_key), []
    while remaining:
        stack, component = [min(remaining)], set()
        while stack:
            key = stack.pop()
            if key in component:
                continue
            component.add(key)
            stack.extend(adjacency[key] - component)
        remaining -= component
        components.append(sorted(component))
    return {"local_evidence_links": sorted(accepted), "unknown_pairs": sorted(unknown),
            "components": sorted(components), "confirmed_navigation_edges": [],
            "scope": "ideal_observed_surface_bridge_software_probe"}
