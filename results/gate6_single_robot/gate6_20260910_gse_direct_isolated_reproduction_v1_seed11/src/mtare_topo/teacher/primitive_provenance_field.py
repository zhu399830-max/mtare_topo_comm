"""Identity-preserving implicit Teacher geometry and LiDAR ray provenance.

The field represents free tunnel volume as a union of edge-level swept
primitives.  Unlike the native Poisson perception mesh, every operand keeps a
stable construction identity.  A ray exits the *union*, so internal surfaces
of overlapping incident primitives cannot become false walls.  Equal active
operands are returned as an explicit ambiguous source set and may be masked by
the Teacher instead of being assigned by an arbitrary nearest-object rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from mtare_topo.teacher.primitive_construction_supervisor import PrimitiveConstructionGraph


@dataclass(frozen=True)
class PrimitiveRayHit:
    distance_m: float
    xyz_m: tuple[float, float, float]
    source_primitive_ids: tuple[str, ...]
    provenance_unique: bool

    def as_dict(self) -> dict:
        return {
            "distance_m": self.distance_m,
            "xyz_m": list(self.xyz_m),
            "source_primitive_ids": list(self.source_primitive_ids),
            "provenance_unique": self.provenance_unique,
        }


def _sample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    arcs = np.append(np.arange(0.0, cumulative[-1], spacing_m), cumulative[-1])
    indices = np.minimum(np.searchsorted(cumulative, arcs, side="right") - 1, len(lengths) - 1)
    ratios = (arcs - cumulative[indices]) / lengths[indices]
    return points[indices] + ratios[:, None] * (points[indices + 1] - points[indices])


class PrimitiveProvenanceField:
    """Sampled swept-tube union with exact operand identity propagation."""

    def __init__(self, construction: PrimitiveConstructionGraph, *, spacing_m: float = 0.025):
        if not math.isfinite(spacing_m) or spacing_m <= 0.0:
            raise ValueError("field spacing must be finite and positive")
        if not construction.primitives:
            raise ValueError("construction graph has no primitives")
        self.spacing_m = float(spacing_m)
        self.primitive_ids = tuple(value.primitive_id for value in construction.primitives)
        if len(self.primitive_ids) != len(set(self.primitive_ids)):
            raise ValueError("primitive identities must be unique")
        self.radii_m = np.asarray([value.radius_m for value in construction.primitives], dtype=np.float64)
        self.trees = tuple(
            cKDTree(_sample_polyline(value.centerline_xyz_m, self.spacing_m))
            for value in construction.primitives
        )

    def operand_signed_distances(self, points_xyz_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.ndim < 2 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("field queries must be finite [...,3]")
        flat = points.reshape(-1, 3)
        values = np.stack(
            [tree.query(flat, k=1, workers=1)[0] - radius for tree, radius in zip(self.trees, self.radii_m)],
            axis=-1,
        )
        return values.reshape(points.shape[:-1] + (len(self.trees),))

    def signed_distance(self, points_xyz_m: np.ndarray) -> np.ndarray:
        return np.min(self.operand_signed_distances(points_xyz_m), axis=-1)

    def active_source_ids(
        self, points_xyz_m: np.ndarray, *, tolerance_m: float | None = None
    ) -> tuple[tuple[str, ...], ...]:
        tolerance = self.spacing_m if tolerance_m is None else float(tolerance_m)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("provenance tolerance must be finite and nonnegative")
        values = self.operand_signed_distances(points_xyz_m).reshape(-1, len(self.trees))
        minimum = np.min(values, axis=1, keepdims=True)
        return tuple(
            tuple(self.primitive_ids[index] for index in np.flatnonzero(row <= tolerance))
            for row in values - minimum
        )

    def ray_exit_hits(
        self,
        origins_xyz_m: np.ndarray,
        directions_xyz: np.ndarray,
        *,
        maximum_m: float = 50.0,
        maximum_steps: int = 512,
        bisection_steps: int = 40,
    ) -> tuple[PrimitiveRayHit | None, ...]:
        origins = np.asarray(origins_xyz_m, dtype=np.float64)
        directions = np.asarray(directions_xyz, dtype=np.float64)
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins and directions must be matching Nx3 arrays")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(~np.isfinite(origins)) or np.any(~np.isfinite(directions)) or np.any(norms <= 0.0):
            raise ValueError("rays must be finite with nonzero direction")
        directions = directions / norms[:, None]
        initial = self.signed_distance(origins)
        if np.any(initial > self.spacing_m):
            raise ValueError("ray origins must start inside the Teacher free-space union")

        count = len(origins)
        travel = np.zeros(count, dtype=np.float64)
        low = np.zeros(count, dtype=np.float64)
        high = np.full(count, np.nan, dtype=np.float64)
        sdf = initial.copy()
        active = np.ones(count, dtype=bool)
        for _ in range(maximum_steps):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            escaped = (sdf[indices] >= 0.0) & (travel[indices] > 0.0)
            if escaped.any():
                finished = indices[escaped]
                high[finished] = travel[finished]
                active[finished] = False
            remaining = indices[~escaped]
            if not len(remaining):
                continue
            low[remaining] = travel[remaining]
            travel[remaining] += np.maximum(
                np.abs(sdf[remaining]) * 0.8, self.spacing_m * 0.5
            )
            over = travel[remaining] > maximum_m
            if over.any():
                active[remaining[over]] = False
            querying = remaining[~over]
            if len(querying):
                sdf[querying] = self.signed_distance(
                    origins[querying] + travel[querying, None] * directions[querying]
                )

        hit_indices = np.flatnonzero(np.isfinite(high))
        for _ in range(bisection_steps):
            if not len(hit_indices):
                break
            midpoint = 0.5 * (low[hit_indices] + high[hit_indices])
            value = self.signed_distance(
                origins[hit_indices] + midpoint[:, None] * directions[hit_indices]
            )
            outside = value >= 0.0
            high[hit_indices[outside]] = midpoint[outside]
            low[hit_indices[~outside]] = midpoint[~outside]

        result: list[PrimitiveRayHit | None] = [None] * count
        if len(hit_indices):
            points = origins[hit_indices] + high[hit_indices, None] * directions[hit_indices]
            sources = self.active_source_ids(points, tolerance_m=self.spacing_m)
            for index, point, source_ids in zip(hit_indices, points, sources):
                result[int(index)] = PrimitiveRayHit(
                    distance_m=float(high[index]),
                    xyz_m=tuple(float(value) for value in point),
                    source_primitive_ids=source_ids,
                    provenance_unique=len(source_ids) == 1,
                )
        return tuple(result)


__all__ = ["PrimitiveProvenanceField", "PrimitiveRayHit"]
