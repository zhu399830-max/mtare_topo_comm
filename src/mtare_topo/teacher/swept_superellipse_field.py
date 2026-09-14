"""Shape-generic, identity-preserving swept-superellipse Teacher field.

The representation uses one finite centreline sweep per construction operand.
An exponent of two is an ellipse; larger exponents continuously approach a
rounded rectangle.  Cross-section parameters use cubic smoothstep along arc
length so endpoint derivatives vanish and compatible operands can meet with a
C1 parameter transition.  Operand identities are never collapsed by the
free-space union.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit
from mtare_topo.teacher.primitive_construction_supervisor import PrimitiveConstructionGraph


def _finite_array(value, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-len(shape_tail):] != shape_tail or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with trailing shape {shape_tail}")
    return array


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


@dataclass(frozen=True)
class SweptSuperellipsePrimitive:
    primitive_id: str
    centerline_xyz_m: np.ndarray
    endpoint_half_axes_m: tuple[tuple[float, float], tuple[float, float]]
    endpoint_shape_exponent: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.primitive_id, str) or not self.primitive_id:
            raise ValueError("primitive_id must be nonempty")
        points = _finite_array(self.centerline_xyz_m, (3,), "centerline")
        if points.ndim != 2 or len(points) < 2:
            raise ValueError("centerline must have shape [N>=2,3]")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-12):
            raise ValueError("centerline contains a degenerate segment")
        axes = _finite_array(self.endpoint_half_axes_m, (2, 2), "endpoint_half_axes_m")
        exponent = _finite_array(self.endpoint_shape_exponent, (2,), "endpoint_shape_exponent")
        if np.any(axes <= 0.0):
            raise ValueError("half axes must be positive")
        if np.any(exponent < 2.0):
            raise ValueError("shape exponents must be >=2")
        object.__setattr__(self, "centerline_xyz_m", points)
        object.__setattr__(self, "endpoint_half_axes_m", tuple(tuple(float(x) for x in row) for row in axes))
        object.__setattr__(self, "endpoint_shape_exponent", tuple(float(x) for x in exponent))

    def parameters_at_fraction(self, fraction: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        fraction_array = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
        weight = _smoothstep(fraction_array)
        axes = np.asarray(self.endpoint_half_axes_m, dtype=np.float64)
        exponent = np.asarray(self.endpoint_shape_exponent, dtype=np.float64)
        return (
            axes[0] + weight[..., None] * (axes[1] - axes[0]),
            exponent[0] + weight * (exponent[1] - exponent[0]),
        )

    def as_dict(self) -> dict:
        return {
            "primitive_id": self.primitive_id,
            "primitive_type": "swept_superellipse_v1",
            "centerline_xyz_m": self.centerline_xyz_m.tolist(),
            "endpoint_half_axes_m": [list(row) for row in self.endpoint_half_axes_m],
            "endpoint_shape_exponent": list(self.endpoint_shape_exponent),
            "parameter_interpolation": "cubic_smoothstep_c1_endpoint",
        }


@dataclass(frozen=True)
class _SampledOperand:
    primitive: SweptSuperellipsePrimitive
    arc_m: np.ndarray
    points: np.ndarray
    tangents: np.ndarray
    lateral: np.ndarray
    vertical: np.ndarray
    half_axes: np.ndarray
    exponent: np.ndarray
    tree: cKDTree


def _sample_operand(primitive: SweptSuperellipsePrimitive, spacing_m: float) -> _SampledOperand:
    source = primitive.centerline_xyz_m
    lengths = np.linalg.norm(np.diff(source, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    arcs = np.unique(np.append(np.arange(0.0, cumulative[-1], spacing_m), cumulative[-1]))
    indices = np.minimum(np.searchsorted(cumulative, arcs, side="right") - 1, len(lengths) - 1)
    ratios = (arcs - cumulative[indices]) / lengths[indices]
    points = source[indices] + ratios[:, None] * (source[indices + 1] - source[indices])
    # Distinct arc floats can interpolate to exactly the same world point,
    # especially at a translated endpoint. Keep the last occurrence (the
    # actual endpoint), without a distance tolerance or geometry snapping.
    keep = np.concatenate((np.any(points[:-1] != points[1:], axis=1), [True]))
    arcs, points = arcs[keep], points[keep]
    if len(points) < 2:
        raise ValueError('sampled operand has fewer than two distinct points')
    tangent = np.empty_like(points)
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangent[1:-1] = points[2:] - points[:-2]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), tangent.shape).copy()
    near_vertical = np.abs(tangent[:, 2]) > 0.95
    up[near_vertical] = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(up, tangent)
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True)
    vertical = np.cross(tangent, lateral)
    vertical /= np.linalg.norm(vertical, axis=1, keepdims=True)
    axes, exponent = primitive.parameters_at_fraction(arcs / cumulative[-1])
    return _SampledOperand(primitive, arcs, points, tangent, lateral, vertical, axes, exponent, cKDTree(points))


class SweptSuperellipseProvenanceField:
    """Approximate finite swept-superellipse union with exact operand IDs."""

    def __init__(self, primitives: Iterable[SweptSuperellipsePrimitive], *, spacing_m: float = 0.01):
        values = tuple(primitives)
        if not values:
            raise ValueError("at least one primitive is required")
        if not math.isfinite(spacing_m) or spacing_m <= 0.0:
            raise ValueError("spacing_m must be finite and positive")
        identities = tuple(value.primitive_id for value in values)
        if len(identities) != len(set(identities)):
            raise ValueError("primitive identities must be unique")
        self.spacing_m = float(spacing_m)
        self.primitive_ids = identities
        self.operands = tuple(_sample_operand(value, self.spacing_m) for value in values)
        self.operand_bounds = tuple(
            (
                np.min(value.points, axis=0) - float(np.max(value.half_axes)) - self.spacing_m,
                np.max(value.points, axis=0) + float(np.max(value.half_axes)) + self.spacing_m,
            )
            for value in self.operands
        )

    @staticmethod
    def _operand_distance(operand: _SampledOperand, flat: np.ndarray) -> np.ndarray:
        _, index = operand.tree.query(flat, k=1, workers=1)
        delta = flat - operand.points[index]
        lateral = np.abs(np.einsum("ij,ij->i", delta, operand.lateral[index]))
        vertical = np.abs(np.einsum("ij,ij->i", delta, operand.vertical[index]))
        axes = operand.half_axes[index]
        exponent = operand.exponent[index]
        normalized = ((lateral / axes[:, 0]) ** exponent + (vertical / axes[:, 1]) ** exponent) ** (1.0 / exponent)
        cross_distance = (normalized - 1.0) * np.min(axes, axis=1)
        axial = np.einsum("ij,ij->i", delta, operand.tangents[index])
        # A finite sweep is clipped by the outward endpoint half-spaces.  The
        # cap term must remain negative on the inward side; clipping it to zero
        # would incorrectly turn every interior query into a surface query.
        endpoint_cap = np.full(len(flat), -np.inf, dtype=np.float64)
        endpoint_cap[index == 0] = -axial[index == 0]
        endpoint_cap[index == len(operand.points) - 1] = axial[index == len(operand.points) - 1]
        return np.maximum(cross_distance, endpoint_cap)

    def operand_signed_distances(self, points_xyz_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.ndim < 2 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("queries must be finite [...,3]")
        flat = points.reshape(-1, 3)
        values = np.stack([self._operand_distance(value, flat) for value in self.operands], axis=-1)
        return values.reshape(points.shape[:-1] + (len(self.operands),))

    def signed_distance(self, points_xyz_m: np.ndarray) -> np.ndarray:
        return np.min(self.operand_signed_distances(points_xyz_m), axis=-1)

    def operand_signed_distances_sparse(self, points_xyz_m: np.ndarray) -> np.ndarray:
        """Exact local operand values with +inf for impossible AABB sources."""

        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.ndim < 2 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("queries must be finite [...,3]")
        flat = points.reshape(-1, 3)
        values = np.full((len(flat), len(self.operands)), np.inf, dtype=np.float64)
        for operand_index, (operand, bounds) in enumerate(zip(self.operands, self.operand_bounds)):
            lower, upper = bounds
            selected = np.flatnonzero(np.all((flat >= lower) & (flat <= upper), axis=1))
            if len(selected):
                values[selected, operand_index] = self._operand_distance(operand, flat[selected])
        return values.reshape(points.shape[:-1] + (len(self.operands),))

    def active_source_ids(self, points_xyz_m: np.ndarray, *, tolerance_m: float | None = None) -> tuple[tuple[str, ...], ...]:
        tolerance = self.spacing_m if tolerance_m is None else float(tolerance_m)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        values = self.operand_signed_distances(points_xyz_m).reshape(-1, len(self.operands))
        minimum = np.min(values, axis=1, keepdims=True)
        return tuple(tuple(self.primitive_ids[i] for i in np.flatnonzero(row <= tolerance)) for row in values - minimum)

    def ray_exit_hits(
        self,
        origins_xyz_m: np.ndarray,
        directions_xyz: np.ndarray,
        *,
        maximum_m: float = 50.0,
        maximum_steps: int = 2048,
        bisection_steps: int = 40,
    ) -> tuple[PrimitiveRayHit | None, ...]:
        origins = np.asarray(origins_xyz_m, dtype=np.float64)
        directions = np.asarray(directions_xyz, dtype=np.float64)
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins and directions must be matching Nx3")
        norms = np.linalg.norm(directions, axis=1)
        if not np.isfinite(origins).all() or not np.isfinite(directions).all() or np.any(norms <= 0.0):
            raise ValueError("rays must be finite and nonzero")
        directions = directions / norms[:, None]
        initial = self.signed_distance(origins)
        if np.any(initial > self.spacing_m):
            raise ValueError("ray origins must start inside the Teacher free-space union")
        count = len(origins)
        travel = np.zeros(count); low = np.zeros(count); high = np.full(count, np.nan)
        value = initial.copy(); active = np.ones(count, dtype=bool)
        for _ in range(maximum_steps):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            escaped = (value[indices] >= 0.0) & (travel[indices] > 0.0)
            finished = indices[escaped]
            high[finished] = travel[finished]; active[finished] = False
            remaining = indices[~escaped]
            if not len(remaining):
                continue
            low[remaining] = travel[remaining]
            travel[remaining] += np.maximum(np.abs(value[remaining]) * 0.65, self.spacing_m * 0.25)
            over = travel[remaining] > maximum_m
            active[remaining[over]] = False
            querying = remaining[~over]
            if len(querying):
                value[querying] = self.signed_distance(origins[querying] + travel[querying, None] * directions[querying])
        hit_indices = np.flatnonzero(np.isfinite(high))
        for _ in range(bisection_steps):
            midpoint = 0.5 * (low[hit_indices] + high[hit_indices])
            query = origins[hit_indices] + midpoint[:, None] * directions[hit_indices]
            outside = self.signed_distance(query) >= 0.0
            high[hit_indices[outside]] = midpoint[outside]
            low[hit_indices[~outside]] = midpoint[~outside]
        result: list[PrimitiveRayHit | None] = [None] * count
        if len(hit_indices):
            points = origins[hit_indices] + high[hit_indices, None] * directions[hit_indices]
            sources = self.active_source_ids(points)
            for index, point, source_ids in zip(hit_indices, points, sources):
                result[int(index)] = PrimitiveRayHit(float(high[index]), tuple(float(x) for x in point), source_ids, len(source_ids) == 1)
        return tuple(result)


def superellipse_primitives_from_circular_construction(
    construction: PrimitiveConstructionGraph,
) -> tuple[SweptSuperellipsePrimitive, ...]:
    """Lift archived circular edge operands into the shape-generic schema."""

    return tuple(
        SweptSuperellipsePrimitive(
            primitive_id=value.primitive_id,
            centerline_xyz_m=value.centerline_xyz_m,
            endpoint_half_axes_m=((value.radius_m, value.radius_m), (value.radius_m, value.radius_m)),
            endpoint_shape_exponent=(2.0, 2.0),
        )
        for value in construction.primitives
    )


__all__ = [
    "SweptSuperellipsePrimitive",
    "SweptSuperellipseProvenanceField",
    "superellipse_primitives_from_circular_construction",
]
