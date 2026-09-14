"""Fast identity-preserving CSG-union ray exits over closed primitive meshes.

Each swept primitive remains a separate closed triangle mesh in an Open3D
raycasting scene.  Ordered multi-hit intersections update per-primitive
occupancy.  A LiDAR return is emitted only when the ray leaves the union of
all occupied primitives, so internal overlap surfaces are skipped without a
destructive boolean mesh and the contributing primitive identities survive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np

from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit
from mtare_topo.teacher.swept_superellipse_field import (
    SweptSuperellipsePrimitive,
    _sample_operand,
)


@dataclass(frozen=True)
class ClosedPrimitiveMesh:
    primitive_id: str
    vertices_xyz_m: np.ndarray
    triangle_vertex_indices: np.ndarray
    triangle_normals: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_xyz_m, dtype=np.float64)
        triangles = np.asarray(self.triangle_vertex_indices, dtype=np.int64)
        normals = np.asarray(self.triangle_normals, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("vertices must be finite [V,3]")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) != len(normals):
            raise ValueError("triangles/normals shape mismatch")
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)) or not np.isfinite(normals).all():
            raise ValueError("invalid triangles or normals")
        object.__setattr__(self, "vertices_xyz_m", vertices)
        object.__setattr__(self, "triangle_vertex_indices", triangles)
        object.__setattr__(self, "triangle_normals", normals)


def mesh_swept_superellipse(
    primitive: SweptSuperellipsePrimitive,
    *,
    axial_spacing_m: float = 0.20,
    angular_segments: int = 48,
) -> ClosedPrimitiveMesh:
    if not math.isfinite(axial_spacing_m) or axial_spacing_m <= 0:
        raise ValueError("axial_spacing_m must be finite and positive")
    if not isinstance(angular_segments, int) or angular_segments < 12 or angular_segments % 4:
        raise ValueError("angular_segments must be an integer >=12 divisible by four")
    sampled = _sample_operand(primitive, axial_spacing_m)
    theta = np.arange(angular_segments, dtype=np.float64) * (2.0 * np.pi / angular_segments)
    cosine = np.cos(theta); sine = np.sin(theta)
    rings = []
    for point, lateral, vertical, axes, exponent in zip(
        sampled.points, sampled.lateral, sampled.vertical, sampled.half_axes, sampled.exponent
    ):
        y = axes[0] * np.sign(cosine) * np.abs(cosine) ** (2.0 / exponent)
        z = axes[1] * np.sign(sine) * np.abs(sine) ** (2.0 / exponent)
        rings.append(point + y[:, None] * lateral + z[:, None] * vertical)
    vertices = np.concatenate((np.asarray(rings).reshape(-1, 3), sampled.points[[0, -1]]), axis=0)
    triangles: list[tuple[int, int, int]] = []
    ring_count = len(sampled.points)
    for ring in range(ring_count - 1):
        base = ring * angular_segments; following = (ring + 1) * angular_segments
        for column in range(angular_segments):
            nxt = (column + 1) % angular_segments
            a, b = base + column, following + column
            c, d = following + nxt, base + nxt
            triangles.extend(((a, d, c), (a, c, b)))
    start_center = ring_count * angular_segments; end_center = start_center + 1
    end_base = (ring_count - 1) * angular_segments
    for column in range(angular_segments):
        nxt = (column + 1) % angular_segments
        triangles.append((start_center, nxt, column))
        triangles.append((end_center, end_base + column, end_base + nxt))
    indices = np.asarray(triangles, dtype=np.int64)
    vectors1 = vertices[indices[:, 1]] - vertices[indices[:, 0]]
    vectors2 = vertices[indices[:, 2]] - vertices[indices[:, 0]]
    normals = np.cross(vectors1, vectors2)
    magnitudes = np.linalg.norm(normals, axis=1)
    if np.any(magnitudes <= 1e-12):
        raise ValueError("generated primitive mesh contains degenerate triangles")
    normals /= magnitudes[:, None]
    return ClosedPrimitiveMesh(primitive.primitive_id, vertices, indices, normals)


class CSGMeshProvenanceRaycaster:
    """Open3D multi-hit raycaster that preserves CSG operand identity."""

    def __init__(
        self,
        meshes: Iterable[ClosedPrimitiveMesh],
        *,
        distance_group_tolerance_m: float = 0.01,
        union_signed_distance: Callable[[np.ndarray], np.ndarray] | None = None,
        surface_inside_tolerance_m: float = 0.05,
        operand_signed_distances: Callable[[np.ndarray], np.ndarray] | None = None,
        surface_probe_m: float = 0.05,
        provenance_tolerance_m: float = 0.025,
        require_unique_qualified_candidate: bool = False,
        rescue_missing_with_interval_winding: bool = False,
    ):
        values = tuple(meshes)
        if not values:
            raise ValueError("at least one closed primitive mesh is required")
        identities = tuple(value.primitive_id for value in values)
        if len(identities) != len(set(identities)):
            raise ValueError("primitive mesh identities must be unique")
        if not math.isfinite(distance_group_tolerance_m) or distance_group_tolerance_m <= 0:
            raise ValueError("distance_group_tolerance_m must be finite and positive")
        if not math.isfinite(surface_inside_tolerance_m) or surface_inside_tolerance_m < 0:
            raise ValueError("surface_inside_tolerance_m must be finite and nonnegative")
        if not math.isfinite(surface_probe_m) or surface_probe_m <= 0:
            raise ValueError("surface_probe_m must be finite and positive")
        if not math.isfinite(provenance_tolerance_m) or provenance_tolerance_m < 0:
            raise ValueError("provenance_tolerance_m must be finite and nonnegative")
        import open3d as o3d
        self.meshes = values; self.primitive_ids = identities
        # Meshes of incident edge operands are generated independently.  Their
        # analytically common port can therefore differ by sub-millimetres in
        # ray distance.  The frozen 1 cm construction resolution groups those
        # entry/exit faces into one occupancy event without bridging a
        # physically separated tunnel.
        self.distance_group_tolerance_m = float(distance_group_tolerance_m)
        self.union_signed_distance = union_signed_distance
        self.surface_inside_tolerance_m = float(surface_inside_tolerance_m)
        self.operand_signed_distances = operand_signed_distances
        self.surface_probe_m = float(surface_probe_m)
        self.provenance_tolerance_m = float(provenance_tolerance_m)
        self.require_unique_qualified_candidate = bool(require_unique_qualified_candidate)
        if rescue_missing_with_interval_winding and (operand_signed_distances is not None or union_signed_distance is not None or require_unique_qualified_candidate):
            raise ValueError('interval rescue cannot bypass signed-field/unique-candidate qualification')
        self.rescue_missing_with_interval_winding = bool(rescue_missing_with_interval_winding)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.geometry_to_operand: dict[int, int] = {}
        for operand, mesh in enumerate(values):
            tensor_mesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(mesh.vertices_xyz_m.astype(np.float32)),
                o3d.core.Tensor(mesh.triangle_vertex_indices.astype(np.uint32)),
            )
            geometry_id = int(self.scene.add_triangles(tensor_mesh))
            self.geometry_to_operand[geometry_id] = operand

    def ray_exit_hits(
        self,
        origins_xyz_m: np.ndarray,
        directions_xyz: np.ndarray,
        initial_inside: np.ndarray,
        *,
        maximum_m: float = 50.0,
    ) -> tuple[PrimitiveRayHit | None, ...]:
        origins = np.asarray(origins_xyz_m, dtype=np.float64)
        directions = np.asarray(directions_xyz, dtype=np.float64)
        inside = np.asarray(initial_inside, dtype=bool)
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins/directions must be matching Nx3")
        if inside.shape != (len(origins), len(self.meshes)):
            raise ValueError("initial_inside must have shape [rays,primitives]")
        norms = np.linalg.norm(directions, axis=1)
        if not np.isfinite(origins).all() or not np.isfinite(directions).all() or np.any(norms <= 0):
            raise ValueError("rays must be finite and nonzero")
        if np.any(~inside.any(axis=1)):
            raise ValueError("every ray origin must start inside the primitive union")
        directions = directions / norms[:, None]
        import open3d as o3d
        rays = np.concatenate((origins, directions), axis=1).astype(np.float32)
        raw = {key: value.numpy() for key, value in self.scene.list_intersections(o3d.core.Tensor(rays)).items()}
        splits = raw["ray_splits"].astype(np.int64)
        candidates: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(len(origins))]
        group_distances: list[list[float]] = [[] for _ in range(len(origins))]
        for ray_index in range(len(origins)):
            start, stop = int(splits[ray_index]), int(splits[ray_index + 1])
            if start == stop:
                continue
            order = np.argsort(raw["t_hit"][start:stop], kind="stable") + start
            occupancy = inside[ray_index].copy(); cursor = 0
            while cursor < len(order):
                first = cursor; distance = float(raw["t_hit"][order[cursor]])
                numerical_slack = 8.0 * np.finfo(np.float32).eps * max(1.0, abs(distance))
                while cursor + 1 < len(order) and abs(float(raw["t_hit"][order[cursor + 1]]) - distance) <= self.distance_group_tolerance_m + numerical_slack:
                    cursor += 1
                group = order[first:cursor + 1]; was_inside = bool(occupancy.any())
                group_distances[ray_index].append(distance)
                exited: set[int] = set()
                group_operands: set[int] = set()
                for geometry_id in np.unique(raw["geometry_ids"][group]):
                    operand = self.geometry_to_operand[int(geometry_id)]
                    group_operands.add(operand)
                    members = group[raw["geometry_ids"][group] == geometry_id]
                    triangle_ids = np.unique(raw["primitive_ids"][members].astype(np.int64))
                    dots = self.meshes[operand].triangle_normals[triangle_ids] @ directions[ray_index]
                    positive = np.any(dots > 1e-7); negative = np.any(dots < -1e-7)
                    if positive and not negative:
                        occupancy[operand] = False; exited.add(operand)
                    elif negative and not positive:
                        occupancy[operand] = True
                    # Opposing normals at the same distance are a tangent or
                    # coincident no-net crossing and leave occupancy unchanged.
                if distance <= maximum_m:
                    if self.operand_signed_distances is not None and group_operands:
                        # Every operand-exit face is a cheap candidate.  The
                        # batched analytic union check below decides whether it
                        # is an internal overlap face or the actual union exit,
                        # avoiding long-lived parity errors on grazing faces.
                        candidates[ray_index].append((distance, tuple(sorted(group_operands))))
                    elif exited and was_inside and not occupancy.any():
                        candidates[ray_index].append((distance, tuple(sorted(exited))))
                cursor += 1
        result: list[PrimitiveRayHit | None] = [None] * len(origins)
        eligible: list[np.ndarray] = []
        qualified_sources: list[list[tuple[int, ...]]] = [
            [source for _, source in values] for values in candidates
        ]
        if self.operand_signed_distances is not None:
            points = np.asarray(
                [origins[ray] + distance * directions[ray] for ray, values in enumerate(candidates) for distance, _ in values],
                dtype=np.float64,
            )
            probes = np.asarray(
                [origins[ray] + (distance + self.surface_probe_m) * directions[ray] for ray, values in enumerate(candidates) for distance, _ in values],
                dtype=np.float64,
            )
            if len(points):
                values = np.asarray(self.operand_signed_distances(np.concatenate((points, probes))), dtype=np.float64)
                if (
                    values.shape != (2 * len(points), len(self.meshes))
                    or np.isnan(values).any()
                    or np.isneginf(values).any()
                ):
                    raise ValueError("operand_signed_distances returned invalid candidate values")
                at_surface, after_surface = values[:len(points)], values[len(points):]
                minima = np.min(at_surface, axis=1)
                flat_eligible = (np.abs(minima) <= self.surface_inside_tolerance_m) & (np.min(after_surface, axis=1) >= 0.0)
                active = [tuple(int(x) for x in np.flatnonzero(row <= minimum + self.provenance_tolerance_m)) for row, minimum in zip(at_surface, minima)]
            else:
                flat_eligible = np.empty(0, dtype=bool); active = []
            offset = 0
            for ray_index, candidate_values in enumerate(candidates):
                ray_eligible = flat_eligible[offset:offset + len(candidate_values)].copy()
                for local, (_, face_sources) in enumerate(candidate_values):
                    complete_sources = active[offset + local]
                    if not set(face_sources).issubset(complete_sources):
                        ray_eligible[local] = False
                    qualified_sources[ray_index][local] = complete_sources
                eligible.append(ray_eligible); offset += len(candidate_values)
        elif self.union_signed_distance is None:
            eligible = [np.ones(len(values), dtype=bool) for values in candidates]
        else:
            points = np.asarray(
                [origins[ray] + distance * directions[ray] for ray, values in enumerate(candidates) for distance, _ in values],
                dtype=np.float64,
            )
            if len(points):
                signed = np.asarray(self.union_signed_distance(points), dtype=np.float64)
                if signed.shape != (len(points),) or not np.isfinite(signed).all():
                    raise ValueError("union_signed_distance returned invalid candidate residuals")
                flat_eligible = signed >= -self.surface_inside_tolerance_m
            else:
                flat_eligible = np.empty(0, dtype=bool)
            offset = 0
            for values in candidates:
                eligible.append(flat_eligible[offset:offset + len(values)])
                offset += len(values)
        for ray_index, values in enumerate(candidates):
            if self.require_unique_qualified_candidate and np.count_nonzero(eligible[ray_index]) != 1:
                continue
            for candidate_index, ((distance, _), accepted) in enumerate(zip(values, eligible[ray_index])):
                if not accepted:
                    continue
                source_operands = qualified_sources[ray_index][candidate_index]
                source_ids = tuple(self.primitive_ids[value] for value in source_operands)
                point = origins[ray_index] + distance * directions[ray_index]
                result[ray_index] = PrimitiveRayHit(distance, tuple(float(x) for x in point), source_ids, len(source_ids) == 1)
                break
        if self.operand_signed_distances is not None:
            path_points: list[np.ndarray] = []
            path_rays: list[int] = []
            for ray_index, hit in enumerate(result):
                if hit is None:
                    continue
                boundaries = [0.0] + [value for value in group_distances[ray_index] if value <= hit.distance_m + self.distance_group_tolerance_m]
                for low, high in zip(boundaries[:-1], boundaries[1:]):
                    if high - low <= self.distance_group_tolerance_m:
                        continue
                    for fraction in (0.25, 0.5, 0.75):
                        path_points.append(
                            origins[ray_index] + (low + fraction * (high - low)) * directions[ray_index]
                        )
                        path_rays.append(ray_index)
            if path_points:
                path_values = np.asarray(self.operand_signed_distances(np.asarray(path_points)), dtype=np.float64)
                if (
                    path_values.shape != (len(path_points), len(self.meshes))
                    or np.isnan(path_values).any()
                    or np.isneginf(path_values).any()
                ):
                    raise ValueError("operand_signed_distances returned invalid path values")
                invalid_rays = {
                    ray_index
                    for ray_index, minimum in zip(path_rays, np.min(path_values, axis=1))
                    if minimum > 0.0
                }
                for ray_index in invalid_rays:
                    result[ray_index] = None
        if self.rescue_missing_with_interval_winding:
            from .mesh_interval_exit import interval_winding_exit
            for ray_index,hit in enumerate(result):
                if hit is not None:continue
                start,stop=int(splits[ray_index]),int(splits[ray_index+1])
                rescued=interval_winding_exit(self.meshes,origins[ray_index],directions[ray_index],
                    inside[ray_index],raw['t_hit'][start:stop],
                    [self.geometry_to_operand[int(x)] for x in raw['geometry_ids'][start:stop]],maximum_m=maximum_m)
                if rescued is not None:
                    distance,operands=rescued
                    ids=tuple(self.primitive_ids[x] for x in operands)
                    point=origins[ray_index]+distance*directions[ray_index]
                    result[ray_index]=PrimitiveRayHit(distance,tuple(float(x) for x in point),ids,len(ids)==1)
        return tuple(result)


__all__ = ["CSGMeshProvenanceRaycaster", "ClosedPrimitiveMesh", "mesh_swept_superellipse"]
