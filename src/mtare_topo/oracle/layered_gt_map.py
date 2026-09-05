"""Pose-conditioned semantic oracle built from a complete development map.

The map is deliberately queried as a height-conditioned local layer.  A
global XY projection is invalid for garages and stacked tunnels because it can
connect surfaces which share XY but are physically separated in Z.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from mtare_topo.oracle.oriented_dae_support import DAEUpwardSupportIndex


@dataclass(frozen=True)
class LayeredGTMapConfig:
    resolution_m: float = 0.2
    local_size_m: float = 24.0
    sensor_above_support_m: float = 0.75
    support_search_below_m: float = 1.2
    support_search_above_m: float = 1.4
    maximum_neighbor_step_m: float = 0.5
    obstacle_min_above_support_m: float = 0.15
    vehicle_height_m: float = 1.5
    collision_inflation_radius_m: float = 0.55
    terrain_support_dilation_radius_m: float = 0.4
    center_snap_radius_m: float = 0.8
    maximum_reach_m: float = 12.0
    exit_minimum_reach_m: float = 5.0
    direction_bins: int = 720
    role_embedding_size: int = 128

    def __post_init__(self) -> None:
        positive = (
            self.resolution_m,
            self.local_size_m,
            self.sensor_above_support_m,
            self.support_search_below_m,
            self.support_search_above_m,
            self.maximum_neighbor_step_m,
            self.obstacle_min_above_support_m,
            self.vehicle_height_m,
            self.collision_inflation_radius_m,
            self.terrain_support_dilation_radius_m,
            self.center_snap_radius_m,
            self.maximum_reach_m,
            self.exit_minimum_reach_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("all metric oracle parameters must be finite and positive")
        cells = self.local_size_m / self.resolution_m
        if not math.isclose(cells, round(cells), abs_tol=1e-9) or int(round(cells)) % 2:
            raise ValueError("local_size_m/resolution_m must be an even integer")
        if self.maximum_reach_m > self.local_size_m / 2.0:
            raise ValueError("maximum reach must fit inside the local map")
        if self.exit_minimum_reach_m > self.maximum_reach_m:
            raise ValueError("exit reach cannot exceed maximum reach")
        if self.direction_bins != 720 or self.role_embedding_size != 128:
            raise ValueError("oracle output contract is fixed at 720 directions and 128 role values")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalLayerEvidence:
    support_height_m: np.ndarray
    support_valid: np.ndarray
    obstacle: np.ndarray
    inflated_obstacle: np.ndarray
    connected_traversable: np.ndarray
    snapped_center_rc: tuple[int, int]
    expected_support_z_m: float
    selected_center_support_z_m: float
    support_node_count: int = 0
    reachable_support_node_count: int = 0
    multilayer_cell_count: int = 0
    maximum_reachable_layers_per_cell: int = 1


@dataclass(frozen=True)
class GTMapOraclePrediction:
    direction_logits: np.ndarray
    count_probabilities: np.ndarray
    role_probabilities: np.ndarray
    z_role: np.ndarray
    reachable_distance_m: np.ndarray
    exit_count: int
    evidence: LocalLayerEvidence


@dataclass(frozen=True)
class _MultilayerLocalState:
    evidence: LocalLayerEvidence
    cell_node_offsets: np.ndarray
    node_z_m: np.ndarray
    node_reachable: np.ndarray

    def node_ids(self, row: int, column: int) -> np.ndarray:
        width = self.evidence.support_valid.shape[1]
        cell = int(row) * width + int(column)
        return np.arange(
            int(self.cell_node_offsets[cell]),
            int(self.cell_node_offsets[cell + 1]),
            dtype=np.int64,
        )


def load_binary_xyz_ply(path: str | Path) -> np.ndarray:
    """Load the exact xyz-only binary-little-endian M-TARE preview format."""

    source = Path(path)
    with source.open("rb") as stream:
        header: list[str] = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"unterminated PLY header: {source}")
            line = raw.decode("ascii", errors="strict").strip()
            header.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("only binary_little_endian PLY is accepted")
        vertex_lines = [line for line in header if line.startswith("element vertex ")]
        properties = [line for line in header if line.startswith("property ")]
        if len(vertex_lines) != 1 or properties != ["property float x", "property float y", "property float z"]:
            raise ValueError("PLY must contain exactly float x/y/z vertices")
        count = int(vertex_lines[0].split()[-1])
        values = np.fromfile(stream, dtype="<f4", count=count * 3)
        if values.size != count * 3 or stream.read(1):
            raise ValueError("PLY vertex byte count does not match its header")
    points = values.reshape(count, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("map point cloud contains non-finite coordinates")
    return points


class LayeredGTMapOracle:
    """Query full-map semantics without collapsing vertically separated layers."""

    def __init__(
        self,
        points_xyz_m: np.ndarray,
        config: LayeredGTMapConfig = LayeredGTMapConfig(),
        oriented_support_index: DAEUpwardSupportIndex | None = None,
    ) -> None:
        points = np.asarray(points_xyz_m, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points must have shape [N,3] and be non-empty")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")
        self.config = config
        self.points = points.copy()
        margin = config.local_size_m / 2.0 + config.resolution_m
        self.origin_xy = self.points[:, :2].min(axis=0).astype(np.float64) - margin
        maximum_xy = self.points[:, :2].max(axis=0).astype(np.float64) + margin
        self.shape_xy = tuple((np.ceil((maximum_xy - self.origin_xy) / config.resolution_m).astype(int) + 1).tolist())

        # Treat origin_xy as a lattice site and quantize to the nearest site.
        # Using floor here creates periodic holes when float32 PLY coordinates
        # nominally lie on 0.2 m boundaries.
        xy_index = np.rint((self.points[:, :2] - self.origin_xy) / config.resolution_m).astype(np.int64)
        flat = xy_index[:, 1] * self.shape_xy[0] + xy_index[:, 0]
        order = np.lexsort((self.points[:, 2], flat))
        flat_sorted = flat[order]
        z_sorted = self.points[order, 2]
        unique, starts = np.unique(flat_sorted, return_index=True)
        self._flat_cells = unique
        self._starts = starts
        self._ends = np.r_[starts[1:], len(flat_sorted)]
        self._z_sorted = z_sorted
        if oriented_support_index is not None:
            tolerance = 64.0 * np.finfo(np.float64).eps
            if (
                oriented_support_index.shape_xy != self.shape_xy
                or not math.isclose(
                    oriented_support_index.resolution_m,
                    config.resolution_m,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                or not np.allclose(
                    oriented_support_index.origin_xy_m,
                    self.origin_xy,
                    rtol=0.0,
                    atol=tolerance,
                )
            ):
                raise ValueError("oriented DAE support lattice does not match the PLY obstacle lattice")
        self.oriented_support_index = oriented_support_index

    @classmethod
    def from_ply(cls, path: str | Path, config: LayeredGTMapConfig = LayeredGTMapConfig()) -> "LayeredGTMapOracle":
        return cls(load_binary_xyz_ply(path), config)

    @classmethod
    def from_ply_and_dae(
        cls,
        ply_path: str | Path,
        dae_path: str | Path,
        config: LayeredGTMapConfig = LayeredGTMapConfig(),
        world_from_mesh: np.ndarray | None = None,
    ) -> "LayeredGTMapOracle":
        points = load_binary_xyz_ply(ply_path)
        oracle = cls(points, config)
        support = DAEUpwardSupportIndex.from_dae(
            dae_path,
            oracle.origin_xy,
            oracle.shape_xy,
            config.resolution_m,
            world_from_mesh=world_from_mesh,
        )
        return cls(points, config, oriented_support_index=support)

    def _z_values(self, flat: int) -> np.ndarray:
        position = int(np.searchsorted(self._flat_cells, flat))
        if position >= len(self._flat_cells) or int(self._flat_cells[position]) != flat:
            return np.empty(0, dtype=np.float32)
        return self._z_sorted[self._starts[position] : self._ends[position]]

    def _support_z_values(self, flat: int) -> np.ndarray:
        if self.oriented_support_index is None:
            return self._z_values(flat)
        return self.oriented_support_index.z_values(flat)

    def _support_candidate_values(self, flat: int) -> tuple[np.ndarray, np.ndarray]:
        if self.oriented_support_index is None:
            values = self._z_values(flat)
            return values, np.zeros(len(values), dtype=np.int64)
        return self.oriented_support_index.candidate_values(flat)

    def support_provenance(self) -> dict[str, Any]:
        if self.oriented_support_index is None:
            return {"support_source": "XYZ_ONLY_PLY_V1"}
        return {
            "support_source": "GAZEBO_DAE_UPWARD_TRIANGLES_V1R",
            **self.oriented_support_index.provenance(),
        }

    @staticmethod
    def _unique_nearest_support(candidates: np.ndarray, expected: float) -> float:
        distances = np.abs(np.asarray(candidates, dtype=np.float64) - expected)
        minimum = float(np.min(distances))
        scale = max(1.0, abs(expected), float(np.max(np.abs(candidates))))
        tolerance = 64.0 * np.finfo(np.float64).eps * scale
        nearest = np.flatnonzero(np.abs(distances - minimum) <= tolerance)
        if len(nearest) != 1:
            raise RuntimeError("ambiguous upward-facing support exists in the local layer")
        return float(candidates[int(nearest[0])])

    def _multilayer_local_state(
        self, sensor_xyz_m: Sequence[float], yaw_deg: float
    ) -> _MultilayerLocalState:
        """Build a layer-preserving local support graph for oriented DAE input."""

        if self.oriented_support_index is None:
            raise RuntimeError("multilayer support requires an oriented DAE index")
        xyz = np.asarray(sensor_xyz_m, dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)) or not math.isfinite(yaw_deg):
            raise ValueError("pose must contain finite xyz and yaw")
        cfg = self.config
        size = int(round(cfg.local_size_m / cfg.resolution_m))
        center = size // 2
        coordinates = (np.arange(size, dtype=np.float64) - center) * cfg.resolution_m
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        yaw = math.radians(yaw_deg)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        world_x = xyz[0] + cosine * xx - sine * yy
        world_y = xyz[1] + sine * xx + cosine * yy
        ix = np.rint((world_x - self.origin_xy[0]) / cfg.resolution_m).astype(np.int64)
        iy = np.rint((world_y - self.origin_xy[1]) / cfg.resolution_m).astype(np.int64)
        inside = (ix >= 0) & (iy >= 0) & (ix < self.shape_xy[0]) & (iy < self.shape_xy[1])
        global_flat = iy * self.shape_xy[0] + ix

        expected = float(xyz[2] - cfg.sensor_above_support_m)
        lower = expected - cfg.support_search_below_m
        upper = expected + cfg.support_search_above_m
        cell_count = size * size
        raw_candidate_z: list[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(cell_count)]
        raw_candidate_surfaces: list[np.ndarray] = [
            np.empty(0, dtype=np.int64) for _ in range(cell_count)
        ]
        raw_support = np.zeros((size, size), dtype=bool)
        for row, column in np.argwhere(inside):
            values, surfaces = self._support_candidate_values(int(global_flat[row, column]))
            selected = (values >= lower) & (values <= upper)
            if np.any(selected):
                cell = int(row) * size + int(column)
                raw_candidate_z[cell] = values[selected]
                raw_candidate_surfaces[cell] = surfaces[selected]
                raw_support[row, column] = True
        if not np.any(raw_support):
            raise RuntimeError("no support exists in the pose-conditioned local layer")

        # Preserve every layer while filling only the same frozen sub-0.4 m
        # sampling holes.  A filled cell copies the complete candidate set of
        # its deterministic nearest raw cell; no height is selected here.
        distance_cells, nearest = ndimage.distance_transform_edt(
            ~raw_support, return_indices=True
        )
        valid_support = raw_support | (
            distance_cells * cfg.resolution_m <= cfg.terrain_support_dilation_radius_m
        )
        candidate_z_by_cell: list[np.ndarray] = []
        candidate_surface_by_cell: list[np.ndarray] = []
        counts = np.zeros(cell_count, dtype=np.int64)
        for row in range(size):
            for column in range(size):
                cell = row * size + column
                if not valid_support[row, column]:
                    values = np.empty(0, dtype=np.float32)
                    surfaces = np.empty(0, dtype=np.int64)
                elif raw_support[row, column]:
                    values = raw_candidate_z[cell]
                    surfaces = raw_candidate_surfaces[cell]
                else:
                    source_row = int(nearest[0, row, column])
                    source_column = int(nearest[1, row, column])
                    source_cell = source_row * size + source_column
                    values = raw_candidate_z[source_cell]
                    surfaces = raw_candidate_surfaces[source_cell]
                candidate_z_by_cell.append(values)
                candidate_surface_by_cell.append(surfaces)
                counts[cell] = len(values)
        offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        if int(offsets[-1]) == 0:
            raise RuntimeError("no multilayer support nodes exist in the local layer")
        node_z = np.concatenate([values for values in candidate_z_by_cell if len(values)]).astype(
            np.float64
        )
        node_surface = np.concatenate(
            [values for values in candidate_surface_by_cell if len(values)]
        ).astype(np.int64)
        node_obstacle = np.zeros(len(node_z), dtype=bool)
        for row, column in np.argwhere(valid_support):
            cell = int(row) * size + int(column)
            ids = np.arange(offsets[cell], offsets[cell + 1], dtype=np.int64)
            values = self._z_values(int(global_flat[row, column]))
            for node_id in ids:
                support_z = node_z[node_id]
                node_obstacle[node_id] = bool(
                    np.any(
                        (values >= support_z + cfg.obstacle_min_above_support_m)
                        & (values <= support_z + cfg.vehicle_height_m)
                    )
                )

        # Inflate only within a height-continuous layer.  This is the exact
        # 0.55 m XY radius and existing 0.5 m neighbor-height contract, not a
        # global projection that can contaminate a stacked floor.
        node_inflated = node_obstacle.copy()
        cell_radius = int(math.ceil(cfg.collision_inflation_radius_m / cfg.resolution_m))
        spatial_offsets = [
            (dr, dc)
            for dr in range(-cell_radius, cell_radius + 1)
            for dc in range(-cell_radius, cell_radius + 1)
            if math.hypot(dr, dc) * cfg.resolution_m <= cfg.collision_inflation_radius_m
        ]
        obstacle_ids = np.flatnonzero(node_obstacle)
        obstacle_cells = np.searchsorted(offsets, obstacle_ids, side="right") - 1
        for obstacle_id, cell in zip(obstacle_ids, obstacle_cells):
            row, column = divmod(int(cell), size)
            obstacle_z = node_z[obstacle_id]
            for dr, dc in spatial_offsets:
                rr, cc = row + dr, column + dc
                if rr < 0 or cc < 0 or rr >= size or cc >= size:
                    continue
                neighbor = rr * size + cc
                ids = np.arange(offsets[neighbor], offsets[neighbor + 1], dtype=np.int64)
                compatible = ids[
                    (node_surface[ids] == node_surface[obstacle_id])
                    & (np.abs(node_z[ids] - obstacle_z) <= cfg.maximum_neighbor_step_m)
                ]
                node_inflated[compatible] = True

        traversable = ~node_inflated
        traversable_cells = np.flatnonzero(
            np.asarray(
                [np.any(traversable[offsets[cell] : offsets[cell + 1]]) for cell in range(cell_count)],
                dtype=bool,
            )
        )
        if not len(traversable_cells):
            raise RuntimeError("no unoccupied support exists in the local layer")
        cell_rows = traversable_cells // size
        cell_columns = traversable_cells % size
        squared = (cell_rows - center) ** 2 + (cell_columns - center) ** 2
        nearest_cell_position = int(np.argmin(squared))
        nearest_traversable_distance_m = (
            math.sqrt(float(squared[nearest_cell_position])) * cfg.resolution_m
        )
        if nearest_traversable_distance_m > cfg.center_snap_radius_m:
            raise RuntimeError(
                "no unique traversable support exists near the vehicle: "
                f"nearest={nearest_traversable_distance_m:.12f}m "
                f"limit={cfg.center_snap_radius_m:.12f}m"
            )
        seed_cell = int(traversable_cells[nearest_cell_position])
        seed_ids = np.arange(offsets[seed_cell], offsets[seed_cell + 1], dtype=np.int64)
        seed_ids = seed_ids[traversable[seed_ids]]
        seed_z = self._unique_nearest_support(node_z[seed_ids], expected)
        seed_matches = seed_ids[node_z[seed_ids] == seed_z]
        if len(seed_matches) != 1:
            raise RuntimeError("ambiguous support node identity exists near the vehicle")
        seed_id = int(seed_matches[0])

        reachable = np.zeros(len(node_z), dtype=bool)
        reachable[seed_id] = True
        queue: deque[int] = deque([seed_id])
        node_cells = np.searchsorted(offsets, np.arange(len(node_z)), side="right") - 1
        while queue:
            node_id = queue.popleft()
            row, column = divmod(int(node_cells[node_id]), size)
            height = node_z[node_id]
            for dr, dc in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                rr, cc = row + dr, column + dc
                if rr < 0 or cc < 0 or rr >= size or cc >= size:
                    continue
                neighbor = rr * size + cc
                ids = np.arange(offsets[neighbor], offsets[neighbor + 1], dtype=np.int64)
                compatible = ids[
                    traversable[ids]
                    & (~reachable[ids])
                    & (np.abs(node_z[ids] - height) <= cfg.maximum_neighbor_step_m)
                ]
                if len(compatible):
                    reachable[compatible] = True
                    queue.extend(int(value) for value in compatible)

        support = np.full((size, size), np.nan, dtype=np.float32)
        obstacle = np.zeros((size, size), dtype=bool)
        inflated = np.zeros((size, size), dtype=bool)
        connected = np.zeros((size, size), dtype=bool)
        reachable_layer_count = np.zeros((size, size), dtype=np.int32)
        for cell in range(cell_count):
            ids = np.arange(offsets[cell], offsets[cell + 1], dtype=np.int64)
            if not len(ids):
                continue
            row, column = divmod(cell, size)
            # Projection is evidence-only.  Direction qualification below
            # consumes every reachable node and never this representative.
            representative = int(ids[np.lexsort((node_z[ids], np.abs(node_z[ids] - expected)))[0]])
            support[row, column] = node_z[representative]
            obstacle[row, column] = node_obstacle[representative]
            inflated[row, column] = node_inflated[representative]
            reachable_ids = ids[reachable[ids]]
            reachable_layer_count[row, column] = len(reachable_ids)
            connected[row, column] = bool(len(reachable_ids))
        multilayer_cells = int(np.count_nonzero(counts > 1))
        seed_row, seed_column = divmod(seed_cell, size)
        evidence = LocalLayerEvidence(
            support_height_m=support,
            support_valid=valid_support,
            obstacle=obstacle,
            inflated_obstacle=inflated,
            connected_traversable=connected,
            snapped_center_rc=(seed_row, seed_column),
            expected_support_z_m=expected,
            selected_center_support_z_m=float(node_z[seed_id]),
            support_node_count=int(len(node_z)),
            reachable_support_node_count=int(np.count_nonzero(reachable)),
            multilayer_cell_count=multilayer_cells,
            maximum_reachable_layers_per_cell=int(reachable_layer_count.max(initial=0)),
        )
        return _MultilayerLocalState(
            evidence=evidence,
            cell_node_offsets=offsets,
            node_z_m=node_z,
            node_reachable=reachable,
        )

    def local_layer(self, sensor_xyz_m: Sequence[float], yaw_deg: float) -> LocalLayerEvidence:
        if self.oriented_support_index is not None:
            return self._multilayer_local_state(sensor_xyz_m, yaw_deg).evidence
        xyz = np.asarray(sensor_xyz_m, dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)) or not math.isfinite(yaw_deg):
            raise ValueError("pose must contain finite xyz and yaw")
        cfg = self.config
        size = int(round(cfg.local_size_m / cfg.resolution_m))
        center = size // 2
        coordinates = (np.arange(size, dtype=np.float64) - center) * cfg.resolution_m
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        yaw = math.radians(yaw_deg)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        world_x = xyz[0] + cosine * xx - sine * yy
        world_y = xyz[1] + sine * xx + cosine * yy
        ix = np.rint((world_x - self.origin_xy[0]) / cfg.resolution_m).astype(np.int64)
        iy = np.rint((world_y - self.origin_xy[1]) / cfg.resolution_m).astype(np.int64)
        inside = (ix >= 0) & (iy >= 0) & (ix < self.shape_xy[0]) & (iy < self.shape_xy[1])
        flat = iy * self.shape_xy[0] + ix

        expected = float(xyz[2] - cfg.sensor_above_support_m)
        support = np.full((size, size), np.nan, dtype=np.float32)
        obstacle = np.zeros((size, size), dtype=bool)
        lower = expected - cfg.support_search_below_m
        upper = expected + cfg.support_search_above_m
        for row, column in np.argwhere(inside):
            support_values = self._support_z_values(int(flat[row, column]))
            candidates = support_values[(support_values >= lower) & (support_values <= upper)]
            if not len(candidates):
                continue
            selected = (
                self._unique_nearest_support(candidates, expected)
                if self.oriented_support_index is not None
                else float(candidates[np.argmin(np.abs(candidates - expected))])
            )
            support[row, column] = selected
            values = self._z_values(int(flat[row, column]))
            obstacle[row, column] = bool(
                np.any(
                    (values >= selected + cfg.obstacle_min_above_support_m)
                    & (values <= selected + cfg.vehicle_height_m)
                )
            )

        raw_support = np.isfinite(support)
        if not np.any(raw_support):
            raise RuntimeError("no support exists in the pose-conditioned local layer")
        # M-TARE preview points are a sampled surface rather than a watertight
        # height field.  Fill only sub-0.4 m sampling holes, using the frozen
        # terrain-teacher support dilation already recorded in the training
        # data card.  Heights come from the nearest point in this selected Z
        # layer, so the operation cannot import a stacked floor.
        support_distance_cells, nearest = ndimage.distance_transform_edt(
            ~raw_support, return_indices=True
        )
        valid_support = raw_support | (
            support_distance_cells * cfg.resolution_m <= cfg.terrain_support_dilation_radius_m
        )
        filled = support[tuple(nearest)]
        support[valid_support & ~raw_support] = filled[valid_support & ~raw_support]
        obstacle &= valid_support
        clearance = ndimage.distance_transform_edt(~obstacle) * cfg.resolution_m
        inflated = clearance <= cfg.collision_inflation_radius_m
        traversable = valid_support & ~inflated
        candidates = np.argwhere(traversable)
        if not len(candidates):
            raise RuntimeError("no unoccupied support exists in the local layer")
        squared = np.sum((candidates - np.asarray([center, center])) ** 2, axis=1)
        snapped = candidates[int(np.argmin(squared))]
        if math.sqrt(float(np.min(squared))) * cfg.resolution_m > cfg.center_snap_radius_m:
            raise RuntimeError("no unique traversable support exists near the vehicle")

        connected = np.zeros_like(traversable)
        queue: list[tuple[int, int]] = [(int(snapped[0]), int(snapped[1]))]
        connected[queue[0]] = True
        head = 0
        while head < len(queue):
            row, column = queue[head]
            head += 1
            height = float(support[row, column])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr, cc = row + dr, column + dc
                if rr < 0 or cc < 0 or rr >= size or cc >= size or connected[rr, cc] or not traversable[rr, cc]:
                    continue
                if abs(float(support[rr, cc]) - height) <= cfg.maximum_neighbor_step_m:
                    connected[rr, cc] = True
                    queue.append((rr, cc))
        return LocalLayerEvidence(
            support_height_m=support,
            support_valid=valid_support,
            obstacle=obstacle,
            inflated_obstacle=inflated,
            connected_traversable=connected,
            snapped_center_rc=(int(snapped[0]), int(snapped[1])),
            expected_support_z_m=expected,
            selected_center_support_z_m=float(support[snapped[0], snapped[1]]),
        )

    @staticmethod
    def _circular_component_count(active: np.ndarray) -> int:
        if not np.any(active):
            return 0
        if np.all(active):
            return 1
        previous = np.roll(active, 1)
        return int(np.sum(active & ~previous))

    def _physical_exit_mask(self, raw_active: np.ndarray) -> np.ndarray:
        """Join sub-vehicle angular cracks without a tuned label threshold.

        At the exit-audit radius, an aperture narrower than the collision
        diameter cannot separate two independent routes.  Circular closing
        uses exactly that geometry-derived angular width.  Narrow positive
        sectors are retained because a corridor centerline can be valid even
        when its free angular sector is narrow at the audit radius.
        """

        cfg = self.config
        aperture_angle = 2.0 * math.atan2(cfg.collision_inflation_radius_m, cfg.exit_minimum_reach_m)
        bins = max(1, int(math.ceil(aperture_angle / (2.0 * math.pi) * cfg.direction_bins)))
        structure = np.ones(bins, dtype=bool)
        tiled = np.tile(np.asarray(raw_active, dtype=bool), 3)
        closed = ndimage.binary_closing(tiled, structure=structure)
        return closed[cfg.direction_bins : 2 * cfg.direction_bins]

    def _predict_multilayer(self, state: _MultilayerLocalState) -> GTMapOraclePrediction:
        """Trace 720 rays without allowing a ray to jump between layers."""

        evidence = state.evidence
        cfg = self.config
        row0, column0 = evidence.snapped_center_rc
        steps = np.arange(0.0, cfg.maximum_reach_m + cfg.resolution_m * 0.5, cfg.resolution_m)
        angles = np.arange(cfg.direction_bins, dtype=np.float64) * (
            2.0 * math.pi / cfg.direction_bins
        )
        columns = np.rint(
            column0 + np.cos(angles)[:, None] * steps[None, :] / cfg.resolution_m
        ).astype(np.int64)
        rows = np.rint(
            row0 - np.sin(angles)[:, None] * steps[None, :] / cfg.resolution_m
        ).astype(np.int64)
        height_limit = cfg.maximum_neighbor_step_m
        reachable_distance = np.zeros(cfg.direction_bins, dtype=np.float32)
        for angle_index in range(cfg.direction_bins):
            active_heights = np.asarray([evidence.selected_center_support_z_m], dtype=np.float64)
            last_distance = 0.0
            for step_index, distance in enumerate(steps):
                row = int(rows[angle_index, step_index])
                column = int(columns[angle_index, step_index])
                if row < 0 or column < 0 or row >= evidence.support_valid.shape[0] or column >= evidence.support_valid.shape[1]:
                    break
                ids = state.node_ids(row, column)
                ids = ids[state.node_reachable[ids]]
                if not len(ids):
                    break
                candidate_heights = state.node_z_m[ids]
                compatible = np.any(
                    np.abs(candidate_heights[:, None] - active_heights[None, :]) <= height_limit,
                    axis=1,
                )
                active_heights = candidate_heights[compatible]
                if not len(active_heights):
                    break
                last_distance = float(distance)
            reachable_distance[angle_index] = last_distance
        active = self._physical_exit_mask(reachable_distance >= cfg.exit_minimum_reach_m)
        exit_count = self._circular_component_count(active)
        logits = np.where(active, 20.0, -20.0).astype(np.float32)
        count_probabilities = np.zeros(6, dtype=np.float32)
        count_probabilities[min(max(exit_count, 1), 6) - 1] = 1.0
        role_probabilities = np.zeros(3, dtype=np.float32)
        role_probabilities[2 if exit_count <= 1 else (0 if exit_count == 2 else 1)] = 1.0
        sample_at = np.linspace(0, cfg.direction_bins, cfg.role_embedding_size, endpoint=False)
        embedding = np.interp(
            sample_at, np.arange(cfg.direction_bins), reachable_distance, period=cfg.direction_bins
        ).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding /= norm
        return GTMapOraclePrediction(
            direction_logits=logits,
            count_probabilities=count_probabilities,
            role_probabilities=role_probabilities,
            z_role=embedding,
            reachable_distance_m=reachable_distance,
            exit_count=exit_count,
            evidence=evidence,
        )

    def predict(self, sensor_xyz_m: Sequence[float], yaw_deg: float) -> GTMapOraclePrediction:
        if self.oriented_support_index is not None:
            return self._predict_multilayer(self._multilayer_local_state(sensor_xyz_m, yaw_deg))
        evidence = self.local_layer(sensor_xyz_m, yaw_deg)
        cfg = self.config
        connected = evidence.connected_traversable
        row0, column0 = evidence.snapped_center_rc
        steps = np.arange(0.0, cfg.maximum_reach_m + cfg.resolution_m * 0.5, cfg.resolution_m)
        angles = np.arange(cfg.direction_bins, dtype=np.float64) * (2.0 * math.pi / cfg.direction_bins)
        columns = np.rint(column0 + np.cos(angles)[:, None] * steps[None, :] / cfg.resolution_m).astype(np.int64)
        rows = np.rint(row0 - np.sin(angles)[:, None] * steps[None, :] / cfg.resolution_m).astype(np.int64)
        inside = (rows >= 0) & (columns >= 0) & (rows < connected.shape[0]) & (columns < connected.shape[1])
        passable = np.zeros_like(inside)
        passable[inside] = connected[rows[inside], columns[inside]]
        continuous = np.logical_and.accumulate(passable, axis=1)
        reachable = continuous.sum(axis=1).astype(np.float64) * cfg.resolution_m - cfg.resolution_m
        reachable = np.clip(reachable, 0.0, cfg.maximum_reach_m).astype(np.float32)
        active = self._physical_exit_mask(reachable >= cfg.exit_minimum_reach_m)
        exit_count = self._circular_component_count(active)

        logits = np.where(active, 20.0, -20.0).astype(np.float32)
        count_probabilities = np.zeros(6, dtype=np.float32)
        count_probabilities[min(max(exit_count, 1), 6) - 1] = 1.0
        role_probabilities = np.zeros(3, dtype=np.float32)
        role_probabilities[2 if exit_count <= 1 else (0 if exit_count == 2 else 1)] = 1.0
        sample_at = np.linspace(0, cfg.direction_bins, cfg.role_embedding_size, endpoint=False)
        embedding = np.interp(sample_at, np.arange(cfg.direction_bins), reachable, period=cfg.direction_bins)
        embedding = embedding.astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding /= norm
        return GTMapOraclePrediction(
            direction_logits=logits,
            count_probabilities=count_probabilities,
            role_probabilities=role_probabilities,
            z_role=embedding,
            reachable_distance_m=reachable,
            exit_count=exit_count,
            evidence=evidence,
        )
