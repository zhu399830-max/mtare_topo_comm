"""Method-independent M-TARE exploration coverage accounting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence, Tuple

import numpy as np


Voxel = Tuple[int, int, int]


@dataclass(frozen=True)
class MTAReCoverageConfig:
    explored_volume_voxel_m: float = 0.5
    translation_record_interval_m: float = 0.2
    yaw_record_interval_rad: float = 10.0
    reference_match_radius_m: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.explored_volume_voxel_m,
            self.translation_record_interval_m,
            self.yaw_record_interval_rad,
            self.reference_match_radius_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("coverage parameters must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageCycle:
    stamp_sec: float
    elapsed_sec: float
    explored_voxels: int
    explored_volume_m3: float
    traveling_distance_m: float
    scan_points: int
    new_voxels: int
    cumulative_point_redundancy: float
    complete_map_surface_recall_diagnostic: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _voxel_keys(points_xyz_m: np.ndarray, leaf_m: float) -> set[Voxel]:
    indices = np.floor(points_xyz_m / leaf_m).astype(np.int64)
    return {tuple(map(int, row)) for row in np.unique(indices, axis=0)}


def _circular_distance_rad(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


class MTAReCoverageAccumulator:
    """Reproduce native 0.5 m explored volume and add auditable diagnostics.

    The primary volume is exactly unique registered-scan voxel count times
    leaf volume, matching visualizationTools.cpp.  Complete-map surface recall
    is explicitly diagnostic because the full preview map includes surfaces
    which may not be reachable or observable from a legal vehicle pose.
    """

    def __init__(
        self,
        config: MTAReCoverageConfig = MTAReCoverageConfig(),
        *,
        reference_surface_xyz_m: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.covered_voxels: set[Voxel] = set()
        self.reference_voxels: set[Voxel] | None = None
        self.matched_reference_voxels: set[Voxel] = set()
        if reference_surface_xyz_m is not None:
            reference = np.asarray(reference_surface_xyz_m, dtype=np.float64)
            if reference.ndim != 2 or reference.shape[1] != 3 or not len(reference) or not np.all(np.isfinite(reference)):
                raise ValueError("reference surface must be a non-empty finite [N,3] array")
            self.reference_voxels = _voxel_keys(reference, config.explored_volume_voxel_m)
        radius_cells = int(math.ceil(config.reference_match_radius_m / config.explored_volume_voxel_m))
        self._reference_offsets = [
            (dx, dy, dz)
            for dx in range(-radius_cells, radius_cells + 1)
            for dy in range(-radius_cells, radius_cells + 1)
            for dz in range(-radius_cells, radius_cells + 1)
            if math.sqrt(dx * dx + dy * dy + dz * dz) * config.explored_volume_voxel_m
            <= config.reference_match_radius_m + 1e-12
        ]
        self.start_stamp_sec: float | None = None
        self.last_stamp_sec: float | None = None
        self.last_recorded_pose: np.ndarray | None = None
        self.last_recorded_yaw_rad: float | None = None
        self.traveling_distance_m = 0.0
        self.total_scan_points = 0
        self.cycles: list[CoverageCycle] = []

    def update(
        self,
        *,
        stamp_sec: float,
        registered_scan_xyz_m: np.ndarray,
        vehicle_xyz_m: Sequence[float],
        vehicle_yaw_rad: float,
    ) -> CoverageCycle:
        if not math.isfinite(stamp_sec) or not math.isfinite(vehicle_yaw_rad):
            raise ValueError("timestamp and yaw must be finite")
        if self.last_stamp_sec is not None and stamp_sec <= self.last_stamp_sec:
            raise ValueError("coverage timestamps must be strictly increasing")
        scan = np.asarray(registered_scan_xyz_m, dtype=np.float64)
        pose = np.asarray(vehicle_xyz_m, dtype=np.float64)
        if scan.ndim != 2 or scan.shape[1] != 3 or not np.all(np.isfinite(scan)):
            raise ValueError("registered scan must be a finite [N,3] array")
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("vehicle pose must contain three finite coordinates")
        if self.start_stamp_sec is None:
            self.start_stamp_sec = float(stamp_sec)
            self.last_recorded_pose = pose.copy()
            self.last_recorded_yaw_rad = float(vehicle_yaw_rad)
        else:
            assert self.last_recorded_pose is not None and self.last_recorded_yaw_rad is not None
            displacement = float(np.linalg.norm(pose - self.last_recorded_pose))
            yaw_change = _circular_distance_rad(vehicle_yaw_rad, self.last_recorded_yaw_rad)
            if displacement >= self.config.translation_record_interval_m or yaw_change >= self.config.yaw_record_interval_rad:
                self.traveling_distance_m += displacement
                self.last_recorded_pose = pose.copy()
                self.last_recorded_yaw_rad = float(vehicle_yaw_rad)

        scan_voxels = _voxel_keys(scan, self.config.explored_volume_voxel_m)
        before = len(self.covered_voxels)
        self.covered_voxels.update(scan_voxels)
        new_voxels = len(self.covered_voxels) - before
        self.total_scan_points += int(len(scan))
        if self.reference_voxels is not None:
            for x, y, z in scan_voxels:
                for dx, dy, dz in self._reference_offsets:
                    candidate = (x + dx, y + dy, z + dz)
                    if candidate in self.reference_voxels:
                        self.matched_reference_voxels.add(candidate)
            recall: float | None = len(self.matched_reference_voxels) / len(self.reference_voxels)
        else:
            recall = None
        elapsed = float(stamp_sec - self.start_stamp_sec)
        cycle = CoverageCycle(
            stamp_sec=float(stamp_sec),
            elapsed_sec=elapsed,
            explored_voxels=len(self.covered_voxels),
            explored_volume_m3=len(self.covered_voxels) * self.config.explored_volume_voxel_m**3,
            traveling_distance_m=float(self.traveling_distance_m),
            scan_points=int(len(scan)),
            new_voxels=int(new_voxels),
            cumulative_point_redundancy=float(1.0 - len(self.covered_voxels) / max(self.total_scan_points, 1)),
            complete_map_surface_recall_diagnostic=recall,
        )
        self.cycles.append(cycle)
        self.last_stamp_sec = float(stamp_sec)
        return cycle

    def summary(self, *, budget_sec: float | None = None) -> dict[str, Any]:
        if not self.cycles:
            raise RuntimeError("cannot summarize empty coverage")
        times = np.asarray([item.elapsed_sec for item in self.cycles], dtype=np.float64)
        volumes = np.asarray([item.explored_volume_m3 for item in self.cycles], dtype=np.float64)
        duration = float(times[-1] if budget_sec is None else budget_sec)
        if not math.isfinite(duration) or duration <= 0.0 or times[-1] > duration + 1e-9:
            raise ValueError("budget must be positive and cover every recorded cycle")
        if times[-1] < duration:
            times = np.r_[times, duration]
            volumes = np.r_[volumes, volumes[-1]]
        auc = float(np.sum((volumes[:-1] + volumes[1:]) * 0.5 * np.diff(times)))
        final = self.cycles[-1]
        return {
            "schema_version": "mtare_method_independent_coverage_v1",
            "config": self.config.to_dict(),
            "duration_sec": duration,
            "cycle_count": len(self.cycles),
            "final_explored_voxels": final.explored_voxels,
            "final_explored_volume_m3": final.explored_volume_m3,
            "coverage_time_auc_m3_s": auc,
            "mean_explored_volume_m3": auc / duration,
            "traveling_distance_m": final.traveling_distance_m,
            "cumulative_point_redundancy": final.cumulative_point_redundancy,
            "complete_map_surface_recall_diagnostic": final.complete_map_surface_recall_diagnostic,
            "reference_denominator_warning": (
                None
                if self.reference_voxels is None
                else "Diagnostic only: complete preview surfaces are not a qualified reachable/observable denominator."
            ),
        }
