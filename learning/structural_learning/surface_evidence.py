"""Ray-free local surface-evidence tensor shared by LAMP and M-TARE adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from learning.local_structural_map.geometry import grid_indices, transform_points_world_to_local
from learning.local_structural_map.schema import Pose2D


@dataclass(frozen=True)
class SurfaceEvidenceConfig:
    contract_version: str = "surface_evidence_v1"
    size_m: float = 20.0
    resolution_m: float = 0.2
    z_min_m: float = -2.5
    z_max_m: float = 4.0
    density_cap: int = 8

    @property
    def grid_size(self) -> int:
        return int(round(self.size_m / self.resolution_m))

    @property
    def channel_names(self) -> tuple[str, ...]:
        return ("surface_mask", "log_density", "mean_height", "height_span")


class SurfaceEvidenceBuilder:
    """Projects world-coordinate surface points into a center-pose local tensor.

    The builder deliberately contains no ray traversal or free-space inference.
    A zero-valued cell means no retained surface point, not free space.
    """

    def __init__(self, config: SurfaceEvidenceConfig) -> None:
        self.config = config

    def build(self, point_sets_world: Iterable[np.ndarray], center_pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
        arrays = [np.asarray(points, dtype=np.float32).reshape(-1, 3) for points in point_sets_world]
        points_world = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 3), dtype=np.float32)
        if len(points_world) == 0:
            shape = (len(self.config.channel_names), self.config.grid_size, self.config.grid_size)
            return np.zeros(shape, dtype=np.float32), points_world
        finite = np.isfinite(points_world).all(axis=1)
        local = transform_points_world_to_local(points_world[finite], center_pose)
        rows, cols, valid_xy = grid_indices(local[:, :2], self.config.size_m, self.config.resolution_m)
        valid_z = (local[:, 2] >= self.config.z_min_m) & (local[:, 2] <= self.config.z_max_m)
        valid = valid_xy & valid_z
        local = local[valid]
        rows, cols = rows[valid], cols[valid]
        n = self.config.grid_size
        tensor = np.zeros((len(self.config.channel_names), n, n), dtype=np.float32)
        if len(local) == 0:
            return tensor, local
        flat = rows.astype(np.int64) * n + cols.astype(np.int64)
        count = np.bincount(flat, minlength=n * n).astype(np.float32)
        zsum = np.bincount(flat, weights=local[:, 2], minlength=n * n).astype(np.float32)
        zmin = np.full(n * n, np.inf, dtype=np.float32)
        zmax = np.full(n * n, -np.inf, dtype=np.float32)
        np.minimum.at(zmin, flat, local[:, 2])
        np.maximum.at(zmax, flat, local[:, 2])
        mask = count > 0
        tensor[0] = mask.reshape(n, n).astype(np.float32)
        tensor[1] = (np.log1p(np.minimum(count, self.config.density_cap)) / np.log1p(self.config.density_cap)).reshape(n, n)
        mean_height = np.zeros(n * n, dtype=np.float32)
        mean_height[mask] = (zsum[mask] / count[mask] - self.config.z_min_m) / (self.config.z_max_m - self.config.z_min_m)
        tensor[2] = np.clip(mean_height, 0.0, 1.0).reshape(n, n)
        span = np.zeros(n * n, dtype=np.float32)
        span[mask] = (zmax[mask] - zmin[mask]) / (self.config.z_max_m - self.config.z_min_m)
        tensor[3] = np.clip(span, 0.0, 1.0).reshape(n, n)
        return tensor, local
