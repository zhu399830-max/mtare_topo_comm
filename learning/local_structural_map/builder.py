from collections import deque
from typing import Deque, Optional

import numpy as np

from .config import LocalMapConfig
from .geometry import grid_indices, transform_points_world_to_local
from .raycast import ray_cells
from .schema import LocalStructuralMap, StandardFrame


class LocalStructuralMapBuilder:
    def __init__(self, config: Optional[LocalMapConfig] = None) -> None:
        self.config = config or LocalMapConfig()
        self._frames: Deque[StandardFrame] = deque()

    def reset(self) -> None:
        self._frames.clear()

    def update(self, frame: StandardFrame) -> LocalStructuralMap:
        clean = self._filtered_copy(frame)
        self._frames.append(clean)
        self._trim(clean.timestamp_ns)
        return self.build(clean)

    def build_single(self, frame: StandardFrame) -> LocalStructuralMap:
        clean = self._filtered_copy(frame)
        return self._build_from_points(clean, clean.points_world, clean.sensor_origin_world)

    def build(self, current: StandardFrame) -> LocalStructuralMap:
        if not self._frames:
            return self.build_single(current)
        pts = np.concatenate([f.points_world for f in self._frames if len(f.points_world)], axis=0)
        return self._build_from_points(current, pts, current.sensor_origin_world)

    def _trim(self, now_ns: int) -> None:
        cfg = self.config
        min_ns = now_ns - int(cfg.accumulation_window_sec * 1e9)
        while self._frames and self._frames[0].timestamp_ns < min_ns:
            self._frames.popleft()
        while len(self._frames) > cfg.max_frames_in_window:
            self._frames.popleft()

    def _filtered_copy(self, frame: StandardFrame) -> StandardFrame:
        cfg = self.config
        pts = frame.points_world
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        rel = pts - frame.sensor_origin_world.reshape(1, 3)
        ranges = np.linalg.norm(rel, axis=1)
        z_rel = pts[:, 2] - frame.pose.z
        keep = (
            (ranges >= cfg.min_range_m)
            & (ranges <= cfg.max_range_m)
            & (np.linalg.norm(rel[:, :2], axis=1) >= cfg.robot_self_radius_m)
            & (z_rel >= cfg.min_z_rel_m)
            & (z_rel <= cfg.max_z_rel_m)
        )
        pts = self._voxel_downsample(pts[keep])
        return StandardFrame(
            timestamp_ns=frame.timestamp_ns,
            points_world=pts,
            sensor_origin_world=frame.sensor_origin_world,
            pose=frame.pose,
            source=frame.source,
            frame_id=frame.frame_id,
            metadata={
                **frame.metadata,
                "valid_points_after_filter": int(keep.sum()),
                "points_after_voxel_downsample": int(len(pts)),
                "raw_points": int(len(frame.points_world)),
                "voxel_size_m": float(cfg.voxel_size_m),
            },
        )

    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        cfg = self.config
        if len(pts) == 0 or cfg.voxel_size_m <= 0:
            return pts.astype(np.float32, copy=False)
        keys = np.floor(pts / cfg.voxel_size_m).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        keep_sorted = np.ones(len(order), dtype=bool)
        keep_sorted[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        selected = np.sort(order[keep_sorted])
        return pts[selected].astype(np.float32, copy=False)

    def _build_from_points(self, frame: StandardFrame, points_world: np.ndarray, origin_world: np.ndarray) -> LocalStructuralMap:
        cfg = self.config
        n = cfg.grid_size
        observed = np.zeros((n, n), dtype=np.float32)
        free = np.zeros((n, n), dtype=np.float32)
        occupied = np.zeros((n, n), dtype=np.float32)
        min_z = np.full((n, n), np.inf, dtype=np.float32)
        max_z = np.full((n, n), -np.inf, dtype=np.float32)
        z_values = [[[] for _ in range(n)] for _ in range(n)]

        if len(points_world):
            local_pts = transform_points_world_to_local(points_world, frame.pose)
            origin_local = transform_points_world_to_local(origin_world.reshape(1, 3), frame.pose)[0]
            rows, cols, valid = grid_indices(local_pts[:, :2], cfg.size_m, cfg.resolution_m)
            local_pts = local_pts[valid]
            rows = rows[valid]
            cols = cols[valid]
            for p, r, c in zip(local_pts, rows, cols):
                cells = ray_cells(origin_local[:2], p[:2], cfg.resolution_m, cfg.size_m, cfg.ray_step_m)
                if len(cells):
                    free[cells[:, 0], cells[:, 1]] = 1.0
                    observed[cells[:, 0], cells[:, 1]] = 1.0
                occupied[r, c] = 1.0
                observed[r, c] = 1.0
                free[r, c] = 0.0
                z = float(p[2])
                min_z[r, c] = min(min_z[r, c], z)
                max_z[r, c] = max(max_z[r, c], z)
                z_values[r][c].append(z)

        height_valid = np.isfinite(min_z) & np.isfinite(max_z) & (occupied > 0)
        z_band_maps = [np.zeros((n, n), dtype=np.float32) for _ in cfg.occupancy_z_bands_m]
        endpoint_height_range = np.zeros((n, n), dtype=np.float32)
        for r, c in np.argwhere(height_valid):
            vals = np.asarray(z_values[int(r)][int(c)], dtype=np.float32)
            endpoint_height_range[r, c] = self._norm_endpoint_height_range(float(np.max(vals) - np.min(vals)))
            for band_idx, (lo, hi) in enumerate(cfg.occupancy_z_bands_m):
                z_band_maps[band_idx][r, c] = float(np.any((vals >= lo) & (vals < hi)))

        channel_map = {
            "observed_mask": observed,
            "free_mask": free,
            "occupied_mask": occupied,
            "endpoint_height_range": endpoint_height_range,
        }
        for band_idx, band_map in enumerate(z_band_maps):
            channel_map[f"occupancy_z_band_{band_idx}"] = band_map
        tensor = np.stack([channel_map[name] for name in cfg.channels], axis=0).astype(np.float32)
        return LocalStructuralMap(
            tensor=tensor,
            channel_names=list(cfg.channels),
            timestamp_ns=frame.timestamp_ns,
            source=frame.source,
            pose=frame.pose,
            metadata={
                **frame.metadata,
                "builder_class": self.__class__.__name__,
                "accumulated_frames": len(self._frames),
                "accumulated_points": int(len(points_world)),
                "height_valid_cells": int(height_valid.sum()),
                "z_bands_m": [list(b) for b in cfg.occupancy_z_bands_m],
            },
        )

    def _norm_endpoint_height_range(self, value: float) -> float:
        cfg = self.config
        hi = max(cfg.endpoint_height_range_norm_max_m, 1e-6)
        return float(np.clip(value / hi, 0.0, 1.0))
