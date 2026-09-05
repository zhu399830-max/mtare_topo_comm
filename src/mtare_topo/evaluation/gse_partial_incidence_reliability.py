"""Runtime-only incidence support for partially observed decision nodes."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def physical_edge_identity(traversal_id: str) -> str:
    if ":d" not in traversal_id:
        raise ValueError("traversal identity has no directed suffix")
    physical, direction = traversal_id.rsplit(":d", 1)
    if direction not in ("0", "1") or not physical:
        raise ValueError("traversal direction contract drift")
    return physical


def physical_incidence_count(traversal_ids: Iterable[str]) -> int:
    values = tuple(str(value) for value in traversal_ids)
    if not values:
        raise ValueError("incidence support must be non-empty")
    return len({physical_edge_identity(value) for value in values})


def unanimous_seed_metric_support(
    seed_center_xyz_m: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    distance_cap_m: float = 4.0,
) -> np.ndarray:
    center = np.asarray(seed_center_xyz_m, dtype=np.float64)
    left_rows = np.asarray(left, dtype=np.int64)
    right_rows = np.asarray(right, dtype=np.int64)
    if (
        center.ndim != 3 or center.shape[0] != 3 or center.shape[2] != 3
        or left_rows.shape != right_rows.shape or left_rows.ndim != 1
        or distance_cap_m <= 0.0 or np.any(left_rows < 0) or np.any(right_rows < 0)
        or np.any(left_rows >= center.shape[1]) or np.any(right_rows >= center.shape[1])
    ):
        raise ValueError("unanimous metric support contract drift")
    distance = np.linalg.norm(center[:, left_rows] - center[:, right_rows], axis=2)
    return np.all(distance <= distance_cap_m + 1e-12, axis=0)


def incidence_commit_allowed(event: str, traversal_ids: Iterable[str], *, junction_support: int = 3) -> bool:
    if event == "terminal":
        return True
    if event != "junction" or junction_support < 1:
        raise ValueError("incidence commit event contract drift")
    return physical_incidence_count(traversal_ids) >= junction_support


__all__ = [
    "incidence_commit_allowed", "physical_edge_identity", "physical_incidence_count",
    "unanimous_seed_metric_support",
]
