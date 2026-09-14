import numpy as np


def ray_cells(origin_xy: np.ndarray, end_xy: np.ndarray, resolution_m: float, size_m: float, step_m: float) -> np.ndarray:
    delta = np.asarray(end_xy, dtype=np.float32) - np.asarray(origin_xy, dtype=np.float32)
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-6:
        return np.empty((0, 2), dtype=np.int32)
    steps = max(1, int(np.ceil(dist / max(step_m, 1e-3))))
    ts = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[:-1]
    pts = origin_xy.reshape(1, 2) + ts.reshape(-1, 1) * delta.reshape(1, 2)
    half = size_m / 2.0
    cols = np.floor((pts[:, 0] + half) / resolution_m).astype(np.int32)
    rows = np.floor((half - pts[:, 1]) / resolution_m).astype(np.int32)
    n = int(round(size_m / resolution_m))
    valid = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
    cells = np.stack([rows[valid], cols[valid]], axis=1)
    if cells.size == 0:
        return cells
    keep = np.ones(len(cells), dtype=bool)
    keep[1:] = np.any(cells[1:] != cells[:-1], axis=1)
    return cells[keep]
