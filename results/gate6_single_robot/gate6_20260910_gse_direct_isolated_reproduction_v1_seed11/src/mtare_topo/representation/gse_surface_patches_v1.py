"""Deterministic observed surface patches; no teacher, files or model inputs.

Voxel boundaries are fixed in the current sensor frame: extraction is NOT
strictly rotation equivariant. All five frames must already be registered by
the caller, who also authenticates timestamps and ray origins. Outside-ROI
returns are retained as rays, never clipped into fabricated surface patches.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfacePatches:
    centers_m: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray
    normal_uncertainty: np.ndarray
    bounds_min_m: np.ndarray
    bounds_max_m: np.ndarray
    roughness_m: np.ndarray
    point_count: np.ndarray
    frame_support: np.ndarray
    point_patch_index: np.ndarray
    ray_endpoints_m: np.ndarray
    ray_origins_m: np.ndarray | None
    ray_frame_index: np.ndarray
    ray_input_index: np.ndarray
    voxel_size_m: float
    roi_radius_m: float


@dataclass(frozen=True)
class PatchRelations:
    neighbor_index: np.ndarray       # M,8; -1 means absent, not blocked
    valid: np.ndarray
    relative_xyz_m: np.ndarray       # neighbor minus receiver
    distance_m: np.ndarray
    normal_abs_dot: np.ndarray       # unsigned PCA normals
    normal_pair_valid: np.ndarray
    ray_evidence: np.ndarray         # M,8,3: free / obstacle / unknown
    evidence_requires_source_audit: bool = True
    physical_connectivity: bool = False


def _readonly(array):
    array.setflags(write=False)
    return array


def extract_surface_patches(points_xyz_m, valid, frame_index, *, ray_origins_m=None,
                            voxel_size_m=.5, roi_radius_m=10., max_patches=4096):
    """Extract one observation of unordered, aligned points from five frames.

    Degenerate cells are retained with normal_valid=False and a zero numeric
    placeholder. This mask MUST accompany normals downstream. The function
    does not infer free space from absent returns or unknown ray origins.
    """
    xyz, valid, frame = points_xyz_m, valid, frame_index
    if (type(xyz) is not np.ndarray or xyz.ndim != 2 or xyz.shape[1] != 3
            or xyz.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or type(valid) is not np.ndarray or valid.shape != (len(xyz),) or valid.dtype != np.bool_
            or type(frame) is not np.ndarray or frame.shape != valid.shape or frame.dtype.kind not in "iu"
            or np.any(frame > 4) or np.any(frame < 0)):
        raise ValueError("N,3 float XYZ / N bool validity / N integer causal frame0..4 required")
    if any(type(v) not in (float, int) or not np.isfinite(v) or v <= 0 for v in (voxel_size_m, roi_radius_m)):
        raise ValueError("finite positive voxel and ROI sizes required")
    if type(max_patches) is not int or not 1 <= max_patches <= 4096:
        raise ValueError("explicit capacity in1..4096 required")
    if not np.isfinite(xyz[valid]).all():
        raise ValueError("nonfinite valid surface return")
    if ray_origins_m is not None and (type(ray_origins_m) is not np.ndarray or ray_origins_m.shape != xyz.shape
            or ray_origins_m.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(ray_origins_m[valid]).all()):
        raise ValueError("explicit same-frame finite ray origins required")
    rays = np.flatnonzero(valid)
    selected = rays[np.linalg.norm(xyz[rays].astype(np.float64), axis=1) <= roi_radius_m]
    points = xyz[selected].astype(np.float64)
    cells = np.floor(points / voxel_size_m).astype(np.int64)
    # Canonical reduction order makes input permutations deterministic, also
    # for floating point summation/PCA, not merely set-equivalent.
    if len(points):
        order = np.lexsort((frame[selected], points[:, 2], points[:, 1], points[:, 0],
                            cells[:, 2], cells[:, 1], cells[:, 0]))
        points, cells, selected = points[order], cells[order], selected[order]
    _, first, sizes = np.unique(cells, axis=0, return_index=True, return_counts=True)
    if len(first) > max_patches:
        raise OverflowError(f"surface patch capacity exceeded:{len(first)}>{max_patches}; never truncate")
    m = len(first)
    centers = np.zeros((m, 3)); normals = np.zeros((m, 3)); normal_valid = np.zeros(m, dtype=bool)
    uncertainty = np.ones(m); low = np.zeros((m, 3)); high = np.zeros((m, 3)); rough = np.zeros(m)
    supports = np.zeros((m, 5), dtype=np.int64); point_to_patch = np.full(len(xyz), -1, dtype=np.int64)
    for patch, (start, count) in enumerate(zip(first, sizes)):
        p = points[start:start + count]; original = selected[start:start + count]
        center = p.mean(0); centers[patch] = center; low[patch] = p.min(0); high[patch] = p.max(0)
        residual = p - center; covariance = residual.T @ residual / count
        values, vectors = np.linalg.eigh(covariance); values = np.maximum(values, 0.)
        rough[patch] = np.sqrt(values[0]); tolerance = 64 * np.finfo(np.float64).eps * max(1., values[-1])
        reliable = count >= 3 and values[1] > tolerance and values[1] - values[0] > tolerance
        if reliable:
            normal = vectors[:, 0]
            # Face the current sensor origin; tangent ties get a canonical sign.
            dot = float(normal @ center)
            if dot > tolerance or (abs(dot) <= tolerance and normal[np.argmax(np.abs(normal))] < 0):
                normal = -normal
            normals[patch] = normal; normal_valid[patch] = True
            uncertainty[patch] = 1. - (values[1] - values[0]) / max(values[-1], tolerance)
        supports[patch] = np.bincount(frame[original].astype(np.int64), minlength=5)
        point_to_patch[original] = patch
    arrays = [centers, normals, normal_valid, uncertainty, low, high, rough,
              sizes.astype(np.int64), supports, point_to_patch, xyz[rays].astype(np.float64).copy(),
              None if ray_origins_m is None else ray_origins_m[rays].astype(np.float64).copy(),
              frame[rays].astype(np.int64).copy(), rays]
    return SurfacePatches(*[None if a is None else _readonly(a) for a in arrays],
                          float(voxel_size_m), float(roi_radius_m))


def build_patch_neighbors(patches, *, k=8):
    """Sparse3D message neighbors, never a traversability/connection graph.

    This initial extractor does not certify line segments using sparse rays.
    Consequently ray relation evidence remains UNKNOWN, even when ray origins
    are available. A source-bound ray-evidence kernel is a separate interface;
    proximity, normal agreement and empty space are never substituted for it.
    """
    if type(patches) is not SurfacePatches or type(k) is not int or k != 8:
        raise ValueError("SurfacePatches and fixed k8 required")
    centers = patches.centers_m; m = len(centers)
    index = np.full((m, k), -1, dtype=np.int64)
    if m > 1:
        tree = cKDTree(centers)
        distance, _ = tree.query(centers, k=min(m, k + 1), workers=1)
        for i in range(m):
            radius = float(distance[i, -1])
            candidates = tree.query_ball_point(centers[i], np.nextafter(radius, np.inf))
            candidates = [j for j in candidates if j != i]
            # Resolve kth-distance ties by geometric coordinates, not input IDs.
            candidates.sort(key=lambda j: (float(np.linalg.norm(centers[j] - centers[i])), *centers[j]))
            chosen = candidates[:k]; index[i, :len(chosen)] = chosen
    mask = index >= 0; safe = np.maximum(index, 0)
    relative = np.zeros((m, k, 3)); normal_dot = np.zeros((m, k)); normal_pair = np.zeros((m, k), dtype=bool)
    if m:
        relative = np.where(mask[..., None], centers[safe] - centers[:, None], 0.)
        normal_pair = mask & patches.normal_valid[:, None] & patches.normal_valid[safe]
        normal_dot = np.where(normal_pair, np.abs(np.sum(patches.normals[:, None] * patches.normals[safe], axis=-1)), 0.)
    distance = np.linalg.norm(relative, axis=-1)
    evidence = np.zeros((m, k, 3)); evidence[..., 2] = 1.
    return PatchRelations(*[_readonly(a) for a in (index, mask, relative, distance,
                           normal_dot, normal_pair, evidence)])


__all__ = ["SurfacePatches", "PatchRelations", "extract_surface_patches", "build_patch_neighbors"]
