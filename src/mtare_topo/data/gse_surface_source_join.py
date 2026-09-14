"""Join saved source candidates by original ray identity, without qualifying labels.

Residuals are evidence, not a threshold or a nearest-source selection rule.
The result deliberately has no ``evidence_known`` field: an active-field source
code is not proof of the actual first-return surface identity.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JoinedSurfaceSources:
    ray_indices: np.ndarray
    candidates: tuple
    scene_residuals_m: tuple
    world_residuals_m: tuple
    missing: np.ndarray
    ambiguous: np.ndarray
    point_label_qualified: bool = False


def join_surface_sources(records, roi_return_indices, feature_return_indices):
    """Bind Nx7 saved residual records to the *feature* return order.

    Both inventories must describe the same unique original rays. Candidate
    order is canonicalized by source index, never by residual or model score.
    Missing candidate rows remain explicit unknown evidence; malformed or
    duplicate ray/source rows fail rather than being silently deduplicated.
    """
    def indices(value):
        a = np.asarray(value)
        if (a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a < 0)
                or len(np.unique(a)) != len(a)):
            raise ValueError('unique nonnegative original ray indices required')
        return a
    roi = indices(roi_return_indices)
    feature = indices(feature_return_indices)
    if not np.array_equal(np.sort(roi), np.sort(feature)):
        raise ValueError('source and feature return inventory mismatch')
    rows = np.asarray(records)
    if rows.ndim != 2 or rows.shape[1] != 7 or rows.dtype.kind not in 'fi':
        raise ValueError('saved Nx7 numeric residual records required')
    if not np.isfinite(rows).all():
        raise ValueError('nonfinite source evidence')
    integral = rows[:, [0, 1, 4]]
    if np.any(integral != np.floor(integral)) or np.any(integral < 0):
        raise ValueError('ray/source/tie-count fields must be nonnegative integers')
    if np.any(rows[:, 2:4] < 0) or np.any(rows[:, 4] < 1) or np.any(rows[:, 5] > rows[:, 6]):
        raise ValueError('invalid saved residual evidence')
    by_ray = {int(i): {} for i in roi}
    for row in rows:
        ray, source = int(row[0]), int(row[1])
        if ray not in by_ray:
            raise ValueError('source row outside frozen ROI')
        if source in by_ray[ray]:
            raise ValueError('duplicate ray/source record')
        by_ray[ray][source] = (float(row[2]), float(row[3]))
    candidates, scene, world = [], [], []
    for ray in feature:
        entries = by_ray[int(ray)]
        ordered = sorted(entries)
        candidates.append(tuple(str(s) for s in ordered))
        scene.append(tuple(entries[s][0] for s in ordered))
        world.append(tuple(entries[s][1] for s in ordered))
    missing = np.array([not c for c in candidates], dtype=bool)
    ambiguous = np.array([len(c) > 1 for c in candidates], dtype=bool)
    rays = feature.copy()
    for a in (rays, missing, ambiguous):
        a.setflags(write=False)
    return JoinedSurfaceSources(rays, tuple(candidates), tuple(scene), tuple(world), missing, ambiguous)
