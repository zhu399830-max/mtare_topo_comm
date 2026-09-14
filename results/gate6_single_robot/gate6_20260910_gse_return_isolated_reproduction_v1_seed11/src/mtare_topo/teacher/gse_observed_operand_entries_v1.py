"""Original-ray entries through complete operand surfaces, including sides.

Caller supplies original caster meshes, float32 packed rays, observed first
returns, and bound source sets. These are geometric witnesses, not labels.
No end-cap-only assumption, occupancy filling, or hidden-ray rendering.
"""
import numpy as np


def observed_operand_entries(caster, packed_rays, *, first_return, valid,
                             return_sources, center_m, radius_m=10.):
    import open3d as o3d
    rays = np.asarray(packed_rays)
    ranges = np.asarray(first_return)
    mask = np.asarray(valid)
    center = np.asarray(center_m)
    if (rays.ndim != 2 or rays.shape[1] != 6 or rays.dtype != np.float32
            or not np.isfinite(rays).all() or np.any(np.linalg.norm(rays[:, 3:], axis=1) == 0)
            or ranges.shape != (len(rays),) or ranges.dtype != np.float32
            or mask.shape != ranges.shape or mask.dtype != bool
            or len(return_sources) != len(rays) or center.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(radius_m) or radius_m <= 0
            or np.any(~np.isfinite(ranges[mask])) or np.any(ranges[mask] < 0)):
        raise ValueError('original packed finite rays, float32 returns and source binding required')
    raw = {k: v.numpy() for k, v in caster.scene.list_intersections(o3d.core.Tensor(rays)).items()}
    hits = {}; ambiguous = set()
    for ri, gi, ti, t in zip(raw['ray_ids'], raw['geometry_ids'], raw['primitive_ids'], raw['t_hit']):
        ri, gi, ti, t = int(ri), int(gi), int(ti), float(t)
        operand = caster.geometry_to_operand[gi]
        source = caster.primitive_ids[operand]
        if not mask[ri] or return_sources[ri] != [source]:
            continue
        limit = float(ranges[ri]) - abs(float(np.spacing(ranges[ri])))
        if not np.isfinite(t) or not 0 < t < limit:
            continue
        xyz = rays[ri, :3].astype(np.float64) + t * rays[ri, 3:].astype(np.float64)
        if np.linalg.norm(xyz - center) >= radius_m:
            continue
        normal = caster.meshes[operand].triangle_normals[ti]
        dot = float(normal @ rays[ri, 3:].astype(np.float64))
        guard = 64*np.finfo(np.float64).eps*np.linalg.norm(normal)*np.linalg.norm(rays[ri, 3:].astype(np.float64))
        key = (ri, operand, t)
        if dot >= -guard:
            ambiguous.add(key)
        else:
            hits.setdefault(key, []).append((ti, xyz.tolist()))
    result = []
    for (ri, operand, t), entries in sorted(hits.items()):
        if (ri, operand, t) in ambiguous:
            continue
        result.append(dict(ray_index=ri, source_id_teacher_only=caster.primitive_ids[operand],
            t=t, intersection_world_m=entries[0][1],
            triangle_indices=sorted({ti for ti, _ in entries})))
    return dict(entries=result, membership=None, physical_separation_certified=False)
