"""Source-mesh intersection adapter for causal, finite-ray diagnostics.

Every source is intersected separately so coincident source interfaces survive.
Mesh/interface identities are teacher-only references, never model candidates.
"""
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .gse_caster_cap_replay_v1 import pack_caster_inputs
from .gse_ordered_interface_crossings_v1 import ordered_crossings


@dataclass(frozen=True)
class SourceInterfaces:
    source_key: str
    mesh: Any
    face_interfaces: Mapping[int, int]


def replay_interfaces(sources, *, origins, directions, first_return, valid,
                      source_frame_indices, roi_center, roi_radius_m=10.):
    """Intersect original full meshes, then select named interface faces.

    first_return must be the original float32 stored caster parameter. A ray's
    origin/direction follows normalize64 -> cast32, just as the original caster.
    ROI membership is evaluated on those actual quantized intersection points.
    """
    sources = tuple(sources)
    origins = np.asarray(origins)
    frames = np.asarray(source_frame_indices)
    center = np.asarray(roi_center, dtype=np.float64)
    if (not sources or origins.ndim != 2 or origins.shape[1:] != (3,)
            or not 0 < len(origins) <= 57600
            or frames.shape != (len(origins),) or frames.dtype.kind not in 'iu'
            or np.any(frames < 0) or np.any(frames > 4)
            or center.shape != (3,) or not np.isfinite(center).all()
            or roi_radius_m != 10.):
        raise ValueError('bounded causal rays, source meshes and fixed 10m ROI required')
    # Validate every source and ownership before invoking the backend.
    packed = []
    owners = {}
    keys = set()
    for source in sources:
        if not isinstance(source.source_key, str) or not source.source_key or source.source_key in keys:
            raise ValueError('unique source key required')
        keys.add(source.source_key)
        vertices, rays = pack_caster_inputs(source.mesh.vertices_xyz_m, origins, directions)
        triangles = np.asarray(source.mesh.triangle_vertex_indices)
        if (triangles.ndim != 2 or triangles.shape[1:] != (3,)
                or triangles.dtype.kind not in 'iu' or not len(triangles)
                or np.any(triangles < 0) or np.any(triangles >= len(vertices))
                or not source.face_interfaces):
            raise ValueError('source triangles and interface face map required')
        face_map = {}
        for face, interface in source.face_interfaces.items():
            if (isinstance(face, bool) or not isinstance(face, (int, np.integer))
                    or not 0 <= face < len(triangles)
                    or isinstance(interface, bool) or not isinstance(interface, (int, np.integer))
                    or interface < 0):
                raise ValueError('integer source face and interface identifiers required')
            interface = int(interface)
            if interface in owners and owners[interface] != source.source_key:
                raise ValueError('interface identity has multiple source owners')
            owners[interface] = source.source_key
            face_map[int(face)] = interface
        packed.append((source, vertices, rays, triangles, face_map))
    # Reuse input contract validation before expensive intersection calls.
    empty = np.empty(0, dtype=np.int64)
    ordered_crossings(ray_ids=empty, interface_ids=empty,
        hit_t=np.empty(0, dtype=np.float32), first_return=first_return,
        valid=valid, inside_roi=np.empty(0, dtype=bool))
    if np.asarray(first_return).shape != (len(origins),):
        raise ValueError('one saved return per ray required')
    import open3d as o3d
    records = []
    for source, vertices, rays, triangles, face_map in packed:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices), o3d.core.Tensor(triangles.astype(np.uint32))))
        raw = {k: v.numpy() for k, v in
               scene.list_intersections(o3d.core.Tensor(rays)).items()}
        if raw['t_hit'].dtype != np.float32:
            raise ValueError('backend hit precision differs from source contract')
        for i in np.flatnonzero(np.isin(raw['primitive_ids'], list(face_map))):
            ray = int(raw['ray_ids'][i])
            face = int(raw['primitive_ids'][i])
            t = float(raw['t_hit'][i])
            xyz = rays[ray, :3].astype(np.float64) + t * rays[ray, 3:].astype(np.float64)
            records.append(dict(ray_index=ray, source_frame_index=int(frames[ray]),
                source_key_teacher_only=source.source_key, triangle_index=face,
                interface_id_teacher_only=face_map[face], t=t,
                intersection_world_m=xyz.tolist(),
                inside_roi=bool(np.linalg.norm(xyz-center) < roi_radius_m)))
    records.sort(key=lambda r: (r['ray_index'], r['t'], r['interface_id_teacher_only'], r['triangle_index']))
    result = ordered_crossings(
        ray_ids=np.array([r['ray_index'] for r in records], dtype=np.int64),
        interface_ids=np.array([r['interface_id_teacher_only'] for r in records], dtype=np.int64),
        hit_t=np.array([r['t'] for r in records], dtype=np.float32),
        first_return=first_return, valid=valid,
        inside_roi=np.array([r['inside_roi'] for r in records], dtype=bool))
    result.update(backend_version=o3d.__version__, raw_interface_intersections=records,
                  interface_owners_teacher_only=owners)
    return result
