"""Hash-bound historical mesh execution in a separate Python process.

No current teacher imports are replaced. This produces surface geometry and
arc provenance only, never return-point labels or structural membership.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

import numpy as np


_WORKER = r'''
import sys, json, io
sys.path.insert(0, sys.argv[1])
import numpy as np
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive, _sample_operand
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
r = json.load(sys.stdin)
if r.get('mode') == 'ray_points':
    from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
    ranges = np.asarray(r['ranges_m'], dtype=np.float32)
    origins = np.asarray(r['sensor_xyz_m'], dtype=np.float64)
    yaw = np.asarray(r['yaw_deg'], dtype=np.float64)
    if ranges.shape != (5,16,720) or origins.shape != (5,3) or yaw.shape != (5,):
        raise ValueError('original five-frame shapes required')
    if not all(np.isfinite(x).all() for x in (ranges, origins, yaw)):
        raise ValueError('nonfinite ray input')
    directions = np.stack([world_directions(lidar_local_directions().reshape(-1,3).astype(np.float64), y) for y in yaw])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(origins[:,None,:], directions.shape)
    distance = ranges.reshape(5,-1,1).astype(np.float64)
    b = io.BytesIO()
    np.savez(b, world_points_xyz_m=origins+distance*directions,
             scene_points_xyz_m=origins.astype(np.float32).astype(np.float64)+distance*directions.astype(np.float32).astype(np.float64))
    sys.stdout.buffer.write(b.getvalue())
    sys.exit(0)
p = SweptSuperellipsePrimitive(**r['primitive'])
m = mesh_swept_superellipse(p, axial_spacing_m=r['spacing'], angular_segments=r['angular'])
s = _sample_operand(p, r['spacing'])
a = np.concatenate((np.repeat(s.arc_m, r['angular']), s.arc_m[[0, -1]]))
if len(a) != len(m.vertices_xyz_m):
    raise ValueError('historical vertex arc mismatch')
f = a[m.triangle_vertex_indices]
b = io.BytesIO()
np.savez(b, vertices_xyz_m=m.vertices_xyz_m,
         scene_vertices_xyz_m=m.vertices_xyz_m.astype(np.float32),
         triangle_vertex_indices=m.triangle_vertex_indices,
         triangle_normals=m.triangle_normals, vertex_arc_m=a,
         triangle_arc_bounds_m=np.stack((f.min(1), f.max(1)), axis=1))
sys.stdout.buffer.write(b.getvalue())
'''


def load_original_surface(repo_root, binding, primitive, *, timeout_s=60):
    """Execute authenticated archive sources with original tessellation settings.

    ``primitive`` is a JSON-compatible construction record, not a model input.
    Only archived Python source is unpacked, into an automatically removed temp
    directory. The child uses -I and an explicit archive-only project path.
    """
    request = dict(primitive=primitive, spacing=binding['axial_spacing_m'],
                   angular=binding['angular_segments'])
    return _execute_archive(repo_root,binding,request,timeout_s=timeout_s)


def load_original_ray_points(repo_root, binding, ranges_m, sensor_xyz_m, yaw_deg, *, timeout_s=60):
    """Two original representations: reported world point and float32 scene ray.

    Invalid-return slots remain slots; callers apply only the recorded validity
    mask, never a teacher semantic mask. Scene direction is not renormalized
    after casting, matching the original Open3D ray parameterization.
    """
    return _execute_archive(repo_root,binding,dict(mode='ray_points',
        ranges_m=np.asarray(ranges_m).tolist(),sensor_xyz_m=np.asarray(sensor_xyz_m).tolist(),
        yaw_deg=np.asarray(yaw_deg).tolist()),timeout_s=timeout_s)


def _execute_archive(repo_root, binding, request, *, timeout_s):
    root = Path(repo_root).resolve()
    archive = (root / binding['archive_path']).resolve()
    if not archive.is_relative_to(root):
        raise ValueError('archive outside repository')
    blob = archive.read_bytes()
    if hashlib.sha256(blob).hexdigest() != binding['archive_sha256']:
        raise ValueError('archive SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z, tempfile.TemporaryDirectory(prefix='gse-original-') as tmp:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive member')
        for name, digest in binding['sources'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('historical source SHA mismatch')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe archive member')
            if name.startswith('src/') and name.endswith('.py'):
                target = Path(tmp) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
        result = subprocess.run([sys.executable, '-I', '-c', _WORKER, str(Path(tmp)/'src')],
                                input=json.dumps(request).encode(), capture_output=True,
                                timeout=timeout_s, check=True)
    with np.load(io.BytesIO(result.stdout), allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for value in arrays.values():
        value.setflags(write=False)
    return arrays
