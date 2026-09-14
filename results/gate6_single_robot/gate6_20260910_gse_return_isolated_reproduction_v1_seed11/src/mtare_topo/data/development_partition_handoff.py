"""NumPy-only partition interchange between native SPG and Torch processes.

No file IO, labels, or model calls. The frozen runner authenticates producers;
the consumer independently requires the exact cache's points and ray indices.
"""
import hashlib
import io
import json
import re
import numpy as np


def produce_partitions(cache_payload, *, source, cache_sha256, geof_backend, partition_backend):
    """Native-side body for the frozen executor; no Torch or filesystem IO."""
    from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
    from mtare_topo.representation.gse_spg_extractor import extract_spg
    if hashlib.sha256(cache_payload).hexdigest()!=cache_sha256:
        raise ValueError('native source cache hash mismatch')
    with np.load(io.BytesIO(cache_payload),allow_pickle=False) as archive:
        if set(archive.files)!={'points_xyz_m','frozen_sensor_context','valid'}:
            raise ValueError('native student-only cache required')
        xyz=archive['points_xyz_m'];valid=archive['valid']
    if (xyz.shape!=(1,57600,3) or xyz.dtype!=np.float32 or not np.isfinite(xyz).all()
            or valid.shape!=(1,57600) or valid.dtype!=np.bool_):
        raise ValueError('native compact geometry layout differs')
    indices=np.flatnonzero(valid[0] & (np.linalg.norm(xyz[0].astype(float),axis=1)<=10.))
    points=xyz[0,indices];frames=indices//11520
    voxel=extract_surface_patches(points,np.ones(len(points),bool),frames)
    spg=extract_spg(points,frames,geof_backend=geof_backend,partition_backend=partition_backend)
    if spg['original_point_to_component'] is None:
        raise ValueError('SPG partition unavailable; never substitute another representation: '+spg['status'])
    return encode_partitions(points,indices,
        dict(r1=voxel.point_patch_index,r2=spg['original_point_to_component']),
        source=source,cache_sha256=cache_sha256)


def _validate(points, indices, assignments, header):
    if set(header) != {'schema','source','cache_sha256'} or header['schema']!='development_partitions_v1':
        raise ValueError('closed partition header required')
    source=header['source']
    if (set(source)!={'task','source_sequence_id','frame_rows'} or not isinstance(source['task'],str)
            or not source['task'] or type(source['source_sequence_id']) is not int or source['source_sequence_id']<0):
        raise ValueError('closed observation source required')
    frames=source['frame_rows']
    if (not isinstance(frames,list) or len(frames)!=5 or any(type(i) is not int or i<0 for i in frames)
            or sorted(set(frames))!=frames or not isinstance(header['cache_sha256'],str)
            or re.fullmatch('[0-9a-f]{64}',header['cache_sha256']) is None):
        raise ValueError('ordered frames and cache hash required')
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)>57600 or points.dtype!=np.float32
            or not np.isfinite(points).all() or np.any(np.linalg.norm(points.astype(float),axis=1)>10.)):
        raise ValueError('exact finite float32 ROI points required')
    if (indices.shape!=(len(points),) or indices.dtype.kind not in 'iu'
            or np.any(indices<0) or np.any(indices>=57600)
            or not np.array_equal(indices,np.unique(indices))):
        raise ValueError('ordered unique original ray indices required')
    if set(assignments)!={'r1','r2'}:
        raise ValueError('both explicit voxel and SPG assignments required')
    for assignment in assignments.values():
        if (assignment.shape!=indices.shape or assignment.dtype.kind not in 'iu'
                or np.any(assignment<0) or len(np.unique(assignment))>4096):
            raise ValueError('bounded full-return partition required; never drop points')


def encode_partitions(points, indices, assignments, *, source, cache_sha256):
    header=dict(schema='development_partitions_v1',source=source,cache_sha256=cache_sha256)
    _validate(points,indices,assignments,header)
    stream=io.BytesIO()
    np.savez_compressed(stream,points_xyz_m=points,source_flat_ray_index=indices,
        r1=assignments['r1'],r2=assignments['r2'],
        header=np.frombuffer(json.dumps(header,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8))
    return stream.getvalue()


def decode_partitions(payload, *, expected_sha256, expected_source, expected_cache_sha256,
                      points_xyz_m, source_flat_ray_index):
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('partition payload hash mismatch')
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        if set(archive.files)!={'points_xyz_m','source_flat_ray_index','r1','r2','header'}:
            raise ValueError('closed student-only partition keys required')
        encoded=archive['header']
        if encoded.dtype!=np.uint8 or encoded.ndim!=1 or encoded.size>65536:
            raise ValueError('bounded JSON header bytes required')
        header=json.loads(encoded.tobytes().decode())
        points=archive['points_xyz_m'];indices=archive['source_flat_ray_index']
        assignments={k:archive[k] for k in ('r1','r2')}
    _validate(points,indices,assignments,header)
    if header['source']!=expected_source or header['cache_sha256']!=expected_cache_sha256:
        raise ValueError('partition belongs to another observation/cache')
    if (not np.array_equal(points,points_xyz_m) or not np.array_equal(indices,source_flat_ray_index)
            or points_xyz_m.dtype!=np.float32):
        raise ValueError('partition/cache point or ray mapping differs')
    for value in assignments.values():value.flags.writeable=False
    return assignments
