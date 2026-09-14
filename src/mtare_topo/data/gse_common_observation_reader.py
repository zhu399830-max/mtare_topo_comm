"""Hash-bound v1 common cache reader. No encoder, teacher or label loading."""
from dataclasses import fields
import hashlib
import io
import zipfile
import numpy as np
import torch
from .gse_membership_fit_reader import _checked
from ..representation.gse_common_observation import CommonObservation
from ..representation.gse_local_ray_segments import LocalRaySegments
from ..representation.gse_surface_patches_v1 import SurfacePatches

RAY_FIELDS=tuple(f.name for f in fields(LocalRaySegments) if f.name not in ('crop_endpoint_is_opening','physical_traversability_qualified'))
PATCH_FIELDS=tuple(f.name for f in fields(SurfacePatches) if f.name not in ('voxel_size_m','roi_radius_m'))
BASE_FIELDS=('registered_returns_xyz_m','surface_return_indices','ray_sensor_token_indices','ray_history_indices','full_sensor_context','full_sensor_valid')
EXPECTED=set(BASE_FIELDS)|{'ray_'+k for k in RAY_FIELDS}|{'patch_'+k for k in PATCH_FIELDS}


def decode_common_observation(payload,*,expected_sha256,device='cpu'):
    if len(payload)>512*1024**2 or hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('cache hash or byte budget drift')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        info=z.infolist()
        if len({v.filename for v in info})!=len(info) or sum(v.file_size for v in info)>512*1024**2:
            raise ValueError('duplicate or oversized cache contents')
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        if set(z.files)!=EXPECTED:raise ValueError('unexpected cache fields; teacher fields forbidden')
        a={k:z[k].copy() for k in z.files}
    if any(v.dtype.kind not in 'bifu' or not np.isfinite(v).all() for v in a.values()):raise ValueError('finite numeric cache arrays required')
    n=len(a['ray_ray_indices']);m=len(a['patch_centers_m']);v=len(a['patch_ray_input_index'])
    shapes={'registered_returns_xyz_m':(57600,3),'full_sensor_context':(900,128),'full_sensor_valid':(900,),
        'ray_ray_indices':(n,),'ray_sensor_token_indices':(n,),'ray_history_indices':(n,),
        'patch_point_patch_index':(57600,),'patch_frame_support':(m,5),
        'patch_ray_input_index':(v,),'patch_ray_frame_index':(v,)}
    shapes.update({'ray_'+k:(n,3) for k in ('start_xyz_m','end_xyz_m','original_return_xyz_m')})
    shapes.update({'ray_'+k:(n,) for k in ('end_is_observed_return','start_distance_from_origin_m','end_distance_from_origin_m')})
    shapes.update({'patch_'+k:(m,3) for k in ('centers_m','normals','bounds_min_m','bounds_max_m')})
    shapes.update({'patch_'+k:(m,) for k in ('normal_valid','normal_uncertainty','roughness_m','point_count')})
    shapes.update({'patch_'+k:(v,3) for k in ('ray_endpoints_m','ray_origins_m')})
    if n>57600 or v>57600 or m>4096 or any(a[k].shape!=shape for k,shape in shapes.items()):raise ValueError('cache shape/capacity drift')
    for k in ('full_sensor_valid','ray_end_is_observed_return','patch_normal_valid'):
        if a[k].dtype!=np.bool_:raise ValueError('boolean mask required')
    for k in ('ray_ray_indices','ray_sensor_token_indices','ray_history_indices','surface_return_indices',
              'patch_point_patch_index','patch_point_count','patch_frame_support','patch_ray_input_index','patch_ray_frame_index'):
        if a[k].dtype.kind not in 'iu':raise ValueError('integer indices/counts required')
    if a['full_sensor_context'].dtype!=np.float32 or a['registered_returns_xyz_m'].dtype!=np.float32:raise ValueError('original float32 cache required')
    for k in ('ray_ray_indices','patch_ray_input_index','surface_return_indices'):
        ids=a[k]
        if ids.ndim!=1 or np.any((ids<0)|(ids>=57600)) or not np.array_equal(ids,np.unique(ids)):raise ValueError('ordered unique ray identity required')
    ids=a['ray_ray_indices'];all_ids=a['patch_ray_input_index'];points=a['registered_returns_xyz_m']
    if not np.isin(ids,all_ids).all():raise ValueError('segment without valid return')
    token=lambda x:(x//11520)*180+(x%720)//4
    if not np.array_equal(a['ray_sensor_token_indices'],token(ids)) or not np.array_equal(a['ray_history_indices'],ids//11520):raise ValueError('segment sensor layout mismatch')
    if not np.array_equal(a['full_sensor_valid'],np.bincount(token(all_ids),minlength=900)>0):raise ValueError('full context mask mismatch')
    if not np.array_equal(a['ray_original_return_xyz_m'],points[ids]) or not np.array_equal(a['patch_ray_endpoints_m'],points[all_ids]):raise ValueError('return coordinates mismatch')
    pi=a['patch_point_patch_index']
    if np.any((pi< -1)|(pi>=m)) or not np.array_equal(a['surface_return_indices'],np.flatnonzero(pi>=0)):raise ValueError('surface index mismatch')
    if not np.isin(a['surface_return_indices'],all_ids).all():raise ValueError('surface without valid return')
    for array in a.values():array.setflags(write=False)
    patches=SurfacePatches(**{k:a['patch_'+k] for k in PATCH_FIELDS},voxel_size_m=.5,roi_radius_m=10.)
    rays=LocalRaySegments(**{k:a['ray_'+k] for k in RAY_FIELDS})
    return CommonObservation(torch.tensor(a['full_sensor_context'],device=device),torch.tensor(a['full_sensor_valid'],device=device),
        points,a['surface_return_indices'],patches,rays,a['ray_sensor_token_indices'],a['ray_history_indices'])


def load_common_observation(root,path,expected_sha256,*,device='cpu'):
    return decode_common_observation(_checked(root,path,expected_sha256),expected_sha256=expected_sha256,device=device)
