"""Teacher-free compact observation to per-patch frozen feature pooling.

Never allocates a full57600x128 context array. Geometry extraction and teacher
purity masks do not select which patches are supplied to the predictor.
"""
import numpy as np
import torch
from .gse_common_observation import CommonObservation


@torch.no_grad()
def pool_patch_observed_features(observation):
    if not isinstance(observation,CommonObservation):raise ValueError('common observation required')
    context=observation.full_sensor_context
    if context.shape!=(900,128) or not torch.isfinite(context).all():raise ValueError('finite compact context required')
    patches=observation.surface_patches;m=len(patches.centers_m)
    mapping=patches.point_patch_index
    if mapping.shape!=(57600,) or mapping.dtype.kind not in 'iu' or np.any(mapping < -1) or np.any(mapping>=m):
        raise ValueError('original full-ray patch indices required')
    selected=np.flatnonzero(mapping>=0)
    if not np.array_equal(selected,observation.surface_return_indices):raise ValueError('surface/patch identity mismatch')
    counts=np.bincount(mapping[selected],minlength=m)
    if not np.array_equal(counts,patches.point_count) or np.any(counts==0):raise ValueError('patch count mismatch')
    # Original five-frame ring/azimuth layout, not teacher or physical identity.
    token=(selected//11520)*180+(selected%720)//4
    valid=observation.full_sensor_valid.detach().cpu()
    if valid.shape!=(900,) or valid.dtype!=torch.bool or not bool(valid[token].all()):
        raise ValueError('surface references missing observed context')
    ctx=context.detach().cpu().double();total=torch.zeros(m,128,dtype=torch.float64)
    for start in range(0,len(selected),1024):
        sl=slice(start,start+1024)
        index=torch.from_numpy(mapping[selected[sl]].astype(np.int64,copy=True))
        total.index_add_(0,index,ctx[token[sl]])
    if m:total/=torch.from_numpy(counts.astype(np.float64))[:,None]
    return total.to(device=context.device,dtype=context.dtype)[None]
