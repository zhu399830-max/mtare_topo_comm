"""Shared observation bundle for future controlled representation comparisons.

No teacher targets enter this interface. Full scan context, measured surfaces,
and observed local ray segments stay separate. This does not modify or resume
the closed ObservedMembershipV1 fitting configuration.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from .gse_local_ray_segments import LocalRaySegments,clip_observed_rays
from .gse_surface_patches_v1 import SurfacePatches,extract_surface_patches


@dataclass(frozen=True)
class CommonObservation:
    full_sensor_context: torch.Tensor
    full_sensor_valid: torch.Tensor
    registered_returns_xyz_m: np.ndarray
    surface_return_indices: np.ndarray
    surface_patches: SurfacePatches
    local_rays: LocalRaySegments
    ray_sensor_token_indices: np.ndarray
    ray_history_indices: np.ndarray
    coordinate_frame: str = 'current_sensor_m'
    teacher_in_forward: bool = False


def assemble_common_observation(compact, relative_translation_current_sensor_m):
    """Consume the original, un-ROI-masked compact encoder output, batch one.

    The caller authenticates frame/weight provenance. Sensor token indices are
    tensor layout only; not physical places, tunnels, or structure instances.
    No expanded 57600x128 point-context tensor is materialized.
    """
    if not isinstance(compact,CompactFrozenDualPathFeatures):raise ValueError('original compact encoder output required')
    xyz,context,valid,index=compact.points_xyz_m,compact.frozen_sensor_context,compact.valid,compact.sensor_token_index
    if xyz.shape!=(1,57600,3) or context.shape!=(1,900,128) or valid.shape!=(1,57600) or index.shape!=(1,57600):
        raise ValueError('single original five-frame compact layout required')
    if valid.dtype!=torch.bool or index.dtype!=torch.long or xyz.dtype not in (torch.float32,torch.float64) or context.dtype!=xyz.dtype:
        raise ValueError('compact dtype drift')
    if any(x.device!=xyz.device for x in (context,valid,index)):raise ValueError('compact device drift')
    if not bool(torch.isfinite(xyz).all() and torch.isfinite(context).all()):raise ValueError('nonfinite compact observation')
    frames=np.repeat(np.arange(5),16*720)
    expected=np.repeat(np.arange(5),16*720)*180+np.tile(np.arange(720)//4,5*16)
    if not np.array_equal(index[0].detach().cpu().numpy(),expected):raise ValueError('sensor layout drift')
    translation=np.asarray(relative_translation_current_sensor_m)
    if translation.shape!=(5,3) or not np.isfinite(translation).all():raise ValueError('five causal ray origins required')
    points=xyz[0].detach().cpu().numpy().copy()
    mask=valid[0].detach().cpu().numpy().copy()
    origins=np.repeat(translation,16*720,axis=0)
    # Crucially pass full return validity: patch extraction itself crops only
    # surfaces and retains outside-return rays in its evidence representation.
    patches=extract_surface_patches(points,mask,frames,ray_origins_m=origins)
    rays=clip_observed_rays(origins,points,mask)
    surface_indices=np.flatnonzero(mask&(np.linalg.norm(points.astype(np.float64),axis=1)<=10.))
    token_counts=torch.zeros(1,900,device=xyz.device,dtype=torch.long)
    token_counts.scatter_add_(1,index,valid.long())
    ray_tokens=expected[rays.ray_indices].copy();history=frames[rays.ray_indices].copy()
    for a in (points,surface_indices,ray_tokens,history):a.setflags(write=False)
    return CommonObservation(context[0].detach(),token_counts[0]>0,points,surface_indices,
                             patches,rays,ray_tokens,history)


@torch.no_grad()
def prepare_common_observation(adapter, student, *, device):
    """Four-field student window only; no label or teacher candidate argument."""
    rv=np.stack((student.ranges_m/np.float32(50),student.valid_mask.astype(np.float32)),axis=1)
    args=[torch.from_numpy(np.array(a,copy=True))[None].to(device) for a in (
        rv,student.relative_translation_current_sensor_m,student.relative_yaw_current_sensor_deg)]
    compact=adapter.extract_compact_features(*args)
    return assemble_common_observation(compact,student.relative_translation_current_sensor_m)
